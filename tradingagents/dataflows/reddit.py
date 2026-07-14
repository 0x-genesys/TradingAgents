"""Reddit data fetcher for ticker-specific discussion posts.

Primary: OAuth2 via registered Reddit app (set REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET in your .env). Register at https://www.reddit.com/prefs/apps

Fallback: HTML scraping via old.reddit.com (no API key needed).

Searches by company name (not ticker) across Indian-focused subreddits for
better results with NSE/BSE stocks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import contextlib
import io
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_UA = "tradingagents/1.0 (by /u/slyflyfox)"

# OAuth endpoints
_OAUTH_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH_SEARCH_API = "https://oauth.reddit.com/r/{sub}/search?{qs}"

# HTML scraping endpoint (no auth required)
_HTML_SEARCH_URL = "https://old.reddit.com/r/{sub}/search?{qs}"

# Indian-focused subreddits for stock discussion
INDIAN_SUBREDDITS = ("IndiaInvestments", "IndianStreetBets", "IndianStockMarket")
# Also search general finance subs for broader coverage
GENERAL_SUBREDDITS = ("wallstreetbets", "stocks")

# Cached access token
_access_token: Optional[str] = None
_token_expiry: float = 0.0


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance."""
    try:
        import yfinance as yf
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            t = yf.Ticker(ticker)
            info = t.info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _search_query(ticker: str) -> str:
    """Build a Reddit search query using company name or ticker.

    Uses company name (with quotes for multi-word names) when available,
    falls back to the ticker symbol.
    """
    company = _get_company_name(ticker, timeout=3.0)
    if company and len(company) > 3:
        # Quote multi-word company names for exact match
        if " " in company:
            return f'"{company}" stock'
        return f"{company} stock"
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    return base


def _get_access_token() -> Optional[str]:
    """Obtain Reddit OAuth token via client credentials grant."""
    global _access_token, _token_expiry
    if _access_token and time.time() < _token_expiry:
        return _access_token

    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None

    import base64
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = Request(
        _OAUTH_TOKEN_URL,
        data=b"grant_type=client_credentials&scope=read",
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": _UA,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        _access_token = body.get("access_token")
        expires_in = body.get("expires_in", 3600)
        _token_expiry = time.time() + expires_in - 60
        return _access_token
    except Exception as exc:
        logger.warning("Reddit OAuth token fetch failed: %s", exc)
        return None


def _fetch_via_oauth(query: str, sub: str, limit: int, timeout: float) -> list[dict]:
    """Fetch posts via Reddit OAuth API."""
    token = _get_access_token()
    if token is None:
        return []
    qs = urlencode({"q": query, "restrict_sr": "on", "sort": "new", "t": "week", "limit": limit})
    url = _OAUTH_SEARCH_API.format(sub=sub, qs=qs)
    req = Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        logger.warning("Reddit OAuth fetch failed for r/%s: %s", sub, exc)
        return []
    children = (payload.get("data") or {}).get("children") or []
    return [c.get("data", {}) for c in children if isinstance(c, dict)]


def _fetch_via_html(query: str, sub: str, limit: int, timeout: float) -> list[dict]:
    """Fetch posts by scraping old.reddit.com HTML search results."""
    encoded_query = quote(query)
    qs = urlencode({"q": query, "restrict_sr": "on", "sort": "new", "t": "week", "limit": limit})
    url = _HTML_SEARCH_URL.format(sub=sub, qs=qs)
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Reddit HTML fetch failed for r/%s: %s", sub, exc)
        return []

    posts = []
    # old.reddit.com uses <div class="search-result ..."> with link+title inside
    # The search-title class links contain the actual post title
    title_links = re.findall(r'<a\s+class="search-title[^"]*"[^>]*>\s*(.*?)\s*</a>', html, re.DOTALL)
    for title_html in title_links[:limit]:
        title = re.sub(r'<[^>]+>', '', title_html).strip()
        if not title:
            continue
        posts.append({
            "title": title,
            "score": 0,
            "num_comments": 0,
            "created_str": "?",
            "selftext": "",
        })
    if not posts:
        # Fallback: look for normal link listings (search results page without search-result class structure)
        result_blocks = re.findall(r'<div class="search-result[^"]*".*?>(.*?)</div>\s*</div>', html, re.DOTALL)
        for block in result_blocks[:limit]:
            title_m = re.search(r'<a\s+class="search-title[^"]*"[^>]*>\s*(.*?)\s*</a>', block, re.DOTALL)
            if not title_m:
                continue
            title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()
            if not title:
                continue
            score_m = re.search(r'<span class="search-score">(\d+)\s+point', block)
            comments_m = re.search(r'(\d+)\s+comment', block)
            posts.append({
                "title": title,
                "score": int(score_m.group(1)) if score_m else 0,
                "num_comments": int(comments_m.group(1)) if comments_m else 0,
                "created_str": "?",
                "selftext": "",
            })
    return posts


def _fetch_subreddit(query: str, sub: str, limit: int, timeout: float) -> list[dict]:
    """Try OAuth first, fall back to HTML scraping."""
    oauth_available = _get_access_token() is not None
    if oauth_available:
        posts = _fetch_via_oauth(query, sub, limit, timeout)
        if posts:
            return posts
    return _fetch_via_html(query, sub, limit, timeout)


def fetch_reddit_posts(
    ticker: str,
    subreddits: Optional[Iterable[str]] = None,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``ticker`` across finance
    subreddits and return them as a formatted plaintext block.

    Uses OAuth if REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are set in the
    environment, otherwise falls back to HTML scraping of old.reddit.com.

    Searches by company name (e.g., "Indian Bank" not "INDIANB") for better
    results with Indian NSE/BSE stocks.
    """
    if subreddits is None:
        subreddits = INDIAN_SUBREDDITS + GENERAL_SUBREDDITS

    query = _search_query(ticker)
    oauth_available = _get_access_token() is not None
    method = "OAuth" if oauth_available else "HTML scrape (old.reddit.com)"

    blocks = []
    total_posts = 0

    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_subreddit(query, sub, limit_per_sub, timeout)
        if not posts:
            logger.info("Reddit: no posts found in r/%s for query '%s'", sub, query)
            continue
        total_posts += len(posts)
        lines = [f"r/{sub} — {len(posts)} recent posts (via {method}, query: \"{query}\"):"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score", 0)
            comments = p.get("num_comments", 0)
            created = p.get("created_str") or p.get("created_utc", "?")
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(
                f"  [{created} · {score:>4}↑ · {comments:>3}c] {title}"
                + (f"\n    body excerpt: {selftext}" if selftext else "")
            )
        blocks.append("\n".join(lines))

    if total_posts == 0:
        sub_list = ", ".join(subreddits)
        return (
            f"<no Reddit posts found for '{query}' across {sub_list} "
            f"in the past 7 days (via {method})>"
        )
    return "\n\n".join(blocks)
