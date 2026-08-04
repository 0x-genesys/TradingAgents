"""StockTwits fetcher with a Google News RSS fallback for Indian tickers.

StockTwits covers US stocks at ``api.stocktwits.com/api/2/streams/symbol/{ticker}.json``.
Indian stocks (.NS / .BO) are not covered on StockTwits; ``fetch_stocktwits_messages``
falls back to Google News RSS headlines plus a simple headline-sentiment scan.
The Google News path is also exposed directly as ``fetch_google_news_headlines``
for the frozen source snapshot consumed by the sentiment and news analysts.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
import contextlib
import io
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_NEWS_API = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Simple positive/negative keyword lists for headline sentiment scoring
_POSITIVE_WORDS = {"buy", "bullish", "surge", "rally", "gain", "profit", "upgrade",
                   "positive", "growth", "breakout", "strong", "record", "outperform",
                   "beat", "raised", "up", "green", "higher", "boost", "soar"}
_NEGATIVE_WORDS = {"sell", "bearish", "crash", "plunge", "loss", "downgrade", "negative",
                   "decline", "weak", "cut", "lower", "red", "drop", "fall", "slump",
                   "warning", "risk", "down", "miss", "debt", "probe", "investigation"}
_LEGAL_NAME_WORDS = {
    "co",
    "company",
    "corporation",
    "inc",
    "india",
    "limited",
    "ltd",
    "plc",
}
_AMBIGUOUS_COMPANY_FOLLOWERS = {"agrotech", "cement", "paper", "realty"}


def _score_headline_sentiment(headline: str) -> str:
    """Classify a headline as Positive, Negative, or Neutral using keyword matching."""
    lower = headline.lower()
    words = set(re.findall(r"[a-z]+", lower))
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    if pos_count > neg_count:
        return "Positive"
    if neg_count > pos_count:
        return "Negative"
    return "Neutral"


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance (short timeout)."""
    try:
        import yfinance as yf
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            t = yf.Ticker(ticker)
            info = t.info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _headline_matches_company(title: str, ticker: str, company_name: str) -> bool:
    """Reject search-result contamination from similarly named companies."""
    normalized_title = " ".join(re.findall(r"[a-z0-9]+", title.lower()))
    compact_title = normalized_title.replace(" ", "")
    base = ticker.upper().replace(".NS", "").replace(".BO", "").lower()
    if base and base in compact_title:
        return True

    name_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", company_name.lower())
        if token not in _LEGAL_NAME_WORDS
    ]
    if not name_tokens:
        return False
    if len(name_tokens) == 1:
        token = name_tokens[0]
        title_tokens = normalized_title.split()
        if token not in title_tokens:
            return False
        token_index = title_tokens.index(token)
        if (
            token_index + 1 < len(title_tokens)
            and title_tokens[token_index + 1] in _AMBIGUOUS_COMPANY_FOLLOWERS
        ):
            return False
        corporate_match = re.search(
            rf"\b{re.escape(token)}\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,2}})\s+"
            r"(?:limited|ltd)\b",
            normalized_title,
        )
        if corporate_match and corporate_match.group(1) not in _LEGAL_NAME_WORDS:
            return False
        return len(token) >= 4
    phrase_length = min(3, len(name_tokens)) if len(name_tokens) > 1 else 1
    company_phrase = " ".join(name_tokens[:phrase_length])
    return len(company_phrase) >= 4 and company_phrase in normalized_title


def _fetch_google_news(
    ticker: str,
    limit: int = 10,
    timeout: float = 10.0,
    *,
    as_of_date: str | None = None,
    company_name: str | None = None,
) -> str:
    """Fetch recent Google News headlines for an Indian stock ticker.

    Uses the company name (looked up via yfinance) for a better search query
    since news articles use company names rather than ticker symbols.
    Returns formatted headlines with a summary sentiment distribution.
    """
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    # Try company name first, fall back to ticker symbol
    resolved_name = company_name or _get_company_name(ticker, timeout=min(timeout, 5.0))
    search_term = resolved_name if resolved_name and len(resolved_name) > 3 else base
    logger.info("Google News: searching for %s using '%s' (from ticker %s)", ticker, search_term, base)
    end_date = datetime.strptime(as_of_date, "%Y-%m-%d").date() if as_of_date else date.today()
    start_date = end_date - timedelta(days=7)
    query = quote(
        f"{search_term} stock after:{start_date.isoformat()} "
        f"before:{(end_date + timedelta(days=1)).isoformat()}"
    )
    url = _NEWS_API.format(query=query)
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("Google News fetch failed for %s: %s", ticker, exc)
        return f"<news headlines unavailable for {ticker}: {type(exc).__name__}>"

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return f"<news headlines unavailable for {ticker}: parse error>"

    # Try Atom namespace, then fall back to plain RSS items
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall(".//atom:entry", ns)
    if not entries:
        entries = root.findall(".//item")

    headlines = []
    seen_titles = set()
    for entry in entries[:limit]:
        title_el = entry.find("title")
        if title_el is None:
            title_el = entry.find("atom:title", ns)
        source_el = entry.find("source")
        if source_el is None:
            source_el = entry.find("atom:source", ns)
        if title_el is None or not title_el.text:
            continue
        title_raw = (title_el.text or "").strip().replace("\n", " ").replace("’", "'")
        if not title_raw:
            continue
        title = title_raw
        if not _headline_matches_company(title, ticker, resolved_name):
            logger.info("Google News: discarded off-company headline for %s: %s", ticker, title)
            continue
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        source = ""
        if source_el is not None:
            src_name = source_el.find("name") or source_el.find("atom:name", ns)
            if src_name is not None and src_name.text:
                source = src_name.text.strip()
        published_el = entry.find("pubDate")
        if published_el is None:
            published_el = entry.find("atom:updated", ns)
        if published_el is not None and published_el.text:
            try:
                published = parsedate_to_datetime(published_el.text).date()
            except (TypeError, ValueError):
                published = None
            if published and not (start_date <= published <= end_date):
                continue
        sentiment = _score_headline_sentiment(title)
        headlines.append((title, source, sentiment))

    if not headlines:
        return f"<no news headlines found for {base} in the past week>"

    total = len(headlines)
    pos = sum(1 for _, _, s in headlines if s == "Positive")
    neg = sum(1 for _, _, s in headlines if s == "Negative")
    neu = total - pos - neg
    pos_pct = round(100 * pos / total) if total else 0
    neg_pct = round(100 * neg / total) if total else 0

    summary = (
        f"Google News headlines for {base} — "
        f"Positive: {pos} ({pos_pct}%) · "
        f"Negative: {neg} ({neg_pct}%) · "
        f"Neutral: {neu} · "
        f"Total: {total} recent headlines"
    )
    lines = [summary, ""]
    for title, source, sentiment in headlines:
        tag = "🟢" if sentiment == "Positive" else ("🔴" if sentiment == "Negative" else "⚪")
        src = f" · {source}" if source else ""
        lines.append(f"{tag} [{sentiment}]{src} {title}")

    return "\n".join(lines)


def fetch_google_news_headlines(
    ticker: str,
    limit: int = 10,
    timeout: float = 10.0,
    *,
    as_of_date: str | None = None,
    company_name: str | None = None,
) -> str:
    """Fetch recent Google News RSS headlines for ``ticker`` in any market.

    StockTwits does not cover Indian tickers; callers that need a Google News
    source for every ticker (US or Indian) use this instead of the StockTwits
    fallback so the content always matches the source label.
    """
    return _fetch_google_news(
        ticker,
        limit=limit,
        timeout=timeout,
        as_of_date=as_of_date,
        company_name=company_name,
    )


def fetch_stocktwits_messages(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
    *,
    as_of_date: str | None = None,
    company_name: str | None = None,
) -> str:
    """Fetch recent StockTwits messages for ``ticker`` and return them as a
    formatted plaintext block ready for prompt injection.

    StockTwits is US-centric and does not cover NSE/BSE stocks. For Indian
    tickers (.NS / .BO) we automatically fall back to Google News RSS
    headlines with a simple sentiment keyword scan — giving the sentiment
    analyst something actionable instead of a data vacuum.

    Returns a placeholder string on any failure — the caller never has to
    special-case None or exceptions.
    """
    upper = ticker.upper()
    # NSE/BSE stocks: fall back to Indian news headlines
    if upper.endswith(".NS") or upper.endswith(".BO"):
        logger.info("StockTwits skipped %s: using Google News fallback for Indian stocks", upper)
        return _fetch_google_news(
            upper,
            timeout=timeout,
            as_of_date=as_of_date,
            company_name=company_name,
        )

    url = _API.format(ticker=upper)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return f"<stocktwits unavailable: {type(exc).__name__}>"

    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages[:limit]:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = (m.get("body") or "").replace("\n", " ").strip()
        if len(body) > 280:
            body = body[:280] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · "
        f"Bearish: {bearish} ({bear_pct}%) · "
        f"Unlabeled: {unlabeled} · "
        f"Total: {total} most-recent messages"
    )
    return summary + "\n\n" + "\n".join(lines)
