"""Google Trends data fetcher for retail-attention signal.

Uses ``pytrends`` (unofficial Google Trends API, no API key needed) to
fetch the search-interest score for a stock's company name over the past
7 days. A rising search-interest score suggests growing retail attention
and can serve as a leading indicator for momentum-driven moves.

Usage:
    from tradingagents.dataflows.google_trends import fetch_google_trends
    trends = fetch_google_trends(ticker, ticker)
    # trends is a formatted str (always safe to print/inject)
"""

from __future__ import annotations

import logging
import time
import contextlib
import io
from typing import Optional

logger = logging.getLogger(__name__)

# Min requests between API calls to avoid rate limiting
_REQUEST_DELAY = 5.0
_last_req: float = 0.0


def _rate_limit():
    """Enforce a minimum delay between Google Trends API requests."""
    global _last_req
    now = time.monotonic()
    elapsed = now - _last_req
    if elapsed < _REQUEST_DELAY:
        time.sleep(_REQUEST_DELAY - elapsed)
    _last_req = time.monotonic()


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance (short timeout)."""
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            t = yf.Ticker(ticker)
            info = t.info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def fetch_google_trends(
    ticker: str,
    timeframe: str = "now 7-d",
    gprop: str = "",
) -> str:
    """Fetch Google Trends search-interest data for the company behind ``ticker``.

    Parameters
    ----------
    ticker:
        The ticker symbol (e.g. ``NTPC.NS``). The company name is looked up
        automatically via yfinance and used as the search keyword.
    timeframe:
        Google Trends timeframe string. Default: "now 7-d" (past 7 days).
        Other options: "now 1-d", "today 1-m", "today 3-m", "today 12-m".
    gprop:
        Optional Google property filter. Default: ``""`` (web search).
        Use ``"news"`` for news-search interest, ``"youtube"`` for YouTube.

    Returns
    -------
    str
        A human-readable formatted text block with interest scores and
        trending-queries context. Never None — degrades gracefully.
    """
    company_name = _get_company_name(ticker)
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    kw = company_name if company_name and len(company_name) > 2 else base

    try:
        from pytrends.request import TrendReq
    except ImportError:
        return "<Google Trends unavailable: install pytrends>"

    try:
        _rate_limit()
        pytrends = TrendReq(hl="en-IN", tz=330)  # India timezone
        pytrends.build_payload(kw_list=[kw], timeframe=timeframe, gprop=gprop)

        # --- Interest over time ---
        interest_df = pytrends.interest_over_time()
        if interest_df.empty:
            return (
                f"<Google Trends: no search-interest data found for "
                f"'{kw}' (from {ticker}) in the past 7 days>"
            )

        # Drop the "isPartial" column if present
        if "isPartial" in interest_df.columns:
            interest_df = interest_df.drop(columns=["isPartial"])

        # --- Compute stats ---
        values = interest_df[kw].tolist()
        current = values[-1] if values else 0
        peak = max(values) if values else 0
        avg = sum(values) / len(values) if values else 0
        min_val = min(values) if values else 0

        # Trend direction: compare first half vs second half
        mid = len(values) // 2
        first_half = sum(values[:mid]) / mid if mid > 0 else 0
        second_half = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0

        if second_half > first_half * 1.2:
            direction = "RISING"
            direction_desc = "Search interest is trending UP over the period"
        elif second_half < first_half * 0.8:
            direction = "FALLING"
            direction_desc = "Search interest is trending DOWN over the period"
        else:
            direction = "STABLE"
            direction_desc = "Search interest is relatively flat over the period"

        # --- Interest by region (top 5) ---
        region_lines: list[str] = []
        try:
            regional_df = pytrends.interest_by_region(resolution="COUNTRY", inc_low_vol=True)
            if not regional_df.empty and kw in regional_df.columns:
                regional_df = regional_df.sort_values(kw, ascending=False).head(5)
                for country, row in regional_df.iterrows():
                    region_lines.append(f"  {country}: {int(row[kw])}")
        except Exception:
            region_lines.append("  (regional breakdown unavailable)")

        # --- Related queries (top rising + top) ---
        rising_lines: list[str] = []
        top_lines: list[str] = []
        try:
            related = pytrends.related_queries()
            if related and kw in related:
                rq = related[kw]
                rising_q = rq.get("rising", None)
                top_q = rq.get("top", None)
                if rising_q is not None and not rising_q.empty:
                    for _, row in rising_q.head(5).iterrows():
                        query = row.get("query", "")
                        value = row.get("value", "")
                        rising_lines.append(f"  {query} ({value})")
                if top_q is not None and not top_q.empty:
                    for _, row in top_q.head(5).iterrows():
                        query = row.get("query", "")
                        value = row.get("value", "")
                        top_lines.append(f"  {query} ({value})")
        except Exception:
            pass

        # --- Build output ---
        parts = [
            f"Google Trends search interest for \"{kw}\" (from {ticker}) — {timeframe}:",
            f"  Current score: {current}",
            f"  7-day peak:    {peak}",
            f"  7-day avg:     {avg:.0f}",
            f"  7-day low:     {min_val}",
            f"  Direction:     {direction} — {direction_desc}",
        ]

        if region_lines:
            parts.append(f"\n  Interest by country (top 5):")
            parts.extend(region_lines)

        if rising_lines:
            parts.append(f"\n  Rising related queries:")
            parts.extend(rising_lines)

        if top_lines:
            parts.append(f"\n  Top related queries:")
            parts.extend(top_lines)

        # --- Score interpretation ---
        if current >= 75:
            note = (
                "  ⚠ High attention: search interest ≥ 75. "
                "Elevated retail attention — may indicate FOMO or panic."
            )
        elif current >= 50:
            note = (
                "  📈 Moderate-high attention: search interest 50–74. "
                "Active retail interest."
            )
        elif current >= 25:
            note = (
                "  📊 Moderate attention: search interest 25–49. "
                "Some retail awareness."
            )
        else:
            note = (
                "  📉 Low attention: search interest < 25. "
                "Limited retail awareness — trade may be purely institutional."
            )
        parts.append(f"\n  {note}")

        parts.append(
            f"\n  Data source: Google Trends (pytrends). "
            f"Scale is 0–100 (100 = peak popularity for the term in this period)."
        )

        return "\n".join(parts)

    except Exception as exc:
        logger.warning("Google Trends fetch failed for %s: %s", ticker, exc)
        return f"<Google Trends unavailable for {ticker}: {type(exc).__name__}: {exc}>"
