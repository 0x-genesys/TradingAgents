"""Tavily Search fallback for company-specific news.

The source is intentionally opt-in. Without ``TAVILY_API_KEY`` it returns
DISABLED, preserving the existing no-fake-data contract.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import requests


_ENDPOINT = "https://api.tavily.com/search"


def _query(ticker: str, company_name: str | None) -> str:
    base = (company_name or ticker.split(".", 1)[0]).strip()
    if not base:
        base = ticker
    return f'"{base}" stock India NSE latest company news'


def fetch_tavily_company_news(
    ticker: str,
    *,
    as_of_date: str,
    company_name: str | None = None,
    limit: int = 10,
    timeout: float = 10.0,
) -> str:
    """Return a compact Tavily search digest for one ticker.

    This is used only as a fallback when Yahoo company news is missing. The
    digest is factual search evidence, not sentiment or a trading opinion.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "<Tavily company news disabled: TAVILY_API_KEY not set>"

    today = date.today()
    try:
        end = date.fromisoformat(as_of_date)
    except ValueError:
        end = today
    if end > today:
        return (
            f"<Tavily company news disabled: as_of_date {as_of_date} is in the future>"
        )
    if end < today - timedelta(days=1):
        return (
            "<Tavily company news disabled: historical point-in-time fallback is "
            f"not supported for as_of_date {as_of_date}>"
        )
    start = end - timedelta(days=7)
    body = {
        "query": _query(ticker, company_name),
        "auto_parameters": False,
        "topic": "news",
        "search_depth": "basic",
        "max_results": max(1, min(int(limit), 20)),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_favicon": False,
        "include_usage": False,
        "country": "india",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(_ENDPOINT, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        return f"<Tavily company news unavailable: HTTP {status}>"
    except Exception as exc:  # noqa: BLE001 - preserve fetch failure as source status.
        return f"<Tavily company news unavailable: {type(exc).__name__}>"

    results = payload.get("results") or []
    rows: list[str] = []
    for item in results[:limit]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        source = str(item.get("source") or "").strip()
        published_date = str(item.get("published_date") or "").strip()
        content = str(item.get("content") or "").strip()
        if not title:
            continue
        meta = " · ".join(part for part in (source, published_date) if part)
        rows.append(
            f"- {title}"
            + (f" ({meta})" if meta else "")
            + (f"\n  {content}" if content else "")
            + (f"\n  URL: {url}" if url else "")
        )

    if not rows:
        return (
            f"No Tavily company news found for {ticker} between "
            f"{start.isoformat()} and {end.isoformat()}"
        )
    return (
        f"Tavily company news fallback for {ticker}; query={body['query']!r}; "
        f"window={start.isoformat()} to {end.isoformat()}\n"
        + "\n".join(rows)
    )
