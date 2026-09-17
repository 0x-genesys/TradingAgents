"""Brave Search fallback for company-specific news.

The source is intentionally opt-in. Without ``BRAVE_SEARCH_API_KEY`` it returns
DISABLED, preserving the existing no-fake-data contract.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import requests


_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"


def _query(ticker: str, company_name: str | None) -> str:
    base = (company_name or ticker.split(".", 1)[0]).strip()
    if not base:
        base = ticker
    return f'"{base}" stock India NSE'


def fetch_brave_company_news(
    ticker: str,
    *,
    as_of_date: str,
    company_name: str | None = None,
    limit: int = 10,
    timeout: float = 10.0,
) -> str:
    """Return a compact Brave news digest for one ticker.

    This is used only as a fallback when Yahoo company news is missing. The
    digest is factual search evidence, not sentiment or a trading opinion.
    """
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        return "<Brave company news disabled: BRAVE_SEARCH_API_KEY not set>"

    today = date.today()
    try:
        end = date.fromisoformat(as_of_date)
    except ValueError:
        end = today
    if end > today:
        return (
            f"<Brave company news disabled: as_of_date {as_of_date} is in the future>"
        )
    if end < today - timedelta(days=1):
        return (
            "<Brave company news disabled: historical point-in-time fallback is "
            f"not supported for as_of_date {as_of_date}>"
        )
    start = end - timedelta(days=7)
    params = {
        "q": _query(ticker, company_name),
        "count": max(1, min(int(limit), 20)),
        "country": "in",
        "search_lang": "en",
        "freshness": "pw",
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    try:
        response = requests.get(_ENDPOINT, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        return f"<Brave company news unavailable: HTTP {status}>"
    except Exception as exc:  # noqa: BLE001 - preserve fetch failure as source status.
        return f"<Brave company news unavailable: {type(exc).__name__}>"

    results = payload.get("results") or []
    rows: list[str] = []
    for item in results[:limit]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        source = item.get("source") or {}
        source_name = str(source.get("name") if isinstance(source, dict) else source or "").strip()
        age = str(item.get("age") or "").strip()
        description = str(item.get("description") or "").strip()
        if not title:
            continue
        meta = " · ".join(part for part in (source_name, age) if part)
        rows.append(
            f"- {title}"
            + (f" ({meta})" if meta else "")
            + (f"\n  {description}" if description else "")
            + (f"\n  URL: {url}" if url else "")
        )

    if not rows:
        return (
            f"No Brave company news found for {ticker} between "
            f"{start.isoformat()} and {end.isoformat()}"
        )
    return (
        f"Brave company news fallback for {ticker}; query={params['q']!r}; "
        f"window={start.isoformat()} to {end.isoformat()}\n"
        + "\n".join(rows)
    )
