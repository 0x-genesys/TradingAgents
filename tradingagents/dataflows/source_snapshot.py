"""Frozen, status-preserving source inputs for one TradingAgents candidate."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from tradingagents.dataflows.google_trends import fetch_google_trends_snapshot
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.telegram import fetch_telegram_messages
from tradingagents.dataflows.yfinance_news import get_news_yfinance


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _snapshot_path(cache_dir: str | Path, ticker: str, trade_date: str) -> Path:
    return (
        Path(cache_dir)
        / "source_snapshots"
        / "v1"
        / trade_date
        / f"{_safe_component(ticker.upper())}.json"
    )


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _is_missing(content: str) -> bool:
    lowered = content.strip().lower()
    return not lowered or lowered.startswith("<") or lowered.startswith("no news found")


def _record(source: str, fetch: Callable[[], str]) -> dict[str, Any]:
    try:
        content = str(fetch())
    except Exception as exc:
        return {
            "source": source,
            "status": "UNAVAILABLE",
            "content": f"<{source} unavailable: {type(exc).__name__}>",
        }
    lowered = content.strip().lower()
    if lowered.startswith("error fetching"):
        return {"source": source, "status": "UNAVAILABLE", "content": content}
    if _is_missing(content):
        status = "DISABLED" if "disabled" in content.lower() else "NO_DATA"
        if "unavailable" in content.lower() or "error fetching" in content.lower():
            status = "UNAVAILABLE"
        return {"source": source, "status": status, "content": content}
    return {"source": source, "status": "OK", "content": content}


def _yfinance_profile(ticker: str) -> tuple[str, dict[str, Any]]:
    import yfinance as yf

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        info = yf.Ticker(ticker).info or {}
    company_name = (info.get("longName") or info.get("shortName") or "").strip()
    fields = {
        "company_name": company_name,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }
    available = {key: value for key, value in fields.items() if value not in (None, "")}
    return company_name, available


def build_source_snapshot(
    ticker: str,
    trade_date: str,
    *,
    cache_dir: str | Path,
    refresh: bool = False,
) -> dict[str, Any]:
    """Fetch once per ticker and persist the exact inputs shared by all LLMs."""
    path = _snapshot_path(cache_dir, ticker, trade_date)
    if not refresh:
        cached = _load(path)
        if cached:
            cached["cache_hit"] = True
            return cached

    try:
        company_name, profile = _yfinance_profile(ticker)
    except Exception as exc:
        company_name, profile = "", {}
        profile_record = {
            "source": "yfinance_profile",
            "status": "UNAVAILABLE",
            "content": f"<Yahoo Finance profile unavailable: {type(exc).__name__}>",
        }
    else:
        profile_record = {
            "source": "yfinance_profile",
            "status": "OK" if profile else "NO_DATA",
            "content": json.dumps(profile, ensure_ascii=False, sort_keys=True),
            "metrics": profile,
        }

    end = datetime.strptime(trade_date, "%Y-%m-%d")
    start_date = (end - timedelta(days=7)).strftime("%Y-%m-%d")
    sources: dict[str, dict[str, Any]] = {
        "yfinance_profile": profile_record,
        "company_news": _record(
            "company_news",
            lambda: get_news_yfinance(ticker, start_date, trade_date),
        ),
        "google_news": _record(
            "google_news",
            lambda: fetch_stocktwits_messages(
                ticker,
                limit=30,
                as_of_date=trade_date,
                company_name=company_name,
            ),
        ),
        "reddit": _record(
            "reddit",
            lambda: fetch_reddit_posts(ticker, as_of_date=trade_date),
        ),
        "telegram": _record(
            "telegram",
            lambda: fetch_telegram_messages(
                ticker,
                as_of_date=trade_date,
                company_name=company_name,
            ),
        ),
    }
    trends = fetch_google_trends_snapshot(
        ticker,
        as_of_date=trade_date,
        company_name=company_name,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    sources["google_trends"] = trends

    tags: list[str] = []
    tag_by_source = {
        "company_news": "MISSING_COMPANY_NEWS",
        "google_news": "MISSING_GOOGLE_NEWS",
        "reddit": "MISSING_REDDIT",
        "telegram": "MISSING_TELEGRAM",
        "google_trends": "MISSING_GOOGLE_TRENDS",
    }
    for source, tag in tag_by_source.items():
        status = sources[source].get("status")
        if status == "STALE":
            tags.append("STALE_GOOGLE_TRENDS")
        elif status != "OK":
            tags.append(tag)

    sentiment_available = any(
        sources[name].get("status") in {"OK", "STALE"}
        for name in ("company_news", "google_news", "reddit", "telegram", "google_trends")
    )
    if not sentiment_available:
        tags.append("MISSING_SENTIMENT")

    primary_available = any(
        sources[name].get("status") == "OK"
        for name in ("yfinance_profile", "company_news", "google_news")
    )
    payload = {
        "schema_version": 1,
        "ticker": ticker,
        "trade_date": trade_date,
        "company_name": company_name,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
        "primary_data_available": primary_available,
        "analysis_status": "COMPLETE" if primary_available else "FAILED",
        "data_quality_tags": sorted(set(tags)),
        "sources": sources,
    }
    _save(path, payload)
    return payload


def refresh_requested() -> bool:
    return os.environ.get("TRADINGAGENTS_REFRESH_SOURCE_SNAPSHOT", "").lower() in {
        "1",
        "true",
        "yes",
    }
