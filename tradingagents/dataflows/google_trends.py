"""Cached, rate-aware Google Trends enrichment.

Google Trends is optional evidence. A missing or rate-limited response must
never acquire a bullish or bearish meaning. The public ``fetch_google_trends``
wrapper remains string-compatible with older callers; new code should use
``fetch_google_trends_snapshot`` so source status is preserved separately from
the human-readable content.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUEST_DELAY_SECONDS = 5.0
_DEFAULT_COOLDOWN_SECONDS = 120.0
_MAX_STALE_HOURS = 36.0
_last_request_at = 0.0
_circuit_until = 0.0
_consecutive_429 = 0


def _wait_before_request() -> None:
    """Apply the process-wide delay before every Google HTTP request."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _REQUEST_DELAY_SECONDS:
        time.sleep(_REQUEST_DELAY_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance."""
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            info = yf.Ticker(ticker).info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _cache_path(cache_dir: str | Path, ticker: str, as_of_date: str) -> Path:
    return (
        Path(cache_dir)
        / "google_trends"
        / _safe_component(ticker.upper())
        / f"{as_of_date}.json"
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fresh_age_hours(payload: dict[str, Any]) -> float | None:
    fetched_at = payload.get("fetched_at")
    if not fetched_at:
        return None
    try:
        fetched = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0


def _latest_stale(cache_dir: str | Path, ticker: str) -> dict[str, Any] | None:
    directory = Path(cache_dir) / "google_trends" / _safe_component(ticker.upper())
    for path in sorted(directory.glob("*.json"), reverse=True):
        payload = _load_json(path)
        if not payload or payload.get("status") != "OK":
            continue
        age = _fresh_age_hours(payload)
        if age is not None and age <= _MAX_STALE_HOURS:
            stale = dict(payload)
            stale["status"] = "STALE"
            stale["cache_age_hours"] = round(age, 2)
            stale["data_quality_tags"] = ["STALE_GOOGLE_TRENDS"]
            return stale
    return None


def _is_429(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_COOLDOWN_SECONDS
    return min(max(value, 1.0), _DEFAULT_COOLDOWN_SECONDS)


def _unavailable(
    ticker: str,
    query: str,
    as_of_date: str,
    reason: str,
    cache_dir: str | Path,
) -> dict[str, Any]:
    stale = _latest_stale(cache_dir, ticker)
    if stale:
        stale["reason"] = reason
        return stale
    return {
        "source": "google_trends",
        "status": "UNAVAILABLE",
        "ticker": ticker,
        "query": query,
        "as_of_date": as_of_date,
        "fetched_at": None,
        "content": f"<Google Trends unavailable for {ticker}: {reason}>",
        "data_quality_tags": ["MISSING_GOOGLE_TRENDS"],
    }


def fetch_google_trends_snapshot(
    ticker: str,
    *,
    as_of_date: str | None = None,
    company_name: str | None = None,
    cache_dir: str | Path = ".cache",
    refresh: bool = False,
    gprop: str = "",
) -> dict[str, Any]:
    """Return a status-preserving cached Google Trends snapshot.

    There is no synchronous retry sleep on 429. The process opens a cooldown
    circuit, returns stale data when available, and lets later tickers retry
    after the naturally long TradingAgents analysis interval.
    """
    global _circuit_until, _consecutive_429

    as_of = as_of_date or date.today().isoformat()
    query = (company_name or _get_company_name(ticker) or ticker.split(".", 1)[0]).strip()
    path = _cache_path(cache_dir, ticker, as_of)
    if not refresh:
        exact = _load_json(path)
        if exact:
            exact["cache_hit"] = True
            return exact

    if math.isinf(_circuit_until) or time.monotonic() < _circuit_until:
        return _unavailable(ticker, query, as_of, "rate-limit cooldown active", cache_dir)

    try:
        from pytrends.request import TrendReq
    except ImportError:
        return _unavailable(ticker, query, as_of, "pytrends is not installed", cache_dir)

    end = date.fromisoformat(as_of)
    start = end - timedelta(days=7)
    timeframe = f"{start.isoformat()} {end.isoformat()}"

    try:
        client = TrendReq(hl="en-IN", tz=330, timeout=(5, 15), retries=0)
        _wait_before_request()
        client.build_payload(kw_list=[query], timeframe=timeframe, geo="IN", gprop=gprop)
        _wait_before_request()
        frame = client.interest_over_time()
    except Exception as exc:
        if _is_429(exc):
            _consecutive_429 += 1
            _circuit_until = (
                math.inf
                if _consecutive_429 >= 2
                else time.monotonic() + _retry_after_seconds(exc)
            )
            logger.warning("Google Trends rate limited for %s", ticker)
            return _unavailable(ticker, query, as_of, "HTTP 429 rate limit", cache_dir)
        logger.warning("Google Trends fetch failed for %s: %s", ticker, exc)
        return _unavailable(ticker, query, as_of, type(exc).__name__, cache_dir)

    _consecutive_429 = 0
    _circuit_until = 0.0
    now = datetime.now(timezone.utc).isoformat()
    if frame.empty or query not in frame.columns:
        payload = {
            "source": "google_trends",
            "status": "NO_DATA",
            "ticker": ticker,
            "query": query,
            "as_of_date": as_of,
            "fetched_at": now,
            "content": f"<Google Trends: no search-interest data for '{query}'>",
            "data_quality_tags": ["MISSING_GOOGLE_TRENDS"],
        }
        _write_json(path, payload)
        return payload

    values = [int(value) for value in frame[query].tolist()]
    current = values[-1]
    peak = max(values)
    average = sum(values) / len(values)
    minimum = min(values)
    midpoint = len(values) // 2
    first = sum(values[:midpoint]) / midpoint if midpoint else 0.0
    second_count = len(values) - midpoint
    second = sum(values[midpoint:]) / second_count if second_count else 0.0
    if second > first * 1.2:
        direction = "RISING"
    elif second < first * 0.8:
        direction = "FALLING"
    else:
        direction = "STABLE"

    content = "\n".join(
        [
            f'Google Trends search interest for "{query}" ({ticker}) - {timeframe}:',
            f"  Current score: {current}",
            f"  7-day peak: {peak}",
            f"  7-day average: {average:.1f}",
            f"  7-day low: {minimum}",
            f"  Direction: {direction}",
            "  Scale: 0-100 within this query and time window.",
        ]
    )
    payload = {
        "source": "google_trends",
        "status": "OK",
        "ticker": ticker,
        "query": query,
        "as_of_date": as_of,
        "fetched_at": now,
        "metrics": {
            "current": current,
            "peak": peak,
            "average": round(average, 2),
            "low": minimum,
            "direction": direction,
        },
        "content": content,
        "data_quality_tags": [],
    }
    _write_json(path, payload)
    return payload


def fetch_google_trends(
    ticker: str,
    timeframe: str = "now 7-d",
    gprop: str = "",
) -> str:
    """Backward-compatible string interface."""
    del timeframe
    snapshot = fetch_google_trends_snapshot(ticker, gprop=gprop)
    return str(snapshot["content"])
