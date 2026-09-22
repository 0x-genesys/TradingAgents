"""Validated local OHLCV cache fallback for TradingAgents market tools.

The daily LSTM pipeline repairs Yahoo gaps with approved NSE sources before
selection. When the orchestrator supplies that cache to TradingAgents, market
tools may use it as a fallback if Yahoo is stale, blank, or unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def configured_cache_dir() -> Path | None:
    value = os.environ.get("TRADINGAGENTS_LOCAL_OHLC_CACHE_DIR", "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def configured_required_session() -> pd.Timestamp | None:
    value = os.environ.get("TRADINGAGENTS_REQUIRED_SESSION", "").strip()
    if not value:
        return None
    return pd.Timestamp(value).normalize()


def has_complete_session(data: pd.DataFrame, session: pd.Timestamp | None) -> bool:
    if session is None:
        return True
    if data is None or data.empty:
        return False

    frame = data.copy()
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    else:
        dates = pd.to_datetime(frame.index, errors="coerce").normalize()

    matching = frame.loc[dates == session]
    if matching.empty:
        return False
    required_columns = [column for column in PRICE_COLUMNS if column in matching.columns]
    if len(required_columns) != len(PRICE_COLUMNS):
        return False
    values = matching.iloc[0][required_columns]
    numeric = pd.to_numeric(values, errors="coerce")
    return not numeric.isna().any()


def yahoo_frame_needs_fallback(data: pd.DataFrame, required_session: pd.Timestamp | None) -> bool:
    if data is None or data.empty:
        return True
    return not has_complete_session(data, required_session)


def _cache_file_for_symbol(symbol: str) -> Path | None:
    cache_dir = configured_cache_dir()
    if cache_dir is None:
        return None
    return cache_dir / f"{symbol.upper()}_latest.csv"


def load_local_ohlcv(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    curr_date: str | None = None,
    required_session: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    cache_file = _cache_file_for_symbol(symbol)
    if cache_file is None or not cache_file.exists():
        return None

    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    data.index.name = "Date"
    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    for column in PRICE_COLUMNS:
        if column not in data.columns:
            return None
        data[column] = pd.to_numeric(data[column], errors="coerce")

    required_session = required_session or configured_required_session()
    if not has_complete_session(data, required_session):
        return None

    if start_date:
        data = data[data["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        # Match yfinance history semantics: end date is exclusive.
        data = data[data["Date"] < pd.Timestamp(end_date)]
    if curr_date:
        data = data[data["Date"] <= pd.Timestamp(curr_date)]

    if data.empty:
        return None

    return data[["Date", *PRICE_COLUMNS]].copy()


def local_cache_source_note(required_session: pd.Timestamp | None = None) -> str:
    session = required_session or configured_required_session()
    if session is None:
        return "# Data source fallback: validated local OHLC cache\n"
    return (
        "# Data source fallback: validated local OHLC cache "
        f"(required_session={session.date()})\n"
    )
