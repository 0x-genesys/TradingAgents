from pathlib import Path

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.y_finance import get_YFin_data_online


def _write_local_cache(cache_dir: Path, symbol: str = "OFSS.NS") -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(
        [
            {
                "Open": 11763.0,
                "High": 11763.0,
                "Low": 11763.0,
                "Close": 11763.0,
                "Volume": 0,
            },
            {
                "Open": 10911.0,
                "High": 10911.0,
                "Low": 10911.0,
                "Close": 10911.0,
                "Volume": 565241,
            },
        ],
        index=[pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-21")],
    )
    data.index.name = "Date"
    data.to_csv(cache_dir / f"{symbol}_latest.csv")


def test_stock_data_falls_back_to_validated_local_cache_when_yahoo_is_blank(
    tmp_path,
    monkeypatch,
) -> None:
    _write_local_cache(tmp_path)
    monkeypatch.setenv("TRADINGAGENTS_LOCAL_OHLC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRADINGAGENTS_REQUIRED_SESSION", "2026-09-21")

    yahoo = pd.DataFrame(
        {
            "Open": [None],
            "High": [None],
            "Low": [None],
            "Close": [None],
            "Volume": [565241],
        },
        index=[pd.Timestamp("2026-09-21")],
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, start, end):
            return yahoo

    monkeypatch.setattr("tradingagents.dataflows.y_finance.yf.Ticker", FakeTicker)

    result = get_YFin_data_online("OFSS.NS", "2026-09-18", "2026-09-22")

    assert "Data source fallback: validated local OHLC cache" in result
    assert "2026-09-21,10911.0,10911.0,10911.0,10911.0,565241" in result


def test_indicator_loader_falls_back_to_local_cache_when_yahoo_cache_is_stale(
    tmp_path,
    monkeypatch,
) -> None:
    local_cache = tmp_path / "local"
    ta_cache = tmp_path / "ta"
    _write_local_cache(local_cache)
    ta_cache.mkdir()
    stale = pd.DataFrame(
        [
            {
                "Date": "2026-09-18",
                "Open": 11763.0,
                "High": 11763.0,
                "Low": 11763.0,
                "Close": 11763.0,
                "Volume": 0,
            }
        ]
    )
    stale.to_csv(
        ta_cache / "OFSS.NS-YFin-data-2021-09-22-2026-09-22.csv",
        index=False,
    )

    monkeypatch.setenv("TRADINGAGENTS_LOCAL_OHLC_CACHE_DIR", str(local_cache))
    monkeypatch.setenv("TRADINGAGENTS_REQUIRED_SESSION", "2026-09-21")
    monkeypatch.setattr(
        "tradingagents.dataflows.stockstats_utils.pd.Timestamp.today",
        lambda: pd.Timestamp("2026-09-22"),
    )
    monkeypatch.setattr(
        "tradingagents.dataflows.stockstats_utils.get_config",
        lambda: {"data_cache_dir": str(ta_cache)},
    )

    data = load_ohlcv("OFSS.NS", "2026-09-22")

    row = data[data["Date"] == pd.Timestamp("2026-09-21")].iloc[0]
    assert row["Close"] == 10911.0
    assert row["Volume"] == 565241
