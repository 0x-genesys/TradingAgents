from __future__ import annotations

import pytest

from tradingagents.graph.propagation import Propagator


@pytest.mark.unit
def test_trade_context_contains_fixed_decision_objective() -> None:
    state = Propagator().create_initial_state(
        "EXAMPLE.NS",
        "2026-08-21",
        trade_horizon_days=7,
        entry_price=100.0,
        profit_target_pct=0.03,
        stop_loss_pct=-0.045,
        trade_strategy="momentum",
    )

    assert state["profit_target_pct"] == 0.03
    assert state["stop_loss_pct"] == -0.045
    assert state["trade_horizon_days"] == 7
    assert state["entry_price"] == 100.0
    assert state["trade_context_note"] == (
        "momentum trade | 7d horizon | Entry: ₹100.00 | Target: +3.0% | "
        "Stop: -4.5% | Objective: reach target before stop within the horizon"
    )


@pytest.mark.unit
def test_trade_context_remains_optional() -> None:
    state = Propagator().create_initial_state("EXAMPLE.NS", "2026-08-21")

    assert state["profit_target_pct"] is None
    assert state["trade_context_note"] == ""
