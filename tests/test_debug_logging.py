from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_debug_stage_markers_emit_meaningful_progress() -> None:
    mock_graph = MagicMock(spec=TradingAgentsGraph)
    mock_graph.config = {
        "max_debate_rounds": 5,
        "max_risk_discuss_rounds": 5,
    }

    emitted = set()

    markers, debate_count, risk_count = TradingAgentsGraph._debug_stage_markers(
        mock_graph,
        {
            "market_report": "market ready",
            "investment_debate_state": {
                "count": 1,
                "current_response": "Bull Analyst: setup still works",
            },
        },
        emitted,
        0,
        0,
    )

    assert "Market Analyst complete" in markers
    assert "Bull Researcher round 1/10" in markers
    assert debate_count == 1
    assert risk_count == 0

    markers, debate_count, risk_count = TradingAgentsGraph._debug_stage_markers(
        mock_graph,
        {
            "investment_plan": "**Recommendation**: Hold",
            "trader_investment_plan": "**Action**: Hold",
            "risk_debate_state": {
                "count": 2,
                "latest_speaker": "Conservative Analyst",
            },
            "final_trade_decision": "**Rating**: Hold",
        },
        emitted,
        debate_count,
        risk_count,
    )

    assert "Research Manager complete" in markers
    assert "Trader complete" in markers
    assert "Conservative Analyst round 2/15" in markers
    assert "Portfolio Manager complete" in markers


@pytest.mark.unit
def test_message_text_handles_continue_placeholder() -> None:
    message = MagicMock()
    message.content = "Continue"

    assert TradingAgentsGraph._message_text(message) == "Continue"
