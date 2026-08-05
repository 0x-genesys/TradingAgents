from __future__ import annotations

import pytest

from tradingagents.agents.utils.agent_utils import (
    apply_portfolio_manager_policy,
    apply_research_manager_policy,
    apply_trader_policy,
    assess_precision_first_evidence,
)


def _neutral_state() -> dict:
    return {
        "market_report": (
            "Price is range-bound near resistance and needs confirmation above the recent high. "
            "Macro volatility remains elevated."
        ),
        "news_report": "No fresh company-specific catalyst was verified in the current window.",
        "fundamentals_report": "Balance sheet is stable, but there is no new event catalyst.",
        "sentiment_report": "Sentiment is mixed with no decisive edge.",
        "investment_debate_state": {
            "history": "Bear Analyst: wait for confirmation. Macro uncertainty remains high.",
        },
        "risk_debate_state": {
            "history": "Conservative Analyst: wait for confirmation. Macro uncertainty remains high.",
        },
    }


def _bullish_state() -> dict:
    return {
        "market_report": (
            "The stock staged a bullish breakout with volume expansion. "
            "MACD turned bullish and price reclaimed the 10 EMA."
        ),
        "news_report": (
            "The company received regulatory approval for a key product launch and signed a new partnership."
        ),
        "fundamentals_report": "Management raised guidance after the product approval.",
        "sentiment_report": "Sentiment improved after the launch headlines.",
        "investment_debate_state": {"history": "Bull Analyst: catalyst plus momentum align."},
        "risk_debate_state": {"history": "Neutral Analyst: upside case is evidence-backed."},
    }


def _bearish_state() -> dict:
    return {
        "market_report": (
            "The stock suffered a bearish crossover, printed lower lows, and faced repeated resistance rejection."
        ),
        "news_report": "Management issued a profit warning after an earnings miss.",
        "fundamentals_report": "Margin pressure accelerated after the guidance cut.",
        "sentiment_report": "Investor sentiment weakened after the downgrade.",
        "investment_debate_state": {"history": "Bear Analyst: verified downside is active."},
        "risk_debate_state": {"history": "Conservative Analyst: downside evidence is clear."},
    }


@pytest.mark.unit
def test_assess_precision_first_evidence_tags_real_edge() -> None:
    evidence = assess_precision_first_evidence(_bullish_state())

    assert evidence["positive_catalyst"] is True
    assert evidence["aligned_momentum"] is True
    assert "VERIFIED_POSITIVE_CATALYST" in evidence["tags"]
    assert "ALIGNED_MOMENTUM" in evidence["tags"]


@pytest.mark.unit
def test_trader_buy_without_edge_is_reframed_to_hold() -> None:
    plan = (
        "**Action**: Buy\n\n"
        "**Reasoning**: The setup could work if confirmation arrives.\n\n"
        "FINAL TRANSACTION PROPOSAL: **BUY**"
    )

    updated, tags = apply_trader_policy(plan, _neutral_state())

    assert "**Action**: Hold" in updated
    assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in updated
    assert "BUY_GATED_NO_EDGE" in tags
    assert "**Policy Adjustment**:" in updated


@pytest.mark.unit
def test_portfolio_manager_sell_without_primary_bear_is_reframed_to_hold() -> None:
    decision = (
        "**Rating**: Sell\n\n"
        "**Executive Summary**: Macro uncertainty remains elevated.\n\n"
        "**Investment Thesis**: The setup still needs confirmation above resistance."
    )

    updated, tags = apply_portfolio_manager_policy(decision, _neutral_state())

    assert "**Rating**: Hold" in updated
    assert "SELL_GATED_NO_PRIMARY_BEAR" in tags
    assert "CONFIRMATION_ONLY_OBJECTION" in tags
    assert "MACRO_ONLY_OBJECTION" in tags


@pytest.mark.unit
def test_research_manager_keeps_bullish_recommendation_when_edge_exists() -> None:
    plan = (
        "**Recommendation**: Buy\n\n"
        "**Rationale**: Verified catalyst and momentum align.\n\n"
        "**Strategic Actions**: Enter within the stated trade window."
    )

    updated, tags = apply_research_manager_policy(plan, _bullish_state())

    assert updated == plan
    assert "BUY_GATED_NO_EDGE" not in tags
    assert "VERIFIED_POSITIVE_CATALYST" in tags
    assert "ALIGNED_MOMENTUM" in tags


@pytest.mark.unit
def test_portfolio_manager_keeps_sell_when_primary_bearish_evidence_exists() -> None:
    decision = (
        "**Rating**: Sell\n\n"
        "**Executive Summary**: Primary downside evidence is active.\n\n"
        "**Investment Thesis**: Breakdown plus earnings miss support a bearish view."
    )

    updated, tags = apply_portfolio_manager_policy(decision, _bearish_state())

    assert updated == decision
    assert "SELL_GATED_NO_PRIMARY_BEAR" not in tags
    assert "VERIFIED_PRIMARY_BEAR_EVIDENCE" in tags
