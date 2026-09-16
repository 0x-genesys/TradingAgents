from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

from tradingagents.agents.analysts import (
    fundamentals_analyst,
    market_analyst,
    news_analyst,
    sentiment_analyst,
)
from tradingagents.agents.managers import portfolio_manager, research_manager
from tradingagents.agents.researchers import bear_researcher, bull_researcher
from tradingagents.agents.risk_mgmt import (
    aggressive_debator,
    conservative_debator,
    neutral_debator,
)
from tradingagents.agents.schemas import (
    LSTMThesisAssessment,
    PortfolioDecision,
    PortfolioRating,
    TraderAction,
    TraderProposal,
    render_pm_decision,
    render_trader_proposal,
)
from tradingagents.agents.trader import trader
from tradingagents.agents.utils.agent_utils import (
    apply_portfolio_manager_policy,
    apply_trader_policy,
    find_trade_arithmetic_issues,
    get_data_quality_instruction,
    sanitize_agent_output,
)
from tradingagents.agents.utils.lstm_context import (
    render_lstm_compact_context,
    render_lstm_market_context,
    render_lstm_report,
    validate_lstm_signal_context,
)
from tradingagents.graph.propagation import Propagator


def _context() -> dict:
    dates = [f"2026-08-{day:02d}" for day in range(5, 20)]
    return {
        "schema_version": 1,
        "run_date": "2026-08-19",
        "model": {
            "family": "momentum LSTM",
            "type": "lstm_meta_labeler",
            "artifact_sha256": "a" * 64,
            "sequence_length": 15,
            "feature_names": ["return_5d_csrank", "breakout_20d"],
            "threshold": 0.6,
            "training_label": "realized_return > 0.030 after stop simulation",
            "training_objective": {
                "definition": "realized_return > PROFIT_TARGET",
                "profit_target_pct": 0.03,
                "stop_loss_pct": -0.045,
                "horizon_exchange_sessions": 7,
                "metadata_complete": True,
            },
        },
        "strategy_contract": {
            "setup": "pullback_within_established_momentum",
            "decision_target_pct": 0.03,
            "stop_loss_pct": -0.045,
            "maximum_horizon_exchange_sessions": 7,
            "objective": "reach target before stop within the horizon",
        },
        "universe": {"scored_count": 190, "above_threshold_count": 12},
        "orchestrator_selection_route": "top_four_by_current_lstm_rank",
        "signal": {
            "ticker": "EXAMPLE.NS",
            "inference_date": "2026-08-19",
            "data_timestamp": "2026-08-19",
            "entry_price": 100.0,
            "score": 0.806,
            "score_semantics": "uncalibrated sigmoid model confidence; not target-hit probability",
            "rank": 2,
            "threshold": 0.6,
            "above_threshold": True,
            "top_four_candidate": True,
            "selection_reason": "Rank > 2",
            "feature_sequence": {
                "dates": dates,
                "feature_names": ["return_5d_csrank", "breakout_20d"],
                "values": [[0.4, 0.0] for _ in dates],
            },
            "feature_sensitivity": {
                "baseline_score": 0.806,
                "positive": [
                    {
                        "feature": "return_5d_csrank",
                        "score_influence": 0.03,
                    }
                ],
                "negative": [],
            },
            "feature_group_influence": {
                "established_momentum": 0.02,
                "recent_pullback": 0.03,
            },
            "latest_raw_features": {"return_5d": -0.02},
            "technical_snapshot": {
                "data_timestamp": "2026-08-19",
                "row_count": 260,
                "complete": True,
                "missing_metrics": [],
                "freshness": {
                    "inference_date": "2026-08-19",
                    "age_calendar_days": 0,
                    "future_data": False,
                },
                "values": {
                    "return_5d": -0.02,
                    "return_10d": 0.01,
                    "return_20d": 0.05,
                    "ema10": 99.0,
                    "sma20": 98.0,
                    "sma50": 95.0,
                    "sma200": 85.0,
                    "vwma20": 98.5,
                    "rsi14": 42.0,
                    "macd_line": -0.2,
                    "macd_signal": -0.1,
                    "macd_histogram": -0.1,
                    "atr14": 2.0,
                    "atr14_pct": 0.02,
                    "volatility_10d": 0.015,
                    "bollinger_middle": 98.0,
                    "bollinger_upper": 104.0,
                    "bollinger_lower": 92.0,
                    "bollinger_position": 0.67,
                    "volume_vs_20d_average": 1.2,
                    "breakout_strength_20d": -0.03,
                    "range_position_20d": 0.7,
                    "close_strength": 0.4,
                    "price_vs_ma20": 1.02,
                    "price_vs_ma50": 1.05,
                },
            },
        },
    }


@pytest.mark.unit
def test_context_validation_and_role_specific_rendering() -> None:
    context = _context()
    validate_lstm_signal_context(context, "EXAMPLE.NS", "2026-08-19")

    compact = render_lstm_compact_context(context)
    market = render_lstm_market_context(context)
    assert "pullback within established momentum" in compact
    assert "uncalibrated sigmoid" in compact
    assert "EXACT MODEL INPUT MATRIX" not in compact
    assert "EXACT MODEL INPUT MATRIX" in market
    assert "return_5d_csrank,breakout_20d" in market
    assert "Current feature summary" in render_lstm_report(context)


@pytest.mark.unit
def test_context_rejects_future_or_mismatched_evidence() -> None:
    future = deepcopy(_context())
    future["signal"]["data_timestamp"] = "2026-08-20"
    with pytest.raises(ValueError, match="future-dated"):
        validate_lstm_signal_context(future, "EXAMPLE.NS", "2026-08-19")

    mismatch = deepcopy(_context())
    mismatch["signal"]["ticker"] = "OTHER.NS"
    with pytest.raises(ValueError, match="does not match"):
        validate_lstm_signal_context(mismatch, "EXAMPLE.NS", "2026-08-19")

    future_row = deepcopy(_context())
    future_row["signal"]["feature_sequence"]["dates"][0] = "2026-08-20"
    with pytest.raises(ValueError, match="strictly increasing|future-dated"):
        validate_lstm_signal_context(future_row, "EXAMPLE.NS", "2026-08-19")


@pytest.mark.unit
def test_propagation_preserves_structured_and_rendered_context() -> None:
    state = Propagator().create_initial_state(
        "EXAMPLE.NS",
        "2026-08-19",
        trade_horizon_days=7,
        entry_price=100.0,
        profit_target_pct=0.03,
        stop_loss_pct=-0.045,
        trade_strategy="pullback_within_established_momentum",
        lstm_signal_context=_context(),
    )

    assert state["lstm_signal_context"]["signal"]["rank"] == 2
    assert "EXACT MODEL INPUT MATRIX" in state["lstm_market_context_note"]
    assert "EXACT MODEL INPUT MATRIX" not in state["lstm_context_note"]
    assert "LSTM Quantitative" not in state["trade_context_note"]


@pytest.mark.unit
def test_propagation_rejects_trade_parameter_mismatch() -> None:
    with pytest.raises(ValueError, match="profit_target_pct conflicts"):
        Propagator().create_initial_state(
            "EXAMPLE.NS",
            "2026-08-19",
            profit_target_pct=0.02,
            lstm_signal_context=_context(),
        )


@pytest.mark.unit
def test_lstm_context_decisions_are_not_rewritten_or_sanitized() -> None:
    state = {"lstm_signal_context": _context()}
    trader_text = "**Action**: Buy\n\n**Reasoning**: LSTM evidence supports reversal."
    pm_text = "**Rating**: Sell\n\n**Investment Thesis**: LSTM thesis rejected."

    assert apply_trader_policy(trader_text, state) == (trader_text, [])
    assert apply_portfolio_manager_policy(pm_text, state) == (pm_text, [])
    assert sanitize_agent_output(trader_text, state) == (trader_text, [])

    invented = "The prior LSTM run had a higher score. Current LSTM rank is #2."
    sanitized, tags = sanitize_agent_output(invented, state)
    assert "prior LSTM run" not in sanitized
    assert "Current LSTM rank is #2" in sanitized
    assert tags == ["REMOVED_UNSUPPORTED_UPSTREAM_CLAIM"]


@pytest.mark.unit
def test_fixed_payoff_arithmetic_is_flagged_without_action_override() -> None:
    state = {
        "profit_target_pct": 0.03,
        "stop_loss_pct": -0.045,
        "lstm_signal_context": _context(),
    }
    bad_math = "The trade can break even with only 38% wins at this payoff."
    good_math = "The trade needs more than 60% target hits before costs to break even."

    assert find_trade_arithmetic_issues(bad_math, state) == [bad_math]
    assert find_trade_arithmetic_issues(good_math, state) == []
    assert "60.00% target hits before costs" in get_data_quality_instruction(state)


@pytest.mark.unit
def test_structured_outputs_render_lstm_assessment() -> None:
    proposal = TraderProposal(
        action=TraderAction.BUY,
        reasoning="Current evidence supports reversal.",
        lstm_thesis_assessment=LSTMThesisAssessment.SUPPORTED,
        candidate_rank=2,
        supporting_evidence=["Momentum structure intact"],
        contradicting_evidence=["Weak five-day return"],
        model_disagreement_reason="None",
    )
    decision = PortfolioDecision(
        rating=PortfolioRating.HOLD,
        executive_summary="Wait for clearer evidence.",
        investment_thesis="The model thesis remains unresolved.",
        lstm_thesis_assessment=LSTMThesisAssessment.INSUFFICIENT_EVIDENCE,
        candidate_rank=2,
        model_disagreement_reason="Current confirmation is insufficient.",
    )

    assert "**LSTM Thesis Assessment**: SUPPORTED" in render_trader_proposal(proposal)
    assert "**Candidate Rank**: 2" in render_pm_decision(decision)
    assert "Current confirmation is insufficient" in render_pm_decision(decision)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module", "field_name"),
    [
        (bull_researcher, "lstm_context_note"),
        (bear_researcher, "lstm_context_note"),
        (research_manager, "lstm_context_note"),
        (trader, "lstm_context_note"),
        (aggressive_debator, "lstm_context_note"),
        (conservative_debator, "lstm_context_note"),
        (neutral_debator, "lstm_context_note"),
        (portfolio_manager, "lstm_context_note"),
    ],
)
def test_downstream_roles_consume_compact_lstm_context(module, field_name: str) -> None:
    assert field_name in inspect.getsource(module)


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    [
        market_analyst,
        fundamentals_analyst,
        news_analyst,
        sentiment_analyst,
    ],
)
def test_evidence_first_analysts_do_not_receive_lstm_persuasion(module) -> None:
    source = inspect.getsource(module)
    assert "lstm_context_note" not in source
    assert "lstm_market_context_note" not in source
    assert "lstm_context_available=False" in source
