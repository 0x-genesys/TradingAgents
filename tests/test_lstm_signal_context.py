from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
from tradingagents.agents.utils.grounding import find_current_evidence_issues
from tradingagents.agents.utils.lstm_context import fixed_trade_facts


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


class CapturePrompts(BaseCallbackHandler):
    def __init__(self):
        self.prompts = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.prompts.append("\n".join(str(m.content) for batch in messages for m in batch))


class ToolModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("Free-text fixture")


def _decision_state():
    state = Propagator().create_initial_state("EXAMPLE.NS", "2026-08-19", lstm_signal_context=_context())
    state.update(investment_plan="Review current evidence.", trader_investment_plan="Review current evidence.")
    return state


@pytest.mark.parametrize("factory", [
    market_analyst.create_market_analyst, news_analyst.create_news_analyst,
    sentiment_analyst.create_sentiment_analyst, fundamentals_analyst.create_fundamentals_analyst,
])
def test_full_analyst_prompts_exclude_model_framing_and_preserve_tool_history(factory):
    state = _decision_state()
    state["messages"] = [HumanMessage(content="EXAMPLE.NS"),
        AIMessage(content="", tool_calls=[{"name": "get_stock_data", "args": {}, "id": "price"}]),
        ToolMessage(content="Verified OHLC: close=100, source session 2026-08-19", tool_call_id="price")]
    capture = CapturePrompts()
    factory(ToolModel(responses=["Current evidence is mixed; reversal is unconfirmed."], callbacks=[capture]))(state)
    for prompt in capture.prompts:
        assert "pullback_within_established_momentum" not in prompt
        assert "pullback within established momentum" not in prompt
        assert "0.806" not in prompt
        assert "score_influence" not in prompt
        assert "EXACT MODEL INPUT MATRIX" not in prompt
        assert "Verified OHLC: close=100" in prompt
        assert "sma200: 85.000000" in prompt
        assert "2026-08-19" in prompt
        assert "7 exchange sessions" in prompt


@pytest.mark.parametrize("bad", [
    "The LSTM definitively validates an intact uptrend and proves institutional buying with an 80.6% target-hit probability.",
    "The five-session return is +5% and MACD histogram is positive.",
    "The LSTM rank strongly exceeds the >60.00% hit rate threshold.",
])
@pytest.mark.parametrize("factory,field,tag", [
    (portfolio_manager.create_portfolio_manager, "final_trade_decision", "FINAL"),
    (trader.create_trader, "trader_investment_plan", "TRADER"),
    (research_manager.create_research_manager, "investment_plan", "RESEARCH_MANAGER"),
])
def test_decision_nodes_repair_evidence_once(factory, field, tag, bad):
    state = _decision_state()
    capture = CapturePrompts()
    good = "**Rating**: Buy\n\n**Action**: Buy\n\nFive-session return is -2%. A reversal remains a hypothesis."
    result = factory(ToolModel(responses=[bad, good], callbacks=[capture]))(state)
    assert len(capture.prompts) == 2
    assert result[field] == good
    assert f"REPAIRED_{tag}_GROUNDING" in result["data_quality_tags"]
    assert "LSTM QUANTITATIVE EVIDENCE" in capture.prompts[0]
    assert "0.806" in capture.prompts[0]


@pytest.mark.parametrize("action", ["Buy", "Hold", "Sell"])
def test_unresolved_issue_visible_without_rewriting_action(action):
    state = _decision_state()
    bad = f"**Rating**: {action}\n\nThe LSTM proves an intact uptrend."
    capture = CapturePrompts()
    result = portfolio_manager.create_portfolio_manager(ToolModel(responses=[bad, bad], callbacks=[capture]))(state)
    assert len(capture.prompts) == 2
    assert result["final_trade_decision"].startswith(bad)
    assert "Unresolved evidence issues" in result["final_trade_decision"]
    assert "INVALID_FINAL_GROUNDING" in result["data_quality_tags"]


@pytest.mark.parametrize("action", ["Buy", "Hold", "Sell"])
def test_valid_rebound_or_rejection_is_preserved(action):
    state = _decision_state()
    valid = f"**Rating**: {action}\n\nThe LSTM does not prove an intact trend. Five-session return is -2%. The model supports a possible rebound, but confirmation remains pending."
    capture = CapturePrompts()
    result = portfolio_manager.create_portfolio_manager(ToolModel(responses=[valid], callbacks=[capture]))(state)
    assert len(capture.prompts) == 1
    assert result["final_trade_decision"] == valid
    assert result["data_quality_tags"] == []


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "fixtures/lstm_grounding_cases.json").read_text()), ids=lambda c: c["ticker"])
def test_saved_cases_flag_reasoning_not_trade_outcomes(case):
    state = _decision_state()
    values = state["lstm_signal_context"]["signal"]["technical_snapshot"]["values"]
    values.update(return_5d=case["return_5d"], macd_histogram=case["macd_histogram"])
    if case["issue"] == "arithmetic":
        assert find_trade_arithmetic_issues(case["claim"], state)
    else:
        assert any(case["issue"] in i for i in find_current_evidence_issues(case["claim"], state))
    # A factual correction can retain BUY even for a weak snapshot. No SMA gate.
    corrected = f"**Rating**: Buy\n\nFive-session return is {case['return_5d'] * 100:.2f}%. MACD histogram is {case['macd_histogram']:.3f}. The rebound is a hypothesis, with future confirmation required."
    result = portfolio_manager.create_portfolio_manager(ToolModel(responses=[case["claim"], corrected]))(state)
    assert result["final_trade_decision"] == corrected
    assert "REPAIRED_FINAL_GROUNDING" in result["data_quality_tags"]


@pytest.mark.parametrize("text,bad", [
    ("A +3% target and -4.5% stop require 60% wins to break even before costs.", False),
    ("A +3% target and -4.5% stop with 0.2% costs require 62.67% wins to break even.", False),
    ("The trade can break even with only 38% wins at this payoff.", True),
    ("The break-even rate is 38%.", True),
    ("At 38% wins this trade cannot break even.", False),
    ("A profitable exit books +3% before 0.2% costs.", False),
])
def test_payoff_checks_only_claimed_win_rates(text, bad):
    state = _decision_state()
    assert bool(find_trade_arithmetic_issues(text, state)) == bad
    if not bad:
        capture = CapturePrompts()
        decision = "**Rating**: Buy\n\n" + text
        result = portfolio_manager.create_portfolio_manager(ToolModel(responses=[decision], callbacks=[capture]))(state)
        assert len(capture.prompts) == 1
        assert result["final_trade_decision"] == decision


def test_fixed_levels_training_semantics_and_custom_targets():
    context = _context()
    levels = fixed_trade_facts(context)
    assert levels["target_price"] == 103
    assert levels["stop_price"] == 95.5
    assert levels["target_distance_atr"] == 1.5
    assert levels["stop_distance_atr"] == 2.25
    compact = render_lstm_compact_context(context)
    assert context["model"]["training_label"] in compact
    assert "distinct definitions" in compact
    assert "target price: 103.0000" in compact
    assert "stop distance: 2.2500 ATR" in compact
    assert "as of 2026-08-19" in compact
    context["strategy_contract"]["decision_target_pct"] = 0.02
    compact = render_lstm_compact_context(context)
    assert "target price: 102.0000" in compact
    assert "69.23% target hits" in compact
    assert "Recorded training objective" in compact


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_atr_cannot_generate_trade_levels(value):
    context = _context()
    context["signal"]["technical_snapshot"]["values"]["atr14"] = value
    with pytest.raises(ValueError):
        validate_lstm_signal_context(context, "EXAMPLE.NS", "2026-08-19")


def test_structured_pm_gets_bounded_repair_and_retains_action():
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = [
        PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="Review", investment_thesis="The LSTM proves institutional absorption."),
        PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="Review", investment_thesis="Institutional flow is unknown; a reversal remains a hypothesis."),
    ]
    result = portfolio_manager.create_portfolio_manager(llm)(_decision_state())
    assert llm.with_structured_output.return_value.invoke.call_count == 2
    llm.invoke.assert_not_called()
    assert result["final_trade_decision"].startswith("**Rating**: Buy")
    assert "REPAIRED_FINAL_GROUNDING" in result["data_quality_tags"]


def test_fixed_price_atr_and_ma_conflicts():
    state = _decision_state()
    bad = "Target price: $104. Target distance is 2.0 ATR. The target maps directly to MA20/MA50."
    issues = find_current_evidence_issues(bad, state)
    assert any("FIXED_LEVEL_CONFLICT" in i for i in issues)
    assert any("ATR_DISTANCE_CONFLICT" in i for i in issues)
    assert any("TARGET_MA_CONFLICT" in i for i in issues)
    assert not find_current_evidence_issues("Target price: $103. Stop price: $95.50. Target distance: 1.50 ATR. Stop distance: 2.25 ATR.", state)


def test_metric_names_are_not_values_or_prefixes_of_other_metrics():
    state = _decision_state()
    text = "RSI14 is near an extreme. SMA200 provides a long-term reference. Price/MA200 is a ratio. Close/SMA200 1.082x. SMA200 (0.966x) describes proximity."
    assert not find_current_evidence_issues(text, state)
