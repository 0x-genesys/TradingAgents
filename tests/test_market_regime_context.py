import json
from unittest.mock import Mock

import pytest
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    PortfolioDecision, TraderProposal, render_pm_decision, render_trader_proposal,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import get_analyst_context, get_data_quality_instruction
from tradingagents.agents.utils.grounding import decision_grounding_issues
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tests.test_lstm_signal_context import (
    _context, CapturePrompts, ToolModel, market_analyst, news_analyst,
    sentiment_analyst, fundamentals_analyst, bull_researcher, bear_researcher,
    research_manager, trader, portfolio_manager, aggressive_debator,
    conservative_debator, neutral_debator,
)


def snapshot():
    return {
        "schema_version": 1, "run_date": "2026-08-19", "data_date": "2026-08-18",
        "status": "COMPLETE",
        "universe": {"name": "LSTM inference universe", "expected_count": 190, "valid_count": 190},
        "values": {"mean_return_5d": -0.03, "mean_return_20d": -0.07,
                   "above_sma50_fraction": 0.2, "above_sma200_fraction": 0.4},
        "source": {"path": "/producer/inference.csv", "sha256": "a" * 64},
        "missing_fields": [], "exclusions": {},
    }


def state_for(context, lstm=True):
    state = Propagator().create_initial_state(
        "EXAMPLE.NS", "2026-08-19", market_regime_context=context,
        lstm_signal_context=dated_lstm_context() if lstm else None,
    )
    state.update(investment_plan="Review evidence.", trader_investment_plan="Review evidence.")
    return state


def dated_lstm_context():
    context = _context()
    signal = context["signal"]
    signal["data_timestamp"] = "2026-08-18"
    signal["feature_sequence"]["dates"] = [f"2026-08-{day:02d}" for day in range(4, 19)]
    signal["technical_snapshot"]["data_timestamp"] = "2026-08-18"
    signal["technical_snapshot"]["freshness"]["age_calendar_days"] = 1
    return context


def test_stale_regime_date_must_match_validated_lstm_market_date():
    context = snapshot()
    context["data_date"] = "2026-08-17"
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "UNAVAILABLE"
    assert "MARKET_REGIME_UNAVAILABLE" in state["data_quality_tags"]
    assert "LSTM" in " ".join(state["market_regime_validation"]["errors"])
    assert state["market_regime_context"] == context
    assert "mean_return_20d:" not in get_analyst_context(state)


def test_same_date_relative_return_is_derived_for_downstream_only():
    state = state_for(snapshot())
    prompt = get_data_quality_instruction(state)
    assert "ticker_minus_universe_return_20d: +12.00 percentage points" in prompt
    assert "2026-08-18" in prompt
    assert "ticker_minus_universe_return_20d" not in get_analyst_context(state)
    stale = snapshot()
    stale["data_date"] = "2026-08-17"
    assert "ticker_minus_universe_return_20d" not in get_data_quality_instruction(state_for(stale))


def test_snapshot_preserved_and_neutral_facts_rendered_without_aliasing():
    context = snapshot()
    state = state_for(context)
    assert state["market_regime_context"] == context
    assert state["market_regime_validation"]["status"] == "COMPLETE"
    note = state["market_regime_context_note"]
    for fact in ("2026-08-18", "190/190", "mean_return_5d: -3.00%",
                 "mean_return_20d: -7.00%", "above_sma50_fraction: 20.00%"):
        assert fact in note
    assert "0.806" not in note and "rank" not in note.lower()
    report = state["market_regime_report"]
    assert json.loads(report.split("```json\n")[1].split("\n```")[0]) == context
    context["values"]["mean_return_20d"] = 0.99
    assert state["market_regime_context"]["values"]["mean_return_20d"] == -0.07


def test_omitted_context_is_identical_to_explicit_none():
    kwargs = dict(lstm_signal_context=_context())
    omitted = Propagator().create_initial_state("EXAMPLE.NS", "2026-08-19", **kwargs)
    explicit = Propagator().create_initial_state("EXAMPLE.NS", "2026-08-19", market_regime_context=None, **kwargs)
    assert omitted == explicit
    assert not any(key.startswith("market_regime") for key in omitted)
    assert "MARKET REGIME" not in get_analyst_context(omitted)
    assert "MARKET REGIME" not in get_data_quality_instruction(omitted)
    assert omitted["data_quality_tags"] == []


@pytest.mark.parametrize("path,value", [
    (("schema_version",), True), (("schema_version",), 2),
    (("run_date",), "2026-08-20"), (("run_date",), "2026-8-19"),
    (("data_date",), "2026-08-19"), (("data_date",), "2026-08-20"),
    (("data_date",), "2026-02-30"), (("data_date",), None),
    (("status",), "BULL"), (("values",), []), (("universe",), None),
    (("universe", "expected_count"), True), (("universe", "valid_count"), 191),
    (("universe", "valid_count"), 189), (("universe", "name"), "Nifty 50"),
    (("values", "mean_return_20d"), float("nan")),
    (("values", "mean_return_20d"), float("inf")),
    (("values", "mean_return_20d"), "-0.07"),
    (("values", "mean_return_20d"), True),
    (("values", "mean_return_20d"), -1.1),
    (("values", "above_sma50_fraction"), 20),
    (("values", "above_sma200_fraction"), -0.1),
    (("source",), None), (("source", "path"), ""),
    (("source", "sha256"), "g" * 64), (("source", "sha256"), None),
    (("missing_fields",), "mean_return_20d"), (("exclusions",), []),
])
def test_invalid_optional_context_is_unavailable_not_fatal(path, value):
    context = snapshot()
    target = context
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "UNAVAILABLE"
    assert "MARKET_REGIME_UNAVAILABLE" in state["data_quality_tags"]
    assert "mean_return_5d:" not in state["market_regime_context_note"]
    assert "missing facts" in get_data_quality_instruction(state).lower()
    assert state["analysis_status"] == "COMPLETE"


@pytest.mark.parametrize("context", [{}, [], "bad"])
def test_empty_and_wrong_shape_context_does_not_silently_omit(context):
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "UNAVAILABLE"
    assert state["market_regime_context"] == context


def test_partial_exposes_only_available_facts_and_coverage():
    context = snapshot()
    context.update(status="PARTIAL", missing_fields=["above_sma200_fraction"], exclusions={"stale": 10})
    context["universe"]["valid_count"] = 180
    context["values"]["above_sma200_fraction"] = None
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "PARTIAL"
    assert state["data_quality_tags"] == ["MARKET_REGIME_PARTIAL"]
    assert "180/190" in state["market_regime_context_note"]
    assert "mean_return_20d: -7.00%" in state["market_regime_context_note"]
    assert "above_sma200_fraction: unavailable" in state["market_regime_context_note"]


def test_unavailable_never_leaks_direction_even_with_supplied_values():
    context = snapshot()
    context["status"] = "UNAVAILABLE"
    state = state_for(context)
    assert "mean_return_20d:" not in get_analyst_context(state)
    assert "no direction" in get_analyst_context(state).lower()
    assert "MARKET_REGIME_UNAVAILABLE" in state["data_quality_tags"]


@pytest.mark.parametrize("factory", [
    market_analyst.create_market_analyst, news_analyst.create_news_analyst,
    sentiment_analyst.create_sentiment_analyst, fundamentals_analyst.create_fundamentals_analyst,
])
def test_all_independent_analyst_prompts_receive_only_neutral_regime_facts(factory):
    capture = CapturePrompts()
    factory(ToolModel(responses=["Observed facts reviewed."], callbacks=[capture]))(state_for(snapshot()))
    assert capture.prompts
    for prompt in capture.prompts:
        assert "mean_return_20d: -7.00%" in prompt
        assert "2026-08-18" in prompt
        assert "0.806" not in prompt and "Candidate Rank" not in prompt
        assert "pullback within established momentum" not in prompt
        assert "REGIME STRATEGY CAVEAT" not in prompt


@pytest.mark.parametrize("factory", [
    bull_researcher.create_bull_researcher, bear_researcher.create_bear_researcher,
    research_manager.create_research_manager, trader.create_trader,
    portfolio_manager.create_portfolio_manager, aggressive_debator.create_aggressive_debator,
    conservative_debator.create_conservative_debator, neutral_debator.create_neutral_debator,
])
def test_downstream_prompts_receive_facts_and_conditional_strategy_caveat(factory):
    capture = CapturePrompts()
    factory(ToolModel(responses=["**Rating**: Hold\n**Action**: Hold"], callbacks=[capture]))(state_for(snapshot()))
    prompt = capture.prompts[0]
    assert "mean_return_20d: -7.00%" in prompt
    assert "REGIME STRATEGY CAVEAT" in prompt
    assert "high score does not establish safety" in prompt
    assert "not calibrated bull/bear probabilities" in prompt
    assert "not an automatic HOLD" in prompt


@pytest.mark.parametrize("factory", [trader.create_trader, portfolio_manager.create_portfolio_manager])
@pytest.mark.parametrize("status", ["COMPLETE", "UNAVAILABLE"])
def test_entry_synthesis_reconciles_strongest_contradiction_and_horizon(factory, status):
    context = snapshot()
    context["status"] = status
    capture = CapturePrompts()
    factory(ToolModel(responses=["**Rating**: Hold\n**Action**: Hold"], callbacks=[capture]))(state_for(context))
    prompt = capture.prompts[0]
    assert "strongest contradictory evidence" in prompt
    assert "within 7 exchange sessions" in prompt
    assert "confirmed reversal, catalyst, or relative strength" in prompt


def test_regime_only_context_does_not_enable_lstm_caveat():
    state = state_for(snapshot(), lstm=False)
    prompt = get_data_quality_instruction(state)
    assert "mean_return_20d: -7.00%" in prompt
    assert "REGIME STRATEGY CAVEAT" not in prompt


@pytest.mark.parametrize("assessment", ["SUPPORTIVE", "ADVERSE", "MIXED", "UNKNOWN"])
@pytest.mark.parametrize("model,renderer,kwargs", [
    (TraderProposal, render_trader_proposal, {"action": "Buy", "reasoning": "Entry evidence."}),
    (PortfolioDecision, render_pm_decision, {"rating": "Sell", "executive_summary": "Avoid.", "investment_thesis": "Adverse evidence."}),
])
def test_optional_regime_schema_and_rendering(assessment, model, renderer, kwargs):
    output = model(**kwargs, regime_fit_assessment=assessment, regime_fit_reason="Dated breadth vs entry support.")
    text = renderer(output)
    assert f"**Regime Fit Assessment**: {assessment}" in text
    assert "**Regime Fit Reason**: Dated breadth vs entry support." in text
    assert "Regime Fit" not in renderer(model(**kwargs))
    with pytest.raises(ValidationError):
        model(**kwargs, regime_fit_assessment="BULLISH")


@pytest.mark.parametrize("claim,invalid", [
    ("Universe mean_return_20d: +7.00%", True),
    ("Universe mean_return_20d: -7.00%", False),
    ("Universe mean_return_20d: -0.07", False),
    ("Universe above_sma50_fraction: 80%", True),
    ("Universe above_sma50_fraction: 20%", False),
    ("Universe mean 5-day return is +3%", True),
    ("Universe mean 5-day return is -3%", False),
    ("If universe mean_return_20d: +7%, the scenario changes.", False),
    ("The ticker 20-day return is +5%.", False),
])
def test_bounded_grounding_checks_regime_numbers_not_ticker_or_hypotheses(claim, invalid):
    issues = decision_grounding_issues(claim, state_for(snapshot()))
    assert any("REGIME_NUMERIC_CONFLICT" in issue for issue in issues) is invalid


@pytest.mark.parametrize("action", ["Buy", "Hold", "Sell"])
@pytest.mark.parametrize("factory,field", [
    (trader.create_trader, "trader_investment_plan"),
    (portfolio_manager.create_portfolio_manager, "final_trade_decision"),
])
def test_grounding_bounded_repair_never_rewrites_original_action(action, factory, field):
    bad = f"**Rating**: {action}\n**Action**: {action}\nUniverse mean_return_20d: +7%."
    changed = "**Rating**: Hold\n**Action**: Hold\nUniverse mean_return_20d: -7%."
    capture = CapturePrompts()
    result = factory(ToolModel(responses=[bad, changed], callbacks=[capture]))(state_for(snapshot()))
    assert len(capture.prompts) == 2
    assert f"**Rating**: {action}" in result[field]
    assert f"**Action**: {action}" in result[field]
    if action != "Hold":
        assert result[field].startswith(bad)
        assert "Unresolved evidence issues" in result[field]


@pytest.mark.parametrize("action", ["Buy", "Hold", "Sell"])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("regime_status", ["COMPLETE", "PARTIAL", "UNAVAILABLE", "STALE", "OMITTED"])
def test_public_graph_path_persists_exact_snapshot_and_reports(tmp_path, action, debug, regime_status):
    from cli.main import save_report_to_disk

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
    graph.debug = debug
    graph.propagator = Propagator()
    graph.memory_log = Mock()
    graph._resolve_pending_entries = Mock()
    graph.resolve_source_snapshot = Mock(return_value={})
    graph._checkpointer_ctx = None
    graph.log_states_dict = {}
    from tradingagents.graph.signal_processing import SignalProcessor
    graph.signal_processor = SignalProcessor()

    workflow = StateGraph(AgentState)
    def prepare(state):
        return {"investment_plan": "Review evidence.", "market_report": "Dated OHLC observed."}
    workflow.add_node("prepare", prepare)
    workflow.add_node("trader", trader.create_trader(ToolModel(responses=[f"**Action**: {action}"])))
    workflow.add_node("pm", portfolio_manager.create_portfolio_manager(ToolModel(responses=[f"**Rating**: {action}"])))
    workflow.set_entry_point("prepare")
    workflow.add_edge("prepare", "trader")
    workflow.add_edge("trader", "pm")
    workflow.add_edge("pm", END)
    graph.graph = workflow.compile()
    context = snapshot()
    if regime_status == "PARTIAL":
        context["status"] = "PARTIAL"
        context["universe"]["valid_count"] = 180
        context["exclusions"] = {"stale": 10}
    elif regime_status == "UNAVAILABLE":
        context["status"] = "UNAVAILABLE"
    elif regime_status == "STALE":
        context["data_date"] = "2026-08-17"
    kwargs = {} if regime_status == "OMITTED" else {"market_regime_context": context}
    state, signal = graph.propagate("EXAMPLE.NS", "2026-08-19", lstm_signal_context=dated_lstm_context(), **kwargs)
    assert signal == action
    assert state["analysis_status"] == "COMPLETE"
    saved = json.loads(next(tmp_path.rglob("full_states_log_*.json")).read_text())
    report_path = save_report_to_disk(state, "EXAMPLE.NS", tmp_path / "report")
    report = report_path.read_text()
    if regime_status == "OMITTED":
        assert "market_regime_context" not in state
        assert "market_regime_context" not in saved
        assert "Market Regime Context" not in report
        return
    assert state["market_regime_context"] == context
    assert saved["market_regime_context"] == context
    if regime_status in ("UNAVAILABLE", "STALE"):
        assert "MARKET_REGIME_UNAVAILABLE" in state["data_quality_tags"]
        assert "mean_return_20d:" not in report
    else:
        assert "mean_return_20d: -7.00%" in report
    assert json.loads(report.split("```json\n")[1].split("\n```")[0]) == context


@pytest.mark.parametrize("key", ["source", "universe", "values", "missing_fields", "exclusions", "data_date"])
def test_missing_required_snapshot_field_is_nonfatal(key):
    context = snapshot()
    del context[key]
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "UNAVAILABLE"
    assert state["market_regime_context"] == context


def test_non_json_optional_input_is_auditable_and_serializable():
    context = snapshot()
    context["exclusions"] = {"unexpected": {object()}}
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "UNAVAILABLE"
    assert "invalid_input_repr" in state["market_regime_context"]
    json.dumps(state)


def test_regime_only_grounding_does_not_require_lstm_context():
    state = state_for(snapshot(), lstm=False)
    assert any("REGIME_NUMERIC_CONFLICT" in issue for issue in decision_grounding_issues(
        "Universe mean_return_20d: +7.00%", state,
    ))


def test_regime_metric_never_compared_with_ticker_metric():
    state = state_for(snapshot())
    assert decision_grounding_issues("Universe mean 5-day return is -3%.", state) == []


def test_partial_missing_metric_is_not_checked_as_a_supplied_number():
    context = snapshot()
    context["status"] = "PARTIAL"
    del context["values"]["mean_return_20d"]
    context["missing_fields"] = ["mean_return_20d"]
    state = state_for(context)
    assert state["market_regime_validation"]["status"] == "PARTIAL"
    assert "mean_return_20d: unavailable" in get_analyst_context(state)
    assert not any("REGIME_NUMERIC_CONFLICT" in issue for issue in decision_grounding_issues(
        "Universe mean_return_20d: +7.00%", state,
    ))
