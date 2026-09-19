from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.analysts.sentiment_analyst import (
    _build_system_message,
    _grounded_fallback_report,
    _unsupported_claim_issues,
    create_sentiment_analyst,
)
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    enforce_exact_tool_ticker,
    find_unsupported_optional_source_claims,
    find_unsupported_upstream_claims,
    get_data_quality_instruction,
    sanitize_agent_output,
    sanitize_unsupported_source_claims,
)
from tradingagents.dataflows.source_snapshot import build_source_snapshot
from tradingagents.dataflows.tavily_news import fetch_tavily_company_news
from tradingagents.dataflows.stocktwits import _headline_matches_company
from tradingagents.graph.trading_graph import (
    TradingAgentsGraph,
    validate_replay_source_snapshot,
)


def _trends(status: str = "UNAVAILABLE") -> dict:
    return {
        "source": "google_trends",
        "status": status,
        "content": "<Google Trends unavailable>",
        "data_quality_tags": ["MISSING_GOOGLE_TRENDS"],
    }


@pytest.mark.unit
def test_tavily_company_news_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = fetch_tavily_company_news(
        "EXAMPLE.NS",
        as_of_date="2026-01-15",
        company_name="Example Limited",
    )

    assert "TAVILY_API_KEY not set" in result


@pytest.mark.unit
def test_tavily_company_news_does_not_backfill_historical_dates(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    result = fetch_tavily_company_news(
        "EXAMPLE.NS",
        as_of_date="2026-01-15",
        company_name="Example Limited",
    )

    assert "historical point-in-time fallback is not supported" in result


@pytest.mark.integration
def test_tavily_company_news_live_integration() -> None:
    if not os.environ.get("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")

    result = fetch_tavily_company_news(
        "INFY.NS",
        as_of_date=date.today().isoformat(),
        company_name="Infosys Limited",
        limit=2,
    )

    assert "TAVILY_API_KEY not set" not in result
    assert "historical point-in-time fallback is not supported" not in result
    assert "Tavily company news" in result or "No Tavily company news found" in result


@pytest.mark.unit
def test_optional_sources_add_tags_without_failing(tmp_path) -> None:
    with (
        patch(
            "tradingagents.dataflows.source_snapshot._yfinance_profile",
            return_value=("Example Limited", {"company_name": "Example Limited"}),
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.get_news_yfinance",
            return_value="## EXAMPLE.NS News\nSupported company headline",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_tavily_company_news",
            return_value="Should not be fetched",
        ) as tavily_news,
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_news_headlines",
            return_value="Google News headlines for EXAMPLE.NS",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_reddit_posts",
            return_value="<no Reddit posts found>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_telegram_messages",
            return_value="<Telegram disabled>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_trends_snapshot",
            return_value=_trends(),
        ),
    ):
        snapshot = build_source_snapshot(
            "EXAMPLE.NS",
            "2026-08-03",
            cache_dir=tmp_path,
        )

    assert snapshot["analysis_status"] == "COMPLETE"
    assert snapshot["primary_data_available"] is True
    assert snapshot["sources"]["tavily_company_news"]["status"] == "NOT_NEEDED"
    assert tavily_news.call_count == 0
    assert "MISSING_TELEGRAM" in snapshot["data_quality_tags"]
    assert "MISSING_REDDIT" in snapshot["data_quality_tags"]
    assert "MISSING_GOOGLE_TRENDS" in snapshot["data_quality_tags"]
    assert "MISSING_SENTIMENT" not in snapshot["data_quality_tags"]


@pytest.mark.unit
def test_google_news_filter_rejects_other_hdfc_and_unrelated_companies() -> None:
    assert _headline_matches_company(
        "HDFC Life Insurance shares rise after earnings",
        "HDFCLIFE.NS",
        "HDFC Life Insurance Company Limited",
    )
    assert not _headline_matches_company(
        "HDFC AMC Share Price - Live NSE",
        "HDFCLIFE.NS",
        "HDFC Life Insurance Company Limited",
    )
    assert not _headline_matches_company(
        "Juniper Green Energy raises funds ahead of IPO",
        "HDFCLIFE.NS",
        "HDFC Life Insurance Company Limited",
    )
    assert _headline_matches_company(
        "Emami expects new-age brands to drive growth",
        "EMAMILTD.NS",
        "Emami Limited",
    )
    assert not _headline_matches_company(
        "Emami Paper Mills Ltd locks at upper circuit",
        "EMAMILTD.NS",
        "Emami Limited",
    )


@pytest.mark.unit
def test_no_primary_data_is_failed_edge_case(tmp_path) -> None:
    unavailable = "<source unavailable: network error>"
    with (
        patch(
            "tradingagents.dataflows.source_snapshot._yfinance_profile",
            side_effect=ConnectionError("offline"),
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.get_news_yfinance",
            return_value="Error fetching news for EXAMPLE.NS: offline",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_tavily_company_news",
            return_value="<Tavily company news disabled: TAVILY_API_KEY not set>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_news_headlines",
            return_value=unavailable,
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_reddit_posts",
            return_value=unavailable,
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_telegram_messages",
            return_value=unavailable,
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_trends_snapshot",
            return_value=_trends(),
        ),
    ):
        snapshot = build_source_snapshot(
            "EXAMPLE.NS",
            "2026-08-03",
            cache_dir=tmp_path,
        )

    assert snapshot["analysis_status"] == "FAILED"
    assert snapshot["primary_data_available"] is False
    assert "MISSING_SENTIMENT" in snapshot["data_quality_tags"]


@pytest.mark.unit
def test_tavily_company_news_fallback_replaces_missing_yahoo_company_news(tmp_path) -> None:
    with (
        patch(
            "tradingagents.dataflows.source_snapshot._yfinance_profile",
            return_value=("Example Limited", {"company_name": "Example Limited"}),
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.get_news_yfinance",
            return_value="No news found for EXAMPLE.NS",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_tavily_company_news",
            return_value="Tavily company news fallback for EXAMPLE.NS\n- Example wins order",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_news_headlines",
            return_value="<Google News unavailable>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_reddit_posts",
            return_value="<no Reddit posts found>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_telegram_messages",
            return_value="<Telegram disabled>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_trends_snapshot",
            return_value=_trends(),
        ),
    ):
        snapshot = build_source_snapshot(
            "EXAMPLE.NS",
            "2026-08-03",
            cache_dir=tmp_path,
        )

    assert snapshot["analysis_status"] == "COMPLETE"
    assert snapshot["primary_data_available"] is True
    assert snapshot["sources"]["company_news"]["status"] == "NO_DATA"
    assert snapshot["sources"]["tavily_company_news"]["status"] == "OK"
    assert "MISSING_COMPANY_NEWS" not in snapshot["data_quality_tags"]
    assert "MISSING_TAVILY_COMPANY_NEWS" not in snapshot["data_quality_tags"]


@pytest.mark.unit
def test_snapshot_is_reused_across_model_runs(tmp_path) -> None:
    with (
        patch(
            "tradingagents.dataflows.source_snapshot._yfinance_profile",
            return_value=("Example Limited", {"company_name": "Example Limited"}),
        ) as profile,
        patch(
            "tradingagents.dataflows.source_snapshot.get_news_yfinance",
            return_value="Supported company headline",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_news_headlines",
            return_value="Supported India headline",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_reddit_posts",
            return_value="<no Reddit posts found>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_telegram_messages",
            return_value="<Telegram disabled>",
        ),
        patch(
            "tradingagents.dataflows.source_snapshot.fetch_google_trends_snapshot",
            return_value=_trends(),
        ),
    ):
        first = build_source_snapshot("EXAMPLE.NS", "2026-08-03", cache_dir=tmp_path)
        second = build_source_snapshot("EXAMPLE.NS", "2026-08-03", cache_dir=tmp_path)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert profile.call_count == 1


def _replay_snapshot(ticker: str = "EXAMPLE.NS", trade_date: str = "2026-09-18") -> dict:
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "primary_data_available": True,
        "sources": {
            "company_news": {
                "status": "OK",
                "content": "Frozen company news snapshot",
            }
        },
        "data_quality_tags": [],
    }


@pytest.mark.unit
def test_injected_replay_snapshot_bypasses_live_source_builder(monkeypatch) -> None:
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    snapshot = _replay_snapshot()

    def fail_live_builder(*args, **kwargs):
        raise AssertionError("live source builder should not run for replay snapshots")

    monkeypatch.setattr(graph, "build_source_snapshot", fail_live_builder)

    assert (
        graph.resolve_source_snapshot(
            "EXAMPLE.NS",
            "2026-09-18",
            sentiment_source_snapshot=snapshot,
        )
        is snapshot
    )


@pytest.mark.unit
def test_normal_run_still_calls_live_source_builder(monkeypatch) -> None:
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    live_snapshot = _replay_snapshot()
    calls = []

    def live_builder(company_name, trade_date, asset_type="stock"):
        calls.append((company_name, trade_date, asset_type))
        return live_snapshot

    monkeypatch.setattr(graph, "build_source_snapshot", live_builder)

    resolved = graph.resolve_source_snapshot("EXAMPLE.NS", "2026-09-18")

    assert resolved is live_snapshot
    assert calls == [("EXAMPLE.NS", "2026-09-18", "stock")]


@pytest.mark.unit
def test_injected_replay_snapshot_rejects_ticker_and_date_mismatch() -> None:
    with pytest.raises(ValueError, match="ticker mismatch"):
        validate_replay_source_snapshot(
            _replay_snapshot(ticker="OTHER.NS"),
            "EXAMPLE.NS",
            "2026-09-18",
        )

    with pytest.raises(ValueError, match="trade_date mismatch"):
        validate_replay_source_snapshot(
            _replay_snapshot(trade_date="2026-09-17"),
            "EXAMPLE.NS",
            "2026-09-18",
        )


@pytest.mark.unit
def test_unavailable_optional_sources_reject_invented_counts() -> None:
    snapshot = {
        "sources": {
            "telegram": {"status": "DISABLED"},
            "google_trends": {"status": "UNAVAILABLE"},
        }
    }
    issues = _unsupported_claim_issues(
        "Telegram has 14 messages. Google Trends score is 73/100.", snapshot
    )
    assert len(issues) == 2


@pytest.mark.unit
def test_unavailable_optional_sources_reject_directional_inference() -> None:
    snapshot = {
        "sources": {
            "telegram": {"status": "DISABLED"},
            "reddit": {"status": "NO_DATA"},
        }
    }

    issues = _unsupported_claim_issues(
        "The lack of data across Reddit and Telegram leaves the trade exposed "
        "to liquidity risk.",
        snapshot,
    )

    assert any("directional evidence" in issue for issue in issues)


@pytest.mark.unit
def test_unavailable_optional_sources_reject_zero_scores_and_generic_gap_risk() -> None:
    snapshot = {
        "sources": {
            "telegram": {"status": "DISABLED"},
            "reddit": {"status": "NO_DATA"},
        }
    }
    report = """
| Reddit | NO_DATA | No posts | 0 (Neutral/Unknown) |
| Telegram | DISABLED | Not enabled | 0 (Neutral/Unknown) |

Disabled or unavailable channels are sentiment blind spots. This raises the
probability of false breakouts and liquidity risk.
"""

    issues = _unsupported_claim_issues(report, snapshot)

    assert any("Reddit" in issue and "zero score" in issue for issue in issues)
    assert any("Telegram" in issue and "zero score" in issue for issue in issues)
    assert any("directional evidence" in issue for issue in issues)


@pytest.mark.unit
def test_optional_source_gaps_are_explicitly_non_directional() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {
                "reddit": {"status": "NO_DATA"},
                "telegram": {"status": "DISABLED"},
                "google_trends": {"status": "OK"},
            }
        }
    }

    instruction = get_data_quality_instruction(state)

    assert "Reddit" not in instruction
    assert "Telegram" not in instruction
    assert "Google Trends" not in instruction


@pytest.mark.unit
def test_unavailable_reddit_and_telegram_are_hidden_from_sentiment_prompt_and_fallback() -> None:
    snapshot = {
        "sources": {
            "company_news": {"status": "NO_DATA", "content": "<no Yahoo news>"},
            "tavily_company_news": {
                "status": "OK",
                "content": "Tavily headline: Example wins order",
            },
            "google_news": {"status": "OK", "content": "Google headline: Example wins order"},
            "reddit": {"status": "NO_DATA", "content": "<no Reddit posts>"},
            "telegram": {"status": "DISABLED", "content": "<Telegram disabled>"},
            "google_trends": {"status": "UNAVAILABLE", "content": "<no trends>"},
        }
    }

    prompt = _build_system_message(
        ticker="EXAMPLE.NS",
        start_date="2026-08-01",
        end_date="2026-08-08",
        snapshot=snapshot,
    )
    fallback, _ = _grounded_fallback_report(snapshot)

    assert "Tavily headline: Example wins order" in prompt
    assert "Google headline: Example wins order" in prompt
    assert "<no Reddit posts>" not in prompt
    assert "<Telegram disabled>" not in prompt
    assert "<no Reddit posts>" not in fallback
    assert "<Telegram disabled>" not in fallback


@pytest.mark.unit
def test_upstream_model_metadata_is_explicitly_excluded() -> None:
    instruction = get_data_quality_instruction({})
    instrument = build_instrument_context("EXAMPLE.NS")

    assert "No upstream selector or prediction model output" in instruction
    assert "Do not infer or mention an LSTM signal" in instruction
    assert "No upstream selector or prediction model output" in instrument


@pytest.mark.unit
def test_upstream_claims_are_removed_and_tagged() -> None:
    text = (
        "The LSTM momentum signal is reliable and its model score supports BUY. "
        "Verified price action is constructive."
    )

    assert len(find_unsupported_upstream_claims(text)) == 1
    sanitized, tags = sanitize_agent_output(text, {})

    assert "LSTM" not in sanitized
    assert "model score" not in sanitized
    assert "Verified price action is constructive" in sanitized
    assert tags == ["REMOVED_UNSUPPORTED_UPSTREAM_CLAIM"]


@pytest.mark.unit
def test_ticker_tool_calls_are_forced_to_exact_graph_instrument() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_fundamentals",
                "args": {"ticker": "BIOON.NS", "curr_date": "2026-08-03"},
                "id": "call-1",
                "type": "tool_call",
            },
            {
                "name": "get_global_news",
                "args": {"curr_date": "2026-08-03"},
                "id": "call-2",
                "type": "tool_call",
            },
        ],
    )

    corrected = enforce_exact_tool_ticker(message, "BIOCON.NS")

    assert corrected is True
    assert message.tool_calls[0]["args"]["ticker"] == "BIOCON.NS"
    assert "ticker" not in message.tool_calls[1]["args"]


@pytest.mark.unit
def test_final_decision_gets_one_optional_source_grounding_repair() -> None:
    llm = FakeListChatModel(
        responses=[
            "Telegram is unavailable, so the lack of buyers is bearish. **Rating**: Sell",
            "Verified market evidence is constructive. **Rating**: Buy",
        ]
    )
    node = create_portfolio_manager(llm)
    state = {
        "company_of_interest": "EXAMPLE.NS",
        "investment_plan": "Buy on verified market strength.",
        "trader_investment_plan": "BUY",
        "risk_debate_state": {
            "history": "Primary evidence is constructive.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "data_quality_tags": ["MISSING_TELEGRAM"],
        "sentiment_source_snapshot": {
            "sources": {"telegram": {"status": "DISABLED"}}
        },
    }

    result = node(state)

    assert "lack of buyers" not in result["final_trade_decision"]
    assert "REPAIRED_FINAL_GROUNDING" in result["data_quality_tags"]


@pytest.mark.unit
def test_final_decision_gets_one_upstream_model_grounding_repair() -> None:
    llm = FakeListChatModel(
        responses=[
            "The LSTM signal and model score support BUY. **Rating**: Buy",
            "Verified market evidence supports HOLD. **Rating**: Hold",
        ]
    )
    node = create_portfolio_manager(llm)
    state = {
        "company_of_interest": "EXAMPLE.NS",
        "investment_plan": "Use only verified ticker evidence.",
        "trader_investment_plan": "HOLD",
        "risk_debate_state": {
            "history": "Primary evidence is balanced.",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 1,
        },
        "data_quality_tags": [],
        "sentiment_source_snapshot": {"sources": {}},
    }

    result = node(state)

    assert "LSTM" not in result["final_trade_decision"]
    assert "model score" not in result["final_trade_decision"]
    assert "REPAIRED_FINAL_GROUNDING" in result["data_quality_tags"]


@pytest.mark.unit
def test_optional_gap_claim_detector_does_not_reject_market_liquidity_evidence() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {"telegram": {"status": "DISABLED"}}
        }
    }

    assert find_unsupported_optional_source_claims(
        "Telegram is unavailable, so there are no buyers and risk is elevated.",
        state,
    )
    assert not find_unsupported_optional_source_claims(
        "Exchange volume is below its 20-session average, indicating thin liquidity.",
        state,
    )


@pytest.mark.unit
def test_downstream_sanitizer_removes_bullish_and_bearish_gap_inferences() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {
                "company_news": {"status": "NO_DATA"},
                "telegram": {"status": "DISABLED"},
            }
        }
    }
    text = (
        "Missing Telegram data means buyers lack conviction. "
        "Social media sentiment gaps signal a lack of retail FUD and leave room for upside. "
        "Exchange volume expanded above its 20-session average."
    )

    sanitized, removed = sanitize_unsupported_source_claims(text, state)

    assert removed is True
    assert "buyers lack conviction" not in sanitized
    assert "lack of retail FUD" not in sanitized
    assert "Exchange volume expanded" in sanitized
    assert sanitized.count("Unsupported source-gap inference removed by validation") == 2


@pytest.mark.unit
def test_gap_scanner_ignores_its_own_validation_marker() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {"telegram": {"status": "DISABLED"}}
        }
    }
    text = (
        "[Unsupported source-gap inference removed: unavailable sources are unknown "
        "and carry zero directional weight.] The verified price trend supports HOLD."
    )

    assert not find_unsupported_optional_source_claims(text, state)


@pytest.mark.unit
def test_gap_scanner_allows_explicit_non_directional_missing_source_treatment() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {
                "company_news": {"status": "NO_DATA"},
                "reddit": {"status": "NO_DATA"},
                "telegram": {"status": "DISABLED"},
            }
        }
    }
    text = (
        "Yahoo Finance company news returned no data. "
        "This is missing/unknown data, not a bearish signal.\n\n"
        "Reddit returned no posts. "
        "This is missing data, not evidence of bearish retail sentiment.\n\n"
        "Telegram is disabled. No directional weight is assigned."
    )

    assert not find_unsupported_optional_source_claims(text, state)


@pytest.mark.unit
def test_gap_scanner_still_rejects_missing_source_direction_after_safe_phrase() -> None:
    state = {
        "sentiment_source_snapshot": {
            "sources": {"telegram": {"status": "DISABLED"}}
        }
    }
    text = (
        "Telegram is disabled and has no directional weight, "
        "but this still raises downside risk because retail demand is unknown."
    )

    assert find_unsupported_optional_source_claims(text, state)


@pytest.mark.unit
def test_sentiment_report_gets_one_grounded_repair_attempt() -> None:
    llm = FakeListChatModel(
        responses=[
            "Telegram has 14 messages. Google Trends score is 73/100.",
            "Yahoo news is supportive. Telegram and Google Trends are unavailable.",
        ]
    )
    node = create_sentiment_analyst(llm)
    result = node(
        {
            "company_of_interest": "EXAMPLE.NS",
            "asset_type": "stock",
            "trade_date": "2026-08-03",
            "messages": [HumanMessage(content="EXAMPLE.NS")],
            "trade_context_note": "momentum trade | 7d horizon",
            "data_quality_tags": ["MISSING_TELEGRAM", "MISSING_GOOGLE_TRENDS"],
            "sentiment_source_snapshot": {
                "company_name": "Example Limited",
                "sources": {
                    "company_news": {"status": "OK", "content": "Supported headline"},
                    "google_news": {"status": "OK", "content": "Supported India headline"},
                    "reddit": {"status": "NO_DATA", "content": "<none>"},
                    "telegram": {"status": "DISABLED", "content": "<disabled>"},
                    "google_trends": {"status": "UNAVAILABLE", "content": "<unavailable>"},
                },
            },
        }
    )

    assert "14 messages" not in result["sentiment_report"]
    assert "73/100" not in result["sentiment_report"]
    assert "REPAIRED_SENTIMENT_REPORT" in result["data_quality_tags"]


@pytest.mark.unit
def test_second_grounding_failure_keeps_valid_sources_in_deterministic_digest() -> None:
    llm = FakeListChatModel(
        responses=[
            "Telegram is unavailable, so liquidity risk is higher.",
            "Missing Telegram data leaves the trade exposed to downside risk.",
        ]
    )
    node = create_sentiment_analyst(llm)
    result = node(
        {
            "company_of_interest": "EXAMPLE.NS",
            "asset_type": "stock",
            "trade_date": "2026-08-03",
            "messages": [HumanMessage(content="EXAMPLE.NS")],
            "data_quality_tags": ["MISSING_TELEGRAM"],
            "sentiment_source_snapshot": {
                "company_name": "Example Limited",
                "sources": {
                    "company_news": {"status": "NO_DATA", "content": "<none>"},
                    "google_news": {"status": "OK", "content": "Verified headline"},
                    "reddit": {"status": "NO_DATA", "content": "<none>"},
                    "telegram": {"status": "DISABLED", "content": "<disabled>"},
                    "google_trends": {"status": "UNAVAILABLE", "content": "<unavailable>"},
                },
            },
        }
    )

    assert "Verified headline" in result["sentiment_report"]
    assert "No sentiment direction is synthesized" in result["sentiment_report"]
    assert "INVALID_SENTIMENT_REPORT" in result["data_quality_tags"]
    assert "FALLBACK_SENTIMENT_DIGEST" in result["data_quality_tags"]
    assert "MISSING_SENTIMENT" not in result["data_quality_tags"]


@pytest.mark.unit
def test_yfinance_profile_keeps_completed_status_with_missing_optional_reports() -> None:
    state = {
        "sentiment_source_snapshot": {"primary_data_available": True},
        "data_quality_tags": ["MISSING_TELEGRAM"],
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "final_trade_decision": "HOLD",
    }
    TradingAgentsGraph._finalize_analysis_quality(state)
    assert state["analysis_status"] == "COMPLETE"
    assert "MISSING_SENTIMENT" in state["data_quality_tags"]


@pytest.mark.unit
def test_no_primary_data_or_reports_is_failed() -> None:
    state = {
        "sentiment_source_snapshot": {"primary_data_available": False},
        "data_quality_tags": ["MISSING_SENTIMENT"],
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "final_trade_decision": "HOLD",
    }
    TradingAgentsGraph._finalize_analysis_quality(state)
    assert state["analysis_status"] == "FAILED"
