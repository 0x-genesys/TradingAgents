from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from tradingagents.agents.analysts.news_analyst import (
    _news_report_issues,
    create_news_analyst,
)


class ToolRecordingFakeChatModel(FakeListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tool_names = [tool.name for tool in tools]
        return self


def _state(*, company_status: str = "NO_DATA", google_status: str = "OK") -> dict:
    return {
        "company_of_interest": "EXAMPLE.NS",
        "asset_type": "stock",
        "trade_date": "2026-08-03",
        "messages": [HumanMessage(content="Analyze EXAMPLE.NS")],
        "trade_context_note": "momentum trade | 7-session horizon",
        "data_quality_tags": ["MISSING_TELEGRAM"],
        "sentiment_source_snapshot": {
            "company_name": "Example Limited",
            "sources": {
                "company_news": {
                    "status": company_status,
                    "content": (
                        "<no Yahoo company news>"
                        if company_status == "NO_DATA"
                        else "Yahoo headline: Example wins a new contract"
                    ),
                },
                "google_news": {
                    "status": google_status,
                    "content": (
                        "Google headline: Example wins regulatory approval"
                        if google_status == "OK"
                        else "<no Google News results>"
                    ),
                },
                "telegram": {"status": "DISABLED", "content": "<disabled>"},
            },
        },
    }


@pytest.mark.unit
def test_stock_news_uses_frozen_snapshot_and_only_live_macro_tool() -> None:
    llm = ToolRecordingFakeChatModel(
        responses=["Google News reports that Example won regulatory approval."]
    )
    result = create_news_analyst(llm)(_state())

    assert llm.bound_tool_names == ["get_global_news"]
    assert "Google headline: Example wins regulatory approval" in result["news_report"]
    assert "Yahoo Finance company news" in result["news_report"]
    assert "India-localized Google News" in result["news_report"]


@pytest.mark.unit
def test_contradictory_no_news_claim_gets_one_grounded_repair() -> None:
    llm = ToolRecordingFakeChatModel(
        responses=[
            "No company-specific news was found. Yahoo Finance returned no items.",
            "Yahoo Finance has no data, while Google News reports regulatory approval.",
        ]
    )
    result = create_news_analyst(llm)(_state())

    assert "No company-specific news was found" not in result["news_report"]
    assert "Google headline: Example wins regulatory approval" in result["news_report"]
    assert "REPAIRED_NEWS_REPORT" in result["data_quality_tags"]
    assert "FALLBACK_NEWS_DIGEST" not in result["data_quality_tags"]


@pytest.mark.unit
def test_accurate_source_specific_gap_does_not_trigger_repair() -> None:
    state = _state()
    report = "Yahoo Finance has no company-specific news, but Google News has one item."

    assert _news_report_issues(report, state["sentiment_source_snapshot"]) == []


@pytest.mark.unit
def test_second_news_grounding_failure_uses_deterministic_digest() -> None:
    llm = ToolRecordingFakeChatModel(
        responses=[
            "No company-specific news was found.",
            "No ticker-specific coverage is available.",
        ]
    )
    result = create_news_analyst(llm)(_state())

    assert "Google headline: Example wins regulatory approval" in result["news_report"]
    assert "failed ticker-news grounding validation twice" in result["news_report"]
    assert "INVALID_NEWS_REPORT" in result["data_quality_tags"]
    assert "FALLBACK_NEWS_DIGEST" in result["data_quality_tags"]
