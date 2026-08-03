from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from tradingagents.agents.analysts.market_analyst import create_market_analyst


class ToolCompatibleFakeChatModel(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self


@pytest.mark.unit
def test_empty_market_response_gets_one_synthesis_retry() -> None:
    llm = ToolCompatibleFakeChatModel(
        responses=["", "Supported technical market report\n\n| Signal | Value |\n|---|---|"]
    )
    node = create_market_analyst(llm)
    result = node(
        {
            "company_of_interest": "EXAMPLE.NS",
            "asset_type": "stock",
            "trade_date": "2026-08-03",
            "messages": [HumanMessage(content="EXAMPLE.NS")],
            "trade_context_note": "momentum trade | 7d horizon",
            "data_quality_tags": [],
            "sentiment_source_snapshot": {"company_name": "Example Limited"},
        }
    )

    assert result["market_report"].startswith("Supported technical market report")
    assert result["data_quality_tags"] == ["REPAIRED_MARKET_REPORT"]
