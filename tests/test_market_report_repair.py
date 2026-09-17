from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from unittest.mock import Mock

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


@pytest.mark.parametrize("ticker,draft", [
    ("NMDC.NS", '<tool_code>\nprint(get_stock_data(ticker="NMDC.NS"))\n</tool_code>'),
    ("ABB.NS", '<tool_code>\nprint(get_stock_data(ticker="ABB.NS", start_date="2026-09-09", end_date="2026-09-16"))\n</tool_code>'),
    ("ABB.NS", '```python\nget_stock_data("ABB.NS")\n```'),
])
def test_september_16_tool_text_is_repaired_or_fails(ticker, draft):
    state = {"company_of_interest": ticker, "trade_date": "2026-09-16",
             "messages": [HumanMessage(content=ticker)]}
    valid = "Technical evidence: price remains below SMA50; reversal is unconfirmed."
    result = create_market_analyst(ToolCompatibleFakeChatModel(responses=[draft, valid]))(state)
    assert result["market_report"] == valid
    assert result["messages"][-1].content == valid
    assert "REPAIRED_MARKET_REPORT" in result["data_quality_tags"]
    with pytest.raises(RuntimeError, match="INVALID_MARKET_REPORT"):
        create_market_analyst(ToolCompatibleFakeChatModel(responses=[draft, draft]))(state)


def test_real_tool_call_keeps_graph_tool_routing():
    from tradingagents.graph.conditional_logic import ConditionalLogic

    llm = Mock()
    llm.bind_tools.return_value = RunnableLambda(lambda _: AIMessage(content="", tool_calls=[
        {"name": "get_stock_data", "args": {"symbol": "NMDC.NS"}, "id": "ohlc"}
    ]))
    state = {"company_of_interest": "NMDC.NS", "trade_date": "2026-09-16",
             "messages": [HumanMessage(content="NMDC.NS")]}
    result = create_market_analyst(llm)(state)
    assert result["market_report"] == ""
    assert ConditionalLogic().should_continue_market(result) == "tools_market"
    llm.invoke.assert_not_called()
