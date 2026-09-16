"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    apply_trader_policy,
    build_instrument_context,
    get_data_quality_instruction,
    get_language_instruction,
    sanitize_agent_output,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(
            company_name,
            asset_type,
            lstm_context_available=bool(state.get("lstm_signal_context")),
        )
        investment_plan = state["investment_plan"]

        ctx = state.get("trade_context_note", "")
        ctx_line = f"\n\n---\nIMPORTANT CONTEXT — Trade parameters: {ctx}\nEvaluate whether the GIVEN entry, target, and stop work for the GIVEN horizon. These levels are constraints — assess whether the target can be reached before the stop, not set ideal replacement levels. If you populate the structured Entry Price or Stop Loss fields, copy the supplied entry and stop exactly; never substitute a technical support, resistance, or early-exit level for the fixed strategy stop." if ctx else ""
        lstm_ctx = state.get("lstm_context_note", "")
        lstm_line = f"\n\n---\n{lstm_ctx}" if lstm_ctx else ""

        messages = [
            {
                "role": "system",
                "content": (
                    ctx_line
                    + lstm_line
                    + "You are a trading agent analyzing market data to make short-term trading decisions. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "Use Hold as the default when the setup lacks a proven edge. "
                    "A BUY requires either a verified positive catalyst or clearly aligned momentum within the trade window. "
                    "A SELL requires verified primary ticker-specific downside evidence that can matter within the trade window. "
                    "Do not turn missing confirmation alone or broad macro caution into a SELL call. "
                    "When LSTM evidence is supplied, fill every LSTM thesis field and explain agreement or disagreement using verified current evidence."
                    + get_data_quality_instruction(state)
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, ticker-specific news, company "
                    f"fundamentals, and sentiment evidence. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )
        trader_plan, output_tags = sanitize_agent_output(
            trader_plan, state
        )
        trader_plan, policy_tags = apply_trader_policy(trader_plan, state)
        tags = list(state.get("data_quality_tags") or [])
        tags.extend(output_tags)
        tags.extend(policy_tags)

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
            "data_quality_tags": sorted(set(tags)),
        }

    return functools.partial(trader_node, name="Trader")
