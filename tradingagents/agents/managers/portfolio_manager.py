"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    find_unsupported_optional_source_claims,
    find_unsupported_upstream_claims,
    get_data_quality_instruction,
    get_language_instruction,
    sanitize_agent_output,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        ctx = state.get("trade_context_note", "")
        ctx_line = f"\n\n---\nIMPORTANT CONTEXT — Trade parameters: {ctx}\nRate this as a SHORT-TERM trade, not a long-term investment. Valuation multiples (P/E, EV/EBITDA) are largely irrelevant for this duration. The question is: does this trade work within the given horizon and stop-loss?" if ctx else ""

        prompt = f"""{ctx_line}As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_data_quality_instruction(state)}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        tags = list(state.get("data_quality_tags") or [])
        unsupported_claims = (
            find_unsupported_optional_source_claims(final_trade_decision, state)
            + find_unsupported_upstream_claims(final_trade_decision)
        )
        if unsupported_claims:
            repair_prompt = (
                prompt
                + "\n\nYour previous draft used unsupported evidence. Rewrite the complete "
                "decision once. Preserve valid ticker-specific evidence and the rating only if "
                "verified evidence supports it. Do not use source absence or an invented upstream "
                "selector/model signal as evidence. Problematic draft excerpts:\n- "
                + "\n- ".join(unsupported_claims)
            )
            final_trade_decision = invoke_structured_or_freetext(
                structured_llm,
                llm,
                repair_prompt,
                render_pm_decision,
                "Portfolio Manager grounding repair",
            )
            remaining_claims = (
                find_unsupported_optional_source_claims(final_trade_decision, state)
                + find_unsupported_upstream_claims(final_trade_decision)
            )
            if remaining_claims:
                tags.append("INVALID_FINAL_GROUNDING")
                final_trade_decision, output_tags = sanitize_agent_output(
                    final_trade_decision, state
                )
                tags.extend(output_tags)
            else:
                tags.append("REPAIRED_FINAL_GROUNDING")

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "data_quality_tags": sorted(set(tags)),
        }

    return portfolio_manager_node
