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
from tradingagents.agents.utils.grounding import repair_decision_grounding
from tradingagents.agents.utils.entry_decision import ENTRY_RATING_SCALE
from tradingagents.agents.utils.agent_utils import (
    apply_portfolio_manager_policy,
    build_instrument_context,
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
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            lstm_context_available=bool(state.get("lstm_signal_context")),
        )

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
        ctx_line = f"\n\n---\nIMPORTANT CONTEXT — Trade parameters: {ctx}\nRate this as a SHORT-TERM trade, not a long-term investment. Valuation multiples (P/E, EV/EBITDA) are largely irrelevant for this duration. The decision objective is to reach the fixed target before the fixed stop within the given horizon; do not invent replacement levels." if ctx else ""
        lstm_ctx = state.get("lstm_context_note", "")
        lstm_line = f"\n\n---\n{lstm_ctx}" if lstm_ctx else ""

        rating_policy = ENTRY_RATING_SCALE if state.get("lstm_signal_context") else """
**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

Be decisive and ground every conclusion in specific evidence from the analysts. Use Hold as the default when the setup lacks a proven short-term edge. A Buy or Overweight rating needs either a verified positive catalyst or clearly aligned momentum. A Sell or Underweight rating needs verified primary ticker-specific downside evidence that can matter within the stated trade window. Missing confirmation alone or broad macro caution are not enough for a bearish rating.
"""

        prompt = f"""{ctx_line}{lstm_line}As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision. When LSTM evidence is supplied, answer whether the pullback reversal is supported, whether established momentum remains intact, and whether the fixed target can occur before the stop within the horizon. Fill every LSTM thesis field. The final rating remains your independent decision.

{instrument_context}

{rating_policy}

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts.{get_data_quality_instruction(state)}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        tags = list(state.get("data_quality_tags") or [])
        final_trade_decision, grounding_tags = repair_decision_grounding(
            final_trade_decision, state, prompt, structured_llm, llm,
            render_pm_decision, "Portfolio Manager",
        )
        tags.extend(grounding_tags)
        final_trade_decision, output_tags = sanitize_agent_output(final_trade_decision, state)
        tags.extend(output_tags)

        final_trade_decision, policy_tags = apply_portfolio_manager_policy(
            final_trade_decision, state
        )
        tags.extend(policy_tags)

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
