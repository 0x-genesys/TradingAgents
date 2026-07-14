# TradingAgents/graph/propagation.py

from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        trade_horizon_days: Optional[int] = None,
        entry_price: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        trade_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        When ``trade_horizon_days`` is provided, a human-readable
        ``trade_context_note`` is built and stored so analysts can frame their
        analysis for the specific trade parameters.  When omitted (None), the
        default long-term analysis is used unchanged.
        """
        # Build human-readable trade context note if horizon is provided.
        trade_context_note = ""
        if trade_horizon_days is not None:
            entry_str = f"₹{entry_price:,.2f}" if entry_price is not None else "N/A"
            stop_str = f"{stop_loss_pct*100:.1f}%" if stop_loss_pct is not None else "N/A"
            strategy_str = trade_strategy or "momentum"
            trade_context_note = (
                f"{strategy_str} trade | {trade_horizon_days}d horizon | "
                f"Entry: {entry_str} | Stop: {stop_str}"
            )

        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
            "trade_horizon_days": trade_horizon_days,
            "entry_price": entry_price,
            "stop_loss_pct": stop_loss_pct,
            "trade_strategy": trade_strategy,
            "trade_context_note": trade_context_note,
        }

    def get_graph_args(self, callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
