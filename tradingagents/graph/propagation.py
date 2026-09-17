# TradingAgents/graph/propagation.py

from typing import Dict, Any, List, Optional
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.agents.utils.lstm_context import (
    render_lstm_compact_context,
    render_lstm_market_context,
    render_lstm_report,
    validate_lstm_signal_context,
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
        profit_target_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        trade_strategy: Optional[str] = None,
        lstm_signal_context: Optional[Dict[str, Any]] = None,
        sentiment_source_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph.

        When ``trade_horizon_days`` is provided, a human-readable
        ``trade_context_note`` is built and stored so analysts can frame their
        analysis for the specific trade parameters.  When omitted (None), the
        default long-term analysis is used unchanged.
        """
        lstm_context_note = ""
        lstm_market_context_note = ""
        lstm_quantitative_evidence = ""
        if lstm_signal_context:
            validate_lstm_signal_context(
                lstm_signal_context,
                company_name,
                str(trade_date),
            )
            signal = lstm_signal_context["signal"]
            contract = lstm_signal_context["strategy_contract"]
            expected = {
                "entry_price": float(signal["entry_price"]),
                "profit_target_pct": float(contract["decision_target_pct"]),
                "stop_loss_pct": float(contract["stop_loss_pct"]),
                "trade_horizon_days": int(
                    contract["maximum_horizon_exchange_sessions"]
                ),
                "trade_strategy": str(contract["setup"]),
            }
            supplied = {
                "entry_price": entry_price,
                "profit_target_pct": profit_target_pct,
                "stop_loss_pct": stop_loss_pct,
                "trade_horizon_days": trade_horizon_days,
                "trade_strategy": trade_strategy,
            }
            for field_name, expected_value in expected.items():
                supplied_value = supplied[field_name]
                if supplied_value is None:
                    continue
                if isinstance(expected_value, float):
                    matches = abs(float(supplied_value) - expected_value) <= 1e-9
                else:
                    matches = supplied_value == expected_value
                if not matches:
                    raise ValueError(
                        f"{field_name} conflicts with canonical LSTM signal context"
                    )
            entry_price = expected["entry_price"]
            profit_target_pct = expected["profit_target_pct"]
            stop_loss_pct = expected["stop_loss_pct"]
            trade_horizon_days = expected["trade_horizon_days"]
            trade_strategy = expected["trade_strategy"]
            lstm_context_note = render_lstm_compact_context(lstm_signal_context)
            lstm_market_context_note = render_lstm_market_context(lstm_signal_context)
            lstm_quantitative_evidence = render_lstm_report(lstm_signal_context)

        # Build human-readable trade context note if horizon is provided.
        trade_context_note = ""
        if trade_horizon_days is not None:
            entry_str = f"₹{entry_price:,.2f}" if entry_price is not None else "N/A"
            target_str = f"+{profit_target_pct*100:.1f}%" if profit_target_pct is not None else "N/A"
            stop_str = f"{stop_loss_pct*100:.1f}%" if stop_loss_pct is not None else "N/A"
            strategy_str = trade_strategy or "momentum"
            trade_context_note = (
                f"{strategy_str} trade | {trade_horizon_days}d horizon | "
                f"Entry: {entry_str} | Target: {target_str} | Stop: {stop_str} | "
                f"Objective: reach target before stop within the horizon"
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
            "profit_target_pct": profit_target_pct,
            "stop_loss_pct": stop_loss_pct,
            "trade_strategy": trade_strategy,
            "trade_context_note": trade_context_note,
            "lstm_signal_context": lstm_signal_context or {},
            "lstm_context_note": lstm_context_note,
            "lstm_market_context_note": lstm_market_context_note,
            "lstm_quantitative_evidence": lstm_quantitative_evidence,
            "analysis_status": "COMPLETE",
            "data_quality_tags": list(
                (sentiment_source_snapshot or {}).get("data_quality_tags", [])
            ),
            "sentiment_source_snapshot": sentiment_source_snapshot or {},
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
