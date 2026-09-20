"""Shared entry semantics for downstream synthesis of LSTM candidates."""


ENTRY_RATING_SCALE = """**Entry Rating Scale** (use exactly one):
- **Buy**: Enter the supplied long trade; evidence supports target before stop within the horizon.
- **Hold**: Stay flat; the entry decision is unresolved.
- **Sell**: Avoid the long entry because verified evidence is adverse; do not open a short.
"""


def get_entry_decision_instruction(state: dict) -> str:
    if not state.get("lstm_signal_context"):
        return ""
    contract = state["lstm_signal_context"]["strategy_contract"]
    target = float(contract["decision_target_pct"]) * 100
    stop = float(contract["stop_loss_pct"]) * 100
    horizon = contract["maximum_horizon_exchange_sessions"]
    return f"""

LSTM CANDIDATE ENTRY POLICY:
Assess a prospective long entry from a flat position at the supplied entry price.
Does the available evidence support {target:+.2f}% before {stop:+.2f}% within
{horizon} exchange sessions, even before a full trend reversal?
The LSTM is a meaningful uncalibrated quantitative prior. Weigh it with dated
market facts and verified risks; neither a high score nor absence of invalidation
alone justifies BUY. A plausible bounce or reachable target alone is insufficient.
Do not require a positive news catalyst, positive MACD, or moving-average reclaim
as a universal entry condition. A pullback can reach the target before full trend
repair. If proposing a confirmation level, compare it with the supplied target:
confirmation at or beyond that target cannot be a prerequisite for this entry.
Assess evidence favoring target first, stop first, and failure to resolve within
the horizon. Weak indicators can matter, but explain their specific impact on
this path rather than treating expected pullback symptoms as automatic rejection.
The binary target/stop break-even rate is payoff arithmetic, not a measured
forecast or a requirement to invent or verify a numerical win probability.

In decision outputs, LSTM Thesis Assessment describes this entry objective:
- SUPPORTED: evidence favors the target-before-stop entry; choose BUY.
- REJECTED: verified adverse evidence argues against the entry; choose SELL.
- INSUFFICIENT_EVIDENCE: the entry remains unresolved; choose HOLD.
These are your judgments, not automatic rules based on model score. Do not call
the entry SUPPORTED merely because the longer-term trend is intact.
Use Buy/Hold/Sell for this workflow, not Overweight/Underweight. HOLD means no
entry, not maintain exposure. Do not invent holdings, suggest trimming or keeping
a position, or recommend a starter purchase while labeling the decision HOLD.
Position sizing and execution management belong to the trading engine.

Separate unavailable facts from uncertainty about the future. Fill Evidence
Limitation with NONE, MISSING_INPUTS, FORECAST_UNCERTAINTY, or BOTH. Fill Missing
Inputs with specific unavailable facts essential to this decision and why they
matter (empty when none); fill Forecast Uncertainty with unresolved path questions
despite available data (empty when none). Use BOTH only when both apply.
INSUFFICIENT_EVIDENCE needs at least one specific reason in those fields.
An uncalibrated score or unconfirmed reversal is not a missing input. Missing
optional news/social sources are not adverse evidence or an automatic HOLD.
For free-text decisions, include **Evidence Limitation**, **Missing Inputs** and
**Forecast Uncertainty** with the same meanings; explicitly write None when empty.
"""
