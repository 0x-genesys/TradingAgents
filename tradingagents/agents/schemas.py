"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class LSTMThesisAssessment(str, Enum):
    """Independent TA assessment of the supplied quantitative setup."""

    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceLimitation(str, Enum):
    NONE = "NONE"
    MISSING_INPUTS = "MISSING_INPUTS"
    FORECAST_UNCERTAINTY = "FORECAST_UNCERTAINTY"
    BOTH = "BOTH"


class EntryEvidence(BaseModel):
    """Explicit uncertainty reasons; omitted for legacy/standalone responses."""

    evidence_limitation: Optional[EvidenceLimitation] = Field(
        default=None,
        description=(
            "Required for LSTM candidate entry decisions. NONE means no material limitation; "
            "MISSING_INPUTS means essential facts are unavailable; FORECAST_UNCERTAINTY "
            "means available facts leave the future path unresolved; BOTH means both apply. "
            "Do not classify an uncalibrated model score or pending reversal as missing data."
        ),
    )
    missing_inputs: Optional[list[str]] = Field(
        default=None,
        description=(
            "For LSTM entry decisions, list specific unavailable facts essential to the "
            "decision and why each matters. Use [] when none. Missing optional sources "
            "alone do not justify a directional decision."
        ),
    )
    forecast_uncertainty: Optional[list[str]] = Field(
        default=None,
        description=(
            "For LSTM entry decisions, list unresolved target/stop/horizon questions "
            "despite available facts. Use [] when none. This is distinct from missing data."
        ),
    )


def _render_evidence_limitations(evidence: EntryEvidence) -> list[str]:
    parts = []
    if evidence.evidence_limitation is not None:
        parts.extend(["", f"**Evidence Limitation**: {evidence.evidence_limitation.value}"])
    for label, reasons in (
        ("Missing Inputs", evidence.missing_inputs),
        ("Forecast Uncertainty", evidence.forecast_uncertainty),
    ):
        if reasons is not None:
            parts.extend(["", f"**{label}**: " + ("; ".join(reasons) or "None")])
    return parts


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(EntryEvidence):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "For a flat-position LSTM candidate, state enter, stay flat, or avoid the "
            "long entry and leave sizing to the trading engine. Otherwise include "
            "position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ] + _render_evidence_limitations(plan))


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(EntryEvidence):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance for standalone analysis; omit for LSTM entry decisions (execution engine owns sizing).",
    )
    lstm_thesis_assessment: Optional[LSTMThesisAssessment] = Field(
        default=None,
        description=(
            "Required when structured LSTM evidence is supplied. State whether current "
            "verified evidence supports (BUY), rejects (SELL/avoid long), or cannot "
            "resolve (HOLD/stay flat) the supplied target-before-stop entry within "
            "the horizon. A plausible setup alone is not SUPPORTED. Uncertainty "
            "needs specific missing_inputs and/or forecast_uncertainty reasons."
        ),
    )
    candidate_rank: Optional[int] = Field(
        default=None,
        description="Copy the supplied LSTM cross-sectional rank when evidence is present.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Verified current evidence supporting the LSTM thesis.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Verified current evidence contradicting the LSTM thesis.",
    )
    model_disagreement_reason: Optional[str] = Field(
        default=None,
        description=(
            "Required when the action disagrees with the LSTM upward thesis; otherwise "
            "briefly state why no disagreement exists."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    if proposal.lstm_thesis_assessment is not None:
        parts.extend([
            "",
            f"**LSTM Thesis Assessment**: {proposal.lstm_thesis_assessment.value}",
            f"**Candidate Rank**: {proposal.candidate_rank if proposal.candidate_rank is not None else 'N/A'}",
            "**Supporting Evidence**: " + ("; ".join(proposal.supporting_evidence) or "None verified"),
            "**Contradicting Evidence**: " + ("; ".join(proposal.contradicting_evidence) or "None verified"),
            f"**Model Disagreement Reason**: {proposal.model_disagreement_reason or 'None'}",
        ])
    parts.extend(_render_evidence_limitations(proposal))
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(EntryEvidence):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. For flat-position "
            "LSTM candidates use Buy (enter), Hold (stay flat), or Sell (avoid long) only."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, key risk levels, and time "
            "horizon. Two to four sentences. For LSTM candidates assume flat, do not "
            "invent holdings, and leave position sizing to the execution engine."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    lstm_thesis_assessment: Optional[LSTMThesisAssessment] = Field(
        default=None,
        description=(
            "Required when structured LSTM evidence is supplied. Independently classify "
            "the target-before-stop entry within the supplied horizon: SUPPORTED "
            "(BUY), REJECTED (SELL/avoid long), or INSUFFICIENT_EVIDENCE (HOLD/stay flat). "
            "An intact long-term trend alone is not SUPPORTED. Uncertainty needs "
            "specific missing_inputs and/or forecast_uncertainty reasons."
        ),
    )
    candidate_rank: Optional[int] = Field(
        default=None,
        description="Copy the supplied LSTM cross-sectional rank when evidence is present.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Verified current evidence supporting the LSTM thesis.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Verified current evidence contradicting the LSTM thesis.",
    )
    model_disagreement_reason: Optional[str] = Field(
        default=None,
        description=(
            "Explain any disagreement between the final rating and the LSTM upward thesis."
        ),
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    if decision.lstm_thesis_assessment is not None:
        parts.extend([
            "",
            f"**LSTM Thesis Assessment**: {decision.lstm_thesis_assessment.value}",
            f"**Candidate Rank**: {decision.candidate_rank if decision.candidate_rank is not None else 'N/A'}",
            "**Supporting Evidence**: " + ("; ".join(decision.supporting_evidence) or "None verified"),
            "**Contradicting Evidence**: " + ("; ".join(decision.contradicting_evidence) or "None verified"),
            f"**Model Disagreement Reason**: {decision.model_disagreement_reason or 'None'}",
        ])
    parts.extend(_render_evidence_limitations(decision))
    return "\n".join(parts)
