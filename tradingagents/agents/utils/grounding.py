"""Bounded evidence checks for decision synthesis, never an action policy.

These checks recognize explicit claims about supplied facts. They do not certify
all natural-language reasoning or estimate the probability of a winning trade.
"""

from __future__ import annotations

import math
import re

from tradingagents.agents.utils.agent_utils import (
    find_trade_arithmetic_issues,
    find_unsupported_optional_source_claims,
    find_unsupported_upstream_claims,
)
from tradingagents.agents.utils.structured import invoke_structured_or_freetext
from tradingagents.agents.utils.lstm_context import fixed_trade_facts


def _clauses(text: str) -> list[str]:
    normalized = text.translate(str.maketrans({"\u2212": "-", "\u2013": "-"}))
    normalized = re.sub(r"[*`]", "", normalized)
    return [s.strip() for s in re.split(
        r"(?<=[.!?])\s+|\n+|;|\b(?:but|yet|however)\b", normalized,
        flags=re.IGNORECASE,
    ) if s.strip()]


def find_current_evidence_issues(text: str, state: dict) -> list[str]:
    context = state.get("lstm_signal_context")
    if not context:
        return []
    signal = context["signal"]
    values = signal["technical_snapshot"]["values"]
    facts_date = signal["technical_snapshot"]["data_timestamp"]
    levels = fixed_trade_facts(context)
    issues = []
    model = r"\b(?:lstm|model|score|rank(?:ing)?|feature|geometry|quantitative prior)\b"
    proof = r"\b(?:confirms?|validates?|proves?|guarantees?|establishes?)\b"
    market_truth = r"intact|exhaustion|institutional|absorption|structural failure|non-destructive|uptrend|not occurring|trend breakdown"
    negation = r"\b(?:cannot|can't|doesn't|does not|not|no evidence to|insufficient evidence to)\s+(?:(?:by itself|itself|alone|necessarily)\s+)?(?:confirm|validate|prove|guarantee|establish)\b"
    aliases = [
        (r"(?:five[- ](?:day|session)|5[- ]?(?:d|day|session)) return", "return_5d", 100),
        (r"(?:twenty[- ](?:day|session)|20[- ]?(?:d|day|session)) return", "return_20d", 100),
        (r"RSI(?:14)?", "rsi14", 1),
        (r"MACD histogram", "macd_histogram", 1),
        (r"(?:price|close)\s*/\s*(?:SMA|MA)20", "price_vs_ma20", 1),
        (r"(?:price|close)\s*/\s*(?:SMA|MA)50", "price_vs_ma50", 1),
        (r"SMA200", "sma200", 1),
        (r"SMA20", "sma20", 1),
        (r"SMA50", "sma50", 1),
    ]
    for clause in _clauses(text):
        # Conditional scenarios and explicit denials are not asserted observations.
        hypothetical = bool(re.match(r"(?:if|suppose|assuming|hypothetically)\b", clause, re.I))
        if not hypothetical:
            for part in re.split(r",\s*(?:but|yet)\s*", clause, flags=re.I):
                if (re.search(model, part, re.I) and re.search(proof, part, re.I)
                        and re.search(market_truth, part, re.I)
                        and not re.search(negation, part, re.I)):
                    issues.append("MODEL_INFLUENCE_NOT_MARKET_PROOF: " + part)
            if re.search(model + r".{0,180}\bexceeds?\b.{0,100}(?:hit rate|win rate|probability)", clause, re.I):
                issues.append("RANK_IS_NOT_HIT_PROBABILITY: " + clause)
            probability = re.search(
                r"(?:probability|win rate|hit rate|hit-rate).{0,25}"
                r"(?:is|of|near|at|above|below|exceeds?|hovers|baseline)\s*.{0,12}\d+(?:\.\d+)?\s*%",
                clause, re.I,
            )
            if (probability and not re.search(r"requir|break.?even|threshold|hypothetical|subjective|uncalibrated estimate", clause, re.I)):
                issues.append("UNSOURCED_HIT_PROBABILITY: label this as subjective judgment, not a measured rate: " + clause)

        for alias, key, scale in aliases:
            expected = values.get(key)
            if expected is None or hypothetical:
                continue
            pattern = rf"(?<!/)\b{alias}\b\s*(?:(?:is|at|of|=|:|reading of)\s*)?([+-]?\d[\d,]*(?:\.\d+)?)"
            for match in re.finditer(pattern, clause, re.I):
                if key in {"sma20", "sma50", "sma200"} and re.match(r"\s*[x%]", clause[match.end():], re.I):
                    continue
                actual = float(match.group(1).replace(",", ""))
                if not math.isclose(actual, float(expected) * scale, abs_tol=0.06, rel_tol=0.001):
                    issues.append(f"NUMERIC_CONFLICT: {key}={float(expected) * scale:.6f} as of {facts_date}; reconcile dated sources: {clause}")
            if key in {"return_5d", "return_20d", "macd_histogram"}:
                sign = re.search(rf"\b{alias}\b\s+(?:is|remains)\s+(?:deeply |firmly )?(positive|negative)\b", clause, re.I)
                if sign and ((sign[1].lower() == "positive" and expected < 0) or (sign[1].lower() == "negative" and expected > 0)):
                    issues.append(f"SIGN_CONFLICT: {key}={expected} as of {facts_date}: {clause}")

        if (re.search(r"(?:volume|participation).{0,100}(?:confirms?|proves?|indicat(?:es|ing)).{0,80}(?:absorption|seller exhaustion|institutional)", clause, re.I)
                and not re.search(negation, clause, re.I)):
            issues.append("UNVERIFIED_ORDER_FLOW: volume alone does not establish absorption or participant identity: " + clause)
        for name in ("target", "stop"):
            price = re.search(rf"\b{name}(?: loss)?(?: price)?\s*(?:is|at|of|:|=)?\s*[\u20b9$]\s*([\d,]+(?:\.\d+)?)", clause, re.I)
            # Respect the precision the author reported (e.g. 520 rounds 520.15).
            tolerance = 0.5 * 10 ** (-len(price[1].split(".")[1]) if price and "." in price[1] else 0)
            if price and not math.isclose(float(price[1].replace(",", "")), levels[name + "_price"], abs_tol=tolerance + 1e-8):
                issues.append(f"FIXED_LEVEL_CONFLICT: {name} price={levels[name + '_price']:.4f}: {clause}")
            # Do not associate a stop mention with a later target's ATR value (or vice versa).
            distances = re.finditer(
                rf"\b{name}(?: distance)?\b(?:(?!\b(?:target|stop)\b).){{0,65}}?"
                r"(\d+(?:\.\d+)?)\s*ATRs?\b", clause, re.I,
            )
            for distance in distances:
                if not math.isclose(float(distance[1]), levels[name + "_distance_atr"], abs_tol=0.06):
                    issues.append(f"ATR_DISTANCE_CONFLICT: {name}={levels[name + '_distance_atr']:.4f} ATR: {clause}")
        if re.search(r"target.{0,100}(?:maps? directly|mapped directly|coincides|equals).{0,50}MA20/MA50", clause, re.I):
            ma20, ma50 = values["sma20"], values["sma50"]
            if all(abs(levels["target_price"] / value - 1) > 0.01 for value in (ma20, ma50) if value > 0):
                issues.append(f"TARGET_MA_CONFLICT: target={levels['target_price']:.4f}, SMA20={ma20:.4f}, SMA50={ma50:.4f}: {clause}")
    return list(dict.fromkeys(issues))


def decision_grounding_issues(text: str, state: dict) -> list[str]:
    return list(dict.fromkeys(
        find_unsupported_optional_source_claims(text, state)
        + find_unsupported_upstream_claims(text, state)
        + find_trade_arithmetic_issues(text, state)
        + find_current_evidence_issues(text, state)
    ))


def repair_decision_grounding(draft, state, prompt, structured_llm, llm, render, agent_name):
    """Request at most one complete correction; retain the model's final action."""
    if not state.get("lstm_signal_context") and agent_name != "Portfolio Manager":
        return draft, []
    issues = decision_grounding_issues(draft, state)
    if not issues:
        return draft, []
    instruction = (
        "Your draft has evidence-validation issues. Rewrite the complete decision once. "
        "Use supplied numerical facts or explicitly reconcile a conflict with a dated source. "
        "Model rank and sensitivity are not measured win rates or proof of institutional "
        "activity, an intact trend, or an observed reversal. Separate observations from "
        "hypotheses and future confirmation. Mark unsupported probability estimates as "
        "subjective judgments. Preserve the action if corrected evidence supports it; "
        "the final BUY/HOLD/SELL decision remains yours.\nIssues:\n- "
        + "\n- ".join(issues) + "\n\nPrevious draft:\n" + draft
    )
    repair_prompt = (
        prompt + [{"role": "user", "content": instruction}]
        if isinstance(prompt, list) else prompt + "\n\n" + instruction
    )
    repaired = invoke_structured_or_freetext(
        structured_llm, llm, repair_prompt, render, agent_name + " grounding repair",
    )
    remaining = decision_grounding_issues(repaired, state)
    prefix = "FINAL" if agent_name == "Portfolio Manager" else agent_name.upper().replace(" ", "_")
    if remaining:
        repaired += "\n\n**Unresolved evidence issues** (validation, not a decision override):\n- " + "\n- ".join(remaining)
        return repaired, [f"INVALID_{prefix}_GROUNDING"]
    return repaired, [f"REPAIRED_{prefix}_GROUNDING"]
