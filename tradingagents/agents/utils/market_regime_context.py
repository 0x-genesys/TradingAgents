"""Optional, producer-validated market facts. No cache reads or trading gates."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import math
import re


REGIME_METRICS = (
    "mean_return_5d", "mean_return_20d",
    "above_sma50_fraction", "above_sma200_fraction",
)


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _fraction(value, breadth=False):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= (0 if breadth else -1) and (not breadth or value <= 1)
    except OverflowError:
        return False


def validate_market_regime_context(context, trade_date: str, lstm_data_date: str | None = None) -> dict:
    """Validate shapes/freshness without raising for malformed optional JSON.

    The producer owns source-byte hashing and the previous completed NSE session
    check. TA has no exchange-calendar dependency and must not read live caches.
    """
    errors = []
    if not isinstance(context, dict):
        return {"status": "UNAVAILABLE", "errors": ["snapshot must be an object"]}
    if type(context.get("schema_version")) is not int or context["schema_version"] != 1:
        errors.append("schema_version must be 1")
    run_day, data_day, trade_day = (_date(context.get("run_date")),
                                    _date(context.get("data_date")), _date(trade_date))
    if run_day is None or trade_day is None or run_day != trade_day:
        errors.append("run_date must match the YYYY-MM-DD trade date")
    if data_day is None or run_day is None or data_day >= run_day:
        errors.append("data_date must precede run_date as a completed NSE session")
    if lstm_data_date is not None and data_day != _date(lstm_data_date[:10]):
        errors.append("data_date must match the validated LSTM signal data_timestamp")
    status = context.get("status")
    if status not in ("COMPLETE", "PARTIAL", "UNAVAILABLE"):
        errors.append("status must be COMPLETE, PARTIAL, or UNAVAILABLE")
    universe = context.get("universe")
    if not isinstance(universe, dict):
        universe = {}
    expected, valid = universe.get("expected_count"), universe.get("valid_count")
    if universe.get("name") != "LSTM inference universe":
        errors.append("universe.name must identify the LSTM inference universe")
    counts_valid = type(expected) is int and type(valid) is int and 0 <= valid <= expected
    if not counts_valid:
        errors.append("universe counts must be integers with 0 <= valid_count <= expected_count")
    elif status in ("COMPLETE", "PARTIAL") and (valid == 0 or (status == "COMPLETE" and valid != expected)):
        errors.append("usable coverage required; COMPLETE requires the full expected universe")
    source = context.get("source")
    if not isinstance(source, dict):
        source = {}
    path, checksum = source.get("path"), source.get("sha256")
    if not isinstance(path, str) or not path.strip() or any(ord(c) < 32 for c in path):
        errors.append("source.path must be a nonempty path without control characters")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum):
        errors.append("source.sha256 must be a 64-character hexadecimal file digest")
    missing = context.get("missing_fields")
    if not isinstance(missing, list) or any(not isinstance(key, str) for key in missing):
        errors.append("missing_fields must be a list of field names")
        missing = []
    if not isinstance(context.get("exclusions"), dict):
        errors.append("exclusions must be an object")
    values = context.get("values")
    if not isinstance(values, dict):
        errors.append("values must be an object")
        values = {}
    present = 0
    for key in REGIME_METRICS:
        value = values.get(key)
        if value is None:
            if status == "COMPLETE" or (status == "PARTIAL" and key not in missing):
                errors.append(f"{key} is missing without a matching missing_fields entry")
        elif not _fraction(value, breadth=key.startswith("above_")):
            errors.append(f"{key} must be a finite fraction in its valid range")
        elif key in missing:
            errors.append(f"{key} is both supplied and declared missing")
        else:
            present += 1
    if status == "COMPLETE" and missing:
        errors.append("COMPLETE cannot have missing_fields")
    if status == "PARTIAL" and not present:
        errors.append("PARTIAL requires at least one usable metric")
    try:
        json.dumps(context, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        errors.append("snapshot must contain finite, JSON-serializable data")
    return {"status": "UNAVAILABLE" if errors else status, "errors": errors}


def render_market_regime_context(context, validation: dict) -> str:
    """Project only validated neutral facts, never rank, score, or direction."""
    status = validation["status"]
    if status == "UNAVAILABLE":
        return (
            "MARKET REGIME CONTEXT: UNAVAILABLE. Missing facts: the optional dated "
            "universe snapshot is unavailable or invalid; no direction can be inferred. "
            "This is missing input, not adverse evidence or forecast uncertainty."
        )
    universe = context["universe"]
    lines = [
        f"MARKET REGIME CONTEXT: {status} | run_date: {context['run_date']} | "
        f"data_date: {context['data_date']}",
        f"Universe: LSTM inference universe | coverage: {universe['valid_count']}/{universe['expected_count']}",
        "Fractions below are displayed as percentages; coverage is the valid subset, not an index proxy.",
    ]
    for key in REGIME_METRICS:
        value = context["values"].get(key)
        lines.append(f"- {key}: {value:.2%}" if value is not None else f"- {key}: unavailable (missing fact)")
    if status == "PARTIAL":
        lines.append("Partial coverage: excluded or missing observations are not directional evidence.")
    return "\n".join(lines)


def market_regime_state(context, trade_date: str, lstm_data_date: str | None = None) -> dict:
    """Keep the exact JSON snapshot separate from its effective validation status."""
    validation = validate_market_regime_context(context, trade_date, lstm_data_date)
    try:
        # Valid contract snapshots stay byte-for-byte equivalent at the JSON-value level.
        archived = deepcopy(context)
        encoded = json.dumps(archived, indent=2)
    except (TypeError, ValueError, OverflowError, RecursionError):
        # Non-JSON input has no exact JSON representation, but must not abort the run.
        archived = {"invalid_input_repr": repr(context)}
        encoded = json.dumps(archived, indent=2)
    note = render_market_regime_context(context, validation)
    report = (
        "## Market Regime Context\n\n" + note
        + "\n\nConsumer validation: " + validation["status"]
        + ("\n- " + "\n- ".join(validation["errors"]) if validation["errors"] else "")
        + "\n\nExact supplied snapshot (audit only; invalid facts are not used):\n```json\n"
        + encoded + "\n```"
    )
    return {
        "market_regime_context": archived,
        "market_regime_validation": validation,
        "market_regime_context_note": note,
        "market_regime_report": report,
    }


def get_market_regime_instruction(state: dict) -> str:
    """Downstream-only strategy caution, conditional on this new input."""
    note = state.get("market_regime_context_note", "")
    if not note:
        return ""
    instruction = "\n\n" + note
    instruction += (
        "\nSeparate missing facts from forecast uncertainty about the future path. "
        "Assess regime_fit_assessment as SUPPORTIVE, ADVERSE, MIXED, or UNKNOWN and "
        "explain regime_fit_reason using dated facts, coverage and uncertainty. "
        "Use UNKNOWN for unavailable regime facts. These are qualitative judgments, "
        "not calibrated bull/bear probabilities. An adverse regime is not an automatic HOLD."
    )
    if state.get("lstm_signal_context"):
        validation = state.get("market_regime_validation") or {}
        context = state.get("market_regime_context") or {}
        signal = state["lstm_signal_context"]["signal"]
        if (validation.get("status") in ("COMPLETE", "PARTIAL")
                and context.get("data_date") == str(signal["data_timestamp"])[:10]):
            mean20 = context["values"].get("mean_return_20d")
            ticker20 = signal["technical_snapshot"]["values"].get("return_20d")
            if mean20 is not None and ticker20 is not None:
                instruction += (
                    f"\nDerived same-date comparison as of {context['data_date']}: "
                    f"ticker_minus_universe_return_20d: {(ticker20 - mean20) * 100:+.2f} percentage points. "
                    "This is a return difference, not proof of safety or a required entry condition."
                )
        horizon = state["lstm_signal_context"]["strategy_contract"]["maximum_horizon_exchange_sessions"]
        instruction += (
            "\nREGIME STRATEGY CAVEAT: The LSTM setup is a pullback within established momentum; "
            "a high score does not establish safety in the observed market regime. "
            "Reconcile observed entry support with the strongest contradictory evidence, "
            f"and explain why the target can precede the stop within {horizon} exchange sessions "
            "(or why that path is unsupported or unresolved). Do not require a "
            "confirmed reversal, catalyst, or relative strength as a universal entry condition. "
            "Missing regime facts do not establish a bull or bear regime."
        )
    return instruction
