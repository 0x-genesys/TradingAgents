"""Validation and role-specific rendering for upstream LSTM signal evidence."""

from __future__ import annotations

from datetime import date
import math
from typing import Any


_REQUIRED_SIGNAL_FIELDS = {
    "ticker",
    "inference_date",
    "data_timestamp",
    "entry_price",
    "score",
    "score_semantics",
    "rank",
    "threshold",
    "selection_reason",
    "above_threshold",
    "top_four_candidate",
    "feature_sequence",
    "latest_raw_features",
    "feature_sensitivity",
    "feature_group_influence",
    "technical_snapshot",
}

_REQUIRED_MODEL_FIELDS = {
    "type",
    "artifact_sha256",
    "sequence_length",
    "feature_names",
    "threshold",
    "training_label",
    "training_objective",
}

_REQUIRED_TECHNICAL_METRICS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "ema10",
    "sma20",
    "sma50",
    "sma200",
    "vwma20",
    "rsi14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "atr14",
    "atr14_pct",
    "volatility_10d",
    "bollinger_middle",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_position",
    "volume_vs_20d_average",
    "breakout_strength_20d",
    "range_position_20d",
    "close_strength",
    "price_vs_ma20",
    "price_vs_ma50",
}

_FORBIDDEN_HISTORY_FIELDS = {
    "prior_runs",
    "previous_runs",
    "previous_outcomes",
    "selection_history",
}


def _find_forbidden_history_field(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_HISTORY_FIELDS:
                return key
            found = _find_forbidden_history_field(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_forbidden_history_field(child)
            if found:
                return found
    return None


def validate_lstm_signal_context(
    context: dict[str, Any],
    ticker: str,
    trade_date: str,
) -> None:
    """Reject malformed, mismatched, future-dated, or stale LSTM evidence."""
    if not isinstance(context, dict):
        raise ValueError("LSTM signal context must be a dictionary")
    if context.get("schema_version") != 1:
        raise ValueError("Unsupported LSTM signal context schema_version")
    forbidden = _find_forbidden_history_field(context)
    if forbidden:
        raise ValueError(f"LSTM signal context must not contain {forbidden}")
    if str(context.get("run_date")) != str(trade_date):
        raise ValueError(
            f"LSTM context run date {context.get('run_date')} does not match {trade_date}"
        )
    signal = context.get("signal")
    if not isinstance(signal, dict):
        raise ValueError("LSTM signal context is missing signal payload")
    missing = sorted(_REQUIRED_SIGNAL_FIELDS - set(signal))
    if missing:
        raise ValueError(f"LSTM signal context is missing fields: {missing}")
    if str(signal.get("ticker", "")).upper() != ticker.upper():
        raise ValueError(
            f"LSTM context ticker {signal.get('ticker')} does not match {ticker}"
        )
    if str(signal.get("inference_date")) != str(trade_date):
        raise ValueError("LSTM signal inference date does not match the trade date")
    try:
        score = float(signal["score"])
        rank = int(signal["rank"])
        signal_threshold = float(signal["threshold"])
    except (TypeError, ValueError) as exc:
        raise ValueError("LSTM score, rank, or threshold is invalid") from exc
    if not math.isfinite(score) or not 0 <= score <= 1 or rank < 1:
        raise ValueError("LSTM score or rank is outside its valid range")
    fixed_trade_facts(context)

    run_day = date.fromisoformat(str(trade_date))
    data_day = date.fromisoformat(str(signal["data_timestamp"])[:10])
    if data_day > run_day:
        raise ValueError("LSTM signal context contains future-dated market data")
    if (run_day - data_day).days > 4:
        raise ValueError(
            f"LSTM signal context is stale: data {data_day} for run {run_day}"
        )

    model = context.get("model") or {}
    missing_model = sorted(_REQUIRED_MODEL_FIELDS - set(model))
    if missing_model:
        raise ValueError(f"LSTM model metadata is missing fields: {missing_model}")
    checksum = str(model.get("artifact_sha256", ""))
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum.lower()):
        raise ValueError("LSTM model artifact checksum is invalid")
    if not math.isclose(signal_threshold, float(model["threshold"]), abs_tol=1e-12):
        raise ValueError("LSTM signal threshold does not match model metadata")
    universe = context.get("universe") or {}
    if rank > int(universe.get("scored_count", 0)):
        raise ValueError("LSTM signal rank exceeds the scored universe")
    sequence = signal.get("feature_sequence") or {}
    feature_names = sequence.get("feature_names")
    dates = sequence.get("dates")
    values = sequence.get("values")
    expected_length = int(model.get("sequence_length", 0))
    if not feature_names or feature_names != model.get("feature_names"):
        raise ValueError("LSTM feature order does not match model metadata")
    if not isinstance(dates, list) or not isinstance(values, list):
        raise ValueError("LSTM feature sequence must include dates and values")
    if len(dates) != expected_length or len(values) != expected_length:
        raise ValueError("LSTM feature sequence length does not match model metadata")
    if any(not isinstance(row, list) or len(row) != len(feature_names) for row in values):
        raise ValueError("LSTM feature sequence matrix has an invalid shape")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for row in values
        for value in row
    ):
        raise ValueError("LSTM feature sequence matrix contains non-finite values")
    sequence_days = [date.fromisoformat(str(value)[:10]) for value in dates]
    if sequence_days != sorted(sequence_days) or len(sequence_days) != len(set(sequence_days)):
        raise ValueError("LSTM feature sequence dates must be strictly increasing")
    if any(sequence_day > run_day for sequence_day in sequence_days):
        raise ValueError("LSTM feature sequence contains future-dated rows")
    if dates[-1] != str(signal["data_timestamp"])[:10]:
        raise ValueError("LSTM sequence and signal data timestamps do not match")
    technical_date = str(
        (signal.get("technical_snapshot") or {}).get("data_timestamp", "")
    )[:10]
    if technical_date != dates[-1]:
        raise ValueError("LSTM sequence and technical snapshot dates do not match")
    technical = signal.get("technical_snapshot") or {}
    if not technical.get("complete"):
        raise ValueError(
            "LSTM technical snapshot is incomplete: "
            f"{technical.get('missing_metrics') or 'unknown metrics'}"
        )
    technical_values = technical.get("values") or {}
    missing_metrics = sorted(_REQUIRED_TECHNICAL_METRICS - set(technical_values))
    if missing_metrics:
        raise ValueError(f"LSTM technical snapshot is missing metrics: {missing_metrics}")
    if any(
        not isinstance(technical_values[key], (int, float))
        or not math.isfinite(float(technical_values[key]))
        for key in _REQUIRED_TECHNICAL_METRICS
    ):
        raise ValueError("LSTM technical snapshot contains non-finite metrics")
    if technical_values["atr14"] <= 0:
        raise ValueError("LSTM ATR14 must be positive")
    freshness = technical.get("freshness") or {}
    if freshness.get("inference_date") != str(trade_date):
        raise ValueError("LSTM technical snapshot freshness date does not match")
    if freshness.get("age_calendar_days") != (run_day - data_day).days:
        raise ValueError("LSTM technical snapshot freshness age does not match")
    if freshness.get("future_data") is not False:
        raise ValueError("LSTM technical snapshot freshness is invalid")
    if int(technical.get("row_count", 0)) < 200:
        raise ValueError("LSTM technical snapshot lacks SMA200 history")

    latest_raw = signal.get("latest_raw_features") or {}
    if not latest_raw or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in latest_raw.values()
    ):
        raise ValueError("LSTM latest raw features are missing or non-finite")
    sensitivity = signal.get("feature_sensitivity") or {}
    baseline_score = sensitivity.get("baseline_score")
    if baseline_score is None or not math.isclose(
        float(baseline_score), score, abs_tol=1e-5
    ):
        raise ValueError("LSTM feature sensitivity baseline does not match the score")


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _sensitivity_lines(signal: dict[str, Any]) -> list[str]:
    sensitivity = signal.get("feature_sensitivity") or {}
    lines = []
    for label, key in (("Positive", "positive"), ("Negative", "negative")):
        items = sensitivity.get(key) or []
        if not items:
            lines.append(f"- {label} score influences: none measured")
            continue
        rendered = ", ".join(
            f"{item.get('feature')} ({float(item.get('score_influence', 0)):+.4f})"
            for item in items[:5]
        )
        lines.append(f"- {label} score influences: {rendered}")
    return lines


def fixed_trade_facts(context: dict[str, Any]) -> dict[str, float]:
    """Compute fixed levels centrally so agents do not reconstruct the arithmetic."""
    try:
        entry = float(context["signal"]["entry_price"])
        contract = context["strategy_contract"]
        target = float(contract["decision_target_pct"])
        stop = float(contract["stop_loss_pct"])
        horizon = contract["maximum_horizon_exchange_sessions"]
        atr = float(context["signal"]["technical_snapshot"]["values"]["atr14"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid fixed trade parameters/ATR") from exc
    if (not all(math.isfinite(x) for x in (entry, target, stop, atr))
            or entry <= 0 or target <= 0 or not -1 < stop < 0 or atr <= 0
            or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0):
        raise ValueError("Invalid fixed trade parameters/ATR")
    return {
        "entry_price": entry,
        "target_price": entry * (1 + target),
        "stop_price": entry * (1 + stop),
        "target_distance_atr": entry * target / atr,
        "stop_distance_atr": entry * -stop / atr,
        "break_even_pct_before_costs": -stop / (target - stop) * 100,
    }


def render_fixed_trade_facts(context: dict[str, Any]) -> str:
    facts = fixed_trade_facts(context)
    return (
        f"Fixed target price: {facts['target_price']:.4f}; "
        f"fixed stop price: {facts['stop_price']:.4f}; "
        f"target distance: {facts['target_distance_atr']:.4f} ATR; "
        f"stop distance: {facts['stop_distance_atr']:.4f} ATR. "
        "ATR is a range measure, not a standard deviation or a win probability."
    )


def render_technical_snapshot(context: dict[str, Any]) -> str:
    """Market facts only, suitable for independent analysts and synthesis."""
    snapshot = context["signal"]["technical_snapshot"]
    return (
        f"CANONICAL MARKET FACTS as of {snapshot['data_timestamp']}:\n"
        + render_fixed_trade_facts(context) + "\n"
        + "\n".join(f"- {key}: {_number(value, 6)}" for key, value in snapshot["values"].items())
        + "\nReturns and ATR percentage are fractions; price/MA fields are ratios. "
        "Report material conflicts with newer tool data with the metric, both values, "
        "dates and source. Do not silently replace these facts."
    )


def render_lstm_compact_context(context: dict[str, Any]) -> str:
    """Render model evidence suitable for non-market TA roles."""
    if not context:
        return ""
    signal = context["signal"]
    model = context.get("model") or {}
    contract = context.get("strategy_contract") or {}
    training = model.get("training_objective") or {}
    levels = fixed_trade_facts(context)
    universe = context.get("universe") or {}
    route = context.get("orchestrator_selection_route") or signal.get("selection_reason")
    groups = signal.get("feature_group_influence") or {}
    technical = (signal.get("technical_snapshot") or {}).get("values") or {}
    group_text = ", ".join(
        f"{name}={float(value):+.4f}" for name, value in groups.items()
    )
    close = technical.get("close")
    sma200 = technical.get("sma200")
    close_to_sma200 = (
        _number(float(close) / float(sma200), 3)
        if close is not None and sma200 not in (None, 0)
        else "N/A"
    )
    feature_summary = ", ".join(
        [
            f"5d return {_pct(technical.get('return_5d'))}",
            f"20d return {_pct(technical.get('return_20d'))}",
            f"RSI14 {_number(technical.get('rsi14'), 1)}",
            f"MACD histogram {_number(technical.get('macd_histogram'), 3)}",
            f"ATR {_pct(technical.get('atr14_pct'))}",
            f"volume/20d {_number(technical.get('volume_vs_20d_average'), 2)}x",
            f"price/MA20 {_number(technical.get('price_vs_ma20'), 3)}x",
            f"price/MA50 {_number(technical.get('price_vs_ma50'), 3)}x",
            f"close/SMA200 {close_to_sma200}x",
        ]
    )

    lines = [
        "LSTM QUANTITATIVE EVIDENCE:",
        "- This ticker was supplied by the upstream momentum LSTM selector.",
        "- Setup hypothesis only: pullback within established momentum, with possible upward reversal inside the fixed horizon.",
        f"- Score: {_number(signal.get('score'), 4)} ({signal.get('score_semantics')})",
        f"- Cross-sectional rank: #{signal.get('rank')} of {universe.get('scored_count', 'N/A')} scored tickers",
        f"- Threshold: {_number(signal.get('threshold'), 3)}; selection route: {route}",
        f"- Model: {model.get('family', 'momentum LSTM')}; input window: {model.get('sequence_length', 'N/A')} exchange sessions",
        f"- Recorded training label: {model.get('training_label', 'not recorded')}",
        f"- Recorded training objective: {training.get('definition', 'not recorded')}; target {_pct(training.get('profit_target_pct'))}; stop {_pct(training.get('stop_loss_pct'))}; horizon {training.get('horizon_exchange_sessions', 'not recorded')} exchange sessions; metadata complete: {training.get('metadata_complete', False)}",
        "- The recorded training label and the paper target-before-stop outcome are distinct definitions. A positive realized-return label after stop simulation does not directly label touching the target first. Neither the score nor cross-sectional rank is a calibrated target-hit probability.",
        f"- Data timestamp: {signal.get('data_timestamp')}; entry: {_number(signal.get('entry_price'), 2)}",
        f"- Decision target: {_pct(contract.get('decision_target_pct'))}; stop: {_pct(contract.get('stop_loss_pct'))}; maximum hold: {contract.get('maximum_horizon_exchange_sessions', 'N/A')} exchange sessions",
        f"- Objective: {contract.get('objective', 'N/A')}",
        f"- Current deterministic facts: {feature_summary}",
        render_technical_snapshot(context),
        f"- Feature-group score influence: {group_text or 'N/A'}",
        *_sensitivity_lines(signal),
        "- Sensitivity and feature-group names describe influence on the model score, not causal proof, trend strength, absorption, institutional buying, or target-hit probability.",
        "- Treat the LSTM as a quantitative prior, not as a prescribed action. Decide BUY/HOLD/SELL independently using verified current evidence.",
        "- Short-term weakness may be part of this pullback setup. Test whether established momentum has structurally failed before treating weak recent returns, EMA10, RSI, or MACD as decisive bearish evidence.",
        f"- Payoff arithmetic for target/stop-only exits: needs more than {levels['break_even_pct_before_costs']:.2f}% target hits before costs for positive expectancy. Costs increase the required rate; horizon exits have their own returns. A closer target does not by itself establish an edge.",
    ]
    return "\n".join(lines)


def render_lstm_market_context(context: dict[str, Any]) -> str:
    """Render full current context, including the exact matrix, for Market Analyst."""
    compact = render_lstm_compact_context(context)
    signal = context["signal"]
    technical = signal.get("technical_snapshot") or {}
    technical_values = technical.get("values") or {}
    raw = signal.get("latest_raw_features") or {}
    sequence = signal.get("feature_sequence") or {}

    technical_text = ", ".join(
        f"{key}={_number(value, 4)}" for key, value in technical_values.items()
    )
    raw_text = ", ".join(
        f"{key}={_number(value, 4)}" for key, value in raw.items()
    )
    matrix_lines = [
        "date," + ",".join(sequence.get("feature_names") or [])
    ]
    for row_date, row in zip(sequence.get("dates") or [], sequence.get("values") or []):
        matrix_lines.append(
            row_date + "," + ",".join(_number(value, 6) for value in row)
        )

    return "\n".join(
        [
            compact,
            "",
            "CANONICAL TECHNICAL SNAPSHOT FROM THE INFERENCE DATA:",
            f"- Complete: {technical.get('complete')}; missing metrics: {technical.get('missing_metrics') or 'none'}",
            f"- {technical_text}",
            "",
            "LATEST RAW MOMENTUM FEATURES:",
            f"- {raw_text}",
            "",
            "EXACT MODEL INPUT MATRIX (rows are exchange sessions; columns retain artifact order):",
            *matrix_lines,
            "",
            "Separate established momentum, the current pullback, confirmation evidence, and structural invalidation in the report. Report any material conflict between tool data and this canonical snapshot.",
        ]
    )


def render_lstm_report(context: dict[str, Any]) -> str:
    """Render the auditable evidence section for saved reports."""
    if not context:
        return ""
    signal = context["signal"]
    model = context.get("model") or {}
    contract = context.get("strategy_contract") or {}
    training = model.get("training_objective") or {}
    technical = (signal.get("technical_snapshot") or {}).get("values") or {}
    route = context.get("orchestrator_selection_route") or signal.get("selection_reason")
    training_target = training.get("profit_target_pct")
    training_target_text = _pct(training_target) if training_target is not None else "not recorded in artifact"
    feature_summary = ", ".join(
        [
            f"5d return {_pct(technical.get('return_5d'))}",
            f"20d return {_pct(technical.get('return_20d'))}",
            f"RSI14 {_number(technical.get('rsi14'), 1)}",
            f"MACD histogram {_number(technical.get('macd_histogram'), 3)}",
            f"ATR {_pct(technical.get('atr14_pct'))}",
            f"volume/20d {_number(technical.get('volume_vs_20d_average'), 2)}x",
            f"price/MA50 {_number(technical.get('price_vs_ma50'), 3)}x",
        ]
    )
    lines = [
        f"- **Setup:** `{contract.get('setup', 'pullback_within_established_momentum')}`",
        f"- **Score:** {_number(signal.get('score'), 4)} - {signal.get('score_semantics')}",
        f"- **Rank:** #{signal.get('rank')} of {(context.get('universe') or {}).get('scored_count', 'N/A')}",
        f"- **Threshold:** {_number(signal.get('threshold'), 3)}",
        f"- **Selection route:** {route}",
        f"- **Inference data:** {signal.get('data_timestamp')}",
        f"- **Model:** {model.get('type', 'LSTM')} `{str(model.get('artifact_sha256', ''))[:12]}`",
        f"- **Training objective:** {model.get('training_label', 'N/A')}; target {training_target_text}",
        f"- **Decision objective:** {contract.get('objective', 'N/A')}",
        f"- **Current feature summary:** {feature_summary}",
        "- **Attribution method:** full-sequence neutral-channel occlusion; sensitivity is not causal proof.",
        *_sensitivity_lines(signal),
    ]
    return "\n".join(lines)
