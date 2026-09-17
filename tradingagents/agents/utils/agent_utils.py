import re

from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_analyst_context(state: dict) -> str:
    """Project neutral dated facts and execution parameters, never model persuasion."""
    context = state.get("lstm_signal_context")
    if not context:
        return state.get("trade_context_note", "")
    from tradingagents.agents.utils.lstm_context import render_technical_snapshot

    signal = context["signal"]
    contract = context["strategy_contract"]
    return (
        f"Trade evaluation | {contract['maximum_horizon_exchange_sessions']} exchange sessions | "
        f"Entry: {signal['entry_price']:.2f} | "
        f"Target: {contract['decision_target_pct']:+.2%} | "
        f"Stop: {contract['stop_loss_pct']:+.2%}. "
        "Assess target before stop from observed evidence. Establish trend and reversal "
        "independently; distinguish an observed reversal from future confirmation.\n"
        + render_technical_snapshot(context)
    )


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    canonical_name: str | None = None,
    lstm_context_available: bool = False,
) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )
    identity = (
        f" The verified company name is `{canonical_name}`. Do not substitute or "
        "discuss a different company."
        if canonical_name and asset_type == "stock"
        else ""
    )
    upstream_note = (
        " Structured current-run LSTM evidence is supplied separately. Use only "
        "that evidence and do not invent prior model runs, outcomes, or features."
        if lstm_context_available
        else (
            " Analyze the instrument independently. No upstream selector or prediction "
            "model output is supplied. Do not infer or mention an LSTM signal, model "
            "score, rank, features, selection history, or selection reason."
        )
    )
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + identity
        + extra_hint
        + upstream_note
    )


_OPTIONAL_SENTIMENT_SOURCES = {
    "reddit": "Reddit",
    "telegram": "Telegram",
    "google_trends": "Google Trends",
}
_SOURCE_LABELS = {
    "company_news": "Yahoo Finance company news",
    "google_news": "Google News",
    **_OPTIONAL_SENTIMENT_SOURCES,
}
_USABLE_SOURCE_STATUSES = {"OK", "STALE"}
_TOOL_TICKER_ARGUMENTS = {
    "get_stock_data": "symbol",
    "get_indicators": "symbol",
    "get_fundamentals": "ticker",
    "get_balance_sheet": "ticker",
    "get_cashflow": "ticker",
    "get_income_statement": "ticker",
    "get_news": "ticker",
    "get_insider_transactions": "ticker",
}
_TRADER_ACTIONS = {"buy", "hold", "sell"}
_PORTFOLIO_RATINGS = {"buy", "overweight", "hold", "underweight", "sell"}

_POSITIVE_CATALYST_PATTERNS = (
    r"\bapproval(?:s)?\b",
    r"\blaunch(?:ed|es|ing)?\b",
    r"\border(?:s)? win\b",
    r"\bcontract(?:s)? (?:win|won|secured|signed)\b",
    r"\bpartnership\b",
    r"\bacquisition\b",
    r"\bbuyback\b",
    r"\bdividend increase\b",
    r"\bearnings beat\b",
    r"\bbeat(?:ing)? estimates\b",
    r"\bguidance (?:raised|increase[sd]?)\b",
    r"\bupgrade[sd]?\b",
    r"\bproduct rollout\b",
    r"\bcapacity expansion\b",
)
_MOMENTUM_PATTERNS = (
    r"\bbullish crossover\b",
    r"\bmacd\b.{0,40}\bbullish\b",
    r"\bbreakout\b",
    r"\breclaim(?:ed|ing)?\b",
    r"\bhigher highs?\b",
    r"\bhigher lows?\b",
    r"\bvolume (?:expansion|pickup|confirmation|surge)\b",
    r"\bsupport (?:held|holding|absorbed)\b",
    r"\boversold bounce\b",
    r"\bmean[- ]reversion bounce\b",
    r"\bprice\b.{0,40}\babove\b.{0,40}\b(?:ema|sma|vwap|middle band)\b",
    r"\brsi\b.{0,40}\brecover(?:ing|ed|y)\b",
)
_PRIMARY_BEARISH_EVENT_PATTERNS = (
    r"\bguidance cut\b",
    r"\bearnings miss\b",
    r"\bmiss(?:ed|ing)? estimates\b",
    r"\bdowngrade[sd]?\b",
    r"\bwarning letter\b",
    r"\bregulatory (?:action|risk|warning|scrutiny)\b",
    r"\bmargin pressure\b",
    r"\bprofit warning\b",
    r"\bdebt (?:spike|stress|concern)\b",
    r"\bpromoter selling\b",
    r"\bpledge(?:d|ing)? shares\b",
    r"\blawsuit\b",
)
_PRIMARY_BEARISH_TECHNICAL_PATTERNS = (
    r"\bbearish crossover\b",
    r"\bmacd\b.{0,40}\bbearish\b",
    r"\bbreakdown\b",
    r"\blower highs?\b",
    r"\blower lows?\b",
    r"\bdistribution\b",
    r"\bprice\b.{0,40}\bbelow\b.{0,40}\b(?:ema|sma|vwap|support)\b",
    r"\bfailed reclaim\b",
    r"\bresistance rejection\b",
    r"\btrend failure\b",
)
_CONFIRMATION_ONLY_PATTERNS = (
    r"\bneed(?:s)? confirmation\b",
    r"\bwait for confirmation\b",
    r"\bnot confirmed\b",
    r"\bneeds? a close above\b",
    r"\bneeds? reclaim\b",
    r"\buntil price proves\b",
    r"\bconfirmation (?:is )?missing\b",
    r"\bwithout confirmation\b",
    r"\bwait-and-see\b",
)
_MACRO_ONLY_PATTERNS = (
    r"\bmacro(?:economic)?\b",
    r"\bglobal\b",
    r"\bgeopolitical\b",
    r"\boil prices?\b",
    r"\brate cuts?\b",
    r"\brate hikes?\b",
    r"\bfed\b",
    r"\binflation\b",
    r"\brecession\b",
    r"\brisk-off\b",
    r"\bmarket-wide\b",
    r"\btariff\b",
)


def enforce_exact_tool_ticker(message, ticker: str) -> bool:
    """Force ticker-bearing tool calls to use the graph's exact instrument."""
    corrected = False
    for tool_call in getattr(message, "tool_calls", []) or []:
        argument_name = _TOOL_TICKER_ARGUMENTS.get(str(tool_call.get("name", "")))
        arguments = tool_call.get("args")
        if not argument_name or not isinstance(arguments, dict):
            continue
        supplied = arguments.get(argument_name)
        if supplied != ticker:
            arguments[argument_name] = ticker
            corrected = True
    return corrected


def get_unavailable_optional_sources(state: dict) -> list[str]:
    """Return optional sentiment sources that are absent from this snapshot.

    Missing optional sources are telemetry, not directional evidence. Yahoo
    company data and company-specific news are the primary evidence contract.
    """
    snapshot = state.get("sentiment_source_snapshot") or {}
    sources = snapshot.get("sources") or {}
    unavailable = []
    for key, label in _OPTIONAL_SENTIMENT_SOURCES.items():
        item = sources.get(key)
        if item and str(item.get("status", "")).upper() not in _USABLE_SOURCE_STATUSES:
            unavailable.append(label)
    return unavailable


def get_unavailable_sources(state: dict) -> list[str]:
    """Return non-optional sources whose snapshot has no usable evidence."""
    snapshot = state.get("sentiment_source_snapshot") or {}
    sources = snapshot.get("sources") or {}
    unavailable = []
    for key, label in _SOURCE_LABELS.items():
        if key in {"reddit", "telegram"}:
            continue
        item = sources.get(key)
        if item and str(item.get("status", "")).upper() not in _USABLE_SOURCE_STATUSES:
            unavailable.append(label)
    return unavailable


def get_data_quality_instruction(state: dict) -> str:
    """Tell every downstream agent how to preserve source and model independence."""
    unavailable = get_unavailable_sources(state)
    target_pct = state.get("profit_target_pct")
    stop_pct = state.get("stop_loss_pct")
    payoff_line = ""
    if isinstance(target_pct, (int, float)) and isinstance(stop_pct, (int, float)):
        target = abs(float(target_pct))
        stop = abs(float(stop_pct))
        if target > 0 and stop > 0:
            breakeven = stop / (target + stop)
            payoff_line = (
                f" For payoff arithmetic, a +{target * 100:.2f}% target versus "
                f"-{stop * 100:.2f}% stop requires more than "
                f"{breakeven * 100:.2f}% target hits before costs."
            )
    if state.get("lstm_signal_context"):
        instructions = (
            "\n\nLSTM EVIDENCE POLICY: The structured context contains factual current-run "
            "model evidence. Give it meaningful weight but make an independent decision. "
            "Do not invent prior runs, historical outcomes, probabilities, features, or "
            "selection reasons that are absent from the supplied context. State evidence "
            "with its metric, value and date. A negative indicator level alone does not "
            "establish acceleration; volume alone does not prove order-book depth, "
            "institutional flows or absorption. Distinguish observed reversal from "
            "future confirmation and mark uncalibrated probability judgments as subjective. Short-term "
            "weakness may be the expected pullback, so test whether established momentum "
            "has structurally failed before treating that weakness as contradictory."
            + payoff_line
        )
    else:
        instructions = (
            "\n\nUPSTREAM INDEPENDENCE POLICY: No upstream selector or prediction model "
            "output is available to you. Assess only the ticker-specific evidence in this "
            "analysis. Do not infer or mention an LSTM signal, model score, rank, features, "
            "selection history, or selection reason."
            + payoff_line
        )
    if unavailable:
        source_list = ", ".join(unavailable)
        instructions += (
            "\n\nSOURCE AVAILABILITY POLICY: "
            f"{source_list} are unavailable or contain no usable ticker-specific data. "
            "This means UNKNOWN, not negative sentiment. Do not infer retail silence, "
            "low participation, low demand, thin liquidity, missing buyers, lack of "
            "momentum fuel, reduced conviction, or bearishness from these gaps. Do not "
            "let the gaps change the rating or BUY/HOLD/SELL decision. Base directional "
            "claims only on verified market, company-news, and fundamentals evidence."
        )
    return instructions


def find_unsupported_upstream_claims(text: str, state: dict | None = None) -> list[str]:
    """Find claims about selector/model inputs that TradingAgents never receives."""
    if state and state.get("lstm_signal_context"):
        forbidden_history = re.compile(
            r"\bprior (?:LSTM|model|TA|TradingAgents?) (?:run|score|rank|decision|outcome)s?\b|"
            r"\bprevious (?:LSTM|model|TA|TradingAgents?) (?:run|score|rank|decision|outcome)s?\b|"
            r"\bselection history\b|\bhistorical (?:target|trade|signal) outcomes?\b",
            flags=re.IGNORECASE,
        )
        return [
            segment.strip()
            for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", text or "")
            if segment.strip() and forbidden_history.search(segment)
        ]
    if not text:
        return []

    cleaned = re.sub(
        r"\[Unsupported upstream-model inference removed[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    upstream_pattern = re.compile(
        r"\bLSTM\b|\bupstream (?:selector|model|signal)\b|"
        r"\b(?:selector|prediction model|model) (?:score|rank|features?|signal)\b|"
        r"\bselection (?:history|reason)\b",
        flags=re.IGNORECASE,
    )
    issues = []
    seen = set()
    for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", cleaned):
        claim = segment.strip()
        if claim and upstream_pattern.search(claim) and claim not in seen:
            seen.add(claim)
            issues.append(claim)
    return issues


def _required_break_even_pct(state: dict) -> float | None:
    target_pct = state.get("profit_target_pct")
    stop_pct = state.get("stop_loss_pct")
    if not isinstance(target_pct, (int, float)) or not isinstance(stop_pct, (int, float)):
        return None
    target = abs(float(target_pct))
    stop = abs(float(stop_pct))
    if target <= 0 or stop <= 0:
        return None
    return stop / (target + stop) * 100.0


def find_trade_arithmetic_issues(text: str, state: dict) -> list[str]:
    """Find payoff arithmetic claims that contradict the fixed target/stop."""
    required_pct = _required_break_even_pct(state)
    if not text or required_pct is None:
        return []
    issues = []
    seen = set()
    break_even_pattern = re.compile(
        r"\b(?:break[ -]?even|breakeven|positive expectancy|profitable)\b",
        flags=re.IGNORECASE,
    )
    # Match a claimed win rate, not the target/stop/cost percentages nearby.
    rate_patterns = [
        re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:target[- ]hits?|wins?|win[- ]rate|hit[- ]rate)", re.I),
        re.compile(r"(?:break[ -]?even(?: rate)?|breakeven|win[- ]rate|hit[- ]rate)\s*(?:is|of|at|:|=|requires?)\s*(?:only |about |roughly |approximately |a |~)*(\d+(?:\.\d+)?)\s*%", re.I),
    ]
    for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        claim = segment.strip()
        if not claim or not break_even_pattern.search(claim):
            continue
        if re.search(r"\b(?:cannot|can't|does not|not enough|insufficient|not profitable)\b", claim, re.I):
            continue
        percentages = [float(match.group(1)) for pattern in rate_patterns for match in pattern.finditer(claim)]
        if any(value < required_pct - 0.5 for value in percentages):
            if claim not in seen:
                seen.add(claim)
                issues.append(claim)
    return issues


def sanitize_unsupported_upstream_claims(
    text: str,
    state: dict | None = None,
) -> tuple[str, bool]:
    """Remove invented upstream selector/model evidence from an agent response."""
    claims = find_unsupported_upstream_claims(text, state)
    if not claims:
        return text, False
    sanitized = text
    replacement = "[Unsupported upstream-model inference removed by validation.]"
    for claim in sorted(claims, key=len, reverse=True):
        sanitized = sanitized.replace(claim, replacement)
    return sanitized, True


def sanitize_agent_output(text: str, state: dict) -> tuple[str, list[str]]:
    """Sanitize source-gap and upstream-model claims and return quality tags."""
    sanitized, removed_source = sanitize_unsupported_source_claims(text, state)
    sanitized, removed_upstream = sanitize_unsupported_upstream_claims(sanitized, state)
    tags = []
    if removed_source:
        tags.append("REMOVED_UNSUPPORTED_SOURCE_CLAIM")
    if removed_upstream:
        tags.append("REMOVED_UNSUPPORTED_UPSTREAM_CLAIM")
    return sanitized, tags


def _remove_safe_source_gap_treatment_phrases(text: str) -> str:
    """Remove allowed non-directional treatments before source-gap validation.

    Missing-source explanations often say a gap is "not bearish" or has
    "no directional weight". Those phrases are the required safe treatment,
    not unsupported source-gap inference. Remove only those guarded phrases so
    remaining directional claims such as "risk is higher" still fail.
    """
    safe_patterns = (
        r"\b(?:has|have|carries|carry|assigned|contributes?)\s+"
        r"(?:no|zero)\s+(?:directional\s+)?(?:weight|signal|score)\b",
        r"\b(?:no|zero)\s+(?:directional\s+)?(?:weight|signal|score)\s+"
        r"(?:is\s+)?(?:assigned|given|applied|used|contributed)?\b",
        r"\bnot\s+(?:a\s+)?(?:bearish|bullish|negative|positive)\s+"
        r"(?:signal|indicator|factor|input|evidence|sentiment)\b",
        r"\bnot\s+evidence\s+of\s+(?:a\s+)?"
        r"(?:bearish|bullish|negative|positive)(?:\s+\w+){0,4}\b",
        r"\bnot\s+treated\s+as\s+(?:a\s+)?"
        r"(?:bearish|bullish|negative|positive)(?:\s+\w+){0,4}\b",
        r"\bnot\s+used\s+as\s+(?:a\s+)?"
        r"(?:bearish|bullish|negative|positive|directional)(?:\s+\w+){0,4}\b",
    )
    scrubbed = text
    for pattern in safe_patterns:
        scrubbed = re.sub(pattern, "", scrubbed, flags=re.IGNORECASE)
    return scrubbed


def find_unsupported_optional_source_claims(text: str, state: dict) -> list[str]:
    """Find claims that turn an unavailable source into directional evidence."""
    unavailable = sorted(
        set(get_unavailable_sources(state) + get_unavailable_optional_sources(state))
    )
    if not text or not unavailable:
        return []

    text = re.sub(
        r"\[Unsupported source-gap inference removed[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    source_pattern = "|".join(re.escape(source) for source in unavailable)
    absence_pattern = (
        r"unavailable|missing|disabled|no (?:usable )?(?:data|posts|messages)|"
        r"absence|lack|gaps?|silence|blind spots?"
    )
    directional_pattern = (
        r"bearish|bullish|negative|positive|risk|liquidity|buyers?|participation|"
        r"demand|momentum|conviction|interest|downside|upside|avoid|sell|hold|buy|"
        r"expos(?:e|ed|ure)|vulnerab|weaken|penalty|tailwind|advantage|supportive|"
        r"fuel|friction|fud|strength|allows?"
    )
    generic_gap_pattern = (
        r"(?:missing|unavailable|disabled|absent|lack of).{0,50}"
        r"(?:sources?|channels?|platforms?|coverage|data|sentiment|social media)|"
        r"(?:sources?|channels?|platforms?|coverage|data|sentiment|social media).{0,50}"
        r"(?:missing|unavailable|disabled|absent|lack)"
        r"|(?:sentiment|social media)\s+gaps?"
    )
    issues = []
    seen = set()
    sentence_segments = [
        segment.strip()
        for segment in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if segment.strip()
    ]
    for segment in sentence_segments:
        lowered = segment.lower()
        source_named = re.search(source_pattern, segment, flags=re.IGNORECASE)
        generic_gap = re.search(generic_gap_pattern, lowered)
        if not source_named and not generic_gap:
            continue
        directional_scan = _remove_safe_source_gap_treatment_phrases(lowered)
        if re.search(absence_pattern, lowered) and re.search(directional_pattern, directional_scan):
            claim = segment.strip()
            if claim and claim not in seen:
                seen.add(claim)
                issues.append(claim)
    for previous, current in zip(sentence_segments, sentence_segments[1:]):
        previous_lower = previous.lower()
        current_lower = current.lower()
        if "source-gap inference removed" in previous_lower:
            continue
        if not re.search(generic_gap_pattern, previous_lower):
            continue
        if not re.search(absence_pattern, previous_lower):
            continue
        if not re.match(
            r"(?:this|that|these|those|therefore|thus|consequently|as a result|such)\b",
            current_lower,
        ):
            continue
        directional_scan = _remove_safe_source_gap_treatment_phrases(current_lower)
        if re.search(directional_pattern, directional_scan) and current not in seen:
            seen.add(current)
            issues.append(current)
    return issues


def sanitize_unsupported_source_claims(text: str, state: dict) -> tuple[str, bool]:
    """Remove unsupported source-gap inferences before another agent can read them."""
    claims = find_unsupported_optional_source_claims(text, state)
    if not claims:
        return text, False
    sanitized = text
    replacement = "[Unsupported source-gap inference removed by validation.]"
    for claim in sorted(claims, key=len, reverse=True):
        sanitized = sanitized.replace(claim, replacement)
    return sanitized, True


def _match_count(text: str, patterns: tuple[str, ...]) -> int:
    if not text:
        return 0
    return sum(1 for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE))


def _normalize_labeled_value(value: str, allowed: set[str], default: str) -> str:
    cleaned = value.strip().strip("*").lower()
    return cleaned if cleaned in allowed else default.lower()


def parse_trader_action(text: str, default: str = "Hold") -> str:
    if not text:
        return default
    patterns = (
        re.compile(r"^\*\*Action\*\*:\s*(Buy|Hold|Sell)\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _normalize_labeled_value(
                match.group(1), _TRADER_ACTIONS, default
            ).capitalize()
    return default


def parse_portfolio_rating(text: str, default: str = "Hold") -> str:
    if not text:
        return default
    patterns = (
        re.compile(
            r"^\*\*(?:Rating|Recommendation)\*\*:\s*(Buy|Overweight|Hold|Underweight|Sell)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _normalize_labeled_value(
                match.group(1), _PORTFOLIO_RATINGS, default
            ).capitalize()
    return default


def _replace_first(
    text: str,
    pattern: re.Pattern[str],
    replacement: str,
) -> tuple[str, bool]:
    updated, count = pattern.subn(replacement, text, count=1)
    return updated, count > 0


def _append_policy_adjustment(text: str, note: str) -> str:
    if note in text:
        return text
    return text.rstrip() + f"\n\n**Policy Adjustment**: {note}"


def rewrite_trader_action(text: str, new_action: str, note: str) -> str:
    updated, action_replaced = _replace_first(
        text,
        re.compile(r"(^\*\*Action\*\*:\s*)(Buy|Hold|Sell)(\s*$)", re.IGNORECASE | re.MULTILINE),
        rf"\g<1>{new_action}\g<3>",
    )
    updated, final_replaced = _replace_first(
        updated,
        re.compile(
            r"(FINAL TRANSACTION PROPOSAL:\s*\*\*)(BUY|HOLD|SELL)(\*\*)",
            re.IGNORECASE,
        ),
        rf"\g<1>{new_action.upper()}\g<3>",
    )
    if not action_replaced:
        updated = f"**Action**: {new_action}\n\n{updated.lstrip()}"
    if not final_replaced:
        updated = updated.rstrip() + f"\n\nFINAL TRANSACTION PROPOSAL: **{new_action.upper()}**"
    return _append_policy_adjustment(updated, note)


def rewrite_portfolio_rating(
    text: str,
    new_rating: str,
    note: str,
    *,
    label: str = "Rating",
) -> str:
    updated, rating_replaced = _replace_first(
        text,
        re.compile(
            rf"(^\*\*{label}\*\*:\s*)(Buy|Overweight|Hold|Underweight|Sell)(\s*$)",
            re.IGNORECASE | re.MULTILINE,
        ),
        rf"\g<1>{new_rating}\g<3>",
    )
    if not rating_replaced:
        updated = f"**{label}**: {new_rating}\n\n{updated.lstrip()}"
    return _append_policy_adjustment(updated, note)


def assess_precision_first_evidence(
    state: dict,
    *,
    candidate_text: str = "",
) -> dict[str, bool | list[str]]:
    market_report = str(state.get("market_report") or "")
    evidence_text = "\n\n".join(
        filter(
            None,
            [
                market_report,
                str(state.get("news_report") or ""),
                str(state.get("fundamentals_report") or ""),
                str(state.get("sentiment_report") or ""),
            ],
        )
    )
    catalyst_text = "\n\n".join(
        filter(
            None,
            [
                str(state.get("news_report") or ""),
                str(state.get("fundamentals_report") or ""),
                str(state.get("sentiment_report") or ""),
            ],
        )
    )
    objection_text = "\n\n".join(
        filter(
            None,
            [
                candidate_text,
                str(state.get("investment_plan") or ""),
                str(state.get("trader_investment_plan") or ""),
                str((state.get("investment_debate_state") or {}).get("history", "")),
                str((state.get("risk_debate_state") or {}).get("history", "")),
            ],
        )
    )

    positive_catalyst = _match_count(catalyst_text, _POSITIVE_CATALYST_PATTERNS) >= 1
    aligned_momentum = _match_count(market_report, _MOMENTUM_PATTERNS) >= 2
    primary_bearish_evidence = (
        _match_count(catalyst_text, _PRIMARY_BEARISH_EVENT_PATTERNS) >= 1
        or _match_count(market_report, _PRIMARY_BEARISH_TECHNICAL_PATTERNS) >= 2
    )
    confirmation_only = (
        not primary_bearish_evidence
        and _match_count(objection_text, _CONFIRMATION_ONLY_PATTERNS) >= 1
    )
    macro_only = (
        not primary_bearish_evidence
        and _match_count(objection_text, _MACRO_ONLY_PATTERNS) >= 1
    )

    tags = []
    if positive_catalyst:
        tags.append("VERIFIED_POSITIVE_CATALYST")
    if aligned_momentum:
        tags.append("ALIGNED_MOMENTUM")
    if primary_bearish_evidence:
        tags.append("VERIFIED_PRIMARY_BEAR_EVIDENCE")
    if confirmation_only:
        tags.append("CONFIRMATION_ONLY_OBJECTION")
    if macro_only:
        tags.append("MACRO_ONLY_OBJECTION")

    return {
        "positive_catalyst": positive_catalyst,
        "aligned_momentum": aligned_momentum,
        "primary_bearish_evidence": primary_bearish_evidence,
        "confirmation_only_objection": confirmation_only,
        "macro_only_objection": macro_only,
        "tags": tags,
    }


def apply_research_manager_policy(plan: str, state: dict) -> tuple[str, list[str]]:
    if state.get("lstm_signal_context"):
        return plan, []
    evidence = assess_precision_first_evidence(state, candidate_text=plan)
    recommendation = parse_portfolio_rating(plan, default="Hold")
    tags = list(evidence["tags"])

    if recommendation in {"Sell", "Underweight"} and not evidence["primary_bearish_evidence"]:
        note = (
            "Reframed to Hold because the analyst reports do not show verified "
            "primary ticker-specific bearish evidence for a short-term bearish call."
        )
        return (
            rewrite_portfolio_rating(plan, "Hold", note, label="Recommendation"),
            tags + ["SELL_GATED_NO_PRIMARY_BEAR"],
        )

    if recommendation in {"Buy", "Overweight"} and not (
        evidence["positive_catalyst"] or evidence["aligned_momentum"]
    ):
        note = (
            "Reframed to Hold because the analyst reports do not show a verified "
            "positive catalyst or aligned momentum for a precision-first bullish call."
        )
        return (
            rewrite_portfolio_rating(plan, "Hold", note, label="Recommendation"),
            tags + ["BUY_GATED_NO_EDGE"],
        )

    return plan, tags


def apply_trader_policy(plan: str, state: dict) -> tuple[str, list[str]]:
    if state.get("lstm_signal_context"):
        return plan, []
    evidence = assess_precision_first_evidence(state, candidate_text=plan)
    action = parse_trader_action(plan, default="Hold")
    tags = list(evidence["tags"])

    if action == "Sell" and not evidence["primary_bearish_evidence"]:
        note = (
            "Reframed to Hold because the analyst reports do not show verified "
            "primary ticker-specific bearish evidence for a short-term SELL call."
        )
        return rewrite_trader_action(plan, "Hold", note), tags + ["SELL_GATED_NO_PRIMARY_BEAR"]

    if action == "Buy" and not (
        evidence["positive_catalyst"] or evidence["aligned_momentum"]
    ):
        note = (
            "Reframed to Hold because the analyst reports do not show a verified "
            "positive catalyst or aligned momentum for a precision-first BUY call."
        )
        return rewrite_trader_action(plan, "Hold", note), tags + ["BUY_GATED_NO_EDGE"]

    return plan, tags


def apply_portfolio_manager_policy(decision: str, state: dict) -> tuple[str, list[str]]:
    if state.get("lstm_signal_context"):
        return decision, []
    evidence = assess_precision_first_evidence(state, candidate_text=decision)
    rating = parse_portfolio_rating(decision, default="Hold")
    tags = list(evidence["tags"])

    if rating in {"Sell", "Underweight"} and not evidence["primary_bearish_evidence"]:
        note = (
            "Reframed to Hold because the analyst reports do not show verified "
            "primary ticker-specific bearish evidence for a short-term bearish rating."
        )
        return (
            rewrite_portfolio_rating(decision, "Hold", note, label="Rating"),
            tags + ["SELL_GATED_NO_PRIMARY_BEAR"],
        )

    if rating in {"Buy", "Overweight"} and not (
        evidence["positive_catalyst"] or evidence["aligned_momentum"]
    ):
        note = (
            "Reframed to Hold because the analyst reports do not show a verified "
            "positive catalyst or aligned momentum for a precision-first bullish rating."
        )
        return (
            rewrite_portfolio_rating(decision, "Hold", note, label="Rating"),
            tags + ["BUY_GATED_NO_EDGE"],
        )

    return decision, tags

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
