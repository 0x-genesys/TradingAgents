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


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    canonical_name: str | None = None,
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
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + identity
        + extra_hint
        + " Analyze the instrument independently. No upstream selector or prediction "
        "model output is supplied. Do not infer or mention an LSTM signal, model "
        "score, rank, features, selection history, or selection reason."
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
    """Return every source whose snapshot has no usable evidence."""
    snapshot = state.get("sentiment_source_snapshot") or {}
    sources = snapshot.get("sources") or {}
    unavailable = []
    for key, label in _SOURCE_LABELS.items():
        item = sources.get(key)
        if item and str(item.get("status", "")).upper() not in _USABLE_SOURCE_STATUSES:
            unavailable.append(label)
    return unavailable


def get_data_quality_instruction(state: dict) -> str:
    """Tell every downstream agent how to preserve source and model independence."""
    unavailable = get_unavailable_sources(state)
    instructions = (
        "\n\nUPSTREAM INDEPENDENCE POLICY: No upstream selector or prediction model "
        "output is available to you. Assess only the ticker-specific evidence in this "
        "analysis. Do not infer or mention an LSTM signal, model score, rank, features, "
        "selection history, or selection reason."
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


def find_unsupported_upstream_claims(text: str) -> list[str]:
    """Find claims about selector/model inputs that TradingAgents never receives."""
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


def sanitize_unsupported_upstream_claims(text: str) -> tuple[str, bool]:
    """Remove invented upstream selector/model evidence from an agent response."""
    claims = find_unsupported_upstream_claims(text)
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
    sanitized, removed_upstream = sanitize_unsupported_upstream_claims(sanitized)
    tags = []
    if removed_source:
        tags.append("REMOVED_UNSUPPORTED_SOURCE_CLAIM")
    if removed_upstream:
        tags.append("REMOVED_UNSUPPORTED_UPSTREAM_CLAIM")
    return sanitized, tags


def find_unsupported_optional_source_claims(text: str, state: dict) -> list[str]:
    """Find claims that turn an unavailable source into directional evidence."""
    unavailable = get_unavailable_sources(state)
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
        if re.search(absence_pattern, lowered) and re.search(directional_pattern, lowered):
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
        if re.search(directional_pattern, current_lower) and current not in seen:
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


        
