"""News analysis over frozen ticker news with optional live macro enrichment."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    enforce_exact_tool_ticker,
    get_global_news,
    get_language_instruction,
    get_news,
    sanitize_agent_output,
)

_USABLE_SOURCE_STATUSES = {"OK", "STALE"}
_BROAD_NO_NEWS_PATTERNS = (
    re.compile(
        r"\bno\s+(?:specific\s+)?(?:company|ticker)(?:-specific)?\s+"
        r"(?:news|reporting|articles?|coverage|analysis(?:\s*/\s*news)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+specific\s+news(?:\s+articles?)?\s+"
        r"(?:was|were|is|are)?\s*(?:found|available|reported|flagged)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+(?:significant|material|newsworthy)\s+(?:news|events?)\b"
        r"(?![^.!?]{0,80}\b(?:macroeconomic|macro|global(?:ly)?|econom(?:ic|y)|market(?:wide)?)\b)"
        r".{0,80}\b(?:found|available|reported|flagged)\b",
        re.IGNORECASE,
    ),
)


def _source(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    return (snapshot.get("sources") or {}).get(name) or {
        "status": "UNAVAILABLE",
        "content": f"<{name} unavailable: source snapshot missing>",
    }


def _is_usable(source: dict[str, Any]) -> bool:
    return str(source.get("status", "UNAVAILABLE")).upper() in _USABLE_SOURCE_STATUSES


def _frozen_ticker_news(snapshot: dict[str, Any]) -> str:
    """Render the exact ticker-news inputs shared with the sentiment analyst."""
    sections = []
    for name, label in (
        ("company_news", "Yahoo Finance company news"),
        ("google_news", "India-localized Google News"),
    ):
        source = _source(snapshot, name)
        status = str(source.get("status", "UNAVAILABLE")).upper()
        content = str(source.get("content") or "<no content>")
        treatment = "Verified ticker evidence" if _is_usable(source) else "Unknown"
        sections.append(
            f"### {label}\n"
            f"Status: {status}\n"
            f"Treatment: {treatment}\n"
            f"<start_of_{name}>\n{content}\n<end_of_{name}>"
        )
    return "\n\n".join(sections)


def _news_report_issues(report: str, snapshot: dict[str, Any]) -> list[str]:
    """Reject claims that contradict usable frozen ticker-news evidence."""
    company_news = _source(snapshot, "company_news")
    google_news = _source(snapshot, "google_news")
    if not (_is_usable(company_news) or _is_usable(google_news)):
        return []

    issues = []
    for segment in re.split(r"(?<=[.!?])\s+|\n+", report):
        text = segment.strip()
        if not text:
            continue
        for pattern in _BROAD_NO_NEWS_PATTERNS:
            if not pattern.search(text):
                continue
            lowered = text.lower()
            source_scoped_and_accurate = (
                "yahoo" in lowered and not _is_usable(company_news)
            ) or ("google" in lowered and not _is_usable(google_news))
            if not source_scoped_and_accurate:
                issues.append(
                    "The report says ticker-specific news is absent even though the "
                    "frozen snapshot contains usable ticker-news evidence"
                )
            break
    return sorted(set(issues))


def _grounded_news_digest(snapshot: dict[str, Any], *, fallback: bool = False) -> str:
    intro = (
        "The model-generated news narrative failed ticker-news grounding validation "
        "twice. It was replaced with this deterministic digest."
        if fallback
        else "These are the frozen ticker-news inputs shared with the Sentiment Analyst."
    )
    return f"## Frozen ticker-news evidence\n\n{intro}\n\n{_frozen_ticker_news(snapshot)}"


def _stock_system_message(
    *,
    ticker: str,
    snapshot: dict[str, Any],
    trade_context_note: str,
) -> str:
    context = (
        f"\nTrade parameters: {trade_context_note}\n" if trade_context_note else ""
    )
    return f"""Analyze news relevant to {ticker} over the past week.{context}

The ticker-news blocks below are the same frozen snapshot used by the Sentiment
Analyst. Their status and content are authoritative. Use evidence in OK or STALE
blocks. NO_DATA, DISABLED, and UNAVAILABLE mean unknown and carry no directional
weight. Never claim that company-specific news is absent when either block is OK
or STALE. Distinguish a source-specific gap, such as Yahoo Finance NO_DATA, from
usable evidence in another source, such as Google News OK.

Do not search for ticker-specific news again. The only available tool is
get_global_news, which may add broader macroeconomic evidence that could affect
this stock within the stated trade horizon.

{_frozen_ticker_news(snapshot)}

Produce a concise report that:
1. Summarizes the verified ticker-specific catalysts and risks.
2. Adds only macro events relevant within the trade horizon.
3. Clearly distinguishes verified evidence from unknown source gaps.
4. Ends with a Markdown table of the key evidence and likely near-term impact.

{get_language_instruction()}"""


def _generic_system_message(asset_label: str, trade_context_note: str) -> str:
    context = (
        f"\nTrade parameters: {trade_context_note}\n" if trade_context_note else ""
    )
    return f"""Analyze recent news and trends over the past week.{context}

Use get_news for {asset_label}-specific searches and get_global_news for broader
macroeconomic news. Include only macro factors that could affect the asset within
the stated trade horizon. Provide specific supporting evidence and end with a
Markdown summary table.

{get_language_instruction()}"""


def create_news_analyst(llm):
    """Create a news analyst that shares frozen stock-news inputs with sentiment."""

    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        snapshot = state.get("sentiment_source_snapshot") or {}
        instrument_context = build_instrument_context(
            state["company_of_interest"],
            asset_type,
            canonical_name=snapshot.get("company_name"),
        )

        if asset_type == "stock":
            tools = [get_global_news]
            system_message = _stock_system_message(
                ticker=state["company_of_interest"],
                snapshot=snapshot,
                trade_context_note=state.get("trade_context_note", ""),
            )
        else:
            tools = [get_news, get_global_news]
            system_message = _generic_system_message(
                asset_label,
                state.get("trade_context_note", ""),
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a financial news analyst. Use the supplied evidence and "
                        "available tools without inventing facts.\n{system_message}\n"
                        "Current date: {current_date}. {instrument_context}\n"
                        "Available tools: {tool_names}."
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(
            system_message=system_message,
            tool_names=", ".join(tool.name for tool in tools),
            current_date=current_date,
            instrument_context=instrument_context,
        )

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        tags = list(state.get("data_quality_tags") or [])
        if enforce_exact_tool_ticker(result, state["company_of_interest"]):
            tags.append("CORRECTED_TOOL_TICKER")

        if len(result.tool_calls) == 0:
            report = str(result.content or "")

        if asset_type == "stock" and report:
            issues = _news_report_issues(report, snapshot)
            if issues:
                repair_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            (
                                "Rewrite the news report so every ticker-news claim agrees "
                                "with the frozen snapshot. Preserve supported macro evidence. "
                                "Do not say ticker-specific news is absent when either frozen "
                                "source is usable. Distinguish source-specific missing data "
                                "from evidence available through another source.\n\n"
                                "Frozen ticker news:\n{frozen_news}"
                            ),
                        ),
                        (
                            "human",
                            "Validation errors:\n{issues}\n\nUnsupported draft:\n{draft}",
                        ),
                    ]
                )
                repaired = (repair_prompt | llm).invoke(
                    {
                        "frozen_news": _frozen_ticker_news(snapshot),
                        "issues": "\n".join(f"- {item}" for item in issues),
                        "draft": report,
                    }
                )
                repaired_report = str(repaired.content or "")
                if _news_report_issues(repaired_report, snapshot):
                    tags.extend(["INVALID_NEWS_REPORT", "FALLBACK_NEWS_DIGEST"])
                    report = _grounded_news_digest(snapshot, fallback=True)
                else:
                    tags.append("REPAIRED_NEWS_REPORT")
                    report = repaired_report
                result = repaired

        if "FALLBACK_NEWS_DIGEST" not in tags:
            report, output_tags = sanitize_agent_output(report, state)
            tags.extend(output_tags)
            if asset_type == "stock" and report:
                report = (
                    _grounded_news_digest(snapshot)
                    + "\n\n## News Analyst interpretation\n\n"
                    + report
                )

        return {
            "messages": [result],
            "news_report": report,
            "data_quality_tags": sorted(set(tags)),
        }

    return news_analyst_node
