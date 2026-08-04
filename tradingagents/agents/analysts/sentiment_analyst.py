"""Grounded sentiment analysis over a frozen, status-preserving source snapshot."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    find_unsupported_optional_source_claims,
    get_language_instruction,
    sanitize_agent_output,
)


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime(
        "%Y-%m-%d"
    )


def _source(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    return (snapshot.get("sources") or {}).get(name) or {
        "status": "UNAVAILABLE",
        "content": f"<{name} unavailable: source snapshot missing>",
    }


def _unsupported_claim_issues(report: str, snapshot: dict[str, Any]) -> list[str]:
    """Find claims that could not have come from the frozen snapshot."""
    issues: list[str] = []
    trends = _source(snapshot, "google_trends")
    telegram = _source(snapshot, "telegram")
    if trends.get("status") not in {"OK", "STALE"} and re.search(
        r"google\s+trends.{0,100}(?:\b\d{1,3}\s*/\s*100\b|\bscore\s*(?:is|:)?\s*\d+)",
        report,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        issues.append("Google Trends has no usable data, but the report states a numeric score")
    if telegram.get("status") != "OK" and re.search(
        r"telegram.{0,100}\b\d+\s+(?:messages?|mentions?|posts?)\b",
        report,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        issues.append("Telegram has no usable data, but the report states a message count")
    optional_labels = {
        "reddit": "Reddit",
        "telegram": "Telegram",
        "google_trends": "Google Trends",
    }
    for source_name, source_label in optional_labels.items():
        source = _source(snapshot, source_name)
        if source.get("status") in {"OK", "STALE"}:
            continue
        score_pattern = (
            rf"{re.escape(source_label)}.{{0,180}}"
            rf"(?:\|\s*0(?:\s*\(|\s*\|)|(?:score|weight|sentiment).{{0,30}}\b0\b)"
        )
        if re.search(score_pattern, report, flags=re.IGNORECASE | re.DOTALL):
            issues.append(
                f"{source_label} has no usable data, but the report assigns a zero score or weight"
            )
    for claim in find_unsupported_optional_source_claims(
        report,
        {"sentiment_source_snapshot": snapshot},
    ):
        issues.append(
            "An unavailable optional source is used as directional evidence: " + claim
        )
    return issues


def create_sentiment_analyst(llm):
    """Create a sentiment analyst with one bounded report-repair attempt."""

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        snapshot = state.get("sentiment_source_snapshot") or {}
        company_name = snapshot.get("company_name")
        instrument_context = build_instrument_context(
            ticker,
            state.get("asset_type", "stock"),
            canonical_name=company_name,
        )
        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            snapshot=snapshot,
            trade_context_note=state.get("trade_context_note", ""),
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial sentiment analyst. Use only the frozen source "
                    "snapshot below. Never invent source values. An unavailable optional "
                    "source is unknown, not bearish.\n{system_message}\n"
                    "Current date: {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(
            system_message=system_message,
            current_date=end_date,
            instrument_context=instrument_context,
        )
        result = (prompt | llm).invoke(state["messages"])
        report = str(result.content)
        issues = _unsupported_claim_issues(report, snapshot)
        tags = list(state.get("data_quality_tags") or [])

        if issues:
            repair_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Rewrite the sentiment report using only the frozen source snapshot. "
                        "Do not infer numeric values or bullish/bearish consequences from "
                        "unavailable sources. Their absence is unknown and must have zero "
                        "directional weight. Preserve useful supported evidence and explicitly "
                        "label missing sources.\n\n"
                        "Frozen inputs:\n{system_message}",
                    ),
                    (
                        "human",
                        "Validation errors:\n{issues}\n\nUnsupported draft:\n{draft}",
                    ),
                ]
            )
            repaired = (repair_prompt | llm).invoke(
                {
                    "system_message": system_message,
                    "issues": "\n".join(f"- {item}" for item in issues),
                    "draft": report,
                }
            )
            repaired_report = str(repaired.content)
            if _unsupported_claim_issues(repaired_report, snapshot):
                tags.extend(["INVALID_SENTIMENT_REPORT", "FALLBACK_SENTIMENT_DIGEST"])
                report, has_usable_source = _grounded_fallback_report(snapshot)
                if not has_usable_source:
                    tags.append("MISSING_SENTIMENT")
                result = repaired
            else:
                tags.append("REPAIRED_SENTIMENT_REPORT")
                report = repaired_report
                result = repaired

        if "FALLBACK_SENTIMENT_DIGEST" not in tags:
            report, output_tags = sanitize_agent_output(report, state)
            tags.extend(output_tags)

        return {
            "messages": [result],
            "sentiment_report": report,
            "data_quality_tags": sorted(set(tags)),
        }

    return sentiment_analyst_node


def _grounded_fallback_report(snapshot: dict[str, Any]) -> tuple[str, bool]:
    """Render valid frozen evidence without a second LLM interpretation."""
    sources = snapshot.get("sources") or {}
    labels = {
        "company_news": "Yahoo Finance company news",
        "google_news": "India-localized Google News",
        "reddit": "Reddit",
        "telegram": "Telegram",
        "google_trends": "Google Trends",
    }
    rows = []
    evidence = []
    has_usable_source = False
    for name, label in labels.items():
        source = sources.get(name) or {}
        status = str(source.get("status", "UNAVAILABLE")).upper()
        usable = status in {"OK", "STALE"}
        has_usable_source = has_usable_source or usable
        rows.append(f"| {label} | {status} | {'Evidence retained' if usable else 'Unknown'} |")
        if usable:
            content = str(source.get("content") or "<no content>")
            evidence.append(f"### {label} ({status})\n\n{content}")

    intro = (
        "The model-generated sentiment narrative failed source-grounding validation "
        "twice. It was replaced with this deterministic digest. Unavailable sources "
        "have zero directional weight. No sentiment direction is synthesized here."
    )
    report = (
        "## Grounded sentiment source digest\n\n"
        + intro
        + "\n\n| Source | Status | Treatment |\n|---|---|---|\n"
        + "\n".join(rows)
    )
    if evidence:
        report += "\n\n## Verified source evidence\n\n" + "\n\n".join(evidence)
    else:
        report += "\n\nNo usable ticker-specific sentiment or news source was available."
    return report, has_usable_source


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    snapshot: dict[str, Any],
    trade_context_note: str = "",
) -> str:
    sources = snapshot.get("sources") or {}
    source_sections = []
    labels = {
        "company_news": "Yahoo Finance company news",
        "google_news": "India-localized Google News",
        "reddit": "Reddit",
        "telegram": "Telegram",
        "google_trends": "Google Trends",
    }
    for key, label in labels.items():
        item = sources.get(key) or {"status": "UNAVAILABLE", "content": "<missing>"}
        source_sections.append(
            f"### {label}\nStatus: {item.get('status', 'UNAVAILABLE')}\n"
            f"<start_of_{key}>\n{item.get('content', '<missing>')}\n<end_of_{key}>"
        )
    context = f"\nTrade parameters: {trade_context_note}\n" if trade_context_note else ""
    return f"""Analyze sentiment for {ticker} from {start_date} through {end_date}.{context}

The blocks below are a frozen snapshot. Status is authoritative. When status is
NO_DATA, DISABLED, or UNAVAILABLE, state that limitation and assign no directional
weight. Do not claim a score, count, post, or message that is not in the block.
Healthy Yahoo Finance company news or India-localized Google News is enough to
produce a useful report. Do not require Telegram, Reddit, or Google Trends.

{chr(10).join(source_sections)}

Produce:
1. Overall Bullish, Bearish, Neutral, or Mixed sentiment with confidence.
2. A source-by-source account that distinguishes evidence from missing data.
3. Cross-source narratives, catalysts, and risks.
4. A Markdown summary table with source status and supported evidence.

{get_language_instruction()}"""


def create_social_media_analyst(llm):
    """Backward-compatible alias."""
    return create_sentiment_analyst(llm)
