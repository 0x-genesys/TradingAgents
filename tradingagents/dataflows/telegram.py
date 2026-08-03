"""Optional Telegram enrichment for Indian stock discussion.

The fetch is bounded, date-aware, and non-interactive. Authentication must be
completed separately so a scheduled run can never pause for a phone or OTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

INDIAN_CHANNELS = (
    "StockMarketFactory",
    "TradeFunda",
    "InvestAajForKal",
    "IndianStocks",
)

_SEARCH_WINDOW_DAYS = 7
_MAX_MESSAGES_PER_CHANNEL = 20


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance."""
    del timeout
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            info = yf.Ticker(ticker).info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _search_terms(ticker: str, company_name: str | None = None) -> list[str]:
    """Return distinct company, ticker, hashtag, and cashtag search terms."""
    company = (company_name or _get_company_name(ticker)).strip()
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    candidates = [company, base, f"#{base}", f"${base}"]
    terms: list[str] = []
    for candidate in candidates:
        if candidate and candidate.casefold() not in {item.casefold() for item in terms}:
            terms.append(candidate)
    return terms or [base]


def _configured_channels(channels: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get("TELEGRAM_CHANNELS", "").strip()
    if not raw:
        return channels
    return tuple(value.strip().lstrip("@") for value in raw.split(",") if value.strip())


def fetch_telegram_messages(
    ticker: str,
    channels: tuple[str, ...] = INDIAN_CHANNELS,
    limit_per_channel: int = _MAX_MESSAGES_PER_CHANNEL,
    max_days: int = _SEARCH_WINDOW_DAYS,
    *,
    as_of_date: str | None = None,
    company_name: str | None = None,
) -> str:
    """Fetch matching messages without blocking the analysis on failure."""
    if os.environ.get("TELEGRAM_ENABLED", "").strip().lower() != "true":
        return "<Telegram disabled: set TELEGRAM_ENABLED=true to enable>"

    api_id = os.environ.get("TELEGRAM_API_ID", "").strip()
    api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash:
        return "<Telegram unavailable: TELEGRAM_API_ID or TELEGRAM_API_HASH is missing>"
    try:
        api_id_int = int(api_id)
    except (TypeError, ValueError):
        return "<Telegram unavailable: TELEGRAM_API_ID must be an integer>"

    selected_channels = _configured_channels(channels)
    if not selected_channels:
        return "<Telegram unavailable: no channels configured>"

    session_file = os.environ.get("TELEGRAM_SESSION_FILE", "telegram_session")
    try:
        timeout_seconds = max(1.0, float(os.environ.get("TELEGRAM_TIMEOUT_SECONDS", "20")))
    except ValueError:
        timeout_seconds = 20.0

    return _sync_fetch(
        api_id_int,
        api_hash,
        session_file,
        ticker,
        selected_channels,
        limit_per_channel,
        max_days,
        as_of_date,
        company_name,
        timeout_seconds,
    )


def _sync_fetch(
    api_id: int,
    api_hash: str,
    session_file: str,
    ticker: str,
    channels: tuple[str, ...],
    limit_per_channel: int,
    max_days: int,
    as_of_date: str | None,
    company_name: str | None,
    timeout_seconds: float,
) -> str:
    try:
        return asyncio.run(
            asyncio.wait_for(
                _fetch_telegram_async(
                    api_id,
                    api_hash,
                    session_file,
                    ticker,
                    channels,
                    limit_per_channel,
                    max_days,
                    as_of_date,
                    company_name,
                ),
                timeout=timeout_seconds,
            )
        )
    except TimeoutError:
        return f"<Telegram unavailable for {ticker}: timed out after {timeout_seconds:g}s>"
    except Exception as exc:
        logger.warning("Telegram fetch failed for %s: %s", ticker, exc)
        return f"<Telegram unavailable for {ticker}: {type(exc).__name__}>"


async def _fetch_telegram_async(
    api_id: int,
    api_hash: str,
    session_file: str,
    ticker: str,
    channels: tuple[str, ...],
    limit_per_channel: int,
    max_days: int,
    as_of_date: str | None,
    company_name: str | None,
) -> str:
    from telethon import TelegramClient

    search_terms = _search_terms(ticker, company_name)
    if as_of_date:
        end = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end += timedelta(days=1)
    else:
        end = datetime.now(timezone.utc)
    cutoff = end - timedelta(days=max_days)
    blocks: list[str] = []
    seen: set[tuple[int, int]] = set()

    async with TelegramClient(session_file, api_id, api_hash) as client:
        if not await client.is_user_authorized():
            return "<Telegram unavailable: session is not authorized>"

        for channel_name in channels:
            try:
                entity = await client.get_entity(channel_name)
            except Exception as exc:
                logger.info("Telegram channel %s unavailable: %s", channel_name, exc)
                continue

            channel_lines: list[str] = []
            for term in search_terms:
                try:
                    messages = client.iter_messages(
                        entity,
                        search=term,
                        offset_date=end,
                        limit=limit_per_channel,
                    )
                    async for message in messages:
                        message_date = message.date
                        if not message_date:
                            continue
                        if message_date.tzinfo is None:
                            message_date = message_date.replace(tzinfo=timezone.utc)
                        if message_date < cutoff or message_date >= end or not message.text:
                            continue
                        key = (getattr(entity, "id", 0), message.id)
                        if key in seen:
                            continue
                        seen.add(key)
                        preview = message.text.strip().replace("\n", " ")[:240]
                        channel_lines.append(
                            f"  [{message_date:%Y-%m-%d %H:%M}] {preview}"
                        )
                        if len(channel_lines) >= limit_per_channel:
                            break
                except Exception as exc:
                    logger.info(
                        "Telegram search failed for %s/%s: %s",
                        channel_name,
                        term,
                        exc,
                    )
                if len(channel_lines) >= limit_per_channel:
                    break

            if channel_lines:
                title = getattr(entity, "title", channel_name)
                blocks.append(f"t.me/{channel_name} / {title}:\n" + "\n".join(channel_lines))

    if not blocks:
        return (
            f"<Telegram: no messages found for {' / '.join(search_terms)} "
            f"from {cutoff.date()} through {(end - timedelta(days=1)).date()}>"
        )
    return (
        f"Telegram messages for {' / '.join(search_terms)} "
        f"({len(seen)} messages):\n" + "\n".join(blocks)
    )
