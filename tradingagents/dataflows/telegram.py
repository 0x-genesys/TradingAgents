"""Telegram channel fetcher for Indian stock discussion.

Uses Telethon (MTProto) to fetch messages from public Indian stock-related
Telegram channels/chats, searching by company name for best results.

Setup:
  1. Create a Telegram API app at https://my.telegram.org/apps (free, instant)
  2. Set TELEGRAM_API_ID and TELEGRAM_API_HASH in your .env
  3. (Optional) Set TELEGRAM_SESSION_FILE for a persistent session file.
     On first run Telethon prompts for phone number + verification code.
     The session file caches the auth so subsequent runs are automatic.

Source channels are configurable via the ``INDIAN_CHANNELS`` constant or
the ``telegram_channels`` parameter to ``fetch_telegram_messages()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import contextlib
import io
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Well-known Indian stock-discussion Telegram channels / groups.
# Public channels can be joined by their @username.
INDIAN_CHANNELS = (
    "StockMarketFactory",
    "TradeFunda",
    "InvestAajForKal",
    "IndianStocks",
)

_SEARCH_WINDOW_DAYS = 7
_MAX_MESSAGES_PER_CHANNEL = 20


def _get_company_name(ticker: str, timeout: float = 5.0) -> str:
    """Look up the company name for a ticker via yfinance (short timeout)."""
    try:
        import yfinance as yf

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            t = yf.Ticker(ticker)
            info = t.info or {}
        return (info.get("longName") or info.get("shortName") or "").strip()
    except Exception:
        return ""


def _search_terms(ticker: str) -> list[str]:
    """Build a list of search terms to look for in Telegram messages.

    Returns the company name (when available) and the bare ticker symbol
    so we catch both institutional discussion and ticker-specific chatter.
    """
    company = _get_company_name(ticker)
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    terms = []
    if company and len(company) > 2:
        terms.append(company)
    if base and base != company:
        terms.append(base)
    if not terms:
        terms.append(base)
    return terms


def fetch_telegram_messages(
    ticker: str,
    channels: tuple[str, ...] = INDIAN_CHANNELS,
    limit_per_channel: int = _MAX_MESSAGES_PER_CHANNEL,
    max_days: int = _SEARCH_WINDOW_DAYS,
) -> str:
    """Fetch recent Telegram messages mentioning ``ticker`` from Indian stock
    channels and return them as a formatted plaintext block.

    Returns a placeholder string on any failure — the caller never has to
    special-case None or exceptions.
    """
    api_id = os.environ.get("TELEGRAM_API_ID", "").strip()
    api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()

    if not api_id or not api_hash:
        return (
            "<Telegram unavailable: set TELEGRAM_API_ID and TELEGRAM_API_HASH "
            "in your .env>"
        )

    if not channels:
        return "<Telegram unavailable: no channels configured>"

    try:
        api_id_int = int(api_id)
    except (ValueError, TypeError):
        return "<Telegram unavailable: TELEGRAM_API_ID must be an integer>"

    session_file = os.environ.get("TELEGRAM_SESSION_FILE") or "telegram_session"

    # Run the async fetch in a sync wrapper
    return _sync_fetch(
        api_id_int, api_hash, session_file, ticker, channels, limit_per_channel, max_days
    )


def _sync_fetch(
    api_id: int,
    api_hash: str,
    session_file: str,
    ticker: str,
    channels: tuple[str, ...],
    limit_per_channel: int,
    max_days: int,
) -> str:
    """Synchronous wrapper for the async Telegram fetch."""
    try:
        return asyncio.run(
            _fetch_telegram_async(
                api_id, api_hash, session_file, ticker, channels, limit_per_channel, max_days
            )
        )
    except Exception as exc:
        logger.warning("Telegram fetch failed for %s: %s", ticker, exc)
        return f"<Telegram unavailable: {type(exc).__name__}: {exc}>"


async def _fetch_telegram_async(
    api_id: int,
    api_hash: str,
    session_file: str,
    ticker: str,
    channels: tuple[str, ...],
    limit_per_channel: int,
    max_days: int,
) -> str:
    """Core async implementation."""
    from telethon import TelegramClient

    search_terms = _search_terms(ticker)
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    total_messages = 0
    blocks = []

    async with TelegramClient(session_file, api_id, api_hash) as client:
        # Check if we're authorized
        if not await client.is_user_authorized():
            # Telethon will raise an interactive prompt via stderr/stdin.
            # In non-interactive contexts we can't proceed.
            logger.warning(
                "Telegram: not authorized. "
                "On first use, run interactively to complete phone login."
            )
            return (
                "<Telegram unavailable: not authorized. "
                "Run the interactive login first:\n"
                "  python -c \"from telethon import TelegramClient; "
                "client = TelegramClient('session', API_ID, API_HASH); "
                "client.start(); client.disconnect()\"\n"
                "or keep a session file from a prior login.>"
            )

        for channel_name in channels:
            try:
                entity = await client.get_entity(channel_name)
            except ValueError as exc:
                logger.info("Telegram: channel '%s' not found: %s", channel_name, exc)
                blocks.append(
                    f"  t.me/{channel_name} — skipped (not found or private)"
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Telegram: error resolving '%s': %s", channel_name, exc
                )
                continue

            channel_title = getattr(entity, "title", channel_name)
            channel_username = getattr(entity, "username", channel_name)

            # Fetch recent messages
            channel_matches = 0
            channel_lines: list[str] = []
            message_count = 0

            try:
                async for message in client.iter_messages(
                    entity, limit=limit_per_channel
                ):
                    if message.date and message.date.replace(tzinfo=timezone.utc) < cutoff:
                        continue
                    if not message.text:
                        continue

                    text = message.text.strip()
                    # Check if any search term appears
                    text_lower = text.lower()
                    matched = False
                    for term in search_terms:
                        if term.lower() in text_lower:
                            matched = True
                            break
                    # Also check raw ticker (e.g. "NTPC", "TATACOMM")
                    if not matched and base.lower() in text_lower:
                        matched = True

                    if not matched:
                        continue

                    message_count += 1
                    channel_matches += 1
                    total_messages += 1

                    msg_date = message.date.strftime("%Y-%m-%d %H:%M") if message.date else "?"
                    sender = (
                        message.sender.first_name or ""
                        if message.sender
                        else ""
                    )
                    sender_str = f" · @{sender}" if sender else ""

                    preview = text[:200].replace("\n", " ").strip()
                    if len(text) > 200:
                        preview += "…"

                    channel_lines.append(f"  [{msg_date}{sender_str}] {preview}")

                    if channel_matches >= limit_per_channel:
                        break

            except Exception as exc:
                logger.warning(
                    "Telegram: error reading messages from '%s': %s",
                    channel_name,
                    exc,
                )
                continue

            if channel_lines:
                blocks.append(
                    f"t.me/{channel_username} / {channel_title} — "
                    f"{message_count} recent messages:"
                )
                blocks.extend(channel_lines)
            else:
                blocks.append(
                    f"  t.me/{channel_username} — no mentions of "
                    f"{' / '.join(search_terms)} found"
                )

    if total_messages == 0:
        channel_str = ", ".join(channels)
        return (
            f"<Telegram: no messages mentioning "
            f"{' / '.join(search_terms)} in any of "
            f"[{channel_str}] in the past {max_days} days>"
        )

    return (
        f"Telegram messages mentioning "
        f"{' / '.join(search_terms)} "
        f"(past {max_days} days, {total_messages} messages found):\n"
        + "\n".join(blocks)
    )
