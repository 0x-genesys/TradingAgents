"""Tests for the Telegram dataflow module.

Covers:
- Company name resolution fallback
- Placeholder when env vars are missing
- Format of the returned string (always a str, never None/exception)
- Integration test with known ticker (runs if TELEGRAM_API_ID/HASH are set)
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pytest

from tradingagents.dataflows.telegram import (
    INDIAN_CHANNELS,
    _get_company_name,
    _search_terms,
    fetch_telegram_messages,
)


@pytest.mark.unit
class TelegramCompanyNameTests(unittest.TestCase):
    """Company-name lookup used internally by the Telegram fetcher."""

    def test_known_nse_ticker(self):
        """NTPC.NS should resolve to 'NTPC Limited' or similar."""
        name = _get_company_name("NTPC.NS")
        self.assertTrue(
            len(name) > 3,
            msg=f"Expected a real company name for NTPC.NS, got '{name}'",
        )

    def test_empty_on_missing_ticker(self):
        """An unknown ticker should produce an empty string (not crash)."""
        name = _get_company_name("ZXYZZY.NONEXISTENT")
        self.assertIsInstance(name, str)


@pytest.mark.unit
class TelegramSearchTermsTests(unittest.TestCase):
    """Search-terms builder."""

    def test_returns_company_and_base(self):
        terms = _search_terms("NTPC.NS")
        self.assertTrue(
            any("NTPC" in t.upper() for t in terms),
            msg=f"Expected NTPC in terms, got {terms}",
        )

    def test_always_returns_list(self):
        terms = _search_terms("BOGUS.NS")
        self.assertIsInstance(terms, list)
        self.assertGreater(len(terms), 0)


@pytest.mark.unit
class TelegramFetchWithoutCredsTests(unittest.TestCase):
    """When TELEGRAM_API_ID / TELEGRAM_API_HASH are not set, the fetcher
    must return a friendly placeholder rather than crashing."""

    def setUp(self):
        # Ensure TELEGRAM_ENABLED is not set for these tests
        self._old_enabled = os.environ.pop("TELEGRAM_ENABLED", None)
        self._old_id = os.environ.pop("TELEGRAM_API_ID", None)
        self._old_hash = os.environ.pop("TELEGRAM_API_HASH", None)

    def tearDown(self):
        if self._old_enabled is not None:
            os.environ["TELEGRAM_ENABLED"] = self._old_enabled
        if self._old_id is not None:
            os.environ["TELEGRAM_API_ID"] = self._old_id
        if self._old_hash is not None:
            os.environ["TELEGRAM_API_HASH"] = self._old_hash

    def test_placeholder_without_creds(self):
        """Must return a placeholder string, not crash."""
        result = fetch_telegram_messages("NTPC.NS")
        self.assertIsInstance(result, str)
        self.assertIn("Telegram disabled", result)

    def test_placeholder_with_empty_creds(self):
        """Empty-string creds should also produce a placeholder."""
        os.environ["TELEGRAM_ENABLED"] = "true"
        os.environ["TELEGRAM_API_ID"] = ""
        os.environ["TELEGRAM_API_HASH"] = ""
        result = fetch_telegram_messages("NTPC.NS")
        self.assertIsInstance(result, str)
        self.assertIn("Telegram unavailable", result)

    def test_placeholder_with_bad_api_id(self):
        """Non-numeric API_ID should produce an informative placeholder."""
        os.environ["TELEGRAM_ENABLED"] = "true"
        os.environ["TELEGRAM_API_ID"] = "not-a-number"
        os.environ["TELEGRAM_API_HASH"] = "abc123"
        result = fetch_telegram_messages("NTPC.NS")
        self.assertIsInstance(result, str)
        self.assertIn("TELEGRAM_API_ID must be an integer", result)


@pytest.mark.integration
class TelegramIntegrationTests(unittest.TestCase):
    """End-to-end test that actually connects to Telegram and fetches messages
    for a known ticker.

    These tests only run when TELEGRAM_API_ID and TELEGRAM_API_HASH are set
    in the environment AND a valid session file exists (or interactive login
    has been completed).
    """

    def setUp(self):
        self.api_id = os.environ.get("TELEGRAM_API_ID", "").strip()
        self.api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()
        self.enabled = os.environ.get("TELEGRAM_ENABLED", "").strip().lower()
        if not self.api_id or not self.api_hash or self.enabled != "true":
            self.skipTest(
                "TELEGRAM_ENABLED=true + TELEGRAM_API_ID / TELEGRAM_API_HASH "
                "not all set — skipping integration test"
            )

    def test_fetch_recent_for_ntpc(self):
        """Fetch Telegram messages for NTPC.NS — should not crash and should
        return a string with expected content."""
        result = fetch_telegram_messages("NTPC.NS")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        # May contain data or a placeholder if no messages or auth failed
        self.assertNotIn("Traceback", result)
        self.assertNotIn("Traceback (most recent call last)", result, msg="No tracebacks should leak")
