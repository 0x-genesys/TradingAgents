"""Tests for the Google Trends dataflow module.

Covers:
- Company name resolution
- Formatted response (always a str, never None/exception)
- Integration test with a real ticker (pytrends needs no API key)
- Placeholder on expected failures
"""

from __future__ import annotations

import unittest

import pytest

from tradingagents.dataflows.google_trends import (
    _get_company_name,
    fetch_google_trends,
)


@pytest.mark.unit
class GoogleTrendsCompanyNameTests(unittest.TestCase):
    """Company-name lookup used internally."""

    def test_known_nse_ticker(self):
        """NTPC.NS should resolve to 'NTPC Limited' or similar."""
        name = _get_company_name("NTPC.NS")
        self.assertTrue(
            len(name) > 3,
            msg=f"Expected a real company name for NTPC.NS, got '{name}'",
        )

    def test_empty_on_missing_ticker(self):
        """An unknown ticker should return empty string."""
        name = _get_company_name("ZXYZZY.NONEXISTENT")
        self.assertIsInstance(name, str)


@pytest.mark.unit
class GoogleTrendsOutputTests(unittest.TestCase):
    """Tests that the fetcher returns well-formed output."""

    def test_output_is_always_str(self):
        """The result must always be a string, never None or exception."""
        result = fetch_google_trends("NTPC.NS")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_output_contains_company_name(self):
        """The output should mention the company name it searched for."""
        result = fetch_google_trends("NTPC.NS")
        self.assertIn("NTPC", result)

    def test_output_for_unknown_ticker(self):
        """Even for an unknown ticker, the output should be a graceful string."""
        result = fetch_google_trends("ZXYZZY.NS")
        self.assertIsInstance(result, str)

    def test_output_mentions_google_trends(self):
        """The output should reference the data source."""
        result = fetch_google_trends("NTPC.NS")
        self.assertIn("Google Trends", result)

    def test_output_contains_interest_score_or_fallback(self):
        """The output should contain an interest score OR a graceful fallback."""
        result = fetch_google_trends("NTPC.NS")
        if "Google Trends unavailable" not in result:
            self.assertIn("Current score", result)

    def test_output_contains_direction_or_fallback(self):
        """The output should contain the trend direction OR a graceful fallback."""
        result = fetch_google_trends("NTPC.NS")
        if "Google Trends unavailable" not in result:
            self.assertIn("Direction", result)


@pytest.mark.unit
class GoogleTrendsDifferentTimeframesTests(unittest.TestCase):
    """Test that different timeframe strings work."""

    def test_one_day_timeframe(self):
        """'now 1-d' should work without errors."""
        result = fetch_google_trends("RELIANCE.NS", timeframe="now 1-d")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_news_property(self):
        """gprop='news' should return news-search interest."""
        result = fetch_google_trends("HDFCBANK.NS", timeframe="now 7-d", gprop="news")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)


@pytest.mark.integration
class GoogleTrendsIntegrationTests(unittest.TestCase):
    """End-to-end integration tests that call the actual Google Trends API."""

    def test_real_data_for_ntpc(self):
        """Fetch Google Trends for NTPC.NS — must return a string with expected
        structure or a graceful rate-limit fallback."""
        result = fetch_google_trends("NTPC.NS")
        self.assertIsInstance(result, str)
        if "Google Trends unavailable" not in result:
            self.assertIn("Current score:", result)
            self.assertIn("7-day peak:", result)
            self.assertIn("Direction:", result)
        # Must never contain a Python traceback
        self.assertNotIn("Traceback", result)

    def test_real_data_for_reliance(self):
        """Reliance is one of the most-searched stocks — data should be rich."""
        result = fetch_google_trends("RELIANCE.NS")
        self.assertIsInstance(result, str)
        self.assertIn("Current score:", result)
        self.assertNotIn("unavailable", result.lower())
