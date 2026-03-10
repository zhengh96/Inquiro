"""Tests for canonical_evidence_hash — unified dedup key 🔑."""

from __future__ import annotations

import pytest

from inquiro.core.canonical_hash import canonical_evidence_hash


class TestHashConsistency:
    """Deterministic hash output for identical inputs ✅."""

    def test_same_inputs_same_hash(self) -> None:
        """Identical inputs always produce the same hash."""
        h1 = canonical_evidence_hash("http://a.com", "summary")
        h2 = canonical_evidence_hash("http://a.com", "summary")
        assert h1 == h2

    def test_different_summaries_different_hash(self) -> None:
        """Different summaries produce different hashes."""
        h1 = canonical_evidence_hash("http://a.com", "summary one")
        h2 = canonical_evidence_hash("http://a.com", "summary two")
        assert h1 != h2

    def test_different_urls_different_hash(self) -> None:
        """Different URLs produce different hashes."""
        h1 = canonical_evidence_hash("http://a.com", "same summary")
        h2 = canonical_evidence_hash("http://b.com", "same summary")
        assert h1 != h2


class TestUrlNormalization:
    """URL normalisation ensures case/whitespace invariance ✅."""

    def test_case_insensitive_url(self) -> None:
        """URL casing does not affect hash."""
        h1 = canonical_evidence_hash("HTTP://EXAMPLE.COM", "test")
        h2 = canonical_evidence_hash("http://example.com", "test")
        assert h1 == h2

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace in URL is stripped."""
        h1 = canonical_evidence_hash("  http://a.com  ", "test")
        h2 = canonical_evidence_hash("http://a.com", "test")
        assert h1 == h2

    def test_none_equals_empty_string(self) -> None:
        """None URL is treated the same as empty string."""
        h1 = canonical_evidence_hash(None, "test summary")
        h2 = canonical_evidence_hash("", "test summary")
        assert h1 == h2


class TestSummaryNormalization:
    """Summary normalisation ensures consistent hashing ✅."""

    def test_case_insensitive_summary(self) -> None:
        """Summary casing does not affect hash."""
        h1 = canonical_evidence_hash("http://a.com", "Test Summary")
        h2 = canonical_evidence_hash("http://a.com", "test summary")
        assert h1 == h2

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace in summary is stripped."""
        h1 = canonical_evidence_hash("http://a.com", "  test  ")
        h2 = canonical_evidence_hash("http://a.com", "test")
        assert h1 == h2

    def test_truncation_at_500_chars(self) -> None:
        """Summaries beyond 500 chars are truncated for hashing."""
        base = "a" * 500
        h1 = canonical_evidence_hash("", base + "EXTRA")
        h2 = canonical_evidence_hash("", base)
        assert h1 == h2

    def test_within_500_chars_distinct(self) -> None:
        """Summaries within 500 chars remain distinct."""
        h1 = canonical_evidence_hash("", "a" * 499 + "x")
        h2 = canonical_evidence_hash("", "a" * 499 + "y")
        assert h1 != h2


class TestHashFormat:
    """Output format validation ✅."""

    def test_64_char_hex(self) -> None:
        """Hash is a 64-character lowercase hex string."""
        h = canonical_evidence_hash("http://a.com", "test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_none_url_still_64_char(self) -> None:
        """None URL still produces a valid 64-char hex hash."""
        h = canonical_evidence_hash(None, "test")
        assert len(h) == 64


class TestHashVersionValidation:
    """hash_version parameter validation ✅."""

    def test_version_2_raises(self) -> None:
        """Unsupported hash_version=2 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported hash_version=2"):
            canonical_evidence_hash("", "test", hash_version=2)

    def test_version_0_raises(self) -> None:
        """Unsupported hash_version=0 raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported hash_version=0"):
            canonical_evidence_hash("", "test", hash_version=0)

    def test_default_version_succeeds(self) -> None:
        """Default hash_version=1 succeeds."""
        h = canonical_evidence_hash("", "test")
        assert len(h) == 64
