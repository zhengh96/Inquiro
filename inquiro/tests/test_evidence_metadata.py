"""Tests for evidence_metadata — NCT and DOI extraction 🏷️."""

from __future__ import annotations

from inquiro.core.evidence_metadata import extract_clinical_trial_id, extract_doi


class TestClinicalTrialExtraction:
    """NCT ID extraction from URL and summary ✅."""

    def test_nct_from_url(self) -> None:
        """Extract NCT ID from URL."""
        result = extract_clinical_trial_id(
            "https://clinicaltrials.gov/ct2/show/NCT12345678", None
        )
        assert result == "NCT12345678"

    def test_nct_from_summary(self) -> None:
        """Extract NCT ID from summary when URL has none."""
        result = extract_clinical_trial_id(
            "https://example.com",
            "Study NCT98765432 showed efficacy",
        )
        assert result == "NCT98765432"

    def test_nct_case_insensitive(self) -> None:
        """NCT extraction is case-insensitive."""
        result = extract_clinical_trial_id("https://example.com/nct11111111", None)
        assert result == "NCT11111111"

    def test_url_priority_over_summary(self) -> None:
        """URL NCT takes priority over summary NCT."""
        result = extract_clinical_trial_id(
            "https://clinicaltrials.gov/NCT11111111",
            "Study NCT22222222 results",
        )
        assert result == "NCT11111111"

    def test_requires_8_digits(self) -> None:
        """NCT IDs must have at least 8 digits."""
        result = extract_clinical_trial_id("https://example.com/NCT1234", None)
        assert result is None

    def test_none_inputs(self) -> None:
        """None inputs handled without error."""
        assert extract_clinical_trial_id(None, None) is None

    def test_no_match_returns_none(self) -> None:
        """No NCT pattern returns None."""
        result = extract_clinical_trial_id("https://example.com", "No trial ID here")
        assert result is None


class TestDoiExtraction:
    """DOI extraction from URL and summary ✅."""

    def test_doi_from_url(self) -> None:
        """Extract DOI from URL."""
        result = extract_doi("https://doi.org/10.1234/test.2024", None)
        assert result == "10.1234/test.2024"

    def test_doi_from_summary(self) -> None:
        """Extract DOI from summary when URL has none."""
        result = extract_doi(
            "https://example.com",
            "Published as 10.5678/journal.pone.12345",
        )
        assert result == "10.5678/journal.pone.12345"

    def test_trailing_period_stripped(self) -> None:
        """Trailing period is stripped from DOI."""
        result = extract_doi(None, "DOI 10.1234/test.")
        assert result == "10.1234/test"

    def test_trailing_comma_stripped(self) -> None:
        """Trailing comma is stripped from DOI."""
        result = extract_doi(None, "See 10.1234/test, for details")
        assert result == "10.1234/test"

    def test_trailing_parenthesis_stripped(self) -> None:
        """Trailing closing parenthesis is stripped from DOI."""
        result = extract_doi(None, "(10.1234/test)")
        assert result == "10.1234/test"

    def test_trailing_semicolon_stripped(self) -> None:
        """Trailing semicolon is stripped from DOI."""
        result = extract_doi(None, "10.1234/test;")
        assert result == "10.1234/test"

    def test_url_priority_over_summary(self) -> None:
        """URL DOI takes priority over summary DOI."""
        result = extract_doi(
            "https://doi.org/10.1111/first",
            "Also 10.2222/second",
        )
        assert result == "10.1111/first"

    def test_none_inputs(self) -> None:
        """None inputs handled without error."""
        assert extract_doi(None, None) is None

    def test_no_match_returns_none(self) -> None:
        """No DOI pattern returns None."""
        result = extract_doi("https://example.com", "No DOI here")
        assert result is None
