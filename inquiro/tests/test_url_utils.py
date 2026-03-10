"""Tests for inquiro.infrastructure.url_utils 🔗.

Covers:
- DOI URL passthrough
- Raw DOI conversion
- PMID conversion
- Generic HTTPS extraction
- No URL returns None
- Edge cases (trailing punctuation, brackets, empty input)
"""

from __future__ import annotations

from inquiro.infrastructure.url_utils import extract_and_normalize_url


# ---------------------------------------------------------------------------
# Priority 1: DOI URL passthrough
# ---------------------------------------------------------------------------


class TestDoiUrlPassthrough:
    """Priority 1 — pre-formed https://doi.org/… URLs."""

    def test_doi_url_clean(self) -> None:
        """Returns canonical DOI URL when already well-formed."""
        url = "https://doi.org/10.1038/s41586-024-07618-3"
        assert extract_and_normalize_url(url) == url

    def test_doi_url_embedded_in_sentence(self) -> None:
        """Extracts DOI URL embedded in surrounding prose."""
        text = "See https://doi.org/10.1038/nature.2024 for full details."
        assert extract_and_normalize_url(text) == "https://doi.org/10.1038/nature.2024"

    def test_doi_url_trailing_period_stripped(self) -> None:
        """Strips trailing period from DOI URL at end of sentence."""
        text = "Published at https://doi.org/10.1000/xyz123."
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/xyz123"

    def test_doi_url_trailing_bracket_stripped(self) -> None:
        """Strips trailing closing bracket from DOI URL."""
        text = "(https://doi.org/10.1000/xyz123)"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/xyz123"

    def test_doi_url_http_variant(self) -> None:
        """Accepts http://doi.org/… and returns it as-is."""
        url = "http://doi.org/10.1234/test"
        assert extract_and_normalize_url(url) == url


# ---------------------------------------------------------------------------
# Priority 2: Raw DOI conversion
# ---------------------------------------------------------------------------


class TestRawDoiConversion:
    """Priority 2 — raw DOI references converted to https://doi.org/…"""

    def test_raw_doi_colon_prefix(self) -> None:
        """Converts 'doi:10.xxxx/…' to canonical DOI URL."""
        text = "doi:10.1038/nature.2024.001"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1038/nature.2024.001"

    def test_raw_doi_colon_space_prefix(self) -> None:
        """Converts 'doi: 10.xxxx/…' (with space) to canonical DOI URL."""
        text = "doi: 10.1016/j.cell.2024.01.001"
        assert (
            extract_and_normalize_url(text)
            == "https://doi.org/10.1016/j.cell.2024.01.001"
        )

    def test_raw_doi_case_insensitive(self) -> None:
        """Handles uppercase 'DOI:' prefix."""
        text = "DOI: 10.1093/nar/gkad001"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1093/nar/gkad001"

    def test_raw_doi_trailing_period_stripped(self) -> None:
        """Strips trailing period from raw DOI before converting."""
        text = "doi:10.1000/xyz."
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/xyz"

    def test_raw_doi_trailing_comma_stripped(self) -> None:
        """Strips trailing comma from raw DOI before converting."""
        text = "Reference: doi:10.1000/abc,"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/abc"

    def test_raw_doi_embedded_in_citation(self) -> None:
        """Extracts raw DOI from a typical citation string."""
        text = "Smith et al. (2024). Nature. doi:10.1038/s41586-024-001."
        assert (
            extract_and_normalize_url(text)
            == "https://doi.org/10.1038/s41586-024-001"
        )


# ---------------------------------------------------------------------------
# Priority 3: PMID conversion
# ---------------------------------------------------------------------------


class TestPmidConversion:
    """Priority 3 — PMID identifiers converted to PubMed URLs."""

    def test_pmid_colon_space(self) -> None:
        """Converts 'PMID: 12345678' to PubMed URL."""
        text = "PMID: 12345678"
        assert (
            extract_and_normalize_url(text)
            == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )

    def test_pmid_no_space(self) -> None:
        """Converts 'PMID:12345678' (no space) to PubMed URL."""
        text = "PMID:87654321"
        assert (
            extract_and_normalize_url(text)
            == "https://pubmed.ncbi.nlm.nih.gov/87654321/"
        )

    def test_pmid_case_insensitive(self) -> None:
        """Handles lowercase 'pmid:' prefix."""
        text = "pmid: 11111111"
        assert (
            extract_and_normalize_url(text)
            == "https://pubmed.ncbi.nlm.nih.gov/11111111/"
        )

    def test_pmid_embedded_in_paragraph(self) -> None:
        """Extracts PMID from surrounding text."""
        text = "This study (PMID: 99887766) demonstrated efficacy."
        assert (
            extract_and_normalize_url(text)
            == "https://pubmed.ncbi.nlm.nih.gov/99887766/"
        )

    def test_doi_url_takes_priority_over_pmid(self) -> None:
        """DOI URL has higher priority than PMID when both present."""
        text = "https://doi.org/10.1000/xyz PMID: 12345678"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/xyz"

    def test_raw_doi_takes_priority_over_pmid(self) -> None:
        """Raw DOI has higher priority than PMID when both present."""
        text = "doi:10.1000/xyz PMID: 12345678"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/xyz"


# ---------------------------------------------------------------------------
# Priority 4: Generic HTTPS extraction
# ---------------------------------------------------------------------------


class TestGenericHttpsExtraction:
    """Priority 4 — generic https://… URLs."""

    def test_generic_https_url(self) -> None:
        """Returns a plain HTTPS URL found in text."""
        text = "Read more at https://example.com/paper"
        assert extract_and_normalize_url(text) == "https://example.com/paper"

    def test_generic_https_trailing_period_stripped(self) -> None:
        """Strips trailing period from generic HTTPS URL."""
        text = "Source: https://example.com/paper."
        assert extract_and_normalize_url(text) == "https://example.com/paper"

    def test_generic_https_trailing_semicolon_stripped(self) -> None:
        """Strips trailing semicolon from generic HTTPS URL."""
        text = "https://example.com/a;"
        assert extract_and_normalize_url(text) == "https://example.com/a"

    def test_generic_https_with_query_string(self) -> None:
        """Preserves query string parameters in generic HTTPS URL."""
        url = "https://example.com/search?q=test&page=1"
        assert extract_and_normalize_url(url) == url


# ---------------------------------------------------------------------------
# No URL returns None
# ---------------------------------------------------------------------------


class TestNoUrlReturnsNone:
    """Cases where no URL can be extracted."""

    def test_plain_text_no_url(self) -> None:
        """Returns None for plain text with no URL or identifier."""
        assert extract_and_normalize_url("No links here.") is None

    def test_empty_string(self) -> None:
        """Returns None for empty string input."""
        assert extract_and_normalize_url("") is None

    def test_whitespace_only(self) -> None:
        """Returns None for whitespace-only string."""
        assert extract_and_normalize_url("   ") is None

    def test_http_url_not_https(self) -> None:
        """Does not match plain http:// URL as generic (priority 4 requires https)."""
        # http://doi.org is matched by priority 1; plain http is not matched
        result = extract_and_normalize_url("http://example.com/page")
        # http://example.com is neither a doi.org URL nor matched by https pattern
        assert result is None

    def test_number_only(self) -> None:
        """Returns None when text contains only a number (not a valid PMID pattern)."""
        assert extract_and_normalize_url("12345678") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and edge-case scenarios."""

    def test_doi_url_with_multiple_paths(self) -> None:
        """Handles DOI URL with complex path segments."""
        url = "https://doi.org/10.1016/j.celrep.2024.113942"
        assert extract_and_normalize_url(url) == url

    def test_summary_with_doi_and_https_url(self) -> None:
        """DOI URL wins over a generic HTTPS URL in same text."""
        text = "Check https://example.com and https://doi.org/10.1234/test"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1234/test"

    def test_brackets_around_doi_url(self) -> None:
        """Strips closing bracket from DOI URL inside square brackets."""
        text = "[https://doi.org/10.1000/abc]"
        assert extract_and_normalize_url(text) == "https://doi.org/10.1000/abc"

    def test_pmid_at_end_of_sentence(self) -> None:
        """Handles PMID at end of sentence (period should not affect number)."""
        text = "Validated in PMID: 11223344."
        assert (
            extract_and_normalize_url(text)
            == "https://pubmed.ncbi.nlm.nih.gov/11223344/"
        )
