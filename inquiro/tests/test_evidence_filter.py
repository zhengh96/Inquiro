"""Tests for Inquiro EvidenceFilter 🧪.

Tests the evidence validity detection and filtering pipeline:
- Valid evidence passes through
- HTTP errors, tool errors, empty results are caught
- FilteredEvidence correctly splits and calculates error_rate
- Invalid records preserve _invalid audit marker
"""

from __future__ import annotations

import pytest

from inquiro.infrastructure.evidence_filter import (
    EvidenceFilter,
)


# ============================================================
# 🧪 Test: is_valid_evidence
# ============================================================


class TestIsValidEvidence:
    """Tests for EvidenceFilter.is_valid_evidence static method 🔍."""

    def test_normal_summary_is_valid(self) -> None:
        """Standard evidence summary should be valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "EGFR is overexpressed in 60% of NSCLC tumors."
            )
            is True
        )

    def test_empty_string_is_invalid(self) -> None:
        """Empty string should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("") is False

    def test_none_is_invalid(self) -> None:
        """None should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence(None) is False

    def test_whitespace_only_is_invalid(self) -> None:
        """Whitespace-only string should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("   \n  \t  ") is False

    def test_empty_results_json_is_invalid(self) -> None:
        """Empty results JSON should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence('{"results": []}') is False
        assert EvidenceFilter.is_valid_evidence('{"results":[]}') is False

    def test_empty_list_is_invalid(self) -> None:
        """Empty list JSON should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("[]") is False

    def test_empty_object_is_invalid(self) -> None:
        """Empty object JSON should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("{}") is False

    def test_http_error_is_invalid(self) -> None:
        """HTTP error format should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("401, message='Unauthorized'") is False
        assert (
            EvidenceFilter.is_valid_evidence("500, message='Internal Server Error'")
            is False
        )

    def test_tool_error_is_invalid(self) -> None:
        """Tool execution error should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "Error executing tool biomcp_search: connection timeout"
            )
            is False
        )

    def test_generic_error_is_invalid(self) -> None:
        """Generic error prefix should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Error: request failed with status 503")
            is False
        )

    def test_multiline_with_error_first_line(self) -> None:
        """Multiline text with error on first line should be invalid ❌."""
        text = "Error: timeout\nBut this line is fine."
        assert EvidenceFilter.is_valid_evidence(text) is False

    def test_json_with_actual_content_is_valid(self) -> None:
        """JSON that contains actual data should be valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence('{"gene": "EGFR", "expression": "high"}')
            is True
        )


# ============================================================
# 🧪 Test: filter method
# ============================================================


class TestFilter:
    """Tests for EvidenceFilter.filter class method 🧹."""

    def test_all_valid_records(self) -> None:
        """All valid records should pass through ✅."""
        records = [
            {"id": "E1", "summary": "Finding about EGFR."},
            {"id": "E2", "summary": "Safety data from Phase II."},
        ]
        result = EvidenceFilter.filter(records)
        assert len(result.valid) == 2
        assert len(result.invalid) == 0
        assert result.error_rate == 0.0

    def test_all_invalid_records(self) -> None:
        """All invalid records should be filtered ❌."""
        records = [
            {"id": "E1", "summary": ""},
            {"id": "E2", "summary": '{"results": []}'},
            {"id": "E3", "summary": "Error: timeout"},
        ]
        result = EvidenceFilter.filter(records)
        assert len(result.valid) == 0
        assert len(result.invalid) == 3
        assert result.error_rate == 1.0

    def test_mixed_records(self) -> None:
        """Mixed records should be correctly separated 📊."""
        records = [
            {"id": "E1", "summary": "Valid evidence here."},
            {"id": "E2", "summary": "401, message='Unauthorized'"},
            {"id": "E3", "summary": "Another valid finding."},
            {"id": "E4", "summary": ""},
        ]
        result = EvidenceFilter.filter(records)
        assert len(result.valid) == 2
        assert len(result.invalid) == 2
        assert result.error_rate == pytest.approx(0.5)

    def test_empty_input(self) -> None:
        """Empty input should produce empty result 🛡️."""
        result = EvidenceFilter.filter([])
        assert len(result.valid) == 0
        assert len(result.invalid) == 0
        assert result.error_rate == 0.0

    def test_invalid_records_have_marker(self) -> None:
        """Invalid records should have _invalid=True marker 🏷️."""
        records = [{"id": "E1", "summary": "Error: fail"}]
        result = EvidenceFilter.filter(records)
        assert len(result.invalid) == 1
        assert result.invalid[0]["_invalid"] is True
        assert result.invalid[0]["id"] == "E1"

    def test_valid_records_not_mutated(self) -> None:
        """Valid records should not have _invalid marker ✅."""
        records = [{"id": "E1", "summary": "Good evidence."}]
        result = EvidenceFilter.filter(records)
        assert len(result.valid) == 1
        assert "_invalid" not in result.valid[0]

    def test_error_rate_calculation(self) -> None:
        """Error rate should be invalid / total 📊."""
        records = [
            {"id": "E1", "summary": "Valid."},
            {"id": "E2", "summary": "Also valid."},
            {"id": "E3", "summary": ""},
        ]
        result = EvidenceFilter.filter(records)
        assert result.error_rate == pytest.approx(1 / 3)


# ============================================================
# 🧪 Test: New invalid patterns (API errors, ThinkTool, entity not found)
# ============================================================


class TestNewInvalidPatterns:
    """Tests for newly added invalid evidence patterns 🧪."""

    def test_input_validation_error_is_invalid(self) -> None:
        """API parameter validation error should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "Input validation error: 'auto' is not one of "
                "['day', 'week', 'month', 'year']"
            )
            is False
        )

    def test_think_tool_response_is_invalid(self) -> None:
        """ThinkTool JSON response should be invalid ❌."""
        think_response = (
            '{\n  "domain": "thinking",\n  "result": '
            '"Added thought 1 to main sequence.",\n  '
            '"thoughtNumber": 1,\n  "nextThoughtNeeded": true\n}'
        )
        assert EvidenceFilter.is_valid_evidence(think_response) is False

    def test_entity_not_found_is_invalid(self) -> None:
        """Entity not found message should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "Drug 'STAT3 inhibitor' not found in MyChem.info"
            )
            is False
        )

    def test_entity_not_found_chinese_is_invalid(self) -> None:
        """Chinese entity not found message should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "在 MyChem.info 中未找到药物 'STAT3 inhibitor'"
            )
            is False
        )

    def test_mcp_validation_error_is_invalid(self) -> None:
        """MCP tool validation error should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("call[search_entities] 的 3 个验证错误")
            is False
        )

    def test_pydantic_validation_error_is_invalid(self) -> None:
        """Pydantic validation error count should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "1 validation error for gene_getterArguments\n"
                "gene_id_or_symbol\n  Field required"
            )
            is False
        )

    def test_normal_text_with_not_found_is_valid(self) -> None:
        """Normal text containing 'not found' in context should be valid ✅.

        This verifies the regex is precise enough to avoid false positives
        on valid evidence that happens to contain 'not found'.
        """
        assert (
            EvidenceFilter.is_valid_evidence(
                "The study found that no significant difference was observed. "
                "Biomarker data was not found in earlier trials."
            )
            is True
        )

    def test_normal_json_with_domain_field_is_valid(self) -> None:
        """JSON with domain field but not ThinkTool should be valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                '{"domain": "clinical", "results": [{"trial": "NCT123"}]}'
            )
            is True
        )

    def test_target_not_found_is_invalid(self) -> None:
        """Target not found pattern should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Protein 'IL17A' not found in UniProt")
            is False
        )

    def test_sentence_ending_with_not_found_is_valid(self) -> None:
        """A regular sentence ending with 'not found' should be valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "Evidence for this mechanism was not found in published literature."
            )
            is True
        )


class TestStatusAndToolErrorPatterns:
    """Tests for status message and tool error filtering 🧪."""

    def test_checkmark_status_is_invalid(self) -> None:
        """Bohrium checkmark status message should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("✅ AI Search Session Created") is False

    def test_robot_status_is_invalid(self) -> None:
        """AI-generated summary status should be invalid ❌."""
        assert EvidenceFilter.is_valid_evidence("🤖 AI-Generated Summary") is False

    def test_unknown_tool_is_invalid(self) -> None:
        """Unknown tool error should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Unknown tool: 'get_target_safety'")
            is False
        )

    def test_search_count_magnifying_glass_is_invalid(self) -> None:
        """Search count summary (🔍) should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "🔍 Found 42 papers for: EGFR inhibitor resistance"
            )
            is False
        )

    def test_search_count_books_is_invalid(self) -> None:
        """Search count summary (📚) should be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "📚 Found 15 papers (sorted by RelevanceScore)."
            )
            is False
        )

    def test_search_count_zero_papers_is_invalid(self) -> None:
        """Zero-result search count should also be invalid ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("🔍 Found 0 papers for: nonexistent query")
            is False
        )

    def test_real_evidence_with_emoji_in_body_is_valid(self) -> None:
        """Real evidence that happens to contain emoji mid-text is valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "The trial showed significant improvement in overall survival."
            )
            is True
        )


class TestPrettyPrintedJsonFiltering:
    """Tests for pretty-printed JSON empty result filtering 🧹."""

    def test_pretty_printed_empty_results(self) -> None:
        """Pretty-printed empty results JSON should be filtered 🧹."""
        # This is the format biomcp actually returns
        pretty = '{\n  "results": []\n}'
        assert not EvidenceFilter.is_valid_evidence(pretty)

    def test_indented_empty_results(self) -> None:
        """Multi-indent empty results should be filtered 🧹."""
        indented = '{\n    "results":   [\n    ]\n}'
        assert not EvidenceFilter.is_valid_evidence(indented)

    def test_pretty_printed_empty_array(self) -> None:
        """Pretty-printed empty array should be filtered 🧹."""
        assert not EvidenceFilter.is_valid_evidence("[\n]")
        assert not EvidenceFilter.is_valid_evidence("[\n  \n]")

    def test_real_json_results_not_filtered(self) -> None:
        """JSON with actual results should NOT be filtered ✅."""
        real = '{"results": [{"title": "Study"}]}'
        assert EvidenceFilter.is_valid_evidence(real)


class TestBohriumAndAPIErrors:
    """Tests for Bohrium API error and generic API error filtering 🔧."""

    def test_bohrium_api_error_is_invalid(self) -> None:
        """Bohrium API error should be filtered ❌."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "Bohrium API error: Unknown error (code=-1)"
            )
            is False
        )

    def test_bohrium_api_error_case_insensitive(self) -> None:
        """Bohrium API error matching is case-insensitive ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("bohrium api error: connection failed")
            is False
        )
        assert EvidenceFilter.is_valid_evidence("BOHRIUM API ERROR: timeout") is False

    def test_generic_api_error_with_unknown(self) -> None:
        """Generic API error with 'Unknown' keyword should be filtered ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("API error: Unknown service response")
            is False
        )

    def test_generic_api_error_with_code(self) -> None:
        """Generic API error with 'code=' should be filtered ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("API error: request failed (code=-1)")
            is False
        )
        assert (
            EvidenceFilter.is_valid_evidence(
                "API error: internal server issue (code=500)"
            )
            is False
        )

    def test_valid_text_with_api_mention_not_filtered(self) -> None:
        """Valid evidence mentioning API should NOT be filtered ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "The study used API-based data extraction methods for analysis"
            )
            is True
        )


class TestMarkdownBoldCountSummaries:
    """Tests for markdown bold search count filtering 🔧."""

    def test_markdown_bold_count_is_invalid(self) -> None:
        """Markdown bold count summary should be filtered ❌."""
        assert EvidenceFilter.is_valid_evidence("Found **13** papers.") is False

    def test_markdown_bold_zero_count_is_invalid(self) -> None:
        """Markdown bold zero count should be filtered ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Found **0** papers matching query")
            is False
        )

    def test_markdown_bold_large_count_is_invalid(self) -> None:
        """Markdown bold large count should be filtered ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Found **142** papers for the search")
            is False
        )

    def test_markdown_bold_case_insensitive(self) -> None:
        """Markdown bold pattern matching is case-insensitive ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("found **25** papers in database") is False
        )
        assert EvidenceFilter.is_valid_evidence("FOUND **99** PAPERS TODAY") is False

    def test_markdown_bold_with_extra_spaces(self) -> None:
        """Markdown bold pattern handles spacing variations ❌."""
        assert (
            EvidenceFilter.is_valid_evidence("Found  **42**  papers matching criteria")
            is False
        )

    def test_valid_evidence_with_markdown_mid_text(self) -> None:
        """Valid evidence with markdown bold in body is NOT filtered ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "The study identified **13** candidate genes associated with disease."
            )
            is True
        )

    def test_valid_evidence_with_found_word(self) -> None:
        """Valid evidence starting with 'found' but not count pattern is valid ✅."""
        assert (
            EvidenceFilter.is_valid_evidence(
                "GPR75 variants associated with obesity protection were found in large cohort"
            )
            is True
        )
