"""Tests for EvidencePipeline deterministic cleaning 🧪.

Covers all three pipeline stages:
    - Content dedup (hash-based)
    - Noise filter (pattern matching)
    - Source tagging (URL classification)
"""

from __future__ import annotations

import pytest

from inquiro.core.evidence_pipeline import (
    MIN_EVIDENCE_LENGTH,
    CleaningStats,
    EvidencePipeline,
    EvidenceTag,
)
from inquiro.core.types import Evidence


# ============================================================================
# 🔧 Fixtures
# ============================================================================


def _make_evidence(
    id: str = "E1",
    summary: str = "A meaningful evidence summary with enough content to pass the filter",
    source: str = "mcp_perplexity",
    url: str = "https://example.com/paper",
    query: str = "test query",
) -> Evidence:
    """Create a test Evidence instance 🔧."""
    return Evidence(
        id=id,
        summary=summary,
        source=source,
        url=url,
        query=query,
    )


# ============================================================================
# 🧹 Full pipeline tests
# ============================================================================


class TestEvidencePipelineClean:
    """Test full pipeline clean() method 🧹."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty input produces empty output ✅."""
        pipeline = EvidencePipeline()
        cleaned, stats = pipeline.clean([])
        assert cleaned == []
        assert stats.input_count == 0
        assert stats.output_count == 0

    def test_single_clean_evidence_passes_through(self) -> None:
        """Single valid evidence item passes all stages ✅."""
        ev = _make_evidence()
        pipeline = EvidencePipeline()
        cleaned, stats = pipeline.clean([ev])

        assert len(cleaned) == 1
        assert cleaned[0].id == "E1"
        assert stats.input_count == 1
        assert stats.output_count == 1
        assert stats.dedup_removed == 0
        assert stats.noise_removed == 0

    def test_pipeline_removes_duplicates_and_noise(self) -> None:
        """Pipeline removes both duplicates and noise in one pass ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary="Unique evidence about treatment efficacy in NSCLC patients over time",
            ),
            _make_evidence(
                id="E2",
                summary="Unique evidence about treatment efficacy in NSCLC patients over time",
            ),  # Duplicate
            _make_evidence(
                id="E3",
                summary=(
                    "AI Search Session Created - session id 1234 "
                    "was initialized for the perplexity search"
                ),
            ),  # Noise
            _make_evidence(
                id="E4",
                summary="Another unique finding about mechanism of action in target proteins",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, stats = pipeline.clean(evidence)

        assert stats.input_count == 4
        assert stats.dedup_removed == 1
        assert stats.noise_removed == 1
        assert stats.output_count == 2
        assert len(cleaned) == 2


# ============================================================================
# 🔑 Dedup tests
# ============================================================================


class TestDedup:
    """Test content deduplication 🔑."""

    def test_exact_duplicate_removed(self) -> None:
        """Exact duplicate summary text is removed ✅."""
        ev1 = _make_evidence(
            id="E1",
            summary="Identical summary content here about protein binding affinity",
        )
        ev2 = _make_evidence(
            id="E2",
            summary="Identical summary content here about protein binding affinity",
        )

        result, removed = EvidencePipeline._dedup([ev1, ev2])
        assert len(result) == 1
        assert removed == 1
        assert result[0].id == "E1"  # First occurrence wins

    def test_case_insensitive_dedup(self) -> None:
        """Dedup is case-insensitive ✅."""
        ev1 = _make_evidence(
            id="E1",
            summary="Case Test Evidence Summary about peptide therapeutics study",
        )
        ev2 = _make_evidence(
            id="E2",
            summary="case test evidence summary about peptide therapeutics study",
        )

        result, removed = EvidencePipeline._dedup([ev1, ev2])
        assert len(result) == 1
        assert removed == 1

    def test_different_content_preserved(self) -> None:
        """Different evidence content is preserved ✅."""
        ev1 = _make_evidence(
            id="E1",
            summary="First unique evidence piece about STAT3 transcription factor",
        )
        ev2 = _make_evidence(
            id="E2",
            summary="Second unique evidence piece about peptide drug delivery",
        )

        result, removed = EvidencePipeline._dedup([ev1, ev2])
        assert len(result) == 2
        assert removed == 0

    def test_multiple_duplicates_only_first_kept(self) -> None:
        """Only first occurrence of triplicates is kept ✅."""
        evidence = [
            _make_evidence(
                id=f"E{i}",
                summary="Same text repeated about clinical trial results and outcomes",
            )
            for i in range(5)
        ]

        result, removed = EvidencePipeline._dedup(evidence)
        assert len(result) == 1
        assert removed == 4


# ============================================================================
# 🚫 Noise filter tests
# ============================================================================


class TestNoiseFilter:
    """Test noise filtering 🚫."""

    def test_known_noise_pattern_removed(self) -> None:
        """Known noise pattern is filtered out ✅."""
        ev = _make_evidence(
            id="E1",
            summary="AI Search Session Created - This is not real evidence",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_no_papers_found_removed(self) -> None:
        """'No papers found' noise pattern is filtered ✅."""
        ev = _make_evidence(
            id="E1",
            summary="No papers found matching the query criteria in database",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_short_content_removed(self) -> None:
        """Evidence below minimum length is filtered ✅."""
        ev = _make_evidence(id="E1", summary="Too short")
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_valid_content_preserved(self) -> None:
        """Valid evidence content is preserved ✅."""
        ev = _make_evidence(
            id="E1",
            summary=(
                "STAT3 is a transcription factor involved in cell "
                "proliferation, survival, and immune regulation"
            ),
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 1
        assert removed == 0

    def test_noise_pattern_case_insensitive(self) -> None:
        """Noise pattern matching is case-insensitive ✅."""
        ev = _make_evidence(
            id="E1",
            summary="follow-up question submitted to the search engine for more results",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_exact_threshold_length(self) -> None:
        """Evidence at exact minimum length threshold passes ✅."""
        ev = _make_evidence(
            id="E1",
            summary="x" * MIN_EVIDENCE_LENGTH,  # Exactly at threshold
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 1
        assert removed == 0

    def test_bohrium_api_error_removed(self) -> None:
        """Bohrium API error is filtered as noise ✅."""
        ev = _make_evidence(
            id="E1",
            summary="Bohrium API error: Unknown error (code=-1) occurred during processing",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_generic_api_error_removed(self) -> None:
        """Generic API error patterns are filtered as noise ✅."""
        ev = _make_evidence(
            id="E1",
            summary="API error: request timeout occurred (code=408) please retry later",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_markdown_bold_count_removed(self) -> None:
        """Markdown bold count summaries are filtered as noise ✅."""
        ev = _make_evidence(
            id="E1",
            summary="Found **13** papers matching the search criteria in the database",
        )
        result, removed = EvidencePipeline()._filter_noise([ev])
        assert len(result) == 0
        assert removed == 1

    def test_search_count_emoji_removed(self) -> None:
        """Search count with emoji prefix is filtered as noise ✅."""
        ev1 = _make_evidence(
            id="E1",
            summary="🔍 Found 42 papers for: EGFR inhibitor resistance mechanisms",
        )
        ev2 = _make_evidence(
            id="E2",
            summary="📚 Found 15 papers (sorted by RelevanceScore) from the query.",
        )
        result, removed = EvidencePipeline()._filter_noise([ev1, ev2])
        assert len(result) == 0
        assert removed == 2


# ============================================================================
# 🏷️ Source tagging tests
# ============================================================================


class TestSourceTagging:
    """Test URL-based source classification 🏷️."""

    @pytest.mark.parametrize(
        "url,expected_tag",
        [
            # 🎓 Academic sources
            (
                "https://pubmed.ncbi.nlm.nih.gov/12345",
                EvidenceTag.ACADEMIC,
            ),
            (
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC123",
                EvidenceTag.ACADEMIC,
            ),
            ("https://doi.org/10.1038/nrd4580", EvidenceTag.ACADEMIC),
            (
                "https://www.frontiersin.org/articles/10.3389",
                EvidenceTag.ACADEMIC,
            ),
            ("https://www.nature.com/articles/s41586", EvidenceTag.ACADEMIC),
            (
                "https://www.sciencedirect.com/science/article",
                EvidenceTag.ACADEMIC,
            ),
            # 📄 Patent sources
            (
                "https://patents.google.com/patent/US123",
                EvidenceTag.PATENT,
            ),
            ("https://www.uspto.gov/patents", EvidenceTag.PATENT),
            # 🏥 Clinical trial sources
            (
                "https://clinicaltrials.gov/ct2/show/NCT123",
                EvidenceTag.CLINICAL_TRIAL,
            ),
            (
                "https://www.centerwatch.com/clinical-trials",
                EvidenceTag.CLINICAL_TRIAL,
            ),
            # 🏛️ Regulatory sources
            ("https://www.fda.gov/drugs/new-drugs", EvidenceTag.REGULATORY),
            (
                "https://www.ema.europa.eu/en/medicines",
                EvidenceTag.REGULATORY,
            ),
            # 🌐 Other sources
            ("https://www.example.com", EvidenceTag.OTHER),
            ("https://en.wikipedia.org/wiki/STAT3", EvidenceTag.OTHER),
            ("", EvidenceTag.OTHER),
        ],
    )
    def test_url_classification(self, url: str, expected_tag: EvidenceTag) -> None:
        """URL is classified to the correct tag ✅."""
        tag = EvidencePipeline.classify_url(url)
        assert tag == expected_tag

    def test_tag_returned_in_mapping(self) -> None:
        """Tag is returned in the evidence ID -> EvidenceTag mapping ✅."""
        ev = _make_evidence(url="https://pubmed.ncbi.nlm.nih.gov/12345")
        tag_map = EvidencePipeline._tag_sources([ev])
        assert len(tag_map) == 1
        assert tag_map[ev.id] == EvidenceTag.ACADEMIC

    def test_tag_distribution_in_stats(self) -> None:
        """Tag distribution is reported in CleaningStats ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "Academic paper about protein structure and binding "
                    "affinity measurements in crystal structures"
                ),
                url="https://pubmed.ncbi.nlm.nih.gov/123",
            ),
            _make_evidence(
                id="E2",
                summary=(
                    "Patent filing for novel peptide-based therapeutics "
                    "targeting signal transduction pathways"
                ),
                url="https://patents.google.com/patent/US789",
            ),
            _make_evidence(
                id="E3",
                summary=(
                    "Clinical trial results for new STAT3 inhibitor "
                    "design with phase 2 efficacy endpoints"
                ),
                url="https://clinicaltrials.gov/ct2/show/NCT456",
            ),
        ]
        pipeline = EvidencePipeline()
        _, stats = pipeline.clean(evidence)

        assert stats.tag_distribution.get("academic") == 1
        assert stats.tag_distribution.get("patent") == 1
        assert stats.tag_distribution.get("clinical_trial") == 1


# ============================================================================
# 📊 Statistics tests
# ============================================================================


class TestCleaningStats:
    """Test CleaningStats model 📊."""

    def test_stats_model_serialization(self) -> None:
        """CleaningStats serializes correctly ✅."""
        stats = CleaningStats(
            input_count=10,
            output_count=7,
            dedup_removed=2,
            noise_removed=1,
            tag_distribution={"academic": 3, "other": 4},
        )
        dumped = stats.model_dump()
        assert dumped["input_count"] == 10
        assert dumped["output_count"] == 7
        assert dumped["tag_distribution"]["academic"] == 3


# ============================================================================
# 🏷️ Tag backfill and metadata extraction tests
# ============================================================================


class TestTagBackfillAndMetadata:
    """Test tag backfill to Evidence objects and metadata extraction 🏷️."""

    def test_tag_backfill_to_evidence(self) -> None:
        """Tags are backfilled onto Evidence.evidence_tag ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "Academic study on protein binding affinity "
                    "measurement using surface plasmon resonance"
                ),
                url="https://pubmed.ncbi.nlm.nih.gov/12345678",
            ),
            _make_evidence(
                id="E2",
                summary=(
                    "Web article about general research findings "
                    "on pharmacological interventions for treatment"
                ),
                url="https://example.com/article",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean(evidence)

        tags = {ev.id: ev.evidence_tag for ev in cleaned}
        assert tags["E1"] == "academic"
        assert tags["E2"] == "other"

    def test_all_tag_values_backfilled(self) -> None:
        """All 5 tag categories are correctly backfilled ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "Published research findings from controlled "
                    "laboratory experiments with detailed methodology"
                ),
                url="https://doi.org/10.1234/test",
            ),
            _make_evidence(
                id="E2",
                summary=(
                    "Patent filing for novel compound structure "
                    "with improved pharmacokinetic properties"
                ),
                url="https://patents.google.com/patent/US12345",
            ),
            _make_evidence(
                id="E3",
                summary=(
                    "Phase 3 randomized controlled trial results "
                    "showing superior efficacy outcomes in patients"
                ),
                url="https://clinicaltrials.gov/ct2/show/NCT99999999",
            ),
            _make_evidence(
                id="E4",
                summary=(
                    "Regulatory agency approval documentation and "
                    "labeling requirements for pharmaceutical product"
                ),
                url="https://www.fda.gov/drugs/approvals/test",
            ),
            _make_evidence(
                id="E5",
                summary=(
                    "General web content about health research "
                    "findings without specific academic attribution"
                ),
                url="https://example.com/health",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean(evidence)

        tag_map = {ev.id: ev.evidence_tag for ev in cleaned}
        assert tag_map["E1"] == "academic"
        assert tag_map["E2"] == "patent"
        assert tag_map["E3"] == "clinical_trial"
        assert tag_map["E4"] == "regulatory"
        assert tag_map["E5"] == "other"

    def test_clinical_trial_extraction_during_clean(self) -> None:
        """NCT ID extracted from URL during cleaning ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "Trial results for novel treatment approach "
                    "showing significant efficacy improvement"
                ),
                url="https://clinicaltrials.gov/ct2/show/NCT12345678",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean(evidence)
        assert cleaned[0].clinical_trial_id == "NCT12345678"

    def test_doi_extraction_during_clean(self) -> None:
        """DOI extracted from URL during cleaning ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "Published research on molecular mechanisms "
                    "underlying therapeutic target engagement"
                ),
                url="https://doi.org/10.1038/nature12345",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean(evidence)
        assert cleaned[0].doi == "10.1038/nature12345"

    def test_no_metadata_when_absent(self) -> None:
        """No metadata fields set when patterns not found ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=(
                    "General research findings about protein "
                    "structure and functional characterisation"
                ),
                url="https://example.com/article",
            ),
        ]
        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean(evidence)
        assert cleaned[0].clinical_trial_id is None
        assert cleaned[0].doi is None

    def test_preexisting_metadata_not_overwritten(self) -> None:
        """Pre-set metadata fields are preserved ✅."""
        ev = _make_evidence(
            id="E1",
            summary=(
                "Study NCT99999999 showed significant results "
                "in treating the target disease indication"
            ),
            url="https://example.com",
        )
        ev.clinical_trial_id = "NCT00000000"
        ev.doi = "10.9999/preset"

        pipeline = EvidencePipeline()
        cleaned, _ = pipeline.clean([ev])
        assert cleaned[0].clinical_trial_id == "NCT00000000"
        assert cleaned[0].doi == "10.9999/preset"
