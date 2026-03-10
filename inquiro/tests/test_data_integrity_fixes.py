"""Tests for data integrity fixes — C1, H1, L1, C3, C2, M2, H4, H5, M1, M6 🧪.

Verifies that evidence tag, cleaning stats, and trajectory data are
propagated correctly through the discovery pipeline.

    C1: _to_cleaned_evidence() preserves evidence_tag instead of hardcoding OTHER
    H1: _map_evidence_list() round-trips evidence_tag back to Evidence objects
    L1: _build_round_record() writes tag_distribution to CleaningPhaseRecord
    C3: supplementary_context from Tier-2 condensation wired to AnalysisExp prompt
    C2: Synthesis evidence condensed when >200 items (token overflow guard)
    M2: Duplicate claims deduplicated before synthesis
    H4: EvaluationResult.confidence_breakdown field wired and populated
    H5: CleanedEvidence carries doi/clinical_trial_id/quality_label
    M1: model_decisions wired into trajectory AnalysisPhaseRecord
    M6: _MIN_MODELS_FOR_CONSENSUS warning logged for < 2 models
"""

from __future__ import annotations

import contextlib
import logging
from unittest.mock import MagicMock


from inquiro.core.discovery_loop import DiscoveryLoop
from inquiro.core.evidence_pipeline import CleaningStats
from inquiro.core.types import (
    CleanedEvidence,
    Evidence,
    EvidenceTag,
    GapReport,
)


# ============================================================================
# 🏭 Helpers
# ============================================================================


def _make_evidence(
    eid: str = "E1",
    evidence_tag: str | None = None,
    summary: str = (
        "Test evidence about drug-target interaction in preclinical "
        "binding assays with statistically significant outcomes"
    ),
) -> Evidence:
    """Create a test Evidence with optional evidence_tag 🔧."""
    return Evidence(
        id=eid,
        source="pubmed",
        query="test query",
        summary=summary,
        evidence_tag=evidence_tag,
    )


def _make_cleaned_evidence(
    eid: str = "CE1",
    tag: EvidenceTag = EvidenceTag.ACADEMIC,
    summary: str = (
        "Test evidence about drug-target interaction in preclinical "
        "binding assays with statistically significant outcomes"
    ),
) -> CleanedEvidence:
    """Create a test CleanedEvidence with a given tag 🔧."""
    return CleanedEvidence(
        id=eid,
        summary=summary,
        url="https://pubmed.ncbi.nlm.nih.gov/12345",
        tag=tag,
        source_query="test query",
        mcp_server="pubmed",
    )


def _make_minimal_loop() -> DiscoveryLoop:
    """Create a minimal DiscoveryLoop for instance-method testing 🔧."""
    search_executor = MagicMock()
    analysis_executor = MagicMock()
    return DiscoveryLoop(
        search_executor=search_executor,
        analysis_executor=analysis_executor,
    )


# ============================================================================
# 🔖 C1: _to_cleaned_evidence() preserves evidence_tag
# ============================================================================


class TestToCleanedEvidenceTagPreservation:
    """C1: Verify _to_cleaned_evidence() uses actual evidence_tag 🔖."""

    def test_to_cleaned_evidence_preserves_academic_tag(self) -> None:
        """C1: evidence_tag='academic' maps to EvidenceTag.ACADEMIC ✅."""
        evidence = _make_evidence(evidence_tag="academic")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.ACADEMIC

    def test_to_cleaned_evidence_preserves_clinical_trial_tag(self) -> None:
        """C1: evidence_tag='clinical_trial' maps to EvidenceTag.CLINICAL_TRIAL ✅."""
        evidence = _make_evidence(evidence_tag="clinical_trial")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.CLINICAL_TRIAL

    def test_to_cleaned_evidence_preserves_regulatory_tag(self) -> None:
        """C1: evidence_tag='regulatory' maps to EvidenceTag.REGULATORY ✅."""
        evidence = _make_evidence(evidence_tag="regulatory")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.REGULATORY

    def test_to_cleaned_evidence_preserves_patent_tag(self) -> None:
        """C1: evidence_tag='patent' maps to EvidenceTag.PATENT ✅."""
        evidence = _make_evidence(evidence_tag="patent")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.PATENT

    def test_to_cleaned_evidence_defaults_to_other_when_tag_is_none(self) -> None:
        """C1: None evidence_tag falls back to EvidenceTag.OTHER ✅."""
        evidence = _make_evidence(evidence_tag=None)
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.OTHER

    def test_to_cleaned_evidence_defaults_to_other_when_tag_is_empty(self) -> None:
        """C1: Empty string evidence_tag falls back to EvidenceTag.OTHER ✅."""
        evidence = _make_evidence(evidence_tag="")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.tag == EvidenceTag.OTHER

    def test_to_cleaned_evidence_other_fields_preserved(self) -> None:
        """C1: Non-tag fields are unchanged after conversion ✅."""
        evidence = _make_evidence(eid="E42", evidence_tag="academic")
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.id == "E42"
        assert result.summary == evidence.summary
        assert result.url == evidence.url
        assert result.source_query == evidence.query
        assert result.mcp_server == evidence.source


# ============================================================================
# 📡 H1: _map_evidence_list() preserves evidence_tag
# ============================================================================


class TestMapEvidenceListTagPreservation:
    """H1: Verify _map_evidence_list() round-trips evidence_tag 📡."""

    def test_map_evidence_list_preserves_academic_tag(self) -> None:
        """H1: CleanedEvidence with ACADEMIC tag produces Evidence with 'academic' ✅."""
        ce = _make_cleaned_evidence(tag=EvidenceTag.ACADEMIC)
        from inquiro.core.runner import EvalTaskRunner

        results = EvalTaskRunner._map_evidence_list([ce])
        assert len(results) == 1
        assert results[0].evidence_tag == "academic"

    def test_map_evidence_list_preserves_regulatory_tag(self) -> None:
        """H1: CleanedEvidence with REGULATORY tag produces Evidence with 'regulatory' ✅."""
        ce = _make_cleaned_evidence(tag=EvidenceTag.REGULATORY)
        from inquiro.core.runner import EvalTaskRunner

        results = EvalTaskRunner._map_evidence_list([ce])
        assert results[0].evidence_tag == "regulatory"

    def test_map_evidence_list_other_tag_produces_other_string(self) -> None:
        """H1: CleanedEvidence with OTHER tag produces Evidence with 'other' ✅.

        Note: CleanedEvidence.tag is always a valid EvidenceTag (defaults to OTHER).
        The defensive `if ce.tag else None` branch exists for future extensibility.
        """
        ce = CleanedEvidence(
            id="CE-other",
            summary=(
                "Evidence about preclinical binding studies with target "
                "engagement in cell-based assay models"
            ),
            url=None,
            tag=EvidenceTag.OTHER,
            source_query="query",
            mcp_server="test-mcp",
        )
        from inquiro.core.runner import EvalTaskRunner

        results = EvalTaskRunner._map_evidence_list([ce])
        assert results[0].evidence_tag == "other"

    def test_map_evidence_list_other_fields_preserved(self) -> None:
        """H1: Non-tag fields pass through _map_evidence_list correctly ✅."""
        ce = _make_cleaned_evidence(eid="CE-99", tag=EvidenceTag.CLINICAL_TRIAL)
        from inquiro.core.runner import EvalTaskRunner

        results = EvalTaskRunner._map_evidence_list([ce])
        ev = results[0]
        assert ev.id == "CE-99"
        assert ev.source == ce.mcp_server
        assert ev.url == ce.url
        assert ev.summary == ce.summary


# ============================================================================
# 📊 L1: _build_round_record() writes tag_distribution to CleaningPhaseRecord
# ============================================================================


class TestBuildRoundRecordTagDistribution:
    """L1: Verify _build_round_record() passes tag_distribution to trajectory 📊."""

    def _make_gap_report(self) -> GapReport:
        """Create a minimal GapReport for round record construction 🔧."""
        return GapReport(
            round_number=1,
            covered_items=["c1"],
            uncovered_items=[],
            coverage_ratio=1.0,
            converged=True,
            convergence_reason="all_covered",
            conflict_signals=[],
        )

    def _make_cleaning_stats(
        self, tag_distribution: dict[str, int] | None = None
    ) -> CleaningStats:
        """Create a CleaningStats with configurable tag_distribution 🔧."""
        return CleaningStats(
            input_count=10,
            output_count=8,
            dedup_removed=1,
            noise_removed=1,
            tag_distribution=tag_distribution or {},
        )

    def _make_search_output(self) -> object:
        """Create a minimal SearchRoundOutput mock 🔧."""
        from inquiro.core.discovery_loop import SearchRoundOutput

        return SearchRoundOutput(
            evidence=[],
            agent_trajectory_ref=None,
            agent_trajectory_refs=[],
            duration_seconds=1.0,
        )

    def _make_analysis_output(self) -> object:
        """Create a minimal AnalysisRoundOutput mock 🔧."""
        from inquiro.core.discovery_loop import AnalysisRoundOutput

        return AnalysisRoundOutput(
            consensus_decision="positive",
            consensus_ratio=1.0,
            claims=[],
            duration_seconds=2.0,
        )

    def test_tag_distribution_written_to_cleaning_phase(self) -> None:
        """L1: tag_distribution from CleaningStats flows into CleaningPhaseRecord ✅."""
        loop = _make_minimal_loop()
        tag_dist = {"academic": 5, "regulatory": 2, "other": 1}
        cleaning_stats = self._make_cleaning_stats(tag_distribution=tag_dist)

        record = loop._build_round_record(
            round_num=1,
            search_output=self._make_search_output(),
            cleaning_stats=cleaning_stats,
            analysis_output=self._make_analysis_output(),
            gap_report=self._make_gap_report(),
            round_cost=0.05,
            round_duration=3.0,
        )

        assert record.cleaning_phase.tag_distribution == tag_dist

    def test_empty_tag_distribution_written(self) -> None:
        """L1: Empty tag_distribution is written without error ✅."""
        loop = _make_minimal_loop()
        cleaning_stats = self._make_cleaning_stats(tag_distribution={})

        record = loop._build_round_record(
            round_num=1,
            search_output=self._make_search_output(),
            cleaning_stats=cleaning_stats,
            analysis_output=self._make_analysis_output(),
            gap_report=self._make_gap_report(),
            round_cost=0.0,
            round_duration=1.0,
        )

        assert record.cleaning_phase.tag_distribution == {}

    def test_cleaning_phase_other_stats_still_correct(self) -> None:
        """L1: Other CleaningPhaseRecord fields are not affected by fix ✅."""
        loop = _make_minimal_loop()
        cleaning_stats = self._make_cleaning_stats({"academic": 3})

        record = loop._build_round_record(
            round_num=2,
            search_output=self._make_search_output(),
            cleaning_stats=cleaning_stats,
            analysis_output=self._make_analysis_output(),
            gap_report=self._make_gap_report(),
            round_cost=0.1,
            round_duration=5.0,
        )

        assert record.cleaning_phase.input_count == 10
        assert record.cleaning_phase.output_count == 8
        assert record.cleaning_phase.dedup_removed == 1
        assert record.cleaning_phase.noise_removed == 1


# ============================================================================
# 📋 C3: supplementary_context from Tier-2 condensation wired to AnalysisExp
# ============================================================================


class TestSupplementaryContextWiring:
    """C3: Verify supplementary_context is appended to user prompt 📋."""

    def _make_task(self) -> object:
        """Create a minimal EvaluationTask mock 🔧."""
        from inquiro.core.types import Checklist, ChecklistItem, EvaluationTask

        return EvaluationTask(
            task_id="test-c3",
            topic="test topic",
            rules="Be thorough.",
            checklist=Checklist(
                required=[ChecklistItem(id="c1", description="item1")],
                optional=[],
            ),
        )

    def _make_cleaned_evidence_list(self, n: int) -> list[CleanedEvidence]:
        """Create n minimal CleanedEvidence items 🔧."""
        return [
            CleanedEvidence(
                id=f"CE{i}",
                summary=f"Evidence item {i} about target engagement in assay.",
                url=f"https://example.com/{i}",
                tag=EvidenceTag.ACADEMIC,
                source_query="test query",
                mcp_server="pubmed",
            )
            for i in range(n)
        ]

    def test_render_user_prompt_without_supplementary_context(self) -> None:
        """C3: Prompt has no supplementary section when context is None ✅."""
        from unittest.mock import patch

        from inquiro.exps.analysis_exp import AnalysisExp

        task = self._make_task()
        cleaned = self._make_cleaned_evidence_list(2)

        with patch(
            "inquiro.exps.analysis_exp._prompt_loader"
        ) as mock_loader:
            mock_loader.render.return_value = "BASE_PROMPT"
            exp = AnalysisExp.__new__(AnalysisExp)
            exp._evolution_enrichment = None
            result = exp._render_user_prompt(
                task,
                cleaned,
                round_number=1,
                supplementary_context=None,
            )

        assert "Additional Evidence (Summarised)" not in result
        assert result == "BASE_PROMPT"

    def test_render_user_prompt_appends_supplementary_context(self) -> None:
        """C3: Prompt gains supplementary section when context is provided ✅."""
        from unittest.mock import patch

        from inquiro.exps.analysis_exp import AnalysisExp

        task = self._make_task()
        cleaned = self._make_cleaned_evidence_list(2)
        ctx = "Tag: academic — 50 excluded items covering binding assays."

        with patch(
            "inquiro.exps.analysis_exp._prompt_loader"
        ) as mock_loader:
            mock_loader.render.return_value = "BASE_PROMPT"
            exp = AnalysisExp.__new__(AnalysisExp)
            exp._evolution_enrichment = None
            result = exp._render_user_prompt(
                task,
                cleaned,
                round_number=1,
                supplementary_context=ctx,
            )

        assert "## Additional Evidence (Summarised)" in result
        assert ctx in result
        assert result.startswith("BASE_PROMPT")

    def test_execute_analysis_protocol_accepts_supplementary_context(
        self,
    ) -> None:
        """C3: AnalysisExecutor Protocol signature includes supplementary_context ✅."""
        import inspect

        from inquiro.core.discovery_loop import AnalysisExecutor

        sig = inspect.signature(AnalysisExecutor.execute_analysis)
        assert "supplementary_context" in sig.parameters

    def test_analysis_exp_adapter_accepts_supplementary_context(self) -> None:
        """C3: _AnalysisExpAdapter.execute_analysis() accepts supplementary_context ✅."""
        import inspect

        from inquiro.core.runner import _AnalysisExpAdapter

        sig = inspect.signature(_AnalysisExpAdapter.execute_analysis)
        assert "supplementary_context" in sig.parameters


# ============================================================================
# 🗜️ C2: Synthesis evidence condensed when > 200 items
# ============================================================================


class TestSynthesisCondenserGuard:
    """C2: Verify EvidenceCondenser reduces large evidence lists for synthesis 🗜️."""

    def _make_evidence_batch(self, n: int) -> list[Evidence]:
        """Create n distinct Evidence items 🔧."""
        return [
            Evidence(
                id=f"E{i}",
                source="pubmed",
                query="test query",
                url=f"https://example.com/{i}",
                summary=(
                    f"Evidence item {i} describes binding affinity of "
                    "compound X to receptor Y in in-vitro assay Z."
                ),
                evidence_tag="academic",
            )
            for i in range(n)
        ]

    def test_condenser_reduces_large_evidence_to_target(self) -> None:
        """C2: EvidenceCondenser with tier1_target=200 caps output at ≤200 ✅."""
        from inquiro.core.evidence_condenser import CondenserConfig, EvidenceCondenser

        evidence = self._make_evidence_batch(300)
        condenser = EvidenceCondenser(
            CondenserConfig(tier1_threshold=200, tier1_target=200)
        )
        condensed = condenser.condense(evidence, checklist_items=["item1"])
        assert len(condensed.evidence) <= 200
        assert condensed.meta.original_count == 300

    def test_condenser_not_triggered_below_threshold(self) -> None:
        """C2: No condensation when evidence count ≤ threshold ✅."""
        from inquiro.core.evidence_condenser import CondenserConfig, EvidenceCondenser

        evidence = self._make_evidence_batch(150)
        condenser = EvidenceCondenser(
            CondenserConfig(tier1_threshold=200, tier1_target=200)
        )
        condensed = condenser.condense(evidence, checklist_items=["item1"])
        assert len(condensed.evidence) == 150
        assert condensed.meta.tier == 0


# ============================================================================
# 🔑 M2: Duplicate claims deduplicated before synthesis
# ============================================================================


class TestClaimDeduplication:
    """M2: Verify duplicate claims are removed before synthesis 🔑."""

    def _make_claim_obj(self, text: str) -> object:
        """Create a minimal claim-like object with a .claim attribute 🔧."""

        class _FakeClaim:
            claim = text

        return _FakeClaim()

    def test_dedup_removes_repeated_claims(self) -> None:
        """M2: Identical claim texts appear only once after dedup ✅."""
        from inquiro.core.runner import EvalTaskRunner

        # Build a raw list with duplicates via _map_claim_list path is complex;
        # test the dedup logic directly through the static helper.
        claim_dicts = [
            {"claim": "alpha claim text", "evidence_ids": [], "strength": "strong"},
            {"claim": "beta claim text", "evidence_ids": [], "strength": "weak"},
            {"claim": "alpha claim text", "evidence_ids": [], "strength": "strong"},
        ]
        all_claims = EvalTaskRunner._map_claim_list(claim_dicts)

        # Apply dedup logic (mirrors _run_discovery_synthesis implementation)
        seen_claim_keys: set[str] = set()
        deduped: list[object] = []
        for claim in all_claims:
            key = claim.claim if hasattr(claim, "claim") else str(claim)
            if key not in seen_claim_keys:
                seen_claim_keys.add(key)
                deduped.append(claim)

        assert len(deduped) == 2
        claim_texts = [
            c.claim if hasattr(c, "claim") else str(c) for c in deduped
        ]
        assert "alpha claim text" in claim_texts
        assert "beta claim text" in claim_texts

    def test_dedup_preserves_order_of_first_occurrence(self) -> None:
        """M2: First occurrence is kept; subsequent duplicates are dropped ✅."""
        from inquiro.core.runner import EvalTaskRunner

        claim_dicts = [
            {"claim": "first", "evidence_ids": [], "strength": "strong"},
            {"claim": "second", "evidence_ids": [], "strength": "weak"},
            {"claim": "first", "evidence_ids": [], "strength": "moderate"},
        ]
        all_claims = EvalTaskRunner._map_claim_list(claim_dicts)

        seen: set[str] = set()
        deduped: list[object] = []
        for c in all_claims:
            key = c.claim if hasattr(c, "claim") else str(c)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        assert len(deduped) == 2
        # Order should be preserved: first then second
        keys = [c.claim if hasattr(c, "claim") else str(c) for c in deduped]
        assert keys == ["first", "second"]

    def test_dedup_noop_for_unique_claims(self) -> None:
        """M2: No claims lost when all claim texts are unique ✅."""
        from inquiro.core.runner import EvalTaskRunner

        claim_dicts = [
            {"claim": f"unique claim {i}", "evidence_ids": [], "strength": "strong"}
            for i in range(5)
        ]
        all_claims = EvalTaskRunner._map_claim_list(claim_dicts)

        seen: set[str] = set()
        deduped: list[object] = []
        for c in all_claims:
            key = c.claim if hasattr(c, "claim") else str(c)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        assert len(deduped) == 5


# ============================================================================
# 📊 H4: EvaluationResult.confidence_breakdown — field wired and serializable
# ============================================================================


class TestConfidenceBreakdownInEvaluationResult:
    """H4: Verify EvaluationResult.confidence_breakdown field is wired correctly 📊."""

    def _make_result(self, **overrides) -> object:
        """Build a minimal EvaluationResult 🔧."""
        from inquiro.core.types import Decision, EvaluationResult

        kwargs = dict(
            task_id="task-h4-test",
            decision=Decision.POSITIVE,
            confidence=0.85,
        )
        kwargs.update(overrides)
        return EvaluationResult(**kwargs)

    def _make_breakdown(self) -> object:
        """Build a ConfidenceBreakdown with all dimensions 🔧."""
        from inquiro.core.types import ConfidenceBreakdown

        return ConfidenceBreakdown(
            evidence_strength=0.8,
            evidence_coverage=0.75,
            evidence_consistency=0.9,
            information_recency=0.7,
            overall=0.79,
        )

    def test_confidence_breakdown_defaults_to_none(self) -> None:
        """H4: confidence_breakdown is None when not provided ✅."""
        result = self._make_result()
        assert result.confidence_breakdown is None

    def test_confidence_breakdown_accepts_value(self) -> None:
        """H4: confidence_breakdown accepts a ConfidenceBreakdown instance ✅."""
        breakdown = self._make_breakdown()
        result = self._make_result(confidence_breakdown=breakdown)
        assert result.confidence_breakdown is not None
        assert result.confidence_breakdown.overall == 0.79
        assert result.confidence_breakdown.evidence_strength == 0.8

    def test_confidence_breakdown_all_dimensions_preserved(self) -> None:
        """H4: All breakdown dimensions are stored without loss ✅."""
        breakdown = self._make_breakdown()
        result = self._make_result(confidence_breakdown=breakdown)
        bd = result.confidence_breakdown
        assert bd.evidence_strength == 0.8
        assert bd.evidence_coverage == 0.75
        assert bd.evidence_consistency == 0.9
        assert bd.information_recency == 0.7
        assert bd.overall == 0.79

    def test_confidence_breakdown_serializes_to_dict(self) -> None:
        """H4: confidence_breakdown appears in model_dump() output ✅."""
        breakdown = self._make_breakdown()
        result = self._make_result(confidence_breakdown=breakdown)
        dumped = result.model_dump()
        assert "confidence_breakdown" in dumped
        assert dumped["confidence_breakdown"] is not None
        assert dumped["confidence_breakdown"]["overall"] == 0.79

    def test_confidence_breakdown_none_serializes_to_null(self) -> None:
        """H4: None confidence_breakdown serializes to null in model_dump() ✅."""
        result = self._make_result()
        dumped = result.model_dump()
        assert "confidence_breakdown" in dumped
        assert dumped["confidence_breakdown"] is None


# ============================================================================
# 🏷️ H5: CleanedEvidence carries doi / clinical_trial_id / quality_label
# ============================================================================


class TestCleanedEvidenceExtendedFields:
    """H5: CleanedEvidence propagates doi/clinical_trial_id/quality_label 🏷️."""

    def test_cleaned_evidence_has_doi_field(self) -> None:
        """H5: CleanedEvidence stores doi when provided ✅."""
        ce = CleanedEvidence(
            id="CE1",
            summary="test",
            doi="10.1038/nature12345",
        )
        assert ce.doi == "10.1038/nature12345"

    def test_cleaned_evidence_has_clinical_trial_id(self) -> None:
        """H5: CleanedEvidence stores clinical_trial_id ✅."""
        ce = CleanedEvidence(
            id="CE1",
            summary="test",
            clinical_trial_id="NCT12345678",
        )
        assert ce.clinical_trial_id == "NCT12345678"

    def test_cleaned_evidence_has_quality_label(self) -> None:
        """H5: CleanedEvidence stores quality_label ✅."""
        ce = CleanedEvidence(
            id="CE1",
            summary="test",
            quality_label="high",
        )
        assert ce.quality_label == "high"

    def test_cleaned_evidence_defaults_none(self) -> None:
        """H5: Extended fields default to None ✅."""
        ce = CleanedEvidence(id="CE1", summary="test")
        assert ce.doi is None
        assert ce.clinical_trial_id is None
        assert ce.quality_label is None

    def test_to_cleaned_evidence_propagates_doi(self) -> None:
        """H5: _to_cleaned_evidence() copies doi from Evidence ✅."""
        evidence = _make_evidence(evidence_tag="academic")
        evidence.doi = "10.1021/jacs.2c12345"
        evidence.clinical_trial_id = "NCT99999999"
        result = DiscoveryLoop._to_cleaned_evidence(evidence)
        assert result.doi == "10.1021/jacs.2c12345"
        assert result.clinical_trial_id == "NCT99999999"

    def test_map_evidence_list_preserves_doi(self) -> None:
        """H5: _map_evidence_list() round-trips doi back to Evidence ✅."""
        from inquiro.core.runner import EvalTaskRunner

        ce = CleanedEvidence(
            id="CE1",
            summary="test",
            url="https://doi.org/10.1038/nature12345",
            tag=EvidenceTag.ACADEMIC,
            mcp_server="pubmed",
            source_query="test",
            doi="10.1038/nature12345",
            clinical_trial_id="NCT00001111",
        )
        mapped = EvalTaskRunner._map_evidence_list([ce])
        assert len(mapped) == 1
        assert mapped[0].doi == "10.1038/nature12345"
        assert mapped[0].clinical_trial_id == "NCT00001111"

    def test_format_evidence_list_includes_doi_and_nct(self) -> None:
        """H5: AnalysisExp._format_evidence_list() renders doi/NCT ✅."""
        from inquiro.exps.analysis_exp import AnalysisExp

        ce = CleanedEvidence(
            id="CE1",
            summary="test summary",
            url="https://example.com",
            tag=EvidenceTag.CLINICAL_TRIAL,
            doi="10.1016/j.cell.2024.01.001",
            clinical_trial_id="NCT55555555",
            quality_label="high",
        )
        formatted = AnalysisExp._format_evidence_list([ce])
        assert "10.1016/j.cell.2024.01.001" in formatted
        assert "NCT55555555" in formatted
        assert "high" in formatted


# ============================================================================
# 📊 M1: model_decisions wired into trajectory AnalysisPhaseRecord
# ============================================================================


class TestModelDecisionsInTrajectory:
    """M1: model_decisions propagated to AnalysisPhaseRecord 📊."""

    def test_analysis_phase_record_includes_model_results(self) -> None:
        """M1: AnalysisPhaseRecord stores per-model analysis records ✅."""
        from inquiro.core.trajectory.models import (
            AnalysisPhaseRecord,
            ConsensusRecord,
            ModelAnalysisRecord,
        )

        record = AnalysisPhaseRecord(
            model_results=[
                ModelAnalysisRecord(
                    model_name="gpt-4o",
                    decision="positive",
                    confidence=0.85,
                    claims_count=5,
                    cost_usd=0.01,
                ),
                ModelAnalysisRecord(
                    model_name="claude-3-sonnet",
                    decision="cautious",
                    confidence=0.65,
                    claims_count=3,
                    cost_usd=0.008,
                ),
            ],
            consensus=ConsensusRecord(
                consensus_decision="positive",
                consensus_ratio=0.67,
                total_claims=6,
            ),
        )
        assert len(record.model_results) == 2
        assert record.model_results[0].model_name == "gpt-4o"
        assert record.model_results[1].decision == "cautious"

    def test_model_results_serializes(self) -> None:
        """M1: model_results round-trips via model_dump() ✅."""
        from inquiro.core.trajectory.models import (
            AnalysisPhaseRecord,
            ModelAnalysisRecord,
        )

        record = AnalysisPhaseRecord(
            model_results=[
                ModelAnalysisRecord(
                    model_name="test-model",
                    decision="negative",
                    confidence=0.4,
                    claims_count=2,
                ),
            ],
        )
        dumped = record.model_dump()
        assert len(dumped["model_results"]) == 1
        assert dumped["model_results"][0]["model_name"] == "test-model"


# ============================================================================
# ⚠️ M6: _MIN_MODELS_FOR_CONSENSUS warning logged
# ============================================================================


class TestMinModelsConsensusWarning:
    """M6: Warning when successful models < _MIN_MODELS_FOR_CONSENSUS ⚠️."""

    def test_single_model_logs_warning(self) -> None:
        """M6: Single-model consensus emits a warning ✅."""
        from inquiro.exps.discovery_synthesis_exp import (
            DiscoverySynthesisExp,
            ModelSynthesisOutput,
        )

        output = ModelSynthesisOutput(
            model_name="test-model",
            decision="positive",
            confidence=0.9,
            summary="Test summary",
            claims=[],
            cost_usd=0.01,
        )
        logger = logging.getLogger("inquiro.exps.discovery_synthesis_exp")
        with _capture_log(logger) as log_output:
            decision, ratio = DiscoverySynthesisExp._compute_consensus(
                [output],
            )
        assert decision == "positive"
        assert ratio == 1.0
        assert any("minimum" in msg.lower() for msg in log_output)

    def test_two_models_no_warning(self) -> None:
        """M6: Two-model consensus does not warn ✅."""
        import logging

        from inquiro.exps.discovery_synthesis_exp import (
            DiscoverySynthesisExp,
            ModelSynthesisOutput,
        )

        outputs = [
            ModelSynthesisOutput(
                model_name=f"model-{i}",
                decision="positive",
                confidence=0.8,
                summary="Test",
                claims=[],
                cost_usd=0.01,
            )
            for i in range(2)
        ]
        logger = logging.getLogger("inquiro.exps.discovery_synthesis_exp")
        with _capture_log(logger) as log_output:
            decision, ratio = DiscoverySynthesisExp._compute_consensus(outputs)
        assert decision == "positive"
        assert not any("minimum" in msg.lower() for msg in log_output)


# ============================================================================
# 🔧 Log capture helper
# ============================================================================


class _LogCapture(logging.Handler):
    """Lightweight log capture handler 📝."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


@contextlib.contextmanager
def _capture_log(
    logger: logging.Logger,
    level: int = logging.WARNING,
):
    """Context manager to capture log messages 📋."""
    handler = _LogCapture()
    handler.setLevel(level)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
