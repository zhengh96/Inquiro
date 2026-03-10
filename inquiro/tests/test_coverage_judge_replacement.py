"""Tests for LLMCoverageJudge replacement — analysis agent coverage pipeline 🧪.

Covers:
- AggregationEngine._merge_checklist_coverage() majority voting
- DiscoveryLoop._build_pre_computed_coverage() ID→description mapping
- GapAnalysis.analyze() with pre_computed_coverage parameter
- MockCoverageJudge fallback when coverage is unavailable
- End-to-end AnalysisRoundOutput coverage field propagation

Each test is independent and follows Arrange → Act → Assert.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from inquiro.core.aggregation import AggregationEngine
from inquiro.core.gap_analysis import (
    CoverageResult,
    GapAnalysis,
    MockCoverageJudge,
    StopReason,
)
from inquiro.core.types import (
    Checklist,
    ChecklistCoverage,
    ChecklistItem,
    Decision,
    DiscoveryConfig,
    Evidence,
    EvaluationResult,
    EvaluationTask,
    ReasoningClaim,
    EvidenceStrength,
)


# ============================================================================
# 🏗️ Helpers
# ============================================================================


def _default_config(**overrides: Any) -> DiscoveryConfig:
    """Build a DiscoveryConfig with optional overrides 🔧."""
    defaults: dict[str, Any] = {
        "max_rounds": 5,
        "max_cost_per_subitem": 8.0,
        "coverage_threshold": 0.80,
        "convergence_delta": 0.05,
        "min_evidence_per_round": 3,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


def _make_evidence(eid: str = "E1") -> Evidence:
    """Create a minimal Evidence for testing 🧪."""
    return Evidence(
        id=eid,
        source="test-server",
        summary=f"Evidence item {eid}",
        query="test query",
    )


def _make_eval_result(
    covered: list[str] | None = None,
    missing: list[str] | None = None,
) -> EvaluationResult:
    """Create an EvaluationResult with checklist coverage 🧪."""
    coverage = ChecklistCoverage(
        required_covered=covered or [],
        required_missing=missing or [],
    )
    return EvaluationResult(
        task_id="test-task",
        decision=Decision.POSITIVE,
        confidence=0.8,
        reasoning=[
            ReasoningClaim(
                claim="Test claim",
                evidence_ids=["E1"],
                strength=EvidenceStrength.MODERATE,
            )
        ],
        evidence_index=[_make_evidence()],
        checklist_coverage=coverage,
    )


def _make_eval_result_no_coverage() -> EvaluationResult:
    """Create an EvaluationResult without checklist coverage 🧪."""
    return EvaluationResult(
        task_id="test-task",
        decision=Decision.CAUTIOUS,
        confidence=0.6,
        reasoning=[
            ReasoningClaim(
                claim="Test claim",
                evidence_ids=["E1"],
                strength=EvidenceStrength.MODERATE,
            )
        ],
        evidence_index=[_make_evidence()],
    )


# ============================================================================
# 📊 Tests: _merge_checklist_coverage
# ============================================================================


class TestMergeChecklistCoverage:
    """Tests for AggregationEngine._merge_checklist_coverage 📊."""

    def setup_method(self) -> None:
        """Create engine instance 🔧."""
        self.engine = AggregationEngine()

    def test_all_models_agree_covered(self) -> None:
        """All 3 models agree item is covered → covered ✅."""
        results = [
            ("model_a", _make_eval_result(covered=["R1", "R2"], missing=["R3"])),
            ("model_b", _make_eval_result(covered=["R1", "R2"], missing=["R3"])),
            ("model_c", _make_eval_result(covered=["R1", "R2"], missing=["R3"])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        assert sorted(coverage.required_covered) == ["R1", "R2"]
        assert coverage.required_missing == ["R3"]
        assert conflicts == []  # Unanimous

    def test_two_vs_one_covered(self) -> None:
        """2/3 models say covered → covered (majority) with conflict 🗳️."""
        results = [
            ("model_a", _make_eval_result(covered=["R1"], missing=["R2"])),
            ("model_b", _make_eval_result(covered=["R1"], missing=["R2"])),
            ("model_c", _make_eval_result(covered=["R2"], missing=["R1"])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        # R1: 2 covered votes >= ceil(3/2)=2 → covered
        assert "R1" in coverage.required_covered
        # R2: 1 covered vote < 2 → missing
        assert "R2" in coverage.required_missing
        # Both items have non-unanimous votes
        assert sorted(conflicts) == ["R1", "R2"]

    def test_all_models_agree_missing(self) -> None:
        """All models say missing → missing, no conflict ✅."""
        results = [
            ("model_a", _make_eval_result(covered=[], missing=["R1"])),
            ("model_b", _make_eval_result(covered=[], missing=["R1"])),
            ("model_c", _make_eval_result(covered=[], missing=["R1"])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        assert coverage.required_covered == []
        assert coverage.required_missing == ["R1"]
        assert conflicts == []

    def test_no_coverage_data(self) -> None:
        """No model has coverage data → returns None 🚫."""
        results = [
            ("model_a", _make_eval_result_no_coverage()),
            ("model_b", _make_eval_result_no_coverage()),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is None
        assert conflicts == []

    def test_partial_coverage_data(self) -> None:
        """Only some models have coverage → uses available data 📊."""
        results = [
            ("model_a", _make_eval_result(covered=["R1"], missing=["R2"])),
            ("model_b", _make_eval_result_no_coverage()),
            ("model_c", _make_eval_result(covered=["R1", "R2"], missing=[])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        # Only 2 models with data: threshold = ceil(2/2) = 1
        # R1: 2 votes → covered
        assert "R1" in coverage.required_covered
        # R2: 1 vote → covered (>= threshold of 1)
        assert "R2" in coverage.required_covered

    def test_two_models_split(self) -> None:
        """2 models with opposite views → threshold = ceil(2/2) = 1 📊."""
        results = [
            ("model_a", _make_eval_result(covered=["R1"], missing=["R2"])),
            ("model_b", _make_eval_result(covered=["R2"], missing=["R1"])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        # threshold = ceil(2/2) = 1 → both items get 1 vote → both covered
        assert sorted(coverage.required_covered) == ["R1", "R2"]
        assert coverage.required_missing == []
        # Both are conflicts (1 < 2 models)
        assert sorted(conflicts) == ["R1", "R2"]

    def test_single_model(self) -> None:
        """Single model → threshold = 1, all its data passes 📊."""
        results = [
            ("model_a", _make_eval_result(covered=["R1"], missing=["R2", "R3"])),
        ]

        coverage, conflicts = self.engine._merge_checklist_coverage(results)

        assert coverage is not None
        assert coverage.required_covered == ["R1"]
        assert sorted(coverage.required_missing) == ["R2", "R3"]
        assert conflicts == []  # Only 1 model, no disagreement possible

    def test_empty_results(self) -> None:
        """Empty results list → returns None 🚫."""
        coverage, conflicts = self.engine._merge_checklist_coverage([])

        assert coverage is None
        assert conflicts == []


# ============================================================================
# 🗺️ Tests: _build_pre_computed_coverage
# ============================================================================


class TestBuildPreComputedCoverage:
    """Tests for DiscoveryLoop._build_pre_computed_coverage 🗺️."""

    def _make_analysis_output(
        self,
        covered_ids: list[str] | None = None,
        missing_ids: list[str] | None = None,
        conflicts: list[str] | None = None,
    ) -> Any:
        """Create mock AnalysisRoundOutput 🧪."""
        from inquiro.core.discovery_loop import AnalysisRoundOutput

        coverage = None
        if covered_ids is not None or missing_ids is not None:
            coverage = ChecklistCoverage(
                required_covered=covered_ids or [],
                required_missing=missing_ids or [],
            )

        return AnalysisRoundOutput(
            claims=[{"claim": "test", "evidence_ids": ["E1"], "strength": "moderate"}],
            checklist_coverage=coverage,
            coverage_conflicts=conflicts or [],
        )

    def _make_task_with_checklist(
        self,
        items: list[tuple[str, str]],
    ) -> EvaluationTask:
        """Create EvaluationTask with checklist items 🧪.

        Args:
            items: List of (id, description) tuples.
        """
        checklist_items = [
            ChecklistItem(id=item_id, description=desc)
            for item_id, desc in items
        ]
        return EvaluationTask(
            task_id="test-task",
            topic="Test topic",
            checklist=Checklist(required=checklist_items),
        )

    def _get_loop(self) -> Any:
        """Create a minimal DiscoveryLoop for testing 🧪."""
        from inquiro.core.discovery_loop import DiscoveryLoop

        loop = DiscoveryLoop.__new__(DiscoveryLoop)
        return loop

    def test_normal_mapping(self) -> None:
        """IDs correctly mapped to descriptions ✅."""
        loop = self._get_loop()
        task = self._make_task_with_checklist([
            ("R1", "Check target validation"),
            ("R2", "Check safety profile"),
            ("R3", "Check efficacy data"),
        ])
        analysis = self._make_analysis_output(
            covered_ids=["R1", "R3"],
            missing_ids=["R2"],
            conflicts=["R3"],
        )
        checklist = ["Check target validation", "Check safety profile", "Check efficacy data"]

        result = loop._build_pre_computed_coverage(analysis, task, checklist)

        assert result is not None
        assert "Check target validation" in result.covered
        assert "Check efficacy data" in result.covered
        assert "Check safety profile" in result.uncovered
        assert "Check efficacy data" in result.conflict_signals

    def test_no_coverage_returns_none(self) -> None:
        """No coverage data → returns None 🚫."""
        loop = self._get_loop()
        task = self._make_task_with_checklist([("R1", "Item 1")])
        analysis = self._make_analysis_output()
        checklist = ["Item 1"]

        result = loop._build_pre_computed_coverage(analysis, task, checklist)

        assert result is None

    def test_unknown_id_skipped(self) -> None:
        """Unknown checklist IDs are skipped gracefully ⚠️."""
        loop = self._get_loop()
        task = self._make_task_with_checklist([
            ("R1", "Known item"),
        ])
        analysis = self._make_analysis_output(
            covered_ids=["R1", "R99"],
            missing_ids=["R100"],
        )
        checklist = ["Known item"]

        result = loop._build_pre_computed_coverage(analysis, task, checklist)

        assert result is not None
        assert result.covered == ["Known item"]
        assert result.uncovered == []

    def test_empty_checklist_returns_none(self) -> None:
        """Empty checklist → returns None 🚫."""
        loop = self._get_loop()
        task = EvaluationTask(task_id="test", topic="Test")
        analysis = self._make_analysis_output(
            covered_ids=["R1"],
            missing_ids=[],
        )
        checklist: list[str] = []

        result = loop._build_pre_computed_coverage(analysis, task, checklist)

        assert result is None

    def test_all_ids_unmapped_returns_none(self) -> None:
        """All IDs fail to map → returns None (fallback to judge) ⚠️."""
        loop = self._get_loop()
        task = self._make_task_with_checklist([
            ("R1", "Item 1"),
        ])
        # Coverage has IDs that don't exist in checklist
        analysis = self._make_analysis_output(
            covered_ids=["X1"],
            missing_ids=["X2"],
        )
        checklist = ["Item 1"]

        result = loop._build_pre_computed_coverage(analysis, task, checklist)

        assert result is None


# ============================================================================
# 🎯 Tests: GapAnalysis with pre_computed_coverage
# ============================================================================


class TestGapAnalysisPreComputed:
    """Tests for GapAnalysis.analyze() with pre_computed_coverage 🎯."""

    @pytest.mark.asyncio
    async def test_pre_computed_bypasses_judge(self) -> None:
        """Pre-computed coverage skips the coverage judge entirely ✅."""
        # 🔧 Create a judge that would fail if called
        class FailingJudge:
            async def judge_coverage(
                self, checklist: list[str], claims: list[Any], evidence: list[Any]
            ) -> CoverageResult:
                raise AssertionError("Judge should not be called!")

        gap = GapAnalysis(
            coverage_judge=FailingJudge(),
            coverage_judge_mode="fallback",
        )
        config = _default_config(coverage_threshold=0.80)

        pre_coverage = CoverageResult(
            covered=["Item A", "Item B", "Item C"],
            uncovered=["Item D"],
            conflict_signals=["Item C"],
        )

        report = await gap.analyze(
            checklist=["Item A", "Item B", "Item C", "Item D"],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
            pre_computed_coverage=pre_coverage,
        )

        # 📊 3/4 covered = 0.75
        assert report.coverage_ratio == pytest.approx(0.75)
        assert sorted(report.covered_items) == ["Item A", "Item B", "Item C"]
        assert report.uncovered_items == ["Item D"]
        assert report.conflict_signals == ["Item C"]
        assert not report.converged  # 0.75 < 0.80

    @pytest.mark.asyncio
    async def test_pre_computed_triggers_convergence(self) -> None:
        """Pre-computed coverage can trigger convergence ✅."""
        gap = GapAnalysis(coverage_judge_mode="fallback")
        config = _default_config(coverage_threshold=0.80)

        pre_coverage = CoverageResult(
            covered=["A", "B", "C", "D"],
            uncovered=["E"],
        )

        report = await gap.analyze(
            checklist=["A", "B", "C", "D", "E"],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
            pre_computed_coverage=pre_coverage,
        )

        # 📊 4/5 = 0.80 → converged
        assert report.coverage_ratio == pytest.approx(0.80)
        assert report.converged
        assert report.convergence_reason == StopReason.CONVERGED.value

    @pytest.mark.asyncio
    async def test_none_falls_back_to_judge(self) -> None:
        """None pre_computed_coverage falls back to MockCoverageJudge 🧪."""
        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        config = _default_config()

        report = await gap.analyze(
            checklist=["authentication security", "data validation"],
            claims=[
                {"claim": "Strong authentication is implemented for all endpoints"},
            ],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
            pre_computed_coverage=None,
        )

        # 📊 MockCoverageJudge uses keyword matching
        assert report.coverage_ratio >= 0.0
        assert isinstance(report.covered_items, list)
        assert isinstance(report.uncovered_items, list)


# ============================================================================
# 📊 Tests: AggregatedResult coverage fields
# ============================================================================


class TestAggregatedResultCoverageFields:
    """Tests for AggregatedResult checklist_coverage field 📊."""

    def test_aggregate_includes_coverage(self) -> None:
        """aggregate() populates checklist_coverage and coverage_conflicts ✅."""
        engine = AggregationEngine()
        results = [
            ("model_a", _make_eval_result(covered=["R1", "R2"], missing=["R3"])),
            ("model_b", _make_eval_result(covered=["R1"], missing=["R2", "R3"])),
            ("model_c", _make_eval_result(covered=["R1", "R2"], missing=["R3"])),
        ]
        weights = {"model_a": 1.0, "model_b": 1.0, "model_c": 1.0}

        aggregated = engine.aggregate(results=results, weights=weights)

        assert aggregated.checklist_coverage is not None
        # R1: 3/3 → covered
        assert "R1" in aggregated.checklist_coverage.required_covered
        # R2: 2/3 >= ceil(3/2)=2 → covered
        assert "R2" in aggregated.checklist_coverage.required_covered
        # R3: 0/3 → missing
        assert "R3" in aggregated.checklist_coverage.required_missing
        # R2 conflict: model_b disagrees
        assert "R2" in aggregated.coverage_conflicts

    def test_aggregate_no_coverage(self) -> None:
        """aggregate() with no coverage data → None ✅."""
        engine = AggregationEngine()
        results = [
            ("model_a", _make_eval_result_no_coverage()),
            ("model_b", _make_eval_result_no_coverage()),
        ]
        weights = {"model_a": 1.0, "model_b": 1.0}

        aggregated = engine.aggregate(results=results, weights=weights)

        assert aggregated.checklist_coverage is None
        assert aggregated.coverage_conflicts == []


# ============================================================================
# 🔄 Tests: AnalysisRoundOutput coverage fields
# ============================================================================


class TestAnalysisRoundOutputCoverage:
    """Tests for AnalysisRoundOutput checklist_coverage fields 🔄."""

    def test_default_none(self) -> None:
        """Default AnalysisRoundOutput has None coverage ✅."""
        from inquiro.core.discovery_loop import AnalysisRoundOutput

        output = AnalysisRoundOutput()

        assert output.checklist_coverage is None
        assert output.coverage_conflicts == []

    def test_with_coverage(self) -> None:
        """AnalysisRoundOutput can carry coverage data ✅."""
        from inquiro.core.discovery_loop import AnalysisRoundOutput

        coverage = ChecklistCoverage(
            required_covered=["R1"],
            required_missing=["R2"],
        )
        output = AnalysisRoundOutput(
            checklist_coverage=coverage,
            coverage_conflicts=["R2"],
        )

        assert output.checklist_coverage is not None
        assert output.checklist_coverage.required_covered == ["R1"]
        assert output.coverage_conflicts == ["R2"]


# ============================================================================
# 📐 Tests: Majority vote threshold math
# ============================================================================


class TestMajorityVoteThreshold:
    """Tests for correct ceil(n/2) threshold calculation 📐."""

    def test_threshold_1_model(self) -> None:
        """1 model → threshold=1 ✅."""
        assert math.ceil(1 / 2) == 1

    def test_threshold_2_models(self) -> None:
        """2 models → threshold=1 ✅."""
        assert math.ceil(2 / 2) == 1

    def test_threshold_3_models(self) -> None:
        """3 models → threshold=2 ✅."""
        assert math.ceil(3 / 2) == 2

    def test_threshold_4_models(self) -> None:
        """4 models → threshold=2 ✅."""
        assert math.ceil(4 / 2) == 2

    def test_threshold_5_models(self) -> None:
        """5 models → threshold=3 ✅."""
        assert math.ceil(5 / 2) == 3


# ============================================================================
# 🛡️ Tests: CoverageResult model
# ============================================================================


class TestCoverageResultModel:
    """Tests for CoverageResult with conflict_signals 🛡️."""

    def test_conflict_signals_populated(self) -> None:
        """CoverageResult.conflict_signals can carry real data ✅."""
        result = CoverageResult(
            covered=["Item A"],
            uncovered=["Item B"],
            conflict_signals=["Item B"],
        )

        assert result.conflict_signals == ["Item B"]

    def test_default_empty_conflicts(self) -> None:
        """CoverageResult defaults to empty conflict_signals ✅."""
        result = CoverageResult(covered=["A"], uncovered=[])

        assert result.conflict_signals == []
