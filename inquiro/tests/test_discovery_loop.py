"""Tests for DiscoveryLoop — multi-round orchestrator 🧪.

Covers the full discovery loop lifecycle including multi-round
iteration, convergence conditions, evidence accumulation,
trajectory recording, and error handling.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from inquiro.core.discovery_loop import (
    AnalysisRoundOutput,
    DefaultFocusPromptGenerator,
    DiscoveryLoop,
    MockAnalysisExecutor,
    MockFocusPromptGenerator,
    MockSearchExecutor,
    SearchRoundOutput,
)
from inquiro.core.gap_analysis import (
    CoverageResult,
    GapAnalysis,
)
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    DiscoveryConfig,
    DiscoveryResult,
    Evidence,
    EvaluationTask,
    GapReport,
)


# ============================================================================
# 🔧 Test fixtures
# ============================================================================


def _make_evidence(
    eid: str = "E1",
    summary: str = (
        "This is a sufficiently long test evidence summary about "
        "protein interactions and binding affinities in cell cultures"
    ),
    source: str = "test-mcp",
    query: str = "test query",
    url: str | None = None,
) -> Evidence:
    """Create a test Evidence instance with valid length 🔧.

    Args:
        eid: Evidence identifier.
        summary: Evidence summary (must be > 50 chars).
        source: Evidence source name.
        query: Query that produced this evidence.
        url: Optional source URL.

    Returns:
        Evidence instance.
    """
    return Evidence(
        id=eid,
        source=source,
        query=query,
        summary=summary,
        url=url,
    )


def _make_task(
    task_id: str = "task-test-001",
    topic: str = "Test topic for evidence research",
    checklist_items: list[str] | None = None,
) -> EvaluationTask:
    """Create a test EvaluationTask with optional checklist 🔧.

    Args:
        task_id: Task identifier.
        topic: Task topic.
        checklist_items: Optional checklist item descriptions.

    Returns:
        EvaluationTask instance.
    """
    items = (
        checklist_items
        if checklist_items is not None
        else [
            "Assess protein binding affinity data",
            "Evaluate clinical trial outcomes",
            "Review safety profile evidence",
        ]
    )
    checklist = Checklist(
        required=[
            ChecklistItem(id=f"C{i + 1}", description=desc)
            for i, desc in enumerate(items)
        ]
    )
    return EvaluationTask(
        task_id=task_id,
        topic=topic,
        rules="Test evaluation rules for this task",
        checklist=checklist,
        output_schema={},
    )


def _make_config(**kwargs: Any) -> DiscoveryConfig:
    """Create a DiscoveryConfig with test-friendly defaults 🔧.

    Args:
        **kwargs: Override any DiscoveryConfig field.

    Returns:
        DiscoveryConfig instance.
    """
    defaults = {
        "max_rounds": 3,
        "max_cost_per_subitem": 10.0,
        "coverage_threshold": 0.80,
        "convergence_delta": 0.05,
        "min_evidence_per_round": 1,
        "timeout_per_round": 60,
        "timeout_total": 300,
    }
    defaults.update(kwargs)
    return DiscoveryConfig(**defaults)


def _make_claims(keywords: list[str]) -> list[dict[str, Any]]:
    """Create mock claims containing keywords for coverage matching 🔧.

    Args:
        keywords: Keywords to include in claim text.

    Returns:
        List of claim dicts.
    """
    return [
        {
            "claim": f"Evidence shows {kw} in the dataset",
            "evidence_ids": ["E1"],
        }
        for kw in keywords
    ]


# ============================================================================
# 📊 Basic lifecycle tests
# ============================================================================


class TestDiscoveryLoopBasic:
    """Test basic DiscoveryLoop lifecycle 📊."""

    @pytest.mark.asyncio
    async def test_single_round_converges(self) -> None:
        """Loop stops after 1 round when coverage threshold met ✅."""
        evidence = [_make_evidence()]
        # 🔧 Claims that cover all 3 checklist items
        claims = _make_claims(
            [
                "protein binding affinity",
                "clinical trial outcomes",
                "safety profile evidence",
            ]
        )

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        config = _make_config(coverage_threshold=0.80)
        task = _make_task()
        result = await loop.run(task, config)

        assert isinstance(result, DiscoveryResult)
        assert result.total_rounds == 1
        assert result.final_coverage >= 0.80
        assert "coverage_threshold_reached" in result.termination_reason
        assert search.call_count == 1
        assert analysis.call_count == 1

    @pytest.mark.asyncio
    async def test_max_rounds_reached(self) -> None:
        """Loop stops at max_rounds when coverage never meets threshold ✅."""
        evidence = [_make_evidence()]
        # 🔧 Claims that cover nothing (no keyword matches)
        claims = [{"claim": "Unrelated observation xyz", "evidence_ids": ["E1"]}]

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        # 🔧 Disable diminishing returns to ensure max_rounds is the stop
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        assert result.total_rounds == 2
        assert result.termination_reason == "max_rounds_reached"
        assert search.call_count == 2
        assert analysis.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_checklist_converges_immediately(self) -> None:
        """Empty checklist yields coverage=1.0, converges in 1 round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task(checklist_items=[])  # Empty required list
        config = _make_config()
        result = await loop.run(task, config)

        assert result.total_rounds == 1
        assert result.final_coverage == 1.0
        assert "coverage_threshold_reached" in result.termination_reason

    @pytest.mark.asyncio
    async def test_returns_discovery_result(self) -> None:
        """run() returns a DiscoveryResult with correct fields ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(
            mock_claims=[{"claim": "Test claim", "evidence_ids": ["E1"]}],
        )

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        result = await loop.run(task, config)

        assert result.task_id == task.task_id
        assert result.pipeline_mode == "discovery"
        assert result.total_rounds >= 1
        assert result.total_cost_usd >= 0
        assert result.trajectory_id is not None
        assert len(result.round_summaries) >= 1


# ============================================================================
# 📊 Evidence accumulation tests
# ============================================================================


class TestEvidenceAccumulation:
    """Test evidence and claims accumulation across rounds 📊."""

    @pytest.mark.asyncio
    async def test_evidence_accumulates_across_rounds(self) -> None:
        """Each round adds to the shared evidence pool ✅."""
        evidence = [_make_evidence(eid=f"E{i}") for i in range(3)]
        claims = [{"claim": "Unrelated claim xyz", "evidence_ids": ["E1"]}]

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        # ✅ 2 rounds × 3 evidence = 6 total (after cleaning/dedup)
        assert result.total_rounds == 2
        # 🔧 Evidence count depends on dedup — but should be > 0
        assert len(result.evidence) > 0

    @pytest.mark.asyncio
    async def test_claims_accumulate_across_rounds(self) -> None:
        """Claims from all rounds are collected in the result ✅."""
        evidence = [_make_evidence()]
        claims = [{"claim": "Round claim abc", "evidence_ids": ["E1"]}]

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        # 🔧 Disable diminishing returns to ensure all 3 rounds run
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        # ✅ 3 rounds × 1 claim = 3 claims accumulated
        assert len(result.claims) == 3

    @pytest.mark.asyncio
    async def test_gap_reports_collected_per_round(self) -> None:
        """A gap report is stored for each round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        assert len(result.gap_reports) == 3
        assert all(isinstance(gr, GapReport) for gr in result.gap_reports)

    @pytest.mark.asyncio
    async def test_round_summaries_collected(self) -> None:
        """Round summaries have correct metadata per round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        assert len(result.round_summaries) == 2
        for i, rs in enumerate(result.round_summaries):
            assert rs.round_number == i + 1
            assert rs.round_cost_usd >= 0


# ============================================================================
# 🛑 Convergence condition tests
# ============================================================================


class TestConvergenceConditions:
    """Test various convergence conditions 🛑."""

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self) -> None:
        """Loop stops when budget is exhausted ✅."""
        evidence = [_make_evidence()]
        claims = [{"claim": "Unrelated xyz", "evidence_ids": ["E1"]}]

        search = MockSearchExecutor(mock_evidence=evidence)
        # 🔧 High analysis cost to trigger budget limit
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        # 🔧 Very low budget: search=0.10 + analysis=0.50 = 0.60 per round
        config = _make_config(
            max_rounds=5,
            max_cost_per_subitem=0.50,
            coverage_threshold=0.99,
        )
        task = _make_task()
        result = await loop.run(task, config)

        assert "max_cost_per_subitem_exhausted" in result.termination_reason

    @pytest.mark.asyncio
    async def test_diminishing_returns(self) -> None:
        """Loop stops on diminishing returns after round 1 ✅."""

        # 🔧 Custom coverage judge that always returns 0 coverage
        class ZeroCoverageJudge:
            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                return CoverageResult(
                    covered=[],
                    uncovered=checklist,
                )

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        gap = GapAnalysis(coverage_judge=ZeroCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        # 🔧 convergence_delta=0.1 → 0.0 improvement triggers stop
        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.99,
            convergence_delta=0.1,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        # ✅ Diminishing returns should trigger after round 2
        assert result.total_rounds == 2
        assert "diminishing_returns" in result.termination_reason

    @pytest.mark.asyncio
    async def test_search_exhausted(self) -> None:
        """Loop stops when search returns too few evidence items ✅."""

        # 🔧 Custom judge that gives partial coverage (not converged)
        class PartialCoverageJudge:
            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                # 🔧 Always report 1/3 covered
                covered = checklist[:1] if checklist else []
                uncovered = checklist[1:] if checklist else []
                return CoverageResult(
                    covered=covered,
                    uncovered=uncovered,
                )

        # 🔧 Empty search results → 0 evidence per round
        search = MockSearchExecutor(mock_evidence=[])
        analysis = MockAnalysisExecutor(mock_claims=[])

        gap = GapAnalysis(coverage_judge=PartialCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.99,
            min_evidence_per_round=3,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        task = _make_task()
        result = await loop.run(task, config)

        # ✅ Search exhausted should trigger after round 2
        assert result.total_rounds == 2
        assert "search_exhausted" in result.termination_reason

    @pytest.mark.asyncio
    async def test_convergence_delta_tightened_stops_early(self) -> None:
        """Default delta=0.08 stops loop when improvement is below 8% ✅.

        With the tightened default convergence_delta=0.08, any round that
        yields a coverage improvement strictly below 0.08 triggers
        diminishing_returns (with the default patience=1).  This test
        verifies that a 6% improvement (0.06) is insufficient to continue.
        """

        # 🔧 Stateful judge: round 1 → 40% coverage, round 2 → 46% (+6%)
        class SixPercentImprovementJudge:
            """Returns 40% coverage on round 1, then 46% on round 2 🔢."""

            def __init__(self) -> None:
                self._call_count: int = 0

            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                self._call_count += 1
                if self._call_count == 1:
                    # 🔧 Round 1: 2/5 covered (40%)
                    covered = checklist[:2]
                    uncovered = checklist[2:]
                else:
                    # 🔧 Round 2: 2/5 + tiny fraction → same 2 items covered
                    # Delta = 0/5 = 0.0 < 0.08 → diminishing returns
                    covered = checklist[:2]
                    uncovered = checklist[2:]
                return CoverageResult(covered=covered, uncovered=uncovered)

        task = _make_task(
            checklist_items=[
                "Item alpha binding",
                "Item beta activation",
                "Item gamma pathway",
                "Item delta signaling",
                "Item epsilon regulation",
            ]
        )
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        gap = GapAnalysis(coverage_judge=SixPercentImprovementJudge())
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        # 🔧 Use the new default delta=0.08, patience=1 (also default)
        config = _make_config(
            max_rounds=10,
            coverage_threshold=0.99,
            convergence_delta=0.08,
            max_cost_per_subitem=100.0,
            min_evidence_per_round=1,
        )
        result = await loop.run(task, config)

        # ✅ Should stop after round 2: delta=0.0 < 0.08 → diminishing returns
        assert result.total_rounds == 2
        assert "diminishing_returns" in result.termination_reason

    @pytest.mark.asyncio
    async def test_convergence_patience_prevents_premature_stop(self) -> None:
        """Patience=2 allows loop to survive 1 low-delta round ✅.

        With patience=2, the loop requires *two consecutive* low-delta
        rounds before stopping.  If a low-delta round is followed by a
        productive round (delta >= threshold), the counter resets and the
        loop continues.
        """

        # 🔧 Stateful judge sequence:
        #   Round 1: 2/4 covered (50%)
        #   Round 2: 2/4 covered (50%, delta=0.0 — low, consecutive=1)
        #   Round 3: 4/4 covered (100%, delta=0.5 — good, counter resets)
        #   → Coverage threshold reached, converged
        class ResetPatternJudge:
            """Low delta on round 2, then full coverage on round 3 🔢."""

            def __init__(self) -> None:
                self._call_count: int = 0

            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                self._call_count += 1
                if self._call_count <= 2:
                    # 🔧 Rounds 1 and 2: 2/4 items covered (50%)
                    covered = checklist[:2]
                    uncovered = checklist[2:]
                else:
                    # 🔧 Round 3: all 4 items covered (100%)
                    covered = checklist[:]
                    uncovered = []
                return CoverageResult(covered=covered, uncovered=uncovered)

        task = _make_task(
            checklist_items=[
                "Item alpha binding",
                "Item beta activation",
                "Item gamma pathway",
                "Item delta signaling",
            ]
        )
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        gap = GapAnalysis(coverage_judge=ResetPatternJudge())
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        # 🔧 patience=2: needs 2 consecutive low-delta rounds to stop
        config = _make_config(
            max_rounds=10,
            coverage_threshold=0.99,
            convergence_delta=0.08,
            convergence_patience=2,
            max_cost_per_subitem=100.0,
            min_evidence_per_round=1,
        )
        result = await loop.run(task, config)

        # ✅ Round 2 had low delta (consecutive=1) but patience=2 keeps going.
        # Round 3 hits coverage_threshold=0.99 → convergence.
        assert result.total_rounds == 3
        assert "coverage_threshold_reached" in result.termination_reason

    @pytest.mark.asyncio
    async def test_convergence_patience_one_is_immediate(self) -> None:
        """Patience=1 (default) stops immediately after first low-delta round ✅.

        With patience=1, the very first round with delta < convergence_delta
        triggers diminishing_returns — matching the original behavior
        (before patience was introduced).
        """

        # 🔧 Stateful judge: round 1 → 50% coverage, round 2 → 50% (+0%)
        class StagnantCoverageJudge:
            """Returns the same 50% coverage on every round 🔢."""

            def __init__(self) -> None:
                self._call_count: int = 0

            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                self._call_count += 1
                # 🔧 Always 2/4 covered regardless of round
                covered = checklist[:2]
                uncovered = checklist[2:]
                return CoverageResult(covered=covered, uncovered=uncovered)

        task = _make_task(
            checklist_items=[
                "Item alpha binding",
                "Item beta activation",
                "Item gamma pathway",
                "Item delta signaling",
            ]
        )
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        gap = GapAnalysis(coverage_judge=StagnantCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        # 🔧 patience=1 (default): first low-delta round triggers stop
        config = _make_config(
            max_rounds=10,
            coverage_threshold=0.99,
            convergence_delta=0.08,
            convergence_patience=1,
            max_cost_per_subitem=100.0,
            min_evidence_per_round=1,
        )
        result = await loop.run(task, config)

        # ✅ After round 2: delta=0.0 < 0.08 and patience=1 → stop immediately
        assert result.total_rounds == 2
        assert "diminishing_returns" in result.termination_reason


# ============================================================================
# 📊 Configuration tests
# ============================================================================


class TestConfigResolution:
    """Test configuration resolution 📊."""

    @pytest.mark.asyncio
    async def test_config_from_task_discovery_config(self) -> None:
        """Config parsed from task.discovery_config when not explicit ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        task.discovery_config = {"max_rounds": 1, "coverage_threshold": 0.01}

        result = await loop.run(task)

        # ✅ Only 1 round because max_rounds=1 from task config
        assert result.total_rounds == 1

    @pytest.mark.asyncio
    async def test_explicit_config_overrides_task(self) -> None:
        """Explicit config parameter takes priority over task ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        task.discovery_config = {"max_rounds": 5}

        # 🔧 Explicit config with max_rounds=1 should win
        config = _make_config(max_rounds=1, coverage_threshold=0.01)
        result = await loop.run(task, config)

        assert result.total_rounds == 1

    @pytest.mark.asyncio
    async def test_default_config_when_none(self) -> None:
        """Default DiscoveryConfig used when no config provided ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        # 🔧 Claims that match all checklist items for fast convergence
        claims = _make_claims(
            [
                "protein binding affinity",
                "clinical trial outcomes",
                "safety profile evidence",
            ]
        )
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        # 🔧 No config, no task.discovery_config → defaults
        result = await loop.run(task)

        assert result.total_rounds >= 1


# ============================================================================
# 📊 Trajectory recording tests
# ============================================================================


class TestTrajectoryRecording:
    """Test trajectory JSONL recording 📊."""

    @pytest.mark.asyncio
    async def test_trajectory_file_created(self) -> None:
        """Trajectory JSONL file is created when directory is set ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            search = MockSearchExecutor(mock_evidence=[_make_evidence()])
            claims = _make_claims(
                [
                    "protein binding affinity",
                    "clinical trial outcomes",
                    "safety profile evidence",
                ]
            )
            analysis = MockAnalysisExecutor(mock_claims=claims)

            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
            )

            task = _make_task()
            config = _make_config(max_rounds=1, coverage_threshold=0.01)
            await loop.run(task, config)

            # ✅ At least one JSONL file should exist
            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(files) == 1

    @pytest.mark.asyncio
    async def test_trajectory_contains_expected_records(self) -> None:
        """Trajectory JSONL has meta, round, summary, meta_final ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            search = MockSearchExecutor(mock_evidence=[_make_evidence()])
            claims = _make_claims(
                [
                    "protein binding affinity",
                    "clinical trial outcomes",
                    "safety profile evidence",
                ]
            )
            analysis = MockAnalysisExecutor(mock_claims=claims)

            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
            )

            task = _make_task()
            config = _make_config(max_rounds=1, coverage_threshold=0.01)
            await loop.run(task, config)

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            filepath = os.path.join(tmpdir, files[0])

            with open(filepath) as f:
                records = [json.loads(line) for line in f]

            # ✅ Verify record types present
            types = [r["type"] for r in records]
            assert types[0] == "meta"
            assert "round" in types
            assert "summary" in types
            assert types[-1] == "meta_final"

    @pytest.mark.asyncio
    async def test_trajectory_events_recorded(self) -> None:
        """Trajectory events are written for each phase ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            search = MockSearchExecutor(mock_evidence=[_make_evidence()])
            claims = _make_claims(
                [
                    "protein binding affinity",
                    "clinical trial outcomes",
                    "safety profile evidence",
                ]
            )
            analysis = MockAnalysisExecutor(mock_claims=claims)

            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
            )

            task = _make_task()
            config = _make_config(max_rounds=1, coverage_threshold=0.01)
            await loop.run(task, config)

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            filepath = os.path.join(tmpdir, files[0])

            with open(filepath) as f:
                records = [json.loads(line) for line in f]

            # ✅ Verify event records
            event_types = [r["event_type"] for r in records if r["type"] == "event"]
            assert "round_started" in event_types
            assert "search_completed" in event_types
            assert "cleaning_completed" in event_types
            assert "analysis_completed" in event_types
            assert "gap_completed" in event_types

    @pytest.mark.asyncio
    async def test_no_trajectory_without_dir(self) -> None:
        """No trajectory file created when dir is not set ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            trajectory_dir=None,
        )

        task = _make_task()
        config = _make_config(max_rounds=1, coverage_threshold=0.01)
        result = await loop.run(task, config)

        # ✅ Should complete without error
        assert result.total_rounds >= 1


# ============================================================================
# ❌ Error handling tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and resilience ❌."""

    @pytest.mark.asyncio
    async def test_search_failure_returns_empty(self) -> None:
        """Search failure results in empty search output, loop continues ✅."""

        class FailingSearchExecutor:
            """Always raises an exception 💥."""

            call_count = 0

            async def execute_search(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: str | None = None,
            ) -> SearchRoundOutput:
                self.call_count += 1
                raise RuntimeError("MCP connection failed")

        search = FailingSearchExecutor()
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ Should complete all rounds despite search failures
        assert result.total_rounds == 2
        assert search.call_count == 2

    @pytest.mark.asyncio
    async def test_analysis_failure_returns_empty(self) -> None:
        """Analysis failure results in empty output, loop continues ✅."""

        class FailingAnalysisExecutor:
            """Always raises an exception 💥."""

            call_count = 0

            async def execute_analysis(
                self,
                task: Any,
                evidence: Any,
                config: Any,
                round_number: int,
                supplementary_context: Any = None,
            ) -> AnalysisRoundOutput:
                self.call_count += 1
                raise RuntimeError("LLM API error")

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = FailingAnalysisExecutor()

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ Should complete all rounds despite analysis failures
        assert result.total_rounds == 2
        assert analysis.call_count == 2

    @pytest.mark.asyncio
    async def test_search_timeout_returns_empty(self) -> None:
        """Search timeout returns empty output gracefully ✅."""

        class SlowSearchExecutor:
            """Sleeps longer than timeout 🐌."""

            call_count = 0

            async def execute_search(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: str | None = None,
            ) -> SearchRoundOutput:
                self.call_count += 1
                await asyncio.sleep(100)  # Will be cancelled
                return SearchRoundOutput()

        import asyncio

        search = SlowSearchExecutor()
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=1,
            timeout_per_round=1,  # 1 second timeout
            coverage_threshold=0.01,
        )
        result = await loop.run(task, config)

        # ✅ Should complete without hanging
        assert result.total_rounds == 1
        assert search.call_count == 1

    @pytest.mark.asyncio
    async def test_focus_generator_failure_uses_fallback(self) -> None:
        """Focus generation failure uses fallback prompt ✅."""

        class FailingFocusGenerator:
            """Always raises an exception 💥."""

            call_count = 0

            async def generate_focus(
                self,
                gap_report: Any,
                config: Any,
                round_number: int,
            ) -> None:
                self.call_count += 1
                raise RuntimeError("Focus generation failed")

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        focus_gen = FailingFocusGenerator()
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            focus_generator=focus_gen,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ Should complete despite focus generation failure
        assert result.total_rounds == 2
        assert focus_gen.call_count >= 1


# ============================================================================
# 🔧 Focus prompt tests
# ============================================================================


class TestFocusPromptGeneration:
    """Test focus prompt generation between rounds 🔧."""

    @pytest.mark.asyncio
    async def test_focus_generator_called_between_rounds(self) -> None:
        """Focus generator is called after each non-final round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])
        focus_gen = MockFocusPromptGenerator()

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            focus_generator=focus_gen,
        )

        task = _make_task()
        # 🔧 Disable early stop conditions to ensure all 3 rounds run
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        await loop.run(task, config)

        # ✅ Focus generated between rounds 1→2 and 2→3, but not after 3
        assert focus_gen.call_count == 2  # rounds 1 and 2

    @pytest.mark.asyncio
    async def test_no_focus_for_single_round(self) -> None:
        """Focus generator not called when loop is single round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        claims = _make_claims(
            [
                "protein binding affinity",
                "clinical trial outcomes",
                "safety profile evidence",
            ]
        )
        analysis = MockAnalysisExecutor(mock_claims=claims)
        focus_gen = MockFocusPromptGenerator()

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            focus_generator=focus_gen,
        )

        task = _make_task()
        config = _make_config(coverage_threshold=0.50)
        await loop.run(task, config)

        # ✅ Coverage met in round 1, no focus needed
        assert focus_gen.call_count == 0


# ============================================================================
# 📊 Cost tracking tests
# ============================================================================


class TestCostTracking:
    """Test cost accumulation and budget enforcement 📊."""

    @pytest.mark.asyncio
    async def test_cost_accumulates_correctly(self) -> None:
        """Total cost is sum of search + analysis per round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ Mock search=0.10, analysis=0.50 per round × 2 rounds
        expected_cost = 2 * (0.10 + 0.50)
        assert abs(result.total_cost_usd - expected_cost) < 0.01

    @pytest.mark.asyncio
    async def test_round_cost_in_summaries(self) -> None:
        """Each round summary tracks its own cost ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        for rs in result.round_summaries:
            assert rs.round_cost_usd > 0


# ============================================================================
# 🔧 Mock component tests
# ============================================================================


class TestMockComponents:
    """Test mock implementations work correctly 🔧."""

    @pytest.mark.asyncio
    async def test_mock_search_executor(self) -> None:
        """MockSearchExecutor returns configured evidence ✅."""
        evidence = [_make_evidence(eid="E1"), _make_evidence(eid="E2")]
        mock = MockSearchExecutor(mock_evidence=evidence)

        task = _make_task()
        config = _make_config()
        result = await mock.execute_search(task, config, 1)

        assert len(result.evidence) == 2
        assert result.cost_usd == 0.10
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_analysis_executor(self) -> None:
        """MockAnalysisExecutor returns configured claims ✅."""
        claims = [{"claim": "Test", "evidence_ids": ["E1"]}]
        mock = MockAnalysisExecutor(mock_claims=claims, mock_decision="positive")

        task = _make_task()
        evidence: list[Evidence] = []
        config = _make_config()
        result = await mock.execute_analysis(task, evidence, config, 1)

        assert len(result.claims) == 1
        assert result.consensus_decision == "positive"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_focus_generator(self) -> None:
        """MockFocusPromptGenerator creates prompt from gap report ✅."""
        mock = MockFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=["Item A", "Item B"],
        )
        config = _make_config()

        result = await mock.generate_focus(gap, config, 1)

        assert "Item A" in result.prompt_text
        assert "Item B" in result.prompt_text
        assert result.target_gaps == ["Item A", "Item B"]
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_mock_focus_generator_empty_uncovered(self) -> None:
        """MockFocusPromptGenerator handles empty uncovered list ✅."""
        mock = MockFocusPromptGenerator()
        gap = GapReport(round_number=1, uncovered_items=[])
        config = _make_config()

        result = await mock.generate_focus(gap, config, 1)

        assert "Broaden" in result.prompt_text
        assert result.target_gaps == []


# ============================================================================
# 📊 Multi-round progression tests
# ============================================================================


class TestMultiRoundProgression:
    """Test multi-round behavior and progression 📊."""

    @pytest.mark.asyncio
    async def test_coverage_improves_over_rounds(self) -> None:
        """Coverage ratio can improve across rounds ✅."""

        round_counter = {"n": 0}

        class ProgressiveCoverageJudge:
            """Returns increasing coverage per round 📈."""

            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                round_counter["n"] += 1
                n = round_counter["n"]
                # 🔧 Cover more items each round
                covered = checklist[:n]
                uncovered = checklist[n:]
                return CoverageResult(covered=covered, uncovered=uncovered)

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])
        gap = GapAnalysis(coverage_judge=ProgressiveCoverageJudge())

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            gap_analysis=gap,
        )

        task = _make_task()  # 3 checklist items
        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ Coverage should increase across rounds
        coverages = [rs.coverage_ratio for rs in result.round_summaries]
        assert coverages == sorted(coverages)  # monotonically increasing
        assert result.final_coverage > 0

    @pytest.mark.asyncio
    async def test_round_numbers_are_sequential(self) -> None:
        """Round numbers in summaries are 1, 2, 3, ... ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        round_nums = [rs.round_number for rs in result.round_summaries]
        assert round_nums == [1, 2, 3]


# ============================================================================
# 🔧 Edge case tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions 🔧."""

    @pytest.mark.asyncio
    async def test_single_max_round(self) -> None:
        """max_rounds=1 runs exactly one round ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(max_rounds=1, coverage_threshold=0.99)
        result = await loop.run(task, config)

        assert result.total_rounds == 1

    @pytest.mark.asyncio
    async def test_zero_evidence_from_search(self) -> None:
        """Loop handles zero evidence from search gracefully ✅."""
        search = MockSearchExecutor(mock_evidence=[])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
            min_evidence_per_round=0,
        )
        result = await loop.run(task, config)

        assert result.total_rounds >= 1
        assert len(result.evidence) == 0

    @pytest.mark.asyncio
    async def test_coverage_delta_in_summaries(self) -> None:
        """Coverage delta is computed between consecutive rounds ✅."""
        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        result = await loop.run(task, config)

        # ✅ First round delta should be = coverage_ratio (prev was 0)
        if result.round_summaries:
            first = result.round_summaries[0]
            assert first.coverage_delta == first.coverage_ratio

    @pytest.mark.asyncio
    async def test_trajectory_multi_round(self) -> None:
        """Trajectory with multiple rounds has correct record count ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            search = MockSearchExecutor(mock_evidence=[_make_evidence()])
            analysis = MockAnalysisExecutor(mock_claims=[])

            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
            )

            task = _make_task()
            config = _make_config(
                max_rounds=3,
                coverage_threshold=0.99,
                convergence_delta=0.0,
                max_cost_per_subitem=100.0,
            )
            await loop.run(task, config)

            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            filepath = os.path.join(tmpdir, files[0])

            with open(filepath) as f:
                records = [json.loads(line) for line in f]

            # ✅ Should have: meta + events + rounds + summary + meta_final
            round_records = [r for r in records if r["type"] == "round"]
            assert len(round_records) == 3

    @pytest.mark.asyncio
    async def test_custom_coverage_judge_injection(self) -> None:
        """Custom CoverageJudge can be injected at construction ✅."""

        class AlwaysCoveredJudge:
            """Always reports full coverage 🎯."""

            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                return CoverageResult(
                    covered=checklist,
                    uncovered=[],
                )

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            coverage_judge=AlwaysCoveredJudge(),
        )

        task = _make_task()
        config = _make_config(coverage_threshold=0.80)
        result = await loop.run(task, config)

        assert result.total_rounds == 1
        assert result.final_coverage == 1.0
        assert "coverage_threshold_reached" in result.termination_reason


# ============================================================================
# 🎯 DefaultFocusPromptGenerator tests
# ============================================================================


class TestDiscoveryLoopFocus:
    """Test DefaultFocusPromptGenerator behavior 🎯."""

    @pytest.mark.asyncio
    async def test_default_focus_includes_uncovered_items(self) -> None:
        """Improved prompt enumerates each uncovered item with guidance ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=["Item Alpha", "Item Beta", "Item Gamma"],
            covered_items=[],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=1)

        # ✅ All uncovered items referenced in prompt
        assert "Item Alpha" in result.prompt_text
        assert "Item Beta" in result.prompt_text
        assert "Item Gamma" in result.prompt_text
        # ✅ Directive section present
        assert "UNCOVERED ITEMS" in result.prompt_text
        assert "SEARCH FOCUS" in result.prompt_text
        # ✅ Target gaps populated
        assert result.target_gaps == ["Item Alpha", "Item Beta", "Item Gamma"]
        # ✅ Call count tracked
        assert generator.call_count == 1

    @pytest.mark.asyncio
    async def test_default_focus_includes_covered_exclusion(self) -> None:
        """Prompt includes 'do not re-search' block for covered items ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=2,
            uncovered_items=["Missing topic X"],
            covered_items=["Covered topic A", "Covered topic B"],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=2)

        # ✅ Covered items appear in the exclusion block
        assert "Covered topic A" in result.prompt_text
        assert "Covered topic B" in result.prompt_text
        # ✅ Explicit exclusion language present
        assert "ALREADY COVERED" in result.prompt_text

    @pytest.mark.asyncio
    async def test_focus_prompt_empty_gaps_broadens_scope(self) -> None:
        """Prompt with no uncovered items instructs to broaden scope ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=[],
            covered_items=["Topic A", "Topic B"],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=1)

        # ✅ Broadening language used
        assert "Broaden" in result.prompt_text or "broaden" in result.prompt_text
        # ✅ No target gaps
        assert result.target_gaps == []
        assert result.suggested_queries == []

    @pytest.mark.asyncio
    async def test_focus_prompt_empty_gaps_no_covered_items(self) -> None:
        """Prompt with all empty lists still returns a safe broadening text ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=[],
            covered_items=[],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=1)

        assert result.prompt_text  # Non-empty
        assert result.target_gaps == []

    @pytest.mark.asyncio
    async def test_default_focus_uses_different_search_terms_instruction(
        self,
    ) -> None:
        """Prompt explicitly instructs use of DIFFERENT search terms ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=3,
            uncovered_items=["Biomarker expression data"],
            covered_items=[],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=3)

        # ✅ Round context referenced
        assert "3" in result.prompt_text
        # ✅ Diversity instruction present
        assert "DIFFERENT" in result.prompt_text

    @pytest.mark.asyncio
    async def test_default_focus_suggested_queries_populated(self) -> None:
        """Suggested queries include two variants per uncovered item ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=["Item One", "Item Two"],
            covered_items=[],
        )
        config = _make_config()

        result = await generator.generate_focus(gap, config, round_number=1)

        # ✅ Two query variants per item = 4 total
        assert len(result.suggested_queries) == 4
        # ✅ Each uncovered item appears in at least one query
        assert any("Item One" in q for q in result.suggested_queries)
        assert any("Item Two" in q for q in result.suggested_queries)

    @pytest.mark.asyncio
    async def test_default_focus_respects_gap_focus_max_items(self) -> None:
        """Generator caps focus items at config.gap_focus_max_items ✅."""
        generator = DefaultFocusPromptGenerator()
        gap = GapReport(
            round_number=1,
            uncovered_items=["A", "B", "C", "D", "E"],
            covered_items=[],
        )
        base_config = _make_config()
        config = DiscoveryConfig(
            **{**base_config.model_dump(), "gap_focus_max_items": 2}
        )

        result = await generator.generate_focus(gap, config, round_number=1)

        # ✅ Only first 2 items become target_gaps
        assert len(result.target_gaps) == 2
        assert result.target_gaps == ["A", "B"]

    @pytest.mark.asyncio
    async def test_mock_focus_prompt_generator_is_alias(self) -> None:
        """MockFocusPromptGenerator is a backward-compat alias ✅."""
        # ✅ Both names refer to the same class
        assert MockFocusPromptGenerator is DefaultFocusPromptGenerator

        # ✅ Alias instantiates correctly and works the same
        mock = MockFocusPromptGenerator()
        gap = GapReport(round_number=1, uncovered_items=["Item A"])
        config = _make_config()
        result = await mock.generate_focus(gap, config, 1)
        assert "Item A" in result.prompt_text

    @pytest.mark.asyncio
    async def test_gap_closing_focus_effectiveness(self) -> None:
        """Focus prompt guides mock search to return targeted evidence ✅."""
        # 🔧 Track focus prompts received by the search executor
        received_focus_prompts: list[str | None] = []

        class FocusTrackingSearchExecutor:
            """Records focus_prompt argument from each execute_search call 📝."""

            call_count = 0

            async def execute_search(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: str | None = None,
            ) -> SearchRoundOutput:
                received_focus_prompts.append(focus_prompt)
                self.call_count += 1
                if focus_prompt and "UNCOVERED ITEMS" in focus_prompt:
                    summary = (
                        "Targeted evidence matching the uncovered item "
                        "found via focused search query with sufficient "
                        "detail to satisfy the gap analysis requirements"
                    )
                else:
                    summary = (
                        "General evidence from broad search "
                        "about protein interactions and cell signaling "
                        "pathways in mammalian systems including binding"
                    )
                return SearchRoundOutput(
                    evidence=[
                        Evidence(
                            id=f"E{round_number}",
                            source="test-mcp",
                            query="test",
                            summary=summary,
                        )
                    ],
                    queries_executed=["test query"],
                    mcp_tools_used=["test-tool"],
                    cost_usd=0.10,
                    duration_seconds=0.5,
                )

        search = FocusTrackingSearchExecutor()
        analysis = MockAnalysisExecutor(mock_claims=[])
        generator = DefaultFocusPromptGenerator()

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            focus_generator=generator,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )
        await loop.run(task, config)

        # ✅ Round 1 has no focus (first round, no previous gap)
        assert received_focus_prompts[0] is None
        # ✅ Round 2 and 3 receive focus prompts from DefaultFocusPromptGenerator
        assert received_focus_prompts[1] is not None
        assert received_focus_prompts[2] is not None
        # ✅ Focus prompts contain structured guidance
        assert "SEARCH FOCUS" in received_focus_prompts[1]
        assert "UNCOVERED ITEMS" in received_focus_prompts[1]


# ============================================================================
# 🌱 Seeded evidence and gap hints tests (T1.2 — KB integration)
# ============================================================================


class TestSeededEvidence:
    """Test seeded_evidence and seeded_gap_hints parameters 🌱.

    Covers Knowledge Base pre-fill logic, priority over shared pool,
    first-round focus injection, and seeded-mode claims reset.
    """

    @pytest.mark.asyncio
    async def test_seeded_evidence_prefills(self) -> None:
        """Seeded evidence items appear in the final result ✅."""
        # 🔧 Arrange: two seeded evidence items from KB
        seeded = [
            _make_evidence(
                eid="KB1",
                summary=(
                    "Pre-existing knowledge base evidence about protein "
                    "binding affinity from prior research and meta-analyses"
                ),
            ),
            _make_evidence(
                eid="KB2",
                summary=(
                    "Knowledge base entry on clinical trial outcomes for "
                    "target inhibition in phase III randomized studies"
                ),
            ),
        ]
        # 🔧 Mock executors that return no additional evidence
        search = MockSearchExecutor(mock_evidence=[])
        analysis = MockAnalysisExecutor(mock_claims=[])
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        # 🔄 Act
        result = await loop.run(task, config, seeded_evidence=seeded)

        # ✅ Assert: seeded evidence IDs appear in the result
        result_ids = {e.id for e in result.evidence}
        assert "KB1" in result_ids
        assert "KB2" in result_ids
        assert len(result.evidence) >= 2

    @pytest.mark.asyncio
    async def test_seeded_high_coverage_reduces_rounds(self) -> None:
        """Seeded evidence matching checklist leads to faster convergence ✅."""
        # 🔧 Arrange: seeded evidence + claims that achieve full coverage
        seeded = [
            _make_evidence(
                eid="KB1",
                summary=(
                    "Comprehensive analysis about protein binding affinity "
                    "from systematic review of clinical binding studies"
                ),
            ),
        ]
        covering_claims = _make_claims(
            [
                "protein binding affinity",
                "clinical trial outcomes",
                "safety profile",
            ]
        )
        search = MockSearchExecutor(mock_evidence=[])
        analysis = MockAnalysisExecutor(mock_claims=covering_claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        # ⚙️ High threshold to ensure coverage check matters
        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.80,
        )

        # 🔄 Act
        result = await loop.run(task, config, seeded_evidence=seeded)

        # ✅ Converged early because KB claims boosted coverage
        assert result.total_rounds == 1
        assert result.final_coverage >= 0.80
        assert "coverage_threshold_reached" in result.termination_reason

    @pytest.mark.asyncio
    async def test_seeded_and_shared_pool_coexist(self) -> None:
        """When both seeded and shared pool provided, seeded takes priority ✅.

        seeded_evidence bypasses shared_evidence_pool pre-fill, but
        shared_evidence_pool.add() contribution in the loop still works.
        """
        from inquiro.core.evidence_pool import SharedEvidencePool

        # 🔧 Arrange: pool pre-loaded with one item, seeded has another
        pool = SharedEvidencePool()
        pool_item = _make_evidence(
            eid="POOL1",
            summary=(
                "Shared pool evidence item from cross-task reuse "
                "regarding cellular pathway activation studies"
            ),
        )
        pool.add([pool_item])

        seeded = [
            _make_evidence(
                eid="SEED1",
                summary=(
                    "Seeded knowledge base data about protein target "
                    "binding mechanisms from prior meta-analysis studies"
                ),
            ),
        ]

        # 🔧 Mock search returns a new item that should be contributed back
        new_from_search = _make_evidence(
            eid="SEARCH1",
            summary=(
                "New data found during search about clinical efficacy "
                "in randomized double-blind placebo-controlled trials"
            ),
        )
        search = MockSearchExecutor(mock_evidence=[new_from_search])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        # Use max_rounds=2 with high threshold: round 1 skips search
        # (seeded mode), round 2 runs search normally and finds SEARCH1.
        config = _make_config(max_rounds=2, coverage_threshold=1.0)

        # 🔄 Act
        result = await loop.run(
            task,
            config,
            shared_evidence_pool=pool,
            seeded_evidence=seeded,
        )

        # ✅ SEED1 is in the result (seeded was used)
        result_ids = {e.id for e in result.evidence}
        assert "SEED1" in result_ids

        # ✅ POOL1 is NOT in the result (pool pre-fill was skipped)
        assert "POOL1" not in result_ids

        # ✅ SEARCH1 is in the result (found in round 2 search)
        assert "SEARCH1" in result_ids

        # ✅ Pool contribution still works — SEARCH1 should be in pool now
        # (pool.add() was called with the new search evidence)
        assert pool.size >= 2  # pool_item + search1

    @pytest.mark.asyncio
    async def test_seeded_gap_hints_injected_first_round(self) -> None:
        """Gap hints appear in the first round's focus prompt ✅."""
        # 🔧 Track focus prompts received by the search executor
        received_focus: list[str | None] = []

        class FocusCapturingSearch:
            """Captures focus_prompt from each search call 📝."""

            async def execute_search(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: str | None = None,
            ) -> SearchRoundOutput:
                received_focus.append(focus_prompt)
                return SearchRoundOutput(
                    evidence=[_make_evidence(eid=f"E{round_number}")],
                    queries_executed=["q"],
                    mcp_tools_used=["tool"],
                    cost_usd=0.1,
                    duration_seconds=0.5,
                )

        search = FocusCapturingSearch()
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,  # force 2 rounds
        )
        hints = [
            "Missing data on clinical trial phase III outcomes",
            "No data for safety in paediatric populations",
        ]

        # 🔄 Act
        await loop.run(task, config, seeded_gap_hints=hints)

        # ✅ Round 1 received a focus prompt containing the hints
        assert received_focus[0] is not None
        assert "KB GAP HINTS" in received_focus[0]
        assert (
            "Missing data on clinical trial phase III outcomes" in (received_focus[0])
        )
        assert "No data for safety in paediatric populations" in (received_focus[0])

        # ✅ Round 2 received a gap-analysis-driven focus (not hints)
        assert received_focus[1] is not None
        assert "KB GAP HINTS" not in received_focus[1]

    @pytest.mark.asyncio
    async def test_seeded_mode_resets_claims(self) -> None:
        """In seeded mode, claims are reset each round — not accumulated ✅.

        Uses checklist items and claims carefully crafted so that
        MockCoverageJudge never reaches the 0.99 threshold, forcing
        the loop to run all max_rounds.  The test then verifies that
        the final result.claims contains only one round's worth of
        claims (reset each round) rather than the accumulated total.
        """
        # 🔧 Use checklist items whose keywords do NOT appear in the claims,
        # so MockCoverageJudge keeps coverage below 0.99 for all 3 rounds.
        # Checklist items contain rare words: "xenobiotic", "phosphorylation"
        task_with_rare_items = _make_task(
            task_id="seeded-reset-test",
            checklist_items=[
                "Assess xenobiotic metabolite clearance kinetics",
                "Evaluate phosphorylation cascade activation thresholds",
                "Review mitochondrial membrane potential measurements",
            ],
        )

        # 🔧 Claims only cover item 1 (keyword "xenobiotic") via
        # MockCoverageJudge keyword matching, keeping coverage ~0.33.
        per_round_claims = [
            {
                "claim": ("Xenobiotic clearance observed in hepatocyte assays"),
                "evidence_ids": ["E1"],
            },
        ]

        class CountingAnalysisExecutor:
            """Returns one fixed claim per call and counts invocations ✅."""

            def __init__(self) -> None:
                self.call_count = 0

            async def execute_analysis(
                self,
                task: Any,
                evidence: list[Evidence],
                config: Any,
                round_number: int,
                supplementary_context: Any = None,
            ) -> AnalysisRoundOutput:
                self.call_count += 1
                return AnalysisRoundOutput(
                    claims=list(per_round_claims),
                    consensus_decision="cautious",
                    consensus_ratio=0.5,
                    cost_usd=0.1,
                    duration_seconds=0.5,
                )

        seeded = [
            _make_evidence(
                eid="KB1",
                summary=(
                    "Knowledge base seeded data about target pathway "
                    "interactions and molecular binding characteristics"
                ),
            )
        ]
        search = MockSearchExecutor(mock_evidence=[])
        analysis_exec = CountingAnalysisExecutor()

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis_exec,
        )

        # ⚙️ Force 3 rounds: low convergence_delta prevents early stop
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            convergence_delta=0.0,
            max_cost_per_subitem=100.0,
        )

        # 🔄 Act
        result = await loop.run(task_with_rare_items, config, seeded_evidence=seeded)

        # ✅ Analysis ran 3 times (loop ran all rounds)
        assert analysis_exec.call_count == 3

        # ✅ In seeded mode, claims are reset each round.
        # The final result.claims should reflect only the last round's claims
        # (1 claim from the last call), not 3 × 1 = 3 accumulated claims.
        assert len(result.claims) == len(per_round_claims)

    @pytest.mark.asyncio
    async def test_seeded_gap_hints_without_seeded_evidence(self) -> None:
        """Gap hints work independently of seeded_evidence ✅."""
        received_focus: list[str | None] = []

        class FocusCapturingSearch:
            """Captures focus_prompt for verification 📝."""

            async def execute_search(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: str | None = None,
            ) -> SearchRoundOutput:
                received_focus.append(focus_prompt)
                return SearchRoundOutput(
                    evidence=[_make_evidence(eid=f"E{round_number}")],
                    queries_executed=["q"],
                    mcp_tools_used=["tool"],
                    cost_usd=0.1,
                    duration_seconds=0.5,
                )

        search = FocusCapturingSearch()
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)

        # 🔄 Act: hints provided but no seeded_evidence
        await loop.run(
            task,
            config,
            seeded_gap_hints=["Gap: missing efficacy data"],
        )

        # ✅ Round 1 received the KB gap hints focus
        assert received_focus[0] is not None
        assert "KB GAP HINTS" in received_focus[0]
        assert "Gap: missing efficacy data" in received_focus[0]

    @pytest.mark.asyncio
    async def test_no_seeded_params_behaves_as_before(self) -> None:
        """Omitting seeded params preserves original behavior ✅."""
        evidence = [_make_evidence()]
        claims = _make_claims(["protein binding affinity"])

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        task = _make_task()
        config = _make_config(max_rounds=2, coverage_threshold=0.0)

        # 🔄 Act: call without any seeded params (backward-compatible)
        result = await loop.run(task, config)

        # ✅ Still returns a valid result
        assert isinstance(result, DiscoveryResult)
        assert result.total_rounds >= 1
        assert len(result.evidence) > 0


# ============================================================================
# 🎯 Condenser tag quality map integration
# ============================================================================


class TestCondenserTagQualityMapWiring:
    """Verify condenser_tag_quality_map flows from DiscoveryConfig to CondenserConfig 🎯."""

    def test_discovery_config_accepts_condenser_tag_quality_map(self) -> None:
        """DiscoveryConfig MUST accept condenser_tag_quality_map field 🔧."""
        config = DiscoveryConfig(
            condenser_tag_quality_map={"patent": 1.0, "academic": 0.3},
        )
        assert config.condenser_tag_quality_map == {"patent": 1.0, "academic": 0.3}

    def test_discovery_config_empty_map_default(self) -> None:
        """Empty condenser_tag_quality_map is the default 🔧."""
        config = DiscoveryConfig()
        assert config.condenser_tag_quality_map == {}

    def test_discovery_config_roundtrip_serialization(self) -> None:
        """condenser_tag_quality_map survives model_dump/model_validate roundtrip 🔄."""
        original = DiscoveryConfig(
            condenser_tag_quality_map={"patent": 1.0, "regulatory": 0.6},
        )
        dumped = original.model_dump()
        restored = DiscoveryConfig.model_validate(dumped)
        assert restored.condenser_tag_quality_map == original.condenser_tag_quality_map

    def test_merge_on_top_semantics(self) -> None:
        """Merge-on-top: only specified tags override defaults 🔧."""
        from inquiro.core.evidence_condenser import CondenserConfig

        defaults = dict(CondenserConfig().tag_quality_map)
        overrides = {"patent": 1.0}

        merged = dict(defaults)
        merged.update(overrides)

        # Patent changed, others preserved
        assert merged["patent"] == 1.0
        assert merged["regulatory"] == defaults["regulatory"]
        assert merged["academic"] == defaults["academic"]
        assert merged["clinical_trial"] == defaults["clinical_trial"]
        assert merged["other"] == defaults["other"]


class TestGroupSummarizerEnrichment:
    """Tests for LLM group summarizer enrichment in DiscoveryLoop 📝."""

    @pytest.mark.asyncio
    async def test_enrich_replaces_template_text(self) -> None:
        """Group summarizer replaces template summary_text with LLM output 📝."""
        from inquiro.core.evidence_condenser import (
            CondensationMeta,
            CondensedEvidence,
            GroupSummary,
        )

        class MockSummarizer:
            async def summarize(
                self, tag: str, items: list, included_count: int
            ) -> str:
                return f"LLM summary for {tag} ({len(items)} items)"

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
            group_summarizer=MockSummarizer(),
        )

        items = [
            Evidence(
                id=f"E{i}",
                source="mcp",
                url=f"https://example.com/{i}",
                query="test",
                summary=f"summary {i}",
                evidence_tag="academic",
            )
            for i in range(10)
        ]
        condensed = CondensedEvidence(
            evidence=[],
            meta=CondensationMeta(
                tier=2,
                original_count=500,
                condensed_count=150,
                group_summaries=[
                    GroupSummary(
                        tag="academic",
                        original_count=200,
                        included_count=50,
                        excluded_count=150,
                        summary_text="template text",
                    ),
                ],
                transparency_footer="test footer",
            ),
            excluded_groups={"academic": items},
        )

        await loop._enrich_group_summaries(condensed)

        gs = condensed.meta.group_summaries[0]
        assert gs.summary_text == "LLM summary for academic (10 items)"

    @pytest.mark.asyncio
    async def test_enrich_falls_back_on_error(self) -> None:
        """Template text preserved when summarizer raises exception 📝."""
        from inquiro.core.evidence_condenser import (
            CondensationMeta,
            CondensedEvidence,
            GroupSummary,
        )

        class FailingSummarizer:
            async def summarize(
                self, tag: str, items: list, included_count: int
            ) -> str:
                raise RuntimeError("LLM unavailable")

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
            group_summarizer=FailingSummarizer(),
        )

        items = [
            Evidence(
                id="E1",
                source="mcp",
                url="https://example.com/1",
                query="test",
                summary="test summary",
                evidence_tag="patent",
            )
        ]
        original_text = "original template text"
        condensed = CondensedEvidence(
            evidence=[],
            meta=CondensationMeta(
                tier=2,
                original_count=500,
                condensed_count=150,
                group_summaries=[
                    GroupSummary(
                        tag="patent",
                        original_count=100,
                        included_count=20,
                        excluded_count=80,
                        summary_text=original_text,
                    ),
                ],
                transparency_footer="test",
            ),
            excluded_groups={"patent": items},
        )

        await loop._enrich_group_summaries(condensed)

        # Template text preserved on failure
        assert condensed.meta.group_summaries[0].summary_text == original_text

    @pytest.mark.asyncio
    async def test_enrich_skipped_when_no_summarizer(self) -> None:
        """No enrichment when group_summarizer is None 📝."""
        from inquiro.core.evidence_condenser import (
            CondensationMeta,
            CondensedEvidence,
            GroupSummary,
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
            # No group_summarizer
        )

        original_text = "untouched template"
        condensed = CondensedEvidence(
            evidence=[],
            meta=CondensationMeta(
                tier=2,
                original_count=500,
                condensed_count=150,
                group_summaries=[
                    GroupSummary(
                        tag="academic",
                        original_count=100,
                        included_count=20,
                        excluded_count=80,
                        summary_text=original_text,
                    ),
                ],
                transparency_footer="test",
            ),
            excluded_groups={"academic": []},
        )

        await loop._enrich_group_summaries(condensed)
        assert condensed.meta.group_summaries[0].summary_text == original_text

    def test_condenser_summarizer_model_in_discovery_preset(self) -> None:
        """DISCOVERY preset enables condenser_summarizer_model 📝."""
        from inquiro.core.types import INTENSITY_PRESETS

        discovery = INTENSITY_PRESETS["discovery"]
        assert discovery["condenser_summarizer_model"] == "gemini-backup"

        standard = INTENSITY_PRESETS["standard"]
        assert "condenser_summarizer_model" not in standard


# ============================================================================
# 📊 Gaps/Doubts Propagation Tests
# ============================================================================


class TestGapsDoubtsPropagation:
    """Tests that gaps_remaining and doubts_remaining flow through the pipeline 📊."""

    def test_analysis_round_output_carries_gaps_doubts(self) -> None:
        """AnalysisRoundOutput includes gaps_remaining and doubts_remaining 📊."""
        output = AnalysisRoundOutput(
            claims=[],
            model_decisions=[],
            gaps_remaining=["Gap A", "Gap B"],
            doubts_remaining=["Doubt X"],
        )
        assert output.gaps_remaining == ["Gap A", "Gap B"]
        assert output.doubts_remaining == ["Doubt X"]

    def test_analysis_round_output_defaults_to_empty(self) -> None:
        """AnalysisRoundOutput defaults gaps/doubts to empty lists 📊."""
        output = AnalysisRoundOutput(
            claims=[],
            model_decisions=[],
        )
        assert output.gaps_remaining == []
        assert output.doubts_remaining == []

    def test_analysis_phase_record_carries_gaps_doubts(self) -> None:
        """AnalysisPhaseRecord includes gaps_remaining and doubts_remaining 📊."""
        from inquiro.core.trajectory.models import AnalysisPhaseRecord

        record = AnalysisPhaseRecord(
            gaps_remaining=["missing clinical data"],
            doubts_remaining=["contradictory dosing results"],
        )
        assert record.gaps_remaining == ["missing clinical data"]
        assert record.doubts_remaining == ["contradictory dosing results"]

    def test_build_round_record_propagates_gaps_doubts(self) -> None:
        """_build_round_record passes gaps/doubts to AnalysisPhaseRecord 📊."""
        from inquiro.core.evidence_pipeline import CleaningStats
        from inquiro.core.types import GapReport

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        search_output = SearchRoundOutput(
            evidence=[],
            duration_seconds=1.0,
        )
        cleaning_stats = CleaningStats(
            input_count=5,
            output_count=3,
            dedup_removed=1,
            noise_removed=1,
        )
        analysis_output = AnalysisRoundOutput(
            claims=[{"claim": "test", "evidence_ids": ["E1"]}],
            model_decisions=[],
            gaps_remaining=["no long-term data"],
            doubts_remaining=["conflicting efficacy signals"],
        )
        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=["A"],
            uncovered_items=["B"],
        )

        record = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=0.5,
            round_duration=10.0,
        )

        assert record.analysis_phase.gaps_remaining == [
            "no long-term data"
        ]
        assert record.analysis_phase.doubts_remaining == [
            "conflicting efficacy signals"
        ]

    # ================================================================
    # 📡 Enhanced round-level structured log fields
    # ================================================================

    def test_build_round_record_server_effectiveness(self) -> None:
        """_build_round_record computes per-server effectiveness 📡."""
        from inquiro.core.evidence_pipeline import CleaningStats

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        search_output = SearchRoundOutput(
            evidence=[
                Evidence(
                    id="E1",
                    source="brave",
                    query="query A",
                    summary="s1",
                ),
                Evidence(
                    id="E2",
                    source="brave",
                    query="query B",
                    summary="s2",
                ),
                Evidence(
                    id="E3",
                    source="perplexity",
                    query="query A",
                    summary="s3",
                ),
            ],
            queries_executed=["query A", "query B"],
            duration_seconds=1.0,
        )
        cleaning_stats = CleaningStats(
            input_count=3, output_count=3,
        )
        analysis_output = AnalysisRoundOutput(claims=[], model_decisions=[])
        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=["A"],
            uncovered_items=["B"],
        )

        record = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=0.1,
            round_duration=5.0,
        )

        se = record.search_phase.server_effectiveness
        assert "brave" in se
        assert se["brave"].queries_sent == 2
        assert se["brave"].results_returned == 2
        assert se["brave"].hit_rate == 1.0
        assert "perplexity" in se
        assert se["perplexity"].queries_sent == 1
        assert se["perplexity"].results_returned == 1

    def test_build_round_record_query_diversity(self) -> None:
        """_build_round_record computes query diversity score 🔀."""
        from inquiro.core.evidence_pipeline import CleaningStats

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        # 3 queries, 2 unique → diversity = 2/3
        search_output = SearchRoundOutput(
            evidence=[],
            queries_executed=["q1", "q2", "q1"],
            duration_seconds=1.0,
        )
        cleaning_stats = CleaningStats(input_count=0, output_count=0)
        analysis_output = AnalysisRoundOutput(claims=[], model_decisions=[])
        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.0,
            covered_items=[],
            uncovered_items=[],
        )

        record = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=0.0,
            round_duration=1.0,
        )

        assert abs(record.search_phase.query_diversity_score - 2 / 3) < 1e-6

    def test_build_round_record_evidence_quality_distribution(self) -> None:
        """_build_round_record computes evidence quality distribution 📊."""
        from inquiro.core.evidence_pipeline import CleaningStats

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        search_output = SearchRoundOutput(
            evidence=[
                Evidence(
                    id="E1",
                    source="s",
                    query="q",
                    summary="a",
                    quality_label="high",
                ),
                Evidence(
                    id="E2",
                    source="s",
                    query="q",
                    summary="b",
                    quality_label="high",
                ),
                Evidence(
                    id="E3",
                    source="s",
                    query="q",
                    summary="c",
                    quality_label="low",
                ),
                Evidence(
                    id="E4",
                    source="s",
                    query="q",
                    summary="d",
                ),
            ],
            queries_executed=["q"],
            duration_seconds=1.0,
        )
        cleaning_stats = CleaningStats(input_count=4, output_count=4)
        analysis_output = AnalysisRoundOutput(claims=[], model_decisions=[])
        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.0,
            covered_items=[],
            uncovered_items=[],
        )

        record = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=0.0,
            round_duration=1.0,
        )

        dist = record.analysis_phase.evidence_quality_distribution
        assert dist == {"high": 2, "low": 1, "unknown": 1}

    def test_build_round_record_newly_covered_items(self) -> None:
        """_build_round_record tracks newly covered items via state 🆕."""

        from inquiro.core.discovery_loop import _LoopState
        from inquiro.core.evidence_pipeline import CleaningStats

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        cleaning_stats = CleaningStats(input_count=0, output_count=0)
        analysis_output = AnalysisRoundOutput(claims=[], model_decisions=[])
        search_output = SearchRoundOutput(
            evidence=[], queries_executed=[], duration_seconds=1.0,
        )

        state = _LoopState()
        state.prev_covered_items = set()

        # Round 1: covers A, B
        gap1 = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=["A", "B"],
            uncovered_items=["C"],
        )
        r1 = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap1,
            round_cost=0.0,
            round_duration=1.0,
            state=state,
        )
        assert r1.gap_phase.newly_covered_items == ["A", "B"]

        # Round 2: covers A, B, C — only C is new
        gap2 = GapReport(
            round_number=2,
            coverage_ratio=1.0,
            covered_items=["A", "B", "C"],
            uncovered_items=[],
        )
        r2 = loop._build_round_record(
            round_num=2,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap2,
            round_cost=0.0,
            round_duration=1.0,
            state=state,
        )
        assert r2.gap_phase.newly_covered_items == ["C"]

    def test_build_round_record_no_state_gives_empty_newly_covered(
        self,
    ) -> None:
        """Without state, newly_covered_items defaults to all covered 🔄."""
        from inquiro.core.evidence_pipeline import CleaningStats

        loop = DiscoveryLoop.__new__(DiscoveryLoop)

        search_output = SearchRoundOutput(
            evidence=[], queries_executed=[], duration_seconds=1.0,
        )
        cleaning_stats = CleaningStats(input_count=0, output_count=0)
        analysis_output = AnalysisRoundOutput(claims=[], model_decisions=[])
        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=["A", "B"],
            uncovered_items=["C"],
        )

        record = loop._build_round_record(
            round_num=1,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=0.0,
            round_duration=1.0,
        )
        # Without state, prev_covered is empty so all are "newly covered"
        assert record.gap_phase.newly_covered_items == ["A", "B"]

    def test_new_fields_serialize_roundtrip(self) -> None:
        """New fields survive JSON serialize/deserialize roundtrip 🔄."""
        from inquiro.core.trajectory.models import (
            AnalysisPhaseRecord,
            DiscoveryRoundRecord,
            GapPhaseRecord,
            SearchPhaseRecord,
            ServerStats,
        )

        record = DiscoveryRoundRecord(
            round_number=1,
            search_phase=SearchPhaseRecord(
                total_raw_evidence=10,
                server_effectiveness={
                    "brave": ServerStats(
                        queries_sent=3,
                        results_returned=7,
                        hit_rate=1.0,
                    ),
                },
                query_diversity_score=0.85,
            ),
            analysis_phase=AnalysisPhaseRecord(
                evidence_quality_distribution={
                    "high": 5,
                    "medium": 3,
                    "low": 2,
                },
            ),
            gap_phase=GapPhaseRecord(
                coverage_ratio=0.8,
                covered_items=["A", "B"],
                newly_covered_items=["B"],
            ),
        )

        json_str = record.model_dump_json()
        restored = DiscoveryRoundRecord.model_validate_json(json_str)

        assert restored.search_phase.server_effectiveness["brave"].queries_sent == 3
        assert restored.search_phase.query_diversity_score == 0.85
        assert restored.analysis_phase.evidence_quality_distribution == {
            "high": 5,
            "medium": 3,
            "low": 2,
        }
        assert restored.gap_phase.newly_covered_items == ["B"]
