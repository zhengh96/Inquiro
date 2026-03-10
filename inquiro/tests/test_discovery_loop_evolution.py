"""Tests for DiscoveryLoop evolution_provider integration 🧪.

Covers the EvolutionProvider injection into DiscoveryLoop:
- Constructor acceptance of evolution_provider
- on_round_complete callback after each round
- get_search_enrichment / get_analysis_enrichment before each phase
- Enrichment state stored on self for adapters
- Graceful error handling (try/except prevents loop breakage)
- None evolution_provider handled without errors
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inquiro.core.discovery_loop import (
    AnalysisRoundOutput,
    DiscoveryLoop,
    MockAnalysisExecutor,
    MockSearchExecutor,
    SearchRoundOutput,
)
from inquiro.core.trajectory.models import DiscoveryRoundRecord
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    DiscoveryConfig,
    Evidence,
    EvaluationTask,
    GapReport,
    RoundMetrics,
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
) -> Evidence:
    """Create a test Evidence instance with valid length 🔧.

    Args:
        eid: Evidence identifier.
        summary: Evidence summary (must be > 50 chars).

    Returns:
        Evidence instance.
    """
    return Evidence(
        id=eid,
        source="test-mcp",
        query="test query",
        summary=summary,
    )


def _make_task(
    task_id: str = "task-evo-001",
    checklist_items: list[str] | None = None,
) -> EvaluationTask:
    """Create a test EvaluationTask 🔧.

    Args:
        task_id: Task identifier.
        checklist_items: Optional checklist item descriptions.

    Returns:
        EvaluationTask instance.
    """
    items = checklist_items or [
        "Assess binding affinity data",
        "Evaluate trial outcomes",
        "Review safety profile",
    ]
    checklist = Checklist(
        required=[
            ChecklistItem(id=f"C{i + 1}", description=desc)
            for i, desc in enumerate(items)
        ]
    )
    return EvaluationTask(
        task_id=task_id,
        topic="Test topic for evolution integration",
        rules="Test evaluation rules",
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
        "max_rounds": 2,
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
    """Create mock claims for coverage matching 🔧.

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


def _make_mock_evolution() -> MagicMock:
    """Create a mock EvolutionProvider 🧬.

    Returns:
        MagicMock implementing EvolutionProvider protocol methods.
    """
    evo = MagicMock()
    evo.get_search_enrichment = MagicMock(
        return_value="## LEARNED SEARCH STRATEGIES\n- Use keyword X"
    )
    evo.get_analysis_enrichment = MagicMock(
        return_value="## LEARNED ANALYSIS PATTERNS\n- Focus on Y"
    )
    evo.on_round_complete = AsyncMock(return_value=None)
    return evo


# ============================================================================
# 📊 Constructor tests
# ============================================================================


class TestEvolutionProviderInit:
    """Test DiscoveryLoop accepts evolution_provider parameter 🧬."""

    def test_accepts_evolution_provider(self) -> None:
        """DiscoveryLoop stores evolution_provider on self._evolution ✅."""
        evo = _make_mock_evolution()
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
            evolution_provider=evo,
        )
        assert loop._evolution is evo

    def test_none_evolution_provider(self) -> None:
        """DiscoveryLoop works without evolution_provider ✅."""
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
        )
        assert loop._evolution is None

    def test_evolution_with_feedback_provider(self) -> None:
        """Evolution and feedback providers coexist without conflict ✅."""
        evo = _make_mock_evolution()
        feedback = MagicMock()
        feedback.get_system_prompt_hints = MagicMock(return_value="")
        feedback.get_focus_hints = MagicMock(return_value="")
        feedback.get_feedback = MagicMock(return_value=None)

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
            evolution_provider=evo,
            feedback_provider=feedback,
        )
        assert loop._evolution is evo
        assert loop._feedback_provider is feedback


# ============================================================================
# 🧬 on_round_complete tests
# ============================================================================


class TestOnRoundComplete:
    """Test on_round_complete is called correctly 🧬."""

    @pytest.mark.asyncio
    async def test_on_round_complete_called(self) -> None:
        """on_round_complete is called after each round with trajectory ✅."""
        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        evo = _make_mock_evolution()

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
                evolution_provider=evo,
            )

            task = _make_task()
            config = _make_config(max_rounds=1)
            await loop.run(task, config)

            # 🧬 on_round_complete should have been called once
            evo.on_round_complete.assert_awaited_once()
            call_args = evo.on_round_complete.call_args
            assert call_args[0][0] == 1  # round_num
            assert isinstance(
                call_args[0][1],
                DiscoveryRoundRecord,
            )
            assert isinstance(call_args[0][2], RoundMetrics)

    @pytest.mark.asyncio
    async def test_on_round_complete_metrics_populated(self) -> None:
        """RoundMetrics fields are correctly populated ✅."""
        evidence = [_make_evidence("E1"), _make_evidence("E2")]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        evo = _make_mock_evolution()

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
                evolution_provider=evo,
            )

            task = _make_task()
            config = _make_config(max_rounds=1)
            await loop.run(task, config)

            # 📊 Check metrics
            call_args = evo.on_round_complete.call_args
            metrics: RoundMetrics = call_args[0][2]
            assert metrics.evidence_count >= 1
            assert metrics.new_evidence_count >= 1
            assert metrics.round_index == 0  # round 1 → index 0
            assert 0.0 <= metrics.coverage <= 1.0

    @pytest.mark.asyncio
    async def test_on_round_complete_called_without_trajectory(
        self,
    ) -> None:
        """on_round_complete IS called even when no trajectory_dir set ✅.

        P1.4 fix: the evolution hook must not depend on trajectory writing
        being enabled.  The DiscoveryRoundRecord is built from in-memory
        data and passed to the hook regardless of trajectory configuration.
        A warning is emitted to inform the operator.
        """
        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        evo = _make_mock_evolution()

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            evolution_provider=evo,
            # No trajectory_dir — hook must still fire (P1.4)
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        await loop.run(task, config)

        # 🧬 on_round_complete IS called even without trajectory recording
        evo.on_round_complete.assert_awaited_once()
        call_args = evo.on_round_complete.call_args
        assert call_args[0][0] == 1  # round_num
        from inquiro.core.trajectory.models import DiscoveryRoundRecord

        assert isinstance(call_args[0][1], DiscoveryRoundRecord)
        from inquiro.core.types import RoundMetrics

        assert isinstance(call_args[0][2], RoundMetrics)


# ============================================================================
# 🧬 Enrichment tests
# ============================================================================


class TestEvolutionEnrichment:
    """Test enrichment methods are called before phases 🧬."""

    @pytest.mark.asyncio
    async def test_search_enrichment_called(self) -> None:
        """get_search_enrichment called before each search round ✅."""
        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        evo = _make_mock_evolution()

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        await loop.run(task, config)

        # 🧬 get_search_enrichment called at least once
        evo.get_search_enrichment.assert_called()
        call_args = evo.get_search_enrichment.call_args
        assert call_args[0][0] == 1  # round_num

    @pytest.mark.asyncio
    async def test_analysis_enrichment_called(self) -> None:
        """get_analysis_enrichment called before each analysis round ✅."""
        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        evo = _make_mock_evolution()

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        await loop.run(task, config)

        evo.get_analysis_enrichment.assert_called()

    @pytest.mark.asyncio
    async def test_enrichment_stored_on_self(self) -> None:
        """Enrichment values stored as instance attributes ✅."""
        evo = _make_mock_evolution()
        evo.get_search_enrichment.return_value = "search hint"
        evo.get_analysis_enrichment.return_value = "analysis hint"

        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        await loop.run(task, config)

        # 🧬 After run, enrichment attrs should exist
        assert hasattr(loop, "_current_search_enrichment")
        assert hasattr(loop, "_current_analysis_enrichment")

    @pytest.mark.asyncio
    async def test_multi_round_enrichment_uses_gap_items(self) -> None:
        """Second round passes uncovered items from gap report 🧬."""
        evidence = [_make_evidence()]
        # 🔧 Claims that don't cover all items → forces round 2
        claims = _make_claims(["binding affinity"])

        evo = _make_mock_evolution()

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=2)
        await loop.run(task, config)

        # 🧬 get_search_enrichment should be called twice
        assert evo.get_search_enrichment.call_count == 2

        # 📊 First call: no gap_report yet → empty list
        first_call = evo.get_search_enrichment.call_args_list[0]
        assert first_call[0][0] == 1  # round 1
        assert first_call[0][1] == []  # no gaps yet

        # 📊 Second call: gap_report exists → may have uncovered items
        second_call = evo.get_search_enrichment.call_args_list[1]
        assert second_call[0][0] == 2  # round 2


# ============================================================================
# 🧬 Error handling tests
# ============================================================================


class TestEvolutionErrorHandling:
    """Test evolution errors don't break the loop 🧬."""

    @pytest.mark.asyncio
    async def test_search_enrichment_failure_continues(self) -> None:
        """Search enrichment failure doesn't prevent search ✅."""
        evo = _make_mock_evolution()
        evo.get_search_enrichment.side_effect = RuntimeError("boom")

        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        result = await loop.run(task, config)

        # ✅ Loop completed despite enrichment failure
        assert result is not None
        assert len(result.round_summaries) == 1

    @pytest.mark.asyncio
    async def test_analysis_enrichment_failure_continues(self) -> None:
        """Analysis enrichment failure doesn't prevent analysis ✅."""
        evo = _make_mock_evolution()
        evo.get_analysis_enrichment.side_effect = RuntimeError("boom")

        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
            evolution_provider=evo,
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        result = await loop.run(task, config)

        assert result is not None
        assert len(result.round_summaries) == 1

    @pytest.mark.asyncio
    async def test_on_round_complete_failure_continues(self) -> None:
        """on_round_complete failure doesn't stop the loop ✅."""
        evo = _make_mock_evolution()
        evo.on_round_complete.side_effect = RuntimeError("boom")

        evidence = [_make_evidence()]
        claims = _make_claims(["binding affinity"])

        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = DiscoveryLoop(
                search_executor=search,
                analysis_executor=analysis,
                trajectory_dir=tmpdir,
                evolution_provider=evo,
            )

            task = _make_task()
            config = _make_config(max_rounds=2)
            result = await loop.run(task, config)

            # ✅ Loop continued despite on_round_complete failure
            assert result is not None
            assert len(result.round_summaries) == 2

    @pytest.mark.asyncio
    async def test_no_evolution_provider_runs_cleanly(self) -> None:
        """Loop runs without any evolution_provider ✅."""
        evidence = [_make_evidence()]
        claims = _make_claims(
            [
                "binding affinity",
                "trial outcomes",
                "safety profile",
            ]
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
        )

        task = _make_task()
        config = _make_config(max_rounds=1)
        result = await loop.run(task, config)

        assert result is not None
        assert len(result.round_summaries) == 1


# ============================================================================
# 📊 _build_round_record return value tests
# ============================================================================


class TestBuildRoundRecordReturn:
    """Test _build_round_record returns the record 📊."""

    def test_returns_record_without_writer(self) -> None:
        """Returns DiscoveryRoundRecord even when no trajectory_writer is set ✅.

        _build_round_record always returns the record so that the evolution
        hook can consume it regardless of trajectory config.
        """
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(),
        )

        result = loop._build_round_record(
            round_num=1,
            search_output=SearchRoundOutput(cost_usd=0.0),
            cleaning_stats=MagicMock(
                input_count=0,
                output_count=0,
                dedup_removed=0,
                noise_removed=0,
                tag_distribution={},
            ),
            analysis_output=AnalysisRoundOutput(),
            gap_report=GapReport(
                round_number=1,
                coverage_ratio=0.5,
                covered_items=["A"],
                uncovered_items=["B"],
            ),
            round_cost=0.0,
            round_duration=1.0,
        )
        # Must return the record even without a writer
        assert isinstance(result, DiscoveryRoundRecord)
        assert result.round_number == 1

    def test_returns_record_with_writer(self) -> None:
        """Returns DiscoveryRoundRecord when trajectory_writer exists ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loop = DiscoveryLoop(
                search_executor=MockSearchExecutor(),
                analysis_executor=MockAnalysisExecutor(),
                trajectory_dir=tmpdir,
            )

            # 🔧 Manually initialize trajectory writer
            task = _make_task()
            config = _make_config()
            loop._init_trajectory(
                task.task_id,
                "traj-001",
                config,
                task,
            )

            result = loop._build_round_record(
                round_num=1,
                search_output=SearchRoundOutput(cost_usd=0.1),
                cleaning_stats=MagicMock(
                    input_count=5,
                    output_count=3,
                    dedup_removed=1,
                    noise_removed=1,
                    tag_distribution={},
                ),
                analysis_output=AnalysisRoundOutput(),
                gap_report=GapReport(
                    round_number=1,
                    coverage_ratio=0.6,
                    covered_items=["A"],
                    uncovered_items=["B"],
                ),
                round_cost=0.1,
                round_duration=2.0,
            )
            assert isinstance(result, DiscoveryRoundRecord)
            assert result.round_number == 1
            assert result.round_cost_usd == 0.1
