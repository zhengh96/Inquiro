"""Tests for Discovery pipeline SSE progress events 🧪.

Validates that DiscoveryLoop correctly emits SSE events via the
EventEmitter at each lifecycle point:
1. DISCOVERY_STARTED at loop begin
2. DISCOVERY_ROUND_STARTED at each round start
3. DISCOVERY_ROUND_COMPLETED at each round end with data
4. DISCOVERY_COVERAGE_UPDATED with coverage curve accumulation
5. DISCOVERY_CONVERGED when convergence is reached
6. DISCOVERY_COMPLETED at pipeline finish
7. Full lifecycle ordering verification
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from inquiro.core.discovery_loop import (
    DiscoveryLoop,
    MockAnalysisExecutor,
    MockFocusPromptGenerator,
    MockSearchExecutor,
)
from inquiro.core.gap_analysis import (
    CoverageResult,
    GapAnalysis,
)
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    DiscoveryConfig,
    Evidence,
    EvaluationTask,
)
from inquiro.infrastructure.event_emitter import (
    EventEmitter,
    InquiroEvent,
)


# ============================================================================
# 🔧 Test helpers
# ============================================================================


def _make_evidence(
    eid: str = "E1",
    summary: str = (
        "This is a sufficiently long test evidence summary about "
        "protein interactions and binding affinities in cell cultures"
    ),
    source: str = "test-mcp",
    query: str = "test query",
) -> Evidence:
    """Create a test Evidence instance with valid length 🔧.

    Args:
        eid: Evidence identifier.
        summary: Evidence summary (must be > 50 chars).
        source: Evidence source name.
        query: Query that produced this evidence.

    Returns:
        Evidence instance.
    """
    return Evidence(
        id=eid,
        source=source,
        query=query,
        summary=summary,
    )


def _make_task(
    task_id: str = "task-sse-001",
    topic: str = "Test topic for SSE events",
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
        rules="Test evaluation rules",
        checklist=checklist,
        output_schema={},
    )


def _make_config(**kwargs: Any) -> DiscoveryConfig:
    """Create a DiscoveryConfig with test-friendly defaults 🔧.

    Args:
        **kwargs: Overrides for DiscoveryConfig fields.

    Returns:
        DiscoveryConfig with small limits for fast tests.
    """
    defaults: dict[str, Any] = {
        "max_rounds": 3,
        "coverage_threshold": 0.8,
        "max_cost_per_subitem": 5.0,
        "timeout_per_round": 30,
    }
    defaults.update(kwargs)
    return DiscoveryConfig(**defaults)


class _ConvergeOnRound2Judge:
    """Custom coverage judge that converges at round 2 🧪.

    Round 1: covers 1/3 items (0.33 ratio).
    Round 2: covers 3/3 items (1.0 ratio) -> convergence.
    """

    def __init__(self) -> None:
        """Initialize with call counter 🔧."""
        self._call_count = 0

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Return progressive coverage 🔧.

        Args:
            checklist: Checklist items.
            claims: Claims (unused).
            evidence: Evidence (unused).

        Returns:
            CoverageResult with increasing coverage.
        """
        self._call_count += 1
        if self._call_count == 1:
            # 📊 Round 1: partial coverage
            return CoverageResult(
                covered=checklist[:1],
                uncovered=checklist[1:],
            )
        else:
            # 📊 Round 2+: full coverage
            return CoverageResult(
                covered=checklist,
                uncovered=[],
            )


class _NeverConvergeJudge:
    """Custom coverage judge that never reaches the threshold 🧪.

    Always covers only the first item (1/3 = 0.33).
    """

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Return consistently low coverage 🔧.

        Args:
            checklist: Checklist items.
            claims: Claims (unused).
            evidence: Evidence (unused).

        Returns:
            CoverageResult with only first item covered.
        """
        return CoverageResult(
            covered=checklist[:1] if checklist else [],
            uncovered=checklist[1:] if checklist else [],
        )


def _build_loop(
    mock_emitter: MagicMock,
    mock_evidence: list[Evidence] | None = None,
    converge: bool = True,
) -> DiscoveryLoop:
    """Build a DiscoveryLoop wired to a mock EventEmitter 🔧.

    Args:
        mock_emitter: Mock EventEmitter to capture SSE events.
        mock_evidence: Evidence for the mock search executor.
        converge: If True, use a judge that converges at round 2.
            If False, use a judge that never converges.

    Returns:
        Configured DiscoveryLoop instance.
    """
    evidence = mock_evidence or [_make_evidence()]

    judge = _ConvergeOnRound2Judge() if converge else _NeverConvergeJudge()
    gap_analysis = GapAnalysis(coverage_judge=judge)

    return DiscoveryLoop(
        search_executor=MockSearchExecutor(mock_evidence=evidence),
        analysis_executor=MockAnalysisExecutor(),
        gap_analysis=gap_analysis,
        focus_generator=MockFocusPromptGenerator(),
        event_emitter=mock_emitter,
    )


def _extract_calls(
    mock_emitter: MagicMock,
    event_type: InquiroEvent,
) -> list[dict[str, Any]]:
    """Extract emit() calls for a specific event type from mock 🔍.

    Args:
        mock_emitter: Mock EventEmitter whose emit() was recorded.
        event_type: InquiroEvent to filter for.

    Returns:
        List of data dicts passed to emit() for matching events.
    """
    result = []
    for c in mock_emitter.emit.call_args_list:
        args = c[0] if c[0] else ()
        kwargs = c[1] if c[1] else {}
        # 📡 emit(event_type, task_id, data)
        call_event = args[0] if len(args) > 0 else kwargs.get("event_type")
        call_data = args[2] if len(args) > 2 else kwargs.get("data", {})
        if call_event == event_type:
            result.append(call_data or {})
    return result


def _extract_discovery_event_types(
    mock_emitter: MagicMock,
) -> list[InquiroEvent]:
    """Extract all discovery SSE event types in emission order 📋.

    Args:
        mock_emitter: Mock EventEmitter whose emit() was recorded.

    Returns:
        Ordered list of InquiroEvent enums for discovery events only.
    """
    events = []
    for c in mock_emitter.emit.call_args_list:
        args = c[0] if c[0] else ()
        event_type = args[0] if len(args) > 0 else None
        if isinstance(event_type, InquiroEvent) and event_type.value.startswith(
            "discovery_"
        ):
            events.append(event_type)
    return events


# ============================================================================
# 🧪 Test: DISCOVERY_STARTED event
# ============================================================================


class TestDiscoveryStartedEvent:
    """Verify DISCOVERY_STARTED is emitted at loop begin 📡."""

    @pytest.mark.asyncio
    async def test_discovery_started_emitted_once(self) -> None:
        """DISCOVERY_STARTED MUST be emitted exactly once at the
        start of the discovery loop 📡.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=3)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_STARTED)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_discovery_started_contains_config_data(self) -> None:
        """DISCOVERY_STARTED data MUST include max_rounds and
        coverage_threshold from the config 📡.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5, coverage_threshold=0.75)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_STARTED)
        assert len(calls) == 1
        assert calls[0]["max_rounds"] == 5
        assert calls[0]["coverage_threshold"] == 0.75


# ============================================================================
# 🧪 Test: DISCOVERY_ROUND_STARTED event
# ============================================================================


class TestDiscoveryRoundStartedEvent:
    """Verify DISCOVERY_ROUND_STARTED is emitted at each round 📡."""

    @pytest.mark.asyncio
    async def test_round_started_emitted_per_round(self) -> None:
        """DISCOVERY_ROUND_STARTED MUST be emitted once per round 📡."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_ROUND_STARTED)
        # 🔧 Converges at round 2 => 2 round starts
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_round_started_contains_round_number(self) -> None:
        """DISCOVERY_ROUND_STARTED data MUST include round_number 📡."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_ROUND_STARTED)
        assert calls[0]["round_number"] == 1
        assert calls[1]["round_number"] == 2


# ============================================================================
# 🧪 Test: DISCOVERY_ROUND_COMPLETED event
# ============================================================================


class TestDiscoveryRoundCompletedEvent:
    """Verify DISCOVERY_ROUND_COMPLETED with correct data 📊."""

    @pytest.mark.asyncio
    async def test_round_completed_emitted_per_round(self) -> None:
        """DISCOVERY_ROUND_COMPLETED MUST be emitted once per round 📊."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_ROUND_COMPLETED)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_round_completed_contains_expected_fields(
        self,
    ) -> None:
        """DISCOVERY_ROUND_COMPLETED data MUST include round_number,
        coverage, cost_usd, and evidence_count 📊.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_ROUND_COMPLETED)

        # 📊 Round 1 data
        r1 = calls[0]
        assert r1["round_number"] == 1
        assert "coverage" in r1
        assert "cost_usd" in r1
        assert "evidence_count" in r1
        # 📊 1/3 covered = ~0.333
        assert r1["coverage"] == pytest.approx(1.0 / 3.0, abs=0.01)

        # 📊 Round 2 data (full coverage)
        r2 = calls[1]
        assert r2["round_number"] == 2
        assert r2["coverage"] == pytest.approx(1.0)
        assert r2["evidence_count"] > 0

    @pytest.mark.asyncio
    async def test_round_completed_cost_accumulates(self) -> None:
        """DISCOVERY_ROUND_COMPLETED cost MUST be cumulative 📊."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_ROUND_COMPLETED)
        # 📊 Cost should increase across rounds
        assert calls[1]["cost_usd"] >= calls[0]["cost_usd"]


# ============================================================================
# 🧪 Test: DISCOVERY_COVERAGE_UPDATED event
# ============================================================================


class TestDiscoveryCoverageUpdatedEvent:
    """Verify DISCOVERY_COVERAGE_UPDATED with coverage curve 📈."""

    @pytest.mark.asyncio
    async def test_coverage_updated_emitted_per_round(self) -> None:
        """DISCOVERY_COVERAGE_UPDATED MUST be emitted once per
        round 📈.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_COVERAGE_UPDATED)
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_coverage_curve_accumulates(self) -> None:
        """Coverage curve MUST grow with each round 📈."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_COVERAGE_UPDATED)

        # 📈 Round 1: curve has 1 entry
        assert len(calls[0]["coverage_curve"]) == 1
        assert calls[0]["coverage_curve"][0] == pytest.approx(1.0 / 3.0, abs=0.01)

        # 📈 Round 2: curve has 2 entries
        assert len(calls[1]["coverage_curve"]) == 2
        assert calls[1]["coverage_curve"][0] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert calls[1]["coverage_curve"][1] == pytest.approx(1.0)


# ============================================================================
# 🧪 Test: DISCOVERY_CONVERGED event
# ============================================================================


class TestDiscoveryConvergedEvent:
    """Verify DISCOVERY_CONVERGED when convergence is reached 🎯."""

    @pytest.mark.asyncio
    async def test_converged_emitted_on_convergence(self) -> None:
        """DISCOVERY_CONVERGED MUST be emitted when the loop
        converges 🎯.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=True)
        task = _make_task()
        config = _make_config(max_rounds=5, coverage_threshold=0.8)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_CONVERGED)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_converged_includes_reason_and_rounds(self) -> None:
        """DISCOVERY_CONVERGED data MUST include reason and
        total_rounds 🎯.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=True)
        task = _make_task()
        config = _make_config(max_rounds=5, coverage_threshold=0.8)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_CONVERGED)
        assert len(calls) == 1
        assert "reason" in calls[0]
        assert calls[0]["reason"] != ""
        assert "total_rounds" in calls[0]
        assert calls[0]["total_rounds"] == 2

    @pytest.mark.asyncio
    async def test_converged_emitted_with_max_rounds_reason(
        self,
    ) -> None:
        """DISCOVERY_CONVERGED MUST be emitted with reason
        'max_rounds_reached' when loop ends due to max_rounds
        exhaustion (GapAnalysis treats this as convergence) 🎯.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=False)
        task = _make_task()
        config = _make_config(max_rounds=2, coverage_threshold=0.9)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_CONVERGED)
        # 📡 GapAnalysis reports max_rounds as a convergence reason
        assert len(calls) == 1
        assert calls[0]["reason"] == "max_rounds_reached"
        assert calls[0]["total_rounds"] == 2


# ============================================================================
# 🧪 Test: DISCOVERY_COMPLETED event
# ============================================================================


class TestDiscoveryCompletedEvent:
    """Verify DISCOVERY_COMPLETED at pipeline finish ✅."""

    @pytest.mark.asyncio
    async def test_completed_emitted_once(self) -> None:
        """DISCOVERY_COMPLETED MUST be emitted exactly once ✅."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_COMPLETED)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_completed_contains_summary_data(self) -> None:
        """DISCOVERY_COMPLETED data MUST include full summary
        fields ✅.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task()
        config = _make_config(max_rounds=5)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_COMPLETED)
        data = calls[0]
        assert "total_rounds" in data
        assert "final_coverage" in data
        assert "total_cost_usd" in data
        assert "total_evidence" in data
        assert "termination_reason" in data
        assert data["total_rounds"] == 2
        assert data["final_coverage"] == pytest.approx(1.0)
        assert data["total_evidence"] > 0

    @pytest.mark.asyncio
    async def test_completed_emitted_on_max_rounds(self) -> None:
        """DISCOVERY_COMPLETED MUST be emitted when loop ends
        due to max_rounds (GapAnalysis-triggered convergence) ✅.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=False)
        task = _make_task()
        config = _make_config(max_rounds=2, coverage_threshold=0.95)

        await loop.run(task, config)

        calls = _extract_calls(mock_emitter, InquiroEvent.DISCOVERY_COMPLETED)
        assert len(calls) == 1
        assert calls[0]["termination_reason"] == "max_rounds_reached"


# ============================================================================
# 🧪 Test: No emitter (graceful skip)
# ============================================================================


class TestNoEmitterGracefulSkip:
    """Verify loop works without an event emitter 🔇."""

    @pytest.mark.asyncio
    async def test_run_without_emitter_succeeds(self) -> None:
        """DiscoveryLoop MUST run successfully with no event
        emitter configured 🔇.
        """
        _loop = _build_loop(MagicMock(spec=EventEmitter))

        # 🔧 Rebuild without the emitter
        loop_no_emitter = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=[_make_evidence()]),
            analysis_executor=MockAnalysisExecutor(),
            gap_analysis=GapAnalysis(coverage_judge=_ConvergeOnRound2Judge()),
            focus_generator=MockFocusPromptGenerator(),
            event_emitter=None,
        )
        task = _make_task()
        config = _make_config(max_rounds=3)

        result = await loop_no_emitter.run(task, config)

        # ✅ Should complete normally without errors
        assert result.total_rounds >= 1
        assert result.final_coverage > 0.0


# ============================================================================
# 🧪 Test: Full lifecycle ordering
# ============================================================================


class TestFullLifecycleOrdering:
    """Verify end-to-end event ordering for the lifecycle 📋."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_event_order_with_convergence(
        self,
    ) -> None:
        """Events MUST follow the order: started -> (round_started ->
        round_completed -> coverage_updated)* -> converged ->
        completed 📋.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=True)
        task = _make_task()
        config = _make_config(max_rounds=5, coverage_threshold=0.8)

        await loop.run(task, config)

        events = _extract_discovery_event_types(mock_emitter)

        # 📋 Expected order for 2-round convergence
        expected = [
            InquiroEvent.DISCOVERY_STARTED,
            InquiroEvent.DISCOVERY_ROUND_STARTED,
            InquiroEvent.DISCOVERY_ROUND_COMPLETED,
            InquiroEvent.DISCOVERY_COVERAGE_UPDATED,
            InquiroEvent.DISCOVERY_ROUND_STARTED,
            InquiroEvent.DISCOVERY_ROUND_COMPLETED,
            InquiroEvent.DISCOVERY_COVERAGE_UPDATED,
            InquiroEvent.DISCOVERY_CONVERGED,
            InquiroEvent.DISCOVERY_COMPLETED,
        ]
        assert events == expected

    @pytest.mark.asyncio
    async def test_full_lifecycle_max_rounds_includes_converged(
        self,
    ) -> None:
        """When max_rounds is exhausted, CONVERGED MUST be emitted
        (GapAnalysis treats max_rounds as a convergence reason),
        followed by COMPLETED 📋.
        """
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter, converge=False)
        task = _make_task()
        config = _make_config(max_rounds=1, coverage_threshold=0.9)

        await loop.run(task, config)

        events = _extract_discovery_event_types(mock_emitter)

        # 📋 Expected: CONVERGED present with max_rounds reason
        expected = [
            InquiroEvent.DISCOVERY_STARTED,
            InquiroEvent.DISCOVERY_ROUND_STARTED,
            InquiroEvent.DISCOVERY_ROUND_COMPLETED,
            InquiroEvent.DISCOVERY_COVERAGE_UPDATED,
            InquiroEvent.DISCOVERY_CONVERGED,
            InquiroEvent.DISCOVERY_COMPLETED,
        ]
        assert events == expected

    @pytest.mark.asyncio
    async def test_task_id_present_in_all_events(self) -> None:
        """All SSE events MUST include the correct task_id 📋."""
        mock_emitter = MagicMock(spec=EventEmitter)
        loop = _build_loop(mock_emitter)
        task = _make_task(task_id="task-id-check")
        config = _make_config(max_rounds=3)

        await loop.run(task, config)

        # 📋 Check that task_id is always the second positional arg
        for c in mock_emitter.emit.call_args_list:
            args = c[0] if c[0] else ()
            event_type = args[0] if len(args) > 0 else None
            if isinstance(event_type, InquiroEvent) and event_type.value.startswith(
                "discovery_"
            ):
                task_id_arg = args[1] if len(args) > 1 else None
                assert task_id_arg == "task-id-check", (
                    f"task_id mismatch for {event_type}: got {task_id_arg!r}"
                )
