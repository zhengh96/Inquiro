"""Unit tests for ParallelSearchOrchestrator 🧪.

Tests cover:
    - Fallback when enable_parallel_search is False
    - Fallback when query_strategy is None
    - Fallback when only 1 query section exists
    - Parallel dispatch with multiple sections
    - Sub-task splitting preserves alias_expansion, reduces to 1 section each
    - Evidence deduplication across merged results (same URL)
    - Cost summing and duration max across merged results
    - Partial failure handling (fail-open: failed sections are skipped with warning)
    - All sections fail → empty SearchRoundOutput returned
    - asyncio.Semaphore limits concurrent searches to max_parallel
    - _evidence_key deduplication helper behaviour
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from inquiro.core.discovery_loop import SearchRoundOutput
from inquiro.core.types import DiscoveryConfig, Evidence
from inquiro.exps.parallel_search_exp import (
    ParallelSearchOrchestrator,
    _evidence_key,
)


# ============================================================================
# 🏭 Helpers
# ============================================================================


def _make_config(**kwargs: Any) -> DiscoveryConfig:
    """Build a DiscoveryConfig for testing 🔧."""
    defaults: dict[str, Any] = {
        "enable_parallel_search": True,
        "max_parallel_agents": 3,
    }
    defaults.update(kwargs)
    return DiscoveryConfig(**defaults)


def _make_task(
    num_sections: int = 2,
    *,
    include_strategy: bool = True,
) -> Any:
    """Build a minimal EvaluationTask-like object for testing 🔬.

    Uses the real EvaluationTask from types so model_copy() works.
    """
    from inquiro.core.types import (
        AgentConfig,
        CostGuardConfig,
        EvaluationTask,
        QualityGateConfig,
    )

    strategy: dict[str, Any] | None = None
    if include_strategy and num_sections > 0:
        strategy = {
            "sub_item_id": "si_001",
            "alias_expansion": "EGFR; ErbB-1; HER1",
            "query_sections": [
                {
                    "id": f"sec_{i}",
                    "priority": i,
                    "tool_name": "perplexity",
                    "description": f"Section {i}",
                    "content": f"query content {i}",
                }
                for i in range(num_sections)
            ],
            "tool_allocations": [],
            "follow_up_rules": "",
            "evidence_tiers": "",
        }

    return EvaluationTask(
        task_id="task-parallel-test",
        topic="Test topic",
        query_strategy=strategy,
        agent_config=AgentConfig(model="test-model", max_turns=5),
        quality_gate=QualityGateConfig(),
        cost_guard=CostGuardConfig(),
    )


def _make_output(
    evidence_ids: list[str],
    cost: float = 1.0,
    duration: float = 10.0,
    trajectory_ref: str | None = None,
) -> SearchRoundOutput:
    """Build a SearchRoundOutput with simple evidence items 📊."""
    evidence = [
        Evidence(
            id=eid,
            source="test",
            url=f"https://example.com/{eid}",
            query="q",
            summary=f"summary for {eid}",
        )
        for eid in evidence_ids
    ]
    return SearchRoundOutput(
        evidence=evidence,
        queries_executed=[f"query_{eid}" for eid in evidence_ids],
        mcp_tools_used=["perplexity"],
        cost_usd=cost,
        duration_seconds=duration,
        agent_trajectory_ref=trajectory_ref,
    )


async def _noop_single_fn(
    task: Any,
    config: Any,
    round_number: int,
    focus_prompt: str | None,
) -> SearchRoundOutput:
    """Trivial single-search stub returning empty output 🔍."""
    return SearchRoundOutput()


# ============================================================================
# 🔍 _should_parallelize tests
# ============================================================================


class TestShouldParallelize:
    """Tests for the parallel eligibility check 🔍."""

    def test_returns_false_when_parallel_disabled(self) -> None:
        """_should_parallelize returns False when enable_parallel_search=False."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        config = _make_config(enable_parallel_search=False)
        assert orch._should_parallelize(task, config) is False

    def test_returns_false_when_query_strategy_is_none(self) -> None:
        """_should_parallelize returns False when query_strategy is None."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(include_strategy=False)
        config = _make_config(enable_parallel_search=True)
        assert orch._should_parallelize(task, config) is False

    def test_returns_false_when_only_one_section(self) -> None:
        """_should_parallelize returns False when only 1 query section."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=1)
        config = _make_config(enable_parallel_search=True)
        assert orch._should_parallelize(task, config) is False

    def test_returns_true_with_two_sections(self) -> None:
        """_should_parallelize returns True with 2 sections and flag enabled."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=2)
        config = _make_config(enable_parallel_search=True)
        assert orch._should_parallelize(task, config) is True

    def test_returns_true_with_many_sections(self) -> None:
        """_should_parallelize returns True with many sections."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=5)
        config = _make_config(enable_parallel_search=True)
        assert orch._should_parallelize(task, config) is True


# ============================================================================
# 📋 _split_task_by_sections tests
# ============================================================================


class TestSplitTaskBySections:
    """Tests for per-section task splitting 📋."""

    def test_produces_one_sub_task_per_section(self) -> None:
        """Splitting a 3-section strategy yields 3 sub-tasks."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        sub_tasks = orch._split_task_by_sections(task)
        assert len(sub_tasks) == 3

    def test_each_sub_task_has_exactly_one_section(self) -> None:
        """Each sub-task query_strategy contains exactly 1 query section."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=4)
        sub_tasks = orch._split_task_by_sections(task)
        for st in sub_tasks:
            assert st.query_strategy is not None
            sections = st.query_strategy.get("query_sections", [])
            assert len(sections) == 1

    def test_sub_tasks_retain_alias_expansion(self) -> None:
        """Each sub-task keeps the full alias_expansion string."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        original_alias = task.query_strategy["alias_expansion"]
        sub_tasks = orch._split_task_by_sections(task)
        for st in sub_tasks:
            assert st.query_strategy is not None
            assert st.query_strategy["alias_expansion"] == original_alias

    def test_sub_tasks_preserve_original_task_fields(self) -> None:
        """Sub-tasks share the same task_id, topic, and other fields."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=2)
        sub_tasks = orch._split_task_by_sections(task)
        for st in sub_tasks:
            assert st.task_id == task.task_id
            assert st.topic == task.topic

    def test_sections_are_different_across_sub_tasks(self) -> None:
        """Each sub-task receives a different section (by section id)."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        sub_tasks = orch._split_task_by_sections(task)
        section_ids = [st.query_strategy["query_sections"][0]["id"] for st in sub_tasks]
        # All section ids are distinct
        assert len(set(section_ids)) == 3

    def test_original_task_is_not_mutated(self) -> None:
        """Splitting does not mutate the original task's query_strategy."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        original_sections = list(task.query_strategy["query_sections"])
        orch._split_task_by_sections(task)
        assert task.query_strategy["query_sections"] == original_sections


# ============================================================================
# 🔗 _merge_results tests
# ============================================================================


class TestMergeResults:
    """Tests for result merging with deduplication 🔗."""

    def test_combines_evidence_from_all_results(self) -> None:
        """Merge combines non-duplicate evidence from all sub-results."""
        orch = ParallelSearchOrchestrator()
        r1 = _make_output(["E1", "E2"])
        r2 = _make_output(["E3", "E4"])
        merged = orch._merge_results([r1, r2])
        assert len(merged.evidence) == 4
        ids = {ev.id for ev in merged.evidence}
        assert ids == {"E1", "E2", "E3", "E4"}

    def test_deduplicates_evidence_by_url_and_summary(self) -> None:
        """Evidence with the same URL+summary is deduplicated."""
        orch = ParallelSearchOrchestrator()
        # E1 appears in both results with identical url+summary
        ev_shared = Evidence(
            id="E1",
            source="test",
            url="https://example.com/shared",
            query="q",
            summary="same summary",
        )
        r1 = SearchRoundOutput(evidence=[ev_shared], cost_usd=1.0, duration_seconds=5.0)
        r2 = SearchRoundOutput(
            evidence=[
                ev_shared,  # duplicate
                Evidence(
                    id="E2",
                    source="test",
                    url="https://example.com/unique",
                    query="q",
                    summary="different summary",
                ),
            ],
            cost_usd=1.0,
            duration_seconds=6.0,
        )
        merged = orch._merge_results([r1, r2])
        assert len(merged.evidence) == 2  # E1 deduplicated, E2 added

    def test_cost_is_summed(self) -> None:
        """Merged cost_usd equals sum of all sub-result costs."""
        orch = ParallelSearchOrchestrator()
        r1 = _make_output(["E1"], cost=1.5)
        r2 = _make_output(["E2"], cost=2.5)
        r3 = _make_output(["E3"], cost=0.7)
        merged = orch._merge_results([r1, r2, r3])
        assert abs(merged.cost_usd - 4.7) < 1e-9

    def test_duration_is_max(self) -> None:
        """Merged duration_seconds equals max of all sub-result durations."""
        orch = ParallelSearchOrchestrator()
        r1 = _make_output(["E1"], duration=5.0)
        r2 = _make_output(["E2"], duration=12.0)
        r3 = _make_output(["E3"], duration=3.0)
        merged = orch._merge_results([r1, r2, r3])
        assert merged.duration_seconds == 12.0

    def test_queries_are_unioned(self) -> None:
        """Merged queries_executed is the union of all sub-result queries."""
        orch = ParallelSearchOrchestrator()
        r1 = SearchRoundOutput(queries_executed=["q1", "q2"])
        r2 = SearchRoundOutput(queries_executed=["q2", "q3"])  # q2 duplicate
        merged = orch._merge_results([r1, r2])
        assert set(merged.queries_executed) == {"q1", "q2", "q3"}

    def test_mcp_tools_are_unioned(self) -> None:
        """Merged mcp_tools_used is the union of all sub-result tools."""
        orch = ParallelSearchOrchestrator()
        r1 = SearchRoundOutput(mcp_tools_used=["perplexity"])
        r2 = SearchRoundOutput(mcp_tools_used=["perplexity", "pubmed"])
        merged = orch._merge_results([r1, r2])
        assert set(merged.mcp_tools_used) == {"perplexity", "pubmed"}

    def test_trajectory_ref_is_first_non_none(self) -> None:
        """Merged trajectory_ref is the first non-None value."""
        orch = ParallelSearchOrchestrator()
        r1 = _make_output(["E1"], trajectory_ref=None)
        r2 = _make_output(["E2"], trajectory_ref="/path/to/traj.jsonl")
        r3 = _make_output(["E3"], trajectory_ref="/other/traj.jsonl")
        merged = orch._merge_results([r1, r2, r3])
        assert merged.agent_trajectory_ref == "/path/to/traj.jsonl"


# ============================================================================
# 🔄 execute() — fallback path tests
# ============================================================================


class TestExecuteFallback:
    """Tests for the single-search fallback path in execute() 🔄."""

    @pytest.mark.asyncio
    async def test_fallback_when_parallel_disabled(self) -> None:
        """execute() calls single_search_fn once when parallel is disabled."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=3)
        config = _make_config(enable_parallel_search=False)
        expected_output = _make_output(["E1"])

        call_count = 0

        async def counting_fn(t: Any, c: Any, r: int, f: Any) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            return expected_output

        result = await orch.execute(task, config, 1, None, counting_fn)
        assert call_count == 1
        assert result is expected_output

    @pytest.mark.asyncio
    async def test_fallback_when_query_strategy_is_none(self) -> None:
        """execute() falls back to single when query_strategy is None."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(include_strategy=False)
        config = _make_config(enable_parallel_search=True)
        call_count = 0

        async def counting_fn(t: Any, c: Any, r: int, f: Any) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, counting_fn)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_when_single_section(self) -> None:
        """execute() falls back when only 1 section is present."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(num_sections=1)
        config = _make_config(enable_parallel_search=True)
        call_count = 0

        async def counting_fn(t: Any, c: Any, r: int, f: Any) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, counting_fn)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_focus_prompt_passed_through_on_fallback(self) -> None:
        """execute() passes focus_prompt to single_search_fn on fallback."""
        orch = ParallelSearchOrchestrator()
        task = _make_task(include_strategy=False)
        config = _make_config(enable_parallel_search=True)
        received_focus: list[str | None] = []

        async def capturing_fn(
            t: Any,
            c: Any,
            r: int,
            focus: str | None,
        ) -> SearchRoundOutput:
            received_focus.append(focus)
            return SearchRoundOutput()

        await orch.execute(task, config, 1, "round-2 focus hint", capturing_fn)
        assert received_focus == ["round-2 focus hint"]


# ============================================================================
# 🔄 execute() — parallel path tests
# ============================================================================


class TestExecuteParallel:
    """Tests for the parallel dispatch path in execute() 🔄."""

    @pytest.mark.asyncio
    async def test_calls_fn_once_per_section(self) -> None:
        """execute() invokes single_search_fn N times for N sections."""
        orch = ParallelSearchOrchestrator(max_parallel=3)
        task = _make_task(num_sections=3)
        config = _make_config(enable_parallel_search=True, max_parallel_agents=3)
        call_count = 0

        async def counting_fn(
            t: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, counting_fn)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_each_sub_task_has_one_section(self) -> None:
        """execute() sends sub-tasks with exactly 1 section each."""
        orch = ParallelSearchOrchestrator(max_parallel=5)
        task = _make_task(num_sections=4)
        config = _make_config(enable_parallel_search=True, max_parallel_agents=5)
        section_counts: list[int] = []

        async def inspecting_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            if sub_task.query_strategy:
                section_counts.append(
                    len(sub_task.query_strategy.get("query_sections", [])),
                )
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, inspecting_fn)
        assert section_counts == [1, 1, 1, 1]

    @pytest.mark.asyncio
    async def test_results_are_merged(self) -> None:
        """execute() returns merged evidence from all successful sub-searches."""
        orch = ParallelSearchOrchestrator(max_parallel=3)
        task = _make_task(num_sections=3)
        config = _make_config(enable_parallel_search=True)
        idx = 0

        async def sequential_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            nonlocal idx
            idx += 1
            return _make_output([f"E{idx}"], cost=1.0)

        result = await orch.execute(task, config, 1, None, sequential_fn)
        assert len(result.evidence) == 3
        assert abs(result.cost_usd - 3.0) < 1e-9

    @pytest.mark.asyncio
    async def test_partial_failure_fail_open(self) -> None:
        """execute() returns results from successful sections when one fails."""
        orch = ParallelSearchOrchestrator(max_parallel=3)
        task = _make_task(num_sections=3)
        config = _make_config(enable_parallel_search=True)
        call_count = 0

        async def partially_failing_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated network failure")
            return _make_output([f"E{call_count}"])

        result = await orch.execute(task, config, 1, None, partially_failing_fn)
        # 2 succeed, 1 fails → 2 evidence items
        assert len(result.evidence) == 2

    @pytest.mark.asyncio
    async def test_all_sections_fail_returns_empty_output(self) -> None:
        """execute() returns empty SearchRoundOutput when all sections fail."""
        orch = ParallelSearchOrchestrator(max_parallel=3)
        task = _make_task(num_sections=2)
        config = _make_config(enable_parallel_search=True)

        async def always_failing_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            raise RuntimeError("always fails")

        result = await orch.execute(task, config, 1, None, always_failing_fn)
        assert result.evidence == []
        assert result.cost_usd == 0.0


# ============================================================================
# 🚦 Semaphore / max_parallel tests
# ============================================================================


class TestMaxParallel:
    """Tests for concurrent-execution limiting via asyncio.Semaphore 🚦."""

    @pytest.mark.asyncio
    async def test_max_parallel_limits_concurrent_executions(self) -> None:
        """With max_parallel=2 and 4 sections, at most 2 run at once."""
        max_parallel = 2
        orch = ParallelSearchOrchestrator(max_parallel=max_parallel)
        task = _make_task(num_sections=4)
        config = _make_config(enable_parallel_search=True, max_parallel_agents=4)

        concurrent_peak = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def concurrency_tracking_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            nonlocal concurrent_peak, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > concurrent_peak:
                    concurrent_peak = current_concurrent
            # Yield to let other coroutines run
            await asyncio.sleep(0)
            async with lock:
                current_concurrent -= 1
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, concurrency_tracking_fn)
        assert concurrent_peak <= max_parallel

    @pytest.mark.asyncio
    async def test_all_sections_complete_even_with_strict_semaphore(self) -> None:
        """All sections complete even when max_parallel < num_sections."""
        orch = ParallelSearchOrchestrator(max_parallel=1)
        task = _make_task(num_sections=5)
        config = _make_config(enable_parallel_search=True)
        call_count = 0

        async def counting_fn(
            sub_task: Any,
            c: Any,
            r: int,
            f: Any,
        ) -> SearchRoundOutput:
            nonlocal call_count
            call_count += 1
            return SearchRoundOutput()

        await orch.execute(task, config, 1, None, counting_fn)
        assert call_count == 5


# ============================================================================
# 🔑 _evidence_key helper tests
# ============================================================================


class TestEvidenceKey:
    """Tests for the _evidence_key deduplication helper 🔑."""

    def test_same_url_same_summary_yields_same_key(self) -> None:
        """Identical url+summary always yields the same key."""
        ev1 = Evidence(
            id="E1",
            source="s",
            url="https://x.com/a",
            query="q",
            summary="abc",
        )
        ev2 = Evidence(
            id="E2",
            source="s",
            url="https://x.com/a",
            query="q",
            summary="abc",
        )
        assert _evidence_key(ev1) == _evidence_key(ev2)

    def test_different_url_yields_different_key(self) -> None:
        """Different URLs yield different keys."""
        ev1 = Evidence(
            id="E1", source="s", url="https://x.com/a", query="q", summary="abc"
        )
        ev2 = Evidence(
            id="E2", source="s", url="https://x.com/b", query="q", summary="abc"
        )
        assert _evidence_key(ev1) != _evidence_key(ev2)

    def test_empty_url_falls_back_to_summary(self) -> None:
        """Evidence with no URL but identical summary hashes match."""
        ev1 = Evidence(id="E1", source="s", url=None, query="q", summary="same summary")
        ev2 = Evidence(id="E2", source="s", url=None, query="q", summary="same summary")
        assert _evidence_key(ev1) == _evidence_key(ev2)

    def test_returns_hex_string(self) -> None:
        """_evidence_key returns a non-empty hex string."""
        ev = Evidence(id="E1", source="s", url="https://x.com", query="q", summary="s")
        key = _evidence_key(ev)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest length
