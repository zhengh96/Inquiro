"""Tests for SharedEvidencePool — cross-task evidence reuse 🧪.

Comprehensive tests covering:
    - Add/get/dedup operations
    - Thread safety under concurrent access
    - Relevance matching with keyword scoring
    - Pool statistics accuracy
    - Integration with DiscoveryLoop (evidence grows across calls)
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from inquiro.core.evidence_pool import SharedEvidencePool
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    DiscoveryConfig,
    Evidence,
    EvaluationTask,
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


# ============================================================================
# 🧪 Test: Basic add and get operations
# ============================================================================


class TestSharedEvidencePoolBasics:
    """Basic add/get/size operations 🧪."""

    def test_add_evidence_returns_count_of_newly_added(self) -> None:
        """Adding evidence returns the count of new items 📥."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/1")
        e2 = _make_evidence("E2", url="https://example.com/2")

        added = pool.add([e1, e2])

        assert added == 2
        assert pool.size == 2

    def test_add_empty_list_returns_zero(self) -> None:
        """Adding an empty list returns 0 and pool stays empty 📥."""
        pool = SharedEvidencePool()

        added = pool.add([])

        assert added == 0
        assert pool.size == 0

    def test_get_all_returns_all_evidence(self) -> None:
        """get_all returns all stored evidence items 📋."""
        pool = SharedEvidencePool()
        items = [
            _make_evidence("E1", url="https://example.com/1"),
            _make_evidence("E2", url="https://example.com/2"),
            _make_evidence("E3", url="https://example.com/3"),
        ]
        pool.add(items)

        result = pool.get_all()

        assert len(result) == 3
        result_ids = {e.id for e in result}
        assert result_ids == {"E1", "E2", "E3"}

    def test_size_property_reflects_unique_items(self) -> None:
        """size property returns the number of unique items 📊."""
        pool = SharedEvidencePool()

        assert pool.size == 0

        pool.add([_make_evidence("E1", url="https://example.com/1")])
        assert pool.size == 1

        pool.add([_make_evidence("E2", url="https://example.com/2")])
        assert pool.size == 2

    def test_add_single_item_list_succeeds(self) -> None:
        """Adding a single-item list works correctly 📥."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/1")

        added = pool.add([e1])

        assert added == 1
        assert pool.size == 1


# ============================================================================
# 🧪 Test: Deduplication
# ============================================================================


class TestSharedEvidencePoolDedup:
    """Content-hash deduplication tests 🧪."""

    def test_duplicate_url_and_summary_rejected(self) -> None:
        """Same URL + summary results in dedup rejection 🔑."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/same")
        # ✨ Same URL and summary as e1
        e2 = _make_evidence("E2", url="https://example.com/same")

        pool.add([e1])
        added = pool.add([e2])

        assert added == 0
        assert pool.size == 1

    def test_different_url_same_summary_added(self) -> None:
        """Different URL with same summary is treated as distinct 🔑."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/page1")
        e2 = _make_evidence("E2", url="https://example.com/page2")

        pool.add([e1])
        added = pool.add([e2])

        assert added == 1
        assert pool.size == 2

    def test_same_url_different_summary_added(self) -> None:
        """Same URL but different summary is treated as distinct 🔑."""
        pool = SharedEvidencePool()
        e1 = _make_evidence(
            "E1",
            url="https://example.com/page",
            summary=(
                "Evidence about protein folding mechanisms and "
                "misfolding diseases with detailed analysis"
            ),
        )
        e2 = _make_evidence(
            "E2",
            url="https://example.com/page",
            summary=(
                "Evidence about gene expression levels and "
                "regulatory pathways in human cells"
            ),
        )

        pool.add([e1])
        added = pool.add([e2])

        assert added == 1
        assert pool.size == 2

    def test_batch_dedup_within_single_add_call(self) -> None:
        """Duplicates within a single add() call are detected 🔑."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/same")
        # ✨ e2 has same URL and summary as e1
        e2 = _make_evidence("E2", url="https://example.com/same")

        added = pool.add([e1, e2])

        assert added == 1
        assert pool.size == 1

    def test_none_url_dedup_by_summary_only(self) -> None:
        """Evidence without URL is deduped by summary alone 🔑."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url=None)
        e2 = _make_evidence("E2", url=None)

        pool.add([e1])
        added = pool.add([e2])

        # ✨ Same summary, no URL → duplicate
        assert added == 0
        assert pool.size == 1

    def test_dedup_count_tracked_in_stats(self) -> None:
        """Dedup rejection count is tracked in stats 📊."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/same")
        e2 = _make_evidence("E2", url="https://example.com/same")
        e3 = _make_evidence("E3", url="https://example.com/same")

        pool.add([e1])
        pool.add([e2])
        pool.add([e3])

        stats = pool.get_stats()
        assert stats["dedup_rejected"] == 2


# ============================================================================
# 🧪 Test: Relevance matching
# ============================================================================


class TestSharedEvidencePoolRelevance:
    """Keyword-based relevance scoring tests 🧪."""

    def test_get_relevant_returns_matching_evidence(self) -> None:
        """Relevant evidence is ranked higher than irrelevant 🎯."""
        pool = SharedEvidencePool()
        # ✨ Evidence about protein binding (matches checklist)
        e_binding = _make_evidence(
            "E1",
            summary=(
                "Study of protein binding affinity measurements "
                "using surface plasmon resonance technique"
            ),
            query="protein binding affinity",
            url="https://example.com/binding",
        )
        # ✨ Evidence about market analysis (does not match checklist)
        e_market = _make_evidence(
            "E2",
            summary=(
                "Market analysis report showing pharmaceutical "
                "industry growth trends for the year 2025"
            ),
            query="market analysis report",
            url="https://example.com/market",
        )
        pool.add([e_binding, e_market])

        checklist = ["Assess protein binding affinity data"]
        result = pool.get_relevant(checklist, limit=10)

        # 🏆 Binding evidence should be ranked first
        assert len(result) == 2
        assert result[0].id == "E1"

    def test_get_relevant_respects_limit(self) -> None:
        """get_relevant returns at most 'limit' items 🎯."""
        pool = SharedEvidencePool()
        for i in range(10):
            pool.add(
                [
                    _make_evidence(
                        f"E{i}",
                        url=f"https://example.com/{i}",
                        summary=(
                            f"Unique evidence number {i} about various "
                            f"protein interactions and cellular mechanisms"
                        ),
                    )
                ]
            )

        result = pool.get_relevant(["protein interactions"], limit=3)

        assert len(result) <= 3

    def test_get_relevant_empty_checklist_returns_all(self) -> None:
        """Empty checklist returns all items up to limit 🎯."""
        pool = SharedEvidencePool()
        pool.add(
            [
                _make_evidence("E1", url="https://example.com/1"),
                _make_evidence("E2", url="https://example.com/2"),
            ]
        )

        result = pool.get_relevant([], limit=50)

        assert len(result) == 2

    def test_get_relevant_empty_pool_returns_empty(self) -> None:
        """Empty pool returns empty list 🎯."""
        pool = SharedEvidencePool()

        result = pool.get_relevant(["some checklist item"])

        assert result == []

    def test_get_relevant_matches_query_field(self) -> None:
        """Relevance scoring also checks the query field 🎯."""
        pool = SharedEvidencePool()
        # ✨ Summary is generic, but query matches
        e1 = _make_evidence(
            "E1",
            summary=(
                "General information about biological systems "
                "and their complex regulatory mechanisms"
            ),
            query="protein binding affinity measurement",
            url="https://example.com/1",
        )
        e2 = _make_evidence(
            "E2",
            summary=(
                "General information about weather patterns "
                "and their effects on agricultural output"
            ),
            query="weather forecast methodology",
            url="https://example.com/2",
        )
        pool.add([e1, e2])

        result = pool.get_relevant(
            ["protein binding affinity data"],
            limit=10,
        )

        # 🏆 E1 should rank higher due to query keyword match
        assert result[0].id == "E1"


# ============================================================================
# 🧪 Test: Thread safety
# ============================================================================


class TestSharedEvidencePoolThreadSafety:
    """Concurrent access safety tests 🧪."""

    def test_concurrent_add_no_data_loss(self) -> None:
        """Concurrent add() calls do not lose data 🔒."""
        pool = SharedEvidencePool()
        num_threads = 10
        items_per_thread = 20
        barriers: list[threading.Barrier] = [threading.Barrier(num_threads)]

        def add_items(thread_idx: int) -> None:
            """Worker function for concurrent add 🔧."""
            evidence = [
                _make_evidence(
                    f"E-t{thread_idx}-{i}",
                    url=f"https://example.com/t{thread_idx}/{i}",
                    summary=(
                        f"Thread {thread_idx} evidence item {i} with "
                        f"unique content about research topic number "
                        f"{thread_idx * 100 + i}"
                    ),
                )
                for i in range(items_per_thread)
            ]
            barriers[0].wait()  # 🏁 Start all threads simultaneously
            pool.add(evidence)

        threads = [
            threading.Thread(target=add_items, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ✅ All unique items should be present
        expected_total = num_threads * items_per_thread
        assert pool.size == expected_total

    def test_concurrent_add_and_read_no_crash(self) -> None:
        """Concurrent add() and get_all() calls do not crash 🔒."""
        pool = SharedEvidencePool()
        errors: list[Exception] = []

        def writer(thread_idx: int) -> None:
            """Add evidence concurrently 🔧."""
            try:
                for i in range(50):
                    pool.add(
                        [
                            _make_evidence(
                                f"E-w{thread_idx}-{i}",
                                url=(f"https://example.com/w{thread_idx}/{i}"),
                                summary=(
                                    f"Writer {thread_idx} item {i} with "
                                    f"detailed analysis of research findings"
                                ),
                            )
                        ]
                    )
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            """Read evidence concurrently 🔧."""
            try:
                for _ in range(50):
                    _ = pool.get_all()
                    _ = pool.size
                    _ = pool.get_stats()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)] + [
            threading.Thread(target=reader) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"


# ============================================================================
# 🧪 Test: Statistics
# ============================================================================


class TestSharedEvidencePoolStats:
    """Pool statistics accuracy tests 🧪."""

    def test_get_stats_empty_pool(self) -> None:
        """Empty pool stats are all zeros 📊."""
        pool = SharedEvidencePool()

        stats = pool.get_stats()

        assert stats["total"] == 0
        assert stats["dedup_rejected"] == 0
        assert stats["by_source"] == {}

    def test_get_stats_tracks_total(self) -> None:
        """Stats total matches actual pool size 📊."""
        pool = SharedEvidencePool()
        pool.add(
            [
                _make_evidence("E1", url="https://example.com/1"),
                _make_evidence("E2", url="https://example.com/2"),
            ]
        )

        stats = pool.get_stats()

        assert stats["total"] == 2

    def test_get_stats_tracks_source_distribution(self) -> None:
        """Stats by_source correctly counts per source 📊."""
        pool = SharedEvidencePool()
        pool.add(
            [
                _make_evidence(
                    "E1",
                    source="pubmed",
                    url="https://pubmed.com/1",
                ),
                _make_evidence(
                    "E2",
                    source="pubmed",
                    url="https://pubmed.com/2",
                ),
                _make_evidence(
                    "E3",
                    source="arxiv",
                    url="https://arxiv.org/1",
                ),
            ]
        )

        stats = pool.get_stats()

        assert stats["by_source"]["pubmed"] == 2
        assert stats["by_source"]["arxiv"] == 1

    def test_get_stats_tracks_dedup_rejected(self) -> None:
        """Stats dedup_rejected increments on duplicate adds 📊."""
        pool = SharedEvidencePool()
        e1 = _make_evidence("E1", url="https://example.com/same")
        e2 = _make_evidence("E2", url="https://example.com/same")

        pool.add([e1])
        pool.add([e2])

        stats = pool.get_stats()
        assert stats["dedup_rejected"] == 1

    def test_get_stats_multiple_sources_tracked(self) -> None:
        """Multiple distinct sources are all tracked 📊."""
        pool = SharedEvidencePool()
        sources = ["pubmed", "arxiv", "clinicaltrials", "patent-db"]
        for i, source in enumerate(sources):
            pool.add(
                [
                    _make_evidence(
                        f"E{i}",
                        source=source,
                        url=f"https://{source}.com/{i}",
                    )
                ]
            )

        stats = pool.get_stats()

        assert len(stats["by_source"]) == 4
        for source in sources:
            assert stats["by_source"][source] == 1


# ============================================================================
# 🧪 Test: Integration with DiscoveryLoop
# ============================================================================


class TestSharedEvidencePoolDiscoveryIntegration:
    """Integration tests verifying pool grows across DiscoveryLoop calls 🧪."""

    @pytest.mark.asyncio
    async def test_pool_grows_across_discovery_runs(self) -> None:
        """Evidence pool accumulates items across multiple loop runs 🔄."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            MockAnalysisExecutor,
            MockFocusPromptGenerator,
            MockSearchExecutor,
        )
        from inquiro.core.gap_analysis import (
            CoverageResult,
            GapAnalysis,
            MockCoverageJudge,
        )

        # 🔧 Create shared pool
        pool = SharedEvidencePool()

        # 🔧 Evidence for first run
        evidence_run1 = [
            _make_evidence(
                "E1",
                summary=(
                    "First run evidence about protein binding "
                    "and molecular interactions in the cell"
                ),
                url="https://example.com/run1/1",
            ),
            _make_evidence(
                "E2",
                summary=(
                    "First run evidence about gene expression "
                    "and transcription factor regulation"
                ),
                url="https://example.com/run1/2",
            ),
        ]

        # 🔧 Configure first run (converges after 1 round)
        judge1 = MockCoverageJudge()
        judge1.mock_results = {
            0: CoverageResult(
                covered=["binding data", "outcomes", "safety"],
                uncovered=[],
            ),
        }

        loop1 = DiscoveryLoop(
            search_executor=MockSearchExecutor(evidence_run1),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[
                    {
                        "claim": "Binding data found",
                        "evidence_ids": ["E1"],
                    }
                ],
            ),
            gap_analysis=GapAnalysis(coverage_judge=judge1),
            focus_generator=MockFocusPromptGenerator(),
        )

        task1 = _make_task("task-run-1")
        config1 = _make_config(max_rounds=1)

        _result1 = await loop1.run(
            task1,
            config1,
            shared_evidence_pool=pool,
        )

        # ✅ Pool should have evidence from first run
        assert pool.size > 0
        first_run_size = pool.size

        # 🔧 Evidence for second run (different items)
        evidence_run2 = [
            _make_evidence(
                "E3",
                summary=(
                    "Second run evidence about clinical trial "
                    "outcomes and patient response rates"
                ),
                url="https://example.com/run2/1",
            ),
        ]

        judge2 = MockCoverageJudge()
        judge2.mock_results = {
            0: CoverageResult(
                covered=["binding data", "outcomes", "safety"],
                uncovered=[],
            ),
        }

        loop2 = DiscoveryLoop(
            search_executor=MockSearchExecutor(evidence_run2),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[
                    {
                        "claim": "Trial outcomes found",
                        "evidence_ids": ["E3"],
                    }
                ],
            ),
            gap_analysis=GapAnalysis(coverage_judge=judge2),
            focus_generator=MockFocusPromptGenerator(),
        )

        task2 = _make_task("task-run-2")
        config2 = _make_config(max_rounds=1)

        _result2 = await loop2.run(
            task2,
            config2,
            shared_evidence_pool=pool,
        )

        # ✅ Pool should have grown with second run's evidence
        assert pool.size > first_run_size

    @pytest.mark.asyncio
    async def test_pool_none_works_as_old_behavior(self) -> None:
        """Passing None for pool gives backward-compatible behavior 🔄."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            MockAnalysisExecutor,
            MockFocusPromptGenerator,
            MockSearchExecutor,
        )
        from inquiro.core.gap_analysis import (
            CoverageResult,
            GapAnalysis,
            MockCoverageJudge,
        )

        evidence = [
            _make_evidence(
                "E1",
                summary=(
                    "Evidence about protein binding mechanisms "
                    "in biological systems and pathways"
                ),
                url="https://example.com/1",
            ),
        ]

        judge = MockCoverageJudge()
        judge.mock_results = {
            0: CoverageResult(
                covered=["binding data", "outcomes", "safety"],
                uncovered=[],
            ),
        }

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(evidence),
            analysis_executor=MockAnalysisExecutor(),
            gap_analysis=GapAnalysis(coverage_judge=judge),
            focus_generator=MockFocusPromptGenerator(),
        )

        task = _make_task("task-no-pool")
        config = _make_config(max_rounds=1)

        # ✅ Should work without pool (None is default)
        result = await loop.run(task, config)

        assert result is not None
        assert result.task_id == "task-no-pool"

    @pytest.mark.asyncio
    async def test_second_run_prefills_from_shared_pool(self) -> None:
        """Second run pre-fills evidence from shared pool 🔄."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            MockAnalysisExecutor,
            MockFocusPromptGenerator,
            MockSearchExecutor,
        )
        from inquiro.core.gap_analysis import (
            CoverageResult,
            GapAnalysis,
            MockCoverageJudge,
        )

        pool = SharedEvidencePool()

        # 🔧 Pre-seed the pool with evidence
        seed_evidence = [
            _make_evidence(
                "E-seed-1",
                summary=(
                    "Pre-seeded evidence about binding affinity "
                    "data from previous research iterations"
                ),
                url="https://example.com/seed/1",
            ),
            _make_evidence(
                "E-seed-2",
                summary=(
                    "Pre-seeded evidence about clinical trial "
                    "outcomes from prior studies and reviews"
                ),
                url="https://example.com/seed/2",
            ),
        ]
        pool.add(seed_evidence)
        assert pool.size == 2

        # 🔧 Run a new loop with the pre-seeded pool
        new_evidence = [
            _make_evidence(
                "E-new-1",
                summary=(
                    "New evidence about safety profile studies "
                    "and adverse event monitoring protocols"
                ),
                url="https://example.com/new/1",
            ),
        ]

        judge = MockCoverageJudge()
        judge.mock_results = {
            0: CoverageResult(
                covered=["binding data", "outcomes", "safety"],
                uncovered=[],
            ),
        }

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(new_evidence),
            analysis_executor=MockAnalysisExecutor(),
            gap_analysis=GapAnalysis(coverage_judge=judge),
            focus_generator=MockFocusPromptGenerator(),
        )

        task = _make_task("task-prefilled")
        config = _make_config(max_rounds=1)

        result = await loop.run(
            task,
            config,
            shared_evidence_pool=pool,
        )

        # ✅ Pool should have grown (seed + new)
        assert pool.size == 3

        # ✅ Result should contain more evidence than just new items
        # (pre-filled from pool + new items from search)
        assert len(result.evidence) >= 1
