"""Parallel search orchestrator — splits queries by section 🔄.

When a QueryStrategy with multiple sections is available AND parallel
search is enabled, this module spawns multiple search executions
(one per section) and merges their results.  Falls back gracefully
to single search when conditions are not met.

Architecture position:
    DiscoveryLoop
        -> _SearchExpAdapter.execute_search()
            -> ParallelSearchOrchestrator.execute()   <-- this module
                -> single_search_fn (per section, concurrent)
                -> _merge_results()
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from inquiro.core.canonical_hash import canonical_evidence_hash

if TYPE_CHECKING:
    from inquiro.core.discovery_loop import SearchRoundOutput
    from inquiro.core.types import DiscoveryConfig, EvaluationTask

logger = logging.getLogger(__name__)


# ============================================================================
# 🔄 ParallelSearchOrchestrator
# ============================================================================


class ParallelSearchOrchestrator:
    """Orchestrate parallel search across multiple query sections 🔄.

    Wraps a single-search callable and dispatches parallel sub-searches
    when conditions are met.  Used as a drop-in decision layer inside
    ``_SearchExpAdapter.execute_search()`` in the Runner.

    Falls back to a single search invocation when:
    - ``config.enable_parallel_search`` is False
    - ``task.query_strategy`` is None
    - ``task.query_strategy["query_sections"]`` has <= 1 items

    Attributes:
        _max_parallel: Maximum concurrent search coroutines.
    """

    def __init__(self, max_parallel: int = 3) -> None:
        """Initialise orchestrator 🔧.

        Args:
            max_parallel: Maximum concurrent search coroutines.
                Enforced via ``asyncio.Semaphore``.
        """
        self._max_parallel = max(1, max_parallel)

    # ------------------------------------------------------------------ #
    # 🚀 Public entry-point
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None,
        single_search_fn: Callable[..., Awaitable[SearchRoundOutput]],
    ) -> SearchRoundOutput:
        """Execute search — parallel or single depending on conditions 🔄.

        Args:
            task: Evaluation task, potentially carrying a query_strategy.
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            focus_prompt: Optional targeted search guidance string.
            single_search_fn: Async callable with signature
                ``(task, config, round_number, focus_prompt) -> SearchRoundOutput``.
                Invoked once per section (or once for single fallback).

        Returns:
            Merged ``SearchRoundOutput`` combining all parallel sub-results,
            or the direct result from the single fallback path.
        """
        if not self._should_parallelize(task, config):
            logger.info(
                "🔍 ParallelSearchOrchestrator: single-search fallback "
                "(round=%d, task=%s)",
                round_number,
                task.task_id,
            )
            return await single_search_fn(
                task,
                config,
                round_number,
                focus_prompt,
            )

        sub_tasks = self._split_task_by_sections(task)
        logger.info(
            "🔄 ParallelSearchOrchestrator: launching %d parallel searches "
            "(round=%d, task=%s, max_parallel=%d)",
            len(sub_tasks),
            round_number,
            task.task_id,
            self._max_parallel,
        )

        semaphore = asyncio.Semaphore(self._max_parallel)

        async def _bounded(sub_task: EvaluationTask) -> SearchRoundOutput:
            async with semaphore:
                return await single_search_fn(
                    sub_task,
                    config,
                    round_number,
                    focus_prompt,
                )

        raw_results: list[SearchRoundOutput | BaseException] = await asyncio.gather(
            *[_bounded(st) for st in sub_tasks],
            return_exceptions=True,
        )

        successful: list[SearchRoundOutput] = []
        errors: list[str] = []
        for idx, outcome in enumerate(raw_results):
            if isinstance(outcome, BaseException):
                error_msg = f"section {idx}: {type(outcome).__name__}: {outcome}"
                errors.append(error_msg)
                logger.warning(
                    "⚠️ ParallelSearchOrchestrator: section %d failed (task=%s): %s",
                    idx,
                    task.task_id,
                    outcome,
                )
            else:
                successful.append(outcome)

        if not successful:
            logger.error(
                "❌ ParallelSearchOrchestrator: all %d parallel searches "
                "failed — returning empty SearchRoundOutput with error "
                "details (task=%s): %s",
                len(sub_tasks),
                task.task_id,
                errors,
            )
            from inquiro.core.discovery_loop import SearchRoundOutput

            return SearchRoundOutput(section_errors=errors)

        if len(successful) < len(sub_tasks):
            logger.warning(
                "⚠️ ParallelSearchOrchestrator: %d/%d sections succeeded "
                "(task=%s) — merging partial results",
                len(successful),
                len(sub_tasks),
                task.task_id,
            )

        return self._merge_results(successful)

    # ------------------------------------------------------------------ #
    # 🔍 Parallel eligibility check
    # ------------------------------------------------------------------ #

    def _should_parallelize(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
    ) -> bool:
        """Check whether parallel search is warranted 🔍.

        Args:
            task: Evaluation task to inspect.
            config: Discovery pipeline configuration.

        Returns:
            True only when all conditions for parallel execution are met.
        """
        if not config.enable_parallel_search:
            return False
        if task.query_strategy is None:
            return False
        sections = task.query_strategy.get("query_sections", [])
        return len(sections) > 1

    # ------------------------------------------------------------------ #
    # 📋 Task splitting
    # ------------------------------------------------------------------ #

    def _split_task_by_sections(
        self,
        task: EvaluationTask,
    ) -> list[EvaluationTask]:
        """Create per-section sub-tasks 📋.

        Each sub-task receives a deep copy of the original task with a
        modified ``query_strategy`` that contains only one ``QuerySection``
        plus the full ``alias_expansion`` (preserved for entity disambiguation).

        Args:
            task: Original task with a multi-section query_strategy.

        Returns:
            List of sub-tasks — one per QuerySection.
        """
        strategy: dict[str, Any] = task.query_strategy or {}
        sections: list[dict[str, Any]] = strategy.get("query_sections", [])
        alias_expansion: str = strategy.get("alias_expansion", "")

        sub_tasks: list[EvaluationTask] = []
        for section in sections:
            # Deep-copy the base strategy, then replace query_sections
            section_strategy: dict[str, Any] = {
                **strategy,
                "alias_expansion": alias_expansion,
                "query_sections": [section],
            }
            sub_task = task.model_copy(
                deep=True, update={"query_strategy": section_strategy}
            )
            sub_tasks.append(sub_task)

        return sub_tasks

    # ------------------------------------------------------------------ #
    # 🔗 Result merging
    # ------------------------------------------------------------------ #

    def _merge_results(
        self,
        results: list[SearchRoundOutput],
    ) -> SearchRoundOutput:
        """Merge parallel search results with evidence deduplication 🔗.

        Deduplication key: ``md5(url + ":" + summary)``.
        When ``url`` is empty/None the full observation text acts as
        the hash key so content-level duplicates are still caught.

        Cost is summed; duration is max across all sub-results.
        Query lists and MCP tool sets are unioned.

        Args:
            results: Non-empty list of successful SearchRoundOutput objects.

        Returns:
            Single merged SearchRoundOutput.
        """
        from inquiro.core.discovery_loop import SearchRoundOutput

        seen: dict[str, Any] = {}
        merged_evidence: list[Any] = []
        merged_queries: list[str] = []
        merged_tools: list[str] = []
        seen_queries: set[str] = set()
        seen_tools: set[str] = set()
        total_cost: float = 0.0
        max_duration: float = 0.0
        # Collect ALL trajectory refs for post-hoc debugging
        trajectory_refs: list[str] = []

        for result in results:
            for ev in result.evidence:
                key = _evidence_key(ev)
                if key not in seen:
                    seen[key] = True
                    merged_evidence.append(ev)

            for q in result.queries_executed:
                if q not in seen_queries:
                    seen_queries.add(q)
                    merged_queries.append(q)

            for t in result.mcp_tools_used:
                if t not in seen_tools:
                    seen_tools.add(t)
                    merged_tools.append(t)

            total_cost += result.cost_usd
            max_duration = max(max_duration, result.duration_seconds)

            if result.agent_trajectory_ref:
                trajectory_refs.append(result.agent_trajectory_ref)

        # Log all trajectory refs (not just the first)
        if len(trajectory_refs) > 1:
            logger.info(
                "🔗 ParallelSearchOrchestrator: collected %d trajectory refs: %s",
                len(trajectory_refs),
                trajectory_refs,
            )

        logger.info(
            "🔗 ParallelSearchOrchestrator: merged %d sections → "
            "%d unique evidence items (cost=%.4f USD, dur=%.1fs)",
            len(results),
            len(merged_evidence),
            total_cost,
            max_duration,
        )

        return SearchRoundOutput(
            evidence=merged_evidence,
            queries_executed=merged_queries,
            mcp_tools_used=merged_tools,
            cost_usd=total_cost,
            duration_seconds=max_duration,
            agent_trajectory_ref=(trajectory_refs[0] if trajectory_refs else None),
            agent_trajectory_refs=trajectory_refs,
        )


# ============================================================================
# 🔑 Evidence deduplication helper
# ============================================================================


def _evidence_key(evidence: Any) -> str:
    """Compute a stable deduplication key for an evidence item 🔑.

    Delegates to ``canonical_evidence_hash`` for consistent hashing
    across the entire codebase (SHA-256 with URL/summary normalisation).

    Args:
        evidence: An Evidence (or duck-typed) object with optional
            ``url``, ``summary``, and ``observation`` attributes.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    url: str = getattr(evidence, "url", "") or ""
    summary: str = (
        getattr(evidence, "summary", "") or getattr(evidence, "observation", "") or ""
    )
    return canonical_evidence_hash(url, summary)
