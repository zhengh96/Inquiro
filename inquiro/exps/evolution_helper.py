"""Shared evolution lifecycle helper for Exp classes 🧬.

Provides shared evolution enrichment and post-execution evolution
lifecycle methods used by SynthesisExp and the discovery pipeline.

Uses composition pattern (not mixin) to avoid MRO complexity.

Workflow:
    Pre-execution:  ``enrich_with_experiences(prior_context)`` → enriched text
    Post-execution: ``post_execution_evolution(trajectory, result)`` → store + fitness

Both methods are synchronous and use ``run_async()`` internally to
bridge async evolution infrastructure (ExperienceStore, PromptEnricher,
ExperienceExtractor, FitnessEvaluator) into the Exp's sync context.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from inquiro.evolution.types import EnrichmentResult

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM


# ============================================================================
# 🔧 Result protocol — shared interface for EvaluationResult / SynthesisResult
# ============================================================================


@runtime_checkable
class _ResultLike(Protocol):
    """Minimal interface shared by EvaluationResult and SynthesisResult 📋."""

    evidence_index: list[Any]
    confidence: float
    cost: float


# ============================================================================
# 🧬 EvolutionHelper
# ============================================================================


class EvolutionHelper:
    """Shared evolution lifecycle helper for Exp classes 🧬.

    Encapsulates pre-execution enrichment and post-execution evolution
    pipeline (collect → extract → store → fitness).  Designed to be
    instantiated as a member of SynthesisExp and called
    at the appropriate lifecycle points.

    Thread-safety: not thread-safe. Each Exp instance creates its own
    helper, so no sharing across threads.

    Attributes:
        enrichment_result: The latest EnrichmentResult from enrichment.
            Used by post_execution_evolution for fitness tracking.
    """

    def __init__(
        self,
        task: Any,
        llm: BaseLLM,
        cost_tracker: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize EvolutionHelper 🔧.

        Args:
            task: EvaluationTask or SynthesisTask with evolution_profile.
            llm: LLM instance for experience extraction calls.
            cost_tracker: CostTracker for supplementing cost metrics.
            logger: Optional logger; creates one if not provided.
        """
        self._task = task
        self._llm = llm
        self._cost_tracker = cost_tracker
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self.enrichment_result: EnrichmentResult | None = None

    # ------------------------------------------------------------------
    # 🎯 Public API
    # ------------------------------------------------------------------

    def enrich_with_experiences(
        self,
        prior_context: str,
    ) -> str:
        """Enrich prompt with learned experiences from evolution 🧬.

        If evolution_profile is present in the task, queries the
        ExperienceStore for relevant experiences and appends them
        to the prior_context section of the system prompt.

        This method is synchronous because it's called during prompt
        rendering. Uses ``run_async()`` to bridge async store queries.

        Args:
            prior_context: Existing prior context string (may be empty).

        Returns:
            Updated prior_context with learned insights appended.
            Returns original prior_context on any error (non-blocking).
        """
        if not self._task.evolution_profile:
            return prior_context

        try:
            from inquiro.evolution.enricher import PromptEnricher
            from inquiro.evolution.store_factory import get_store

            profile = self._task.evolution_profile

            # 🔍 Check that enrichment template is configured
            enrichment_template = profile.get("enrichment_prompt_template", "")
            if not enrichment_template:
                self._logger.debug(
                    "🧬 Evolution profile present but no enrichment "
                    "template — skipping enrichment"
                )
                return prior_context

            # 🏭 Get store via lazy singleton factory
            store = self.run_async(get_store())
            enricher = PromptEnricher(store)

            profile_config = {
                "enrichment_prompt_template": enrichment_template,
                "enrichment_max_tokens": profile.get("enrichment_max_tokens", 500),
                "enrichment_max_items": profile.get("enrichment_max_items", 10),
                "namespace": profile.get("namespace", ""),
                "prune_min_fitness": profile.get("prune_min_fitness", 0.3),
            }

            # 🔄 Run async enrichment synchronously
            # ⚠️ SynthesisTask may not have sub_item_id — use safe fallback
            sub_item = getattr(self._task, "sub_item_id", "")
            result = self.run_async(
                enricher.enrich(
                    task_context_tags=self._task.context_tags,
                    sub_item=sub_item,
                    profile_config=profile_config,
                )
            )

            # 🧬 Store enrichment result for fitness tracking
            self.enrichment_result = result

            if result.enrichment_text:
                enrichment_section = (
                    "\n\n# LEARNED INSIGHTS\n\n"
                    "The following insights are from previous "
                    "evaluations. Use them to guide your "
                    "strategy.\n\n"
                    f"{result.enrichment_text}"
                )
                self._logger.info(
                    "🧬 Enriched prompt with %d experiences (%d tokens)",
                    len(result.injected_experience_ids),
                    result.token_count,
                )
                return prior_context + enrichment_section

        except Exception as e:
            self._logger.warning(
                "⚠️ Evolution enrichment failed (non-blocking): %s",
                e,
            )

        return prior_context

    def post_execution_evolution(
        self,
        trajectory: Any,
        result: Any,
    ) -> None:
        """Post-execution evolution pipeline: collect → extract → store → fitness 🧬.

        Runs the complete evolution lifecycle after a successful task
        execution:
            1. Collect trajectory data into a TrajectorySnapshot
            2. Extract reusable experiences via LLM
            3. Store new experiences (with deduplication)
            4. Evaluate and update fitness for injected experiences

        This method is non-blocking: all failures are logged as warnings
        and do not affect the task result.

        Args:
            trajectory: Agent execution trajectory (from EvoMaster).
            result: The completed EvaluationResult or SynthesisResult.
        """
        if not self._task.evolution_profile:
            return

        try:
            profile = self._task.evolution_profile

            from inquiro.evolution.collector import TrajectoryCollector
            from inquiro.evolution.extractor import ExperienceExtractor
            from inquiro.evolution.fitness import FitnessEvaluator
            from inquiro.evolution.store_factory import get_store
            from inquiro.evolution.types import ResultMetrics

            # 🏭 Get store via lazy singleton factory
            store = self.run_async(get_store())

            # 📸 Step 1: Collect trajectory snapshot
            # ⚠️ SynthesisTask may not have sub_item_id — use safe fallback
            sub_item = getattr(self._task, "sub_item_id", "")
            collector = TrajectoryCollector()
            snapshot = collector.collect(
                trajectory=trajectory,
                task=self._task,
                context_tags=self._task.context_tags,
                sub_item_id=sub_item,
            )
            # 💰 Supplement cost from cost_tracker
            snapshot.metrics.cost_usd = self._cost_tracker.get_total_cost()

            self._logger.info(
                "🧬 Collected trajectory: %d tool calls, evidence=%d, confidence=%.2f",
                len(snapshot.tool_calls),
                snapshot.metrics.evidence_count,
                snapshot.metrics.confidence,
            )

            # 🧠 Step 2: Extract experiences via LLM
            extraction_template = profile.get("extraction_prompt_template", "")
            if extraction_template:

                async def llm_fn(prompt: str) -> str:
                    """Wrap LLM call for experience extraction 🤖."""
                    from evomaster.utils.types import (
                        Dialog,
                        UserMessage,
                    )

                    dialog = Dialog(messages=[UserMessage(content=prompt)])
                    response = self._llm.query(dialog)
                    if hasattr(response, "content") and response.content:
                        return response.content
                    return str(response)

                extractor = ExperienceExtractor(llm_fn)
                profile_config = {
                    "extraction_prompt_template": extraction_template,
                    "experience_categories": profile.get("experience_categories", []),
                    "max_experiences_per_extraction": profile.get(
                        "max_experiences_per_extraction", 8
                    ),
                    "namespace": profile.get("namespace", ""),
                }

                new_experiences = self.run_async(
                    extractor.extract(snapshot, profile_config)
                )

                # 💾 Step 3: Store new experiences (with dedup)
                stored_count = 0
                for exp in new_experiences:
                    try:
                        is_dup = self.run_async(
                            store.deduplicate(
                                exp.namespace,
                                exp.insight,
                            )
                        )
                        if not is_dup:
                            self.run_async(store.add(exp))
                            stored_count += 1
                    except Exception as store_err:
                        self._logger.warning(
                            "⚠️ Failed to store experience: %s",
                            store_err,
                        )

                self._logger.info(
                    "🧬 Extracted %d experiences, stored %d (after dedup)",
                    len(new_experiences),
                    stored_count,
                )

            # 📊 Step 4: Fitness evaluation for injected experiences
            enrichment = self.enrichment_result
            if enrichment and enrichment.injected_experience_ids:
                fitness_evaluator = FitnessEvaluator(store)

                # 🔍 Build after_metrics from result — handle both
                # EvaluationResult and SynthesisResult gracefully
                checklist_coverage = (
                    result.get_covered_ratio()
                    if hasattr(result, "get_covered_ratio")
                    else 0.0
                )
                search_rounds = getattr(
                    result,
                    "search_rounds",
                    0,
                )

                after_metrics = ResultMetrics(
                    evidence_count=len(result.evidence_index),
                    confidence=result.confidence,
                    cost_usd=result.cost,
                    search_rounds=search_rounds,
                    checklist_coverage=checklist_coverage,
                )
                before_metrics = ResultMetrics(
                    evidence_count=0,
                    confidence=0.0,
                    cost_usd=0.0,
                    search_rounds=0,
                    checklist_coverage=0.0,
                )

                fitness_config = {
                    "fitness_dimensions": profile.get("fitness_dimensions", []),
                    "fitness_learning_rate": profile.get("fitness_learning_rate", 0.3),
                }

                updates = self.run_async(
                    fitness_evaluator.evaluate(
                        enrichment,
                        before_metrics,
                        after_metrics,
                        fitness_config,
                    )
                )

                if updates:
                    self.run_async(store.bulk_update_fitness(updates))
                    self._logger.info(
                        "🧬 Applied fitness updates for %d experiences",
                        len(updates),
                    )

        except Exception as e:
            self._logger.warning(
                "⚠️ Post-execution evolution failed (non-blocking): %s",
                e,
            )

    # ------------------------------------------------------------------
    # 🔄 Async bridging
    # ------------------------------------------------------------------

    @staticmethod
    def run_async(coro: Any) -> Any:
        """Bridge async coroutine into sync context 🔄.

        Handles both cases: when an event loop is already running
        (uses a thread pool) and when no loop is running (uses
        asyncio.run).

        Args:
            coro: Awaitable coroutine to execute.

        Returns:
            Result from the coroutine.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
            ) as pool:
                return pool.submit(asyncio.run, coro).result(
                    timeout=30,
                )
        else:
            return asyncio.run(coro)
