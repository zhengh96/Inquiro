"""UnifiedEvolutionProvider — orchestrates all learning mechanisms 🧬.

Single EvolutionProvider that composes multiple ``BaseMechanism`` instances
and dispatches lifecycle events to each enabled mechanism. Replaces the
monolithic ``TargetMasterEvolutionProvider`` with a pluggable mechanism
pipeline controlled by ``MechanismConfig``.

Key responsibilities:
- Collect trajectory snapshots from round records
- Dispatch ``produce()`` / ``inject()`` / ``on_round_*()`` to each mechanism
- Enforce per-mechanism token budgets
- Maintain A/B test group assignment for the control/treatment split

Usage::

    provider = UnifiedEvolutionProvider(
        mechanisms=[exp_mechanism, bandit, reflector, distiller],
        collector=DiscoveryTrajectoryCollector(),
        fitness_evaluator=FitnessEvaluator(store),
        store=store,
        profile=profile,
        task=task,
    )
    await provider.prepare_enrichment()
"""

from __future__ import annotations

import logging
import random
from typing import Any

from inquiro.core.trajectory.models import DiscoveryRoundRecord, SynthesisRecord
from inquiro.core.types import EvaluationTask, RoundMetrics
from inquiro.evolution.discovery_collector import DiscoveryTrajectoryCollector
from inquiro.evolution.fitness import FitnessEvaluator
from inquiro.evolution.mechanism_config import TOTAL_INJECTION_BUDGET, MechanismConfig
from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.metrics import EvolutionMetricsRecorder, EvolutionRoundMetric
from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import EnrichmentResult, ResultMetrics

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["UnifiedEvolutionProvider"]


# ============================================================================
# 🧬 UnifiedEvolutionProvider
# ============================================================================


class UnifiedEvolutionProvider:
    """Single EvolutionProvider orchestrating all learning mechanisms 🧬.

    Composes N ``BaseMechanism`` instances and dispatches lifecycle
    events to each enabled mechanism. Token budgets are enforced per
    mechanism and truncated when the total exceeds ``TOTAL_INJECTION_BUDGET``.

    Implements the same protocol as ``TargetMasterEvolutionProvider``
    so it can be used as a drop-in replacement.

    Attributes:
        ab_group: A/B test assignment ("control" or "treatment").
    """

    def __init__(
        self,
        mechanisms: list[BaseMechanism],
        collector: DiscoveryTrajectoryCollector,
        fitness_evaluator: FitnessEvaluator,
        store: ExperienceStore,
        profile_config: dict[str, Any],
        mechanism_config: MechanismConfig,
        context_tags: list[str],
        sub_item_id: str,
        task: EvaluationTask,
        metrics_recorder: EvolutionMetricsRecorder | None = None,
        ab_enabled: bool = False,
    ) -> None:
        """Initialize UnifiedEvolutionProvider 🏗️.

        Args:
            mechanisms: List of BaseMechanism instances to orchestrate.
                Each must have ``enabled`` set according to ``mechanism_config``.
            collector: Trajectory collector for round record parsing.
            fitness_evaluator: Fitness evaluator for before/after comparison.
            store: Experience store for persistence.
            profile_config: Evolution profile configuration dict.
            mechanism_config: Per-mechanism budget and enablement config.
            context_tags: Context tags for experience matching.
            sub_item_id: Sub-item identifier being evaluated.
            task: The evaluation task being processed.
            metrics_recorder: Optional recorder for A/B metrics.
            ab_enabled: Enable A/B split (50% control / 50% treatment).
        """
        self._mechanisms = mechanisms
        self._collector = collector
        self._fitness = fitness_evaluator
        self._store = store
        self._profile_config = profile_config
        self._mechanism_config = mechanism_config
        self._context_tags = context_tags
        self._sub_item_id = sub_item_id
        self._task = task
        self._metrics_recorder = metrics_recorder

        # 📦 Cached state
        self._enrichment_result: EnrichmentResult | None = None
        self._prev_metrics: RoundMetrics | None = None

        # 🔬 A/B test assignment
        if ab_enabled:
            self.ab_group = "control" if random.random() < 0.5 else "treatment"
        else:
            self.ab_group = "treatment"

        logger.info(
            "🧬 UnifiedEvolutionProvider initialized: "
            "sub_item=%s, mechanisms=%s, ab_group=%s",
            sub_item_id,
            [m.mechanism_type.value for m in mechanisms if m.enabled],
            self.ab_group,
        )

    @property
    def enabled_mechanisms(self) -> list[BaseMechanism]:
        """Return only enabled mechanisms 📋.

        Returns:
            List of mechanisms with ``enabled == True``.
        """
        return [m for m in self._mechanisms if m.enabled]

    # ====================================================================
    # 🔄 Async Enrichment Preparation
    # ====================================================================

    async def prepare_enrichment(self) -> None:
        """Pre-fetch enrichment data from all mechanisms 🔄.

        Calls ``on_round_start()`` on each mechanism to refresh caches.
        Must be called from an async context before accessing sync getters.
        """
        try:
            for mechanism in self.enabled_mechanisms:
                await mechanism.on_round_start(1)
            logger.debug("✅ Enrichment prepared for all mechanisms")
        except Exception:
            logger.warning(
                "⚠️ Enrichment preparation failed",
                exc_info=True,
            )

    # ====================================================================
    # 🔍 Sync Enrichment Getters (EvolutionProvider Protocol)
    # ====================================================================

    def get_search_enrichment(
        self,
        round_num: int,
        gap_items: list[str],
    ) -> str | None:
        """Collect and merge enrichment from all mechanisms 🔍.

        Returns None for control group (A/B split) even when enrichment
        is available.

        Args:
            round_num: Current Discovery round (1-based).
            gap_items: Uncovered checklist item IDs from gap analysis.

        Returns:
            Combined markdown text from all mechanisms, or None.
        """
        try:
            if self.ab_group == "control":
                return None

            round_context = {
                "round_num": round_num,
                "gap_items": gap_items,
                "sub_item_id": self._sub_item_id,
                "context_tags": self._context_tags,
                "namespace": self._profile_config.get("namespace", ""),
                "available_tools": self._profile_config.get(
                    "available_tools", []
                ),
            }

            sections: list[str] = []
            for mechanism in self.enabled_mechanisms:
                section = mechanism.inject(round_context)
                if section:
                    sections.append(section)

            if not sections:
                return None

            combined = "\n\n".join(sections)
            return self._budget_truncate(combined)
        except Exception:
            logger.warning("⚠️ Search enrichment failed", exc_info=True)
            return None

    def get_analysis_enrichment(self) -> str | None:
        """Return combined enrichment for analysis prompt 🔬.

        Returns None for control group.

        Returns:
            Markdown text for analysis prompt, or None.
        """
        try:
            if self.ab_group == "control":
                return None

            round_context = {
                "round_num": 0,
                "gap_items": [],
                "sub_item_id": self._sub_item_id,
                "context_tags": self._context_tags,
                "namespace": self._profile_config.get("namespace", ""),
            }

            sections: list[str] = []
            for mechanism in self.enabled_mechanisms:
                section = mechanism.inject(round_context)
                if section:
                    sections.append(section)

            if not sections:
                return None

            base = "\n\n".join(sections)
            return (
                "# ANALYSIS INSIGHTS\n"
                "The following lessons from previous evaluations may help "
                "you analyze evidence more accurately.\n\n"
                f"{self._budget_truncate(base)}"
            )
        except Exception:
            logger.warning("⚠️ Analysis enrichment failed", exc_info=True)
            return None

    def get_synthesis_enrichment(self) -> str | None:
        """Return combined enrichment for synthesis prompt 📝.

        Returns None for control group.

        Returns:
            Markdown text for synthesis prompt, or None.
        """
        try:
            if self.ab_group == "control":
                return None

            round_context = {
                "round_num": 0,
                "gap_items": [],
                "sub_item_id": self._sub_item_id,
                "context_tags": self._context_tags,
                "namespace": self._profile_config.get("namespace", ""),
            }

            sections: list[str] = []
            for mechanism in self.enabled_mechanisms:
                section = mechanism.inject(round_context)
                if section:
                    sections.append(section)

            if not sections:
                return None

            base = "\n\n".join(sections)
            return (
                "# SYNTHESIS INSIGHTS\n"
                "The following lessons from previous evaluations may help "
                "produce better synthesis.\n\n"
                f"{self._budget_truncate(base)}"
            )
        except Exception:
            logger.warning("⚠️ Synthesis enrichment failed", exc_info=True)
            return None

    # ====================================================================
    # 🔄 Async Lifecycle Hooks
    # ====================================================================

    async def on_round_complete(
        self,
        round_num: int,
        round_record: DiscoveryRoundRecord,
        round_metrics: RoundMetrics,
    ) -> None:
        """Dispatch round completion to all mechanisms 🔄.

        Steps:
        1. Collect trajectory snapshot from round record
        2. Dispatch ``produce()`` to each mechanism → store experiences
        3. Dispatch ``on_round_end()`` to each mechanism
        4. Evaluate fitness if enrichment was active
        5. Refresh enrichment for next round via ``on_round_start()``
        6. Record metrics

        Args:
            round_num: Completed round number (1-based).
            round_record: Full record of the completed round.
            round_metrics: Computed metrics for fitness comparison.
        """
        try:
            # 1️⃣ Collect trajectory snapshot
            snapshot = self._collector.collect_from_round(
                round_record,
                self._task,
                self._context_tags,
            )

            round_context = {
                "round_num": round_num,
                "gap_items": [],
                "sub_item_id": self._sub_item_id,
                "context_tags": self._context_tags,
                "namespace": self._profile_config.get("namespace", ""),
            }

            # 2️⃣ Dispatch produce() to each mechanism
            total_stored = 0
            for mechanism in self.enabled_mechanisms:
                try:
                    experiences = await mechanism.produce(
                        snapshot, round_context,
                    )
                    for exp in experiences:
                        exp.namespace = self._profile_config.get(
                            "namespace", "default",
                        )
                        is_dup = await self._store.deduplicate(
                            namespace=exp.namespace,
                            new_insight=exp.insight,
                        )
                        if not is_dup:
                            await self._store.add(exp)
                            total_stored += 1
                except Exception:
                    logger.warning(
                        "⚠️ Mechanism %s produce() failed",
                        mechanism.mechanism_type.value,
                        exc_info=True,
                    )

            logger.info(
                "📦 Round %d: stored %d experiences from %d mechanisms",
                round_num,
                total_stored,
                len(self.enabled_mechanisms),
            )

            # 3️⃣ Dispatch on_round_end() to each mechanism
            for mechanism in self.enabled_mechanisms:
                try:
                    await mechanism.on_round_end(
                        round_num, round_record, round_metrics,
                    )
                except Exception:
                    logger.warning(
                        "⚠️ Mechanism %s on_round_end() failed",
                        mechanism.mechanism_type.value,
                        exc_info=True,
                    )

            # 4️⃣ Evaluate fitness if enrichment was active
            if self._enrichment_result and self._prev_metrics:
                before = self._round_to_result_metrics(self._prev_metrics)
                after = self._round_to_result_metrics(round_metrics)
                updates = await self._fitness.evaluate(
                    self._enrichment_result,
                    before,
                    after,
                    self._profile_config,
                )
                if updates:
                    learning_rate = self._profile_config.get(
                        "fitness_learning_rate", 0.3,
                    )
                    await self._fitness.apply_updates(updates, learning_rate)
                    logger.info(
                        "📊 Applied %d fitness updates for round %d",
                        len(updates),
                        round_num,
                    )

            # 5️⃣ Record metrics
            if self._metrics_recorder:
                enrichment_injected = (
                    self.ab_group == "treatment"
                    and self._enrichment_result is not None
                )
                exp_count = (
                    len(self._enrichment_result.injected_experience_ids)
                    if self._enrichment_result
                    else 0
                )
                self._metrics_recorder.record(
                    EvolutionRoundMetric(
                        namespace=self._profile_config.get("namespace", ""),
                        evaluation_id=getattr(
                            self._task, "evaluation_id", "",
                        ),
                        round_index=round_num - 1,
                        ab_group=self.ab_group,
                        enrichment_injected=enrichment_injected,
                        enrichment_experience_count=exp_count,
                        coverage=round_metrics.coverage,
                        evidence_count=round_metrics.evidence_count,
                        cost_usd=round_metrics.cost_usd,
                    )
                )

            # 6️⃣ Refresh enrichment for next round
            self._prev_metrics = round_metrics
            for mechanism in self.enabled_mechanisms:
                try:
                    await mechanism.on_round_start(round_num + 1)
                except Exception:
                    logger.warning(
                        "⚠️ Mechanism %s on_round_start() failed",
                        mechanism.mechanism_type.value,
                        exc_info=True,
                    )

        except Exception:
            logger.warning(
                "⚠️ UnifiedEvolutionProvider on_round_complete failed "
                "for round %d",
                round_num,
                exc_info=True,
            )

    async def on_synthesis_complete(
        self,
        synthesis_record: SynthesisRecord,
        final_metrics: RoundMetrics,
    ) -> None:
        """Final fitness update after synthesis 🏁.

        Args:
            synthesis_record: Full synthesis execution record.
            final_metrics: Final accumulated metrics.
        """
        try:
            if self._enrichment_result and self._prev_metrics:
                before = self._round_to_result_metrics(self._prev_metrics)
                after = self._round_to_result_metrics(final_metrics)
                updates = await self._fitness.evaluate(
                    self._enrichment_result,
                    before,
                    after,
                    self._profile_config,
                )
                if updates:
                    learning_rate = self._profile_config.get(
                        "fitness_learning_rate", 0.3,
                    )
                    await self._fitness.apply_updates(
                        updates, learning_rate,
                    )
                    logger.info(
                        "🏁 Applied %d final fitness updates after synthesis",
                        len(updates),
                    )
        except Exception:
            logger.warning(
                "⚠️ on_synthesis_complete failed",
                exc_info=True,
            )

    # ====================================================================
    # 🔧 Internal Helpers
    # ====================================================================

    def _budget_truncate(self, text: str) -> str:
        """Truncate text to fit within TOTAL_INJECTION_BUDGET 📏.

        Uses simple character-based estimation (4 chars ≈ 1 token).

        Args:
            text: Combined enrichment text.

        Returns:
            Truncated text if over budget, otherwise unchanged.
        """
        max_chars = TOTAL_INJECTION_BUDGET * 4
        if len(text) <= max_chars:
            return text
        logger.debug(
            "✂️ Truncating enrichment: %d chars → %d chars (%d tokens)",
            len(text),
            max_chars,
            TOTAL_INJECTION_BUDGET,
        )
        return text[:max_chars] + "\n\n*(truncated to fit token budget)*"

    @staticmethod
    def _round_to_result_metrics(
        round_metrics: RoundMetrics,
    ) -> ResultMetrics:
        """Convert RoundMetrics to ResultMetrics for fitness evaluator 🔄.

        Args:
            round_metrics: Metrics from a Discovery round.

        Returns:
            ResultMetrics suitable for fitness evaluation.
        """
        return ResultMetrics(
            evidence_count=round_metrics.evidence_count,
            confidence=round_metrics.coverage,
            cost_usd=round_metrics.cost_usd,
            search_rounds=round_metrics.round_index + 1,
            checklist_coverage=round_metrics.coverage,
        )
