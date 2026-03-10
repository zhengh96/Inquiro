"""Multi-dimensional fitness evaluation for experience learning 📊.

The FitnessEvaluator computes fitness signals by comparing task outcomes
before and after experience injection, then applies exponential moving
average (EMA) updates to experience fitness scores.

This module is domain-agnostic — all fitness dimensions (metrics, weights,
directions) are injected via profile_config from the upper layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from inquiro.evolution.types import (
    EnrichmentResult,
    FitnessUpdate,
    ResultMetrics,
)

if TYPE_CHECKING:
    from inquiro.evolution.store import ExperienceStore

# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["FitnessEvaluator"]


logger = logging.getLogger(__name__)


# ============================================================================
# 📊 FitnessEvaluator
# ============================================================================


class FitnessEvaluator:
    """Multi-dimensional fitness evaluation engine 📊.

    Computes fitness signals by comparing ResultMetrics before and after
    experience injection across multiple dimensions (e.g., evidence_count,
    confidence, cost_usd). Uses exponential moving average (EMA) to update
    experience fitness scores over time.

    All domain knowledge (which metrics matter, weights, directions) comes
    from profile_config — this class is purely computational infrastructure.
    """

    def __init__(self, store: ExperienceStore) -> None:
        """Initialize FitnessEvaluator 🎯.

        Args:
            store: ExperienceStore for reading and updating fitness scores.
        """
        self._store = store

    async def evaluate(
        self,
        enrichment_result: EnrichmentResult,
        before_metrics: ResultMetrics,
        after_metrics: ResultMetrics,
        profile_config: dict[str, Any],
    ) -> list[FitnessUpdate]:
        """Evaluate fitness for injected experiences 🔍.

        Compares before/after metrics across all fitness dimensions defined
        in profile_config, computes weighted signal per experience, and
        creates FitnessUpdate records.

        Args:
            enrichment_result: Result of prompt enrichment (tracks injected IDs).
            before_metrics: Metrics from baseline run (no injection).
            after_metrics: Metrics from enriched run (with injection).
            profile_config: Dict with keys:
                - fitness_dimensions: list[dict] with metric_name, weight, direction
                - fitness_learning_rate: float (EMA alpha)

        Returns:
            List of FitnessUpdate records, one per injected experience.

        Raises:
            KeyError: If required profile_config keys are missing.
            ValueError: If fitness_dimensions is empty or malformed.
        """
        # 🔍 Extract config
        fitness_dimensions = profile_config.get("fitness_dimensions", [])
        if not fitness_dimensions:
            logger.warning(
                "No fitness dimensions in profile_config, returning empty updates 🚨"
            )
            return []

        # 🔧 Fix baseline-zero bug: if before_metrics is all zeros (first run
        # or missing baseline), synthesize a reasonable baseline from
        # after_metrics to avoid inflated fitness signals.
        if self._is_zero_baseline(before_metrics):
            before_metrics = self._synthetic_baseline(after_metrics)
            logger.info(
                "🔧 Using synthetic baseline (75%% of after_metrics) "
                "for first-run fitness evaluation"
            )

        injected_ids = enrichment_result.injected_experience_ids
        if not injected_ids:
            logger.debug("No experiences injected, no fitness updates to compute ✅")
            return []

        logger.info(
            "Evaluating fitness for %d experiences across %d dimensions 📊",
            len(injected_ids),
            len(fitness_dimensions),
        )

        # 📊 Compute per-dimension signals
        dimension_signals: list[float] = []
        metric_deltas: dict[str, float] = {}

        for dim in fitness_dimensions:
            metric_name = dim["metric_name"]
            weight = dim["weight"]
            direction = dim["direction"]

            # 🔢 Get metric values
            before_value = getattr(before_metrics, metric_name, None)
            after_value = getattr(after_metrics, metric_name, None)

            if before_value is None or after_value is None:
                logger.warning(
                    "Missing metric '%s' in before/after metrics, skipping dimension ⚠️",
                    metric_name,
                )
                continue

            # 📈 Compute delta
            delta = after_value - before_value
            metric_deltas[metric_name] = delta

            # 🎯 Normalize signal based on direction
            if direction == "higher_is_better":
                # Positive delta = positive signal
                signal = delta
            elif direction == "lower_is_better":
                # Negative delta = positive signal (e.g., cost reduction)
                signal = -delta
            else:
                logger.warning(
                    "Unknown direction '%s' for metric '%s',"
                    " treating as higher_is_better ⚠️",
                    direction,
                    metric_name,
                )
                signal = delta

            # ⚖️ Apply weight
            weighted_signal = signal * weight
            dimension_signals.append(weighted_signal)

            logger.debug(
                "Dimension %s: before=%.3f, after=%.3f,"
                " delta=%.3f, signal=%.3f, weighted=%.3f",
                metric_name,
                before_value,
                after_value,
                delta,
                signal,
                weighted_signal,
            )

        # 🧮 Aggregate signal (sum of weighted dimension signals)
        if not dimension_signals:
            logger.warning(
                "No valid dimension signals computed, returning empty updates 🚨"
            )
            return []

        current_signal = sum(dimension_signals)

        # 🔒 Clamp signal to [0.0, 1.0] for fitness score compatibility
        # Use sigmoid-like normalization to map arbitrary signals to [0, 1]
        # Simple approach: clip to [-1, 1] then shift to [0, 1]
        normalized_signal = max(0.0, min(1.0, (current_signal + 1.0) / 2.0))

        # ✅ Determine helpfulness (positive aggregate signal = helpful)
        was_helpful = current_signal > 0.0

        logger.info(
            "Aggregate signal: %.3f → normalized: %.3f, helpful: %s 📊",
            current_signal,
            normalized_signal,
            was_helpful,
        )

        # 📝 Create FitnessUpdate for each injected experience
        # All injected experiences share the same signal (simple credit assignment v1)
        updates = [
            FitnessUpdate(
                experience_id=exp_id,
                signal=normalized_signal,
                was_helpful=was_helpful,
                metric_deltas=metric_deltas,
            )
            for exp_id in injected_ids
        ]

        logger.info("Created %d fitness updates ✅", len(updates))
        return updates

    @staticmethod
    def _is_zero_baseline(metrics: ResultMetrics) -> bool:
        """Check if before_metrics is an all-zero placeholder 🔍.

        Returns True when evidence_count, confidence, and
        checklist_coverage are all zero — indicating no real
        baseline data is available.

        Args:
            metrics: Before-run metrics to check.

        Returns:
            True if the metrics appear to be an uninitialized baseline.
        """
        return (
            metrics.evidence_count == 0
            and metrics.confidence == 0.0
            and metrics.checklist_coverage == 0.0
        )

    @staticmethod
    def _synthetic_baseline(after: ResultMetrics) -> ResultMetrics:
        """Synthesize a reasonable baseline from after-run metrics 🧮.

        Uses 75% of the after-run values as a synthetic "before" baseline
        for first-run fitness evaluation.  The higher bar (vs the previous
        50%) means new experiences only receive positive reinforcement when
        they show real improvement.  Cost is inflated to 125% so cost
        reductions still register as an improvement signal.

        Args:
            after: After-run metrics to derive baseline from.

        Returns:
            Synthetic ResultMetrics with 75% quality and moderately
            inflated cost.
        """
        return ResultMetrics(
            evidence_count=max(1, int(after.evidence_count * 0.75)),
            confidence=after.confidence * 0.75,
            cost_usd=after.cost_usd * 1.25,
            search_rounds=max(1, after.search_rounds),
            checklist_coverage=after.checklist_coverage * 0.75,
        )

    async def apply_updates(
        self,
        updates: list[FitnessUpdate],
        learning_rate: float,
    ) -> None:
        """Apply fitness updates using EMA formula 📈.

        For each update:
            new_fitness = alpha * signal + (1 - alpha) * old_fitness

        Also increments times_used and times_helpful counters.

        Args:
            updates: List of FitnessUpdate records to apply.
            learning_rate: EMA alpha parameter (0.0 = no update, 1.0 = replace).

        Raises:
            ValueError: If learning_rate is not in (0.0, 1.0].
        """
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0.0, 1.0], got {learning_rate}"
            )

        if not updates:
            logger.debug("No updates to apply, skipping ✅")
            return

        logger.info(
            "Applying %d fitness updates with learning_rate=%.3f 📈",
            len(updates),
            learning_rate,
        )

        for update in updates:
            exp_id = update.experience_id

            # 🔍 Get current fitness from store
            experience = await self._store.get_by_id(exp_id)
            if experience is None:
                logger.warning(
                    "Experience %s not found in store, skipping update ⚠️",
                    exp_id,
                )
                continue

            old_fitness = experience.fitness_score

            # 🧮 Compute new fitness using EMA
            new_fitness = (
                learning_rate * update.signal + (1 - learning_rate) * old_fitness
            )

            # 🔒 Clamp to [0.0, 1.0]
            new_fitness = max(0.0, min(1.0, new_fitness))

            logger.debug(
                "Experience %s: old_fitness=%.3f, signal=%.3f → new_fitness=%.3f",
                exp_id,
                old_fitness,
                update.signal,
                new_fitness,
            )

            # 📝 Update store — pass precomputed value to avoid double EMA
            delta = FitnessUpdate(
                experience_id=exp_id,
                signal=update.signal,
                was_helpful=update.was_helpful,
                metric_deltas=update.metric_deltas,
            )
            await self._store.update_fitness(
                exp_id, delta, precomputed_fitness=new_fitness
            )

        logger.info("Successfully applied %d fitness updates ✅", len(updates))
