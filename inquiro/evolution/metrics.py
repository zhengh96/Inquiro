"""Evolution metrics recorder — track enrichment and A/B test performance 📊.

Provides domain-agnostic metrics collection for the self-evolution system.
All metrics are keyed by namespace and optionally segmented by A/B group.

Models:
    ``EvolutionRoundMetric`` — Per-round metrics snapshot
    ``EvolutionMetricsRecorder`` — In-memory metrics collector with aggregation

The recorder is intentionally in-memory for v1.  Persistence can be added
via periodic flush to ExperienceStore or a separate metrics table later.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = [
    "EvolutionRoundMetric",
    "EvolutionMetricsRecorder",
    "EvolutionMetricsSummary",
]


# ============================================================================
# 📊 Metric Models
# ============================================================================


class EvolutionRoundMetric(BaseModel):
    """Per-round evolution metrics snapshot 📊.

    Recorded once per DiscoveryLoop round (or once per CLASSIC evaluation)
    to track enrichment injection, A/B group assignment, and outcome.
    """

    namespace: str = Field(
        description="Namespace for data isolation",
    )
    evaluation_id: str = Field(
        default="",
        description="Evaluation task ID for grouping",
    )
    round_index: int = Field(
        description="Round number (0-based)",
        ge=0,
    )
    ab_group: str = Field(
        default="treatment",
        description="A/B test group: 'control' or 'treatment'",
    )
    enrichment_injected: bool = Field(
        default=False,
        description="Whether enrichment was injected this round",
    )
    enrichment_experience_count: int = Field(
        default=0,
        description="Number of experiences injected",
        ge=0,
    )
    coverage: float = Field(
        default=0.0,
        description="Checklist coverage ratio after this round (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    evidence_count: int = Field(
        default=0,
        description="Total evidence count after this round",
        ge=0,
    )
    cost_usd: float = Field(
        default=0.0,
        description="Cumulative cost in USD up to this round",
        ge=0.0,
    )
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recording timestamp (UTC)",
    )


class EvolutionMetricsSummary(BaseModel):
    """Aggregated evolution metrics summary 📈.

    Computed from a collection of EvolutionRoundMetric records,
    optionally filtered by A/B group.
    """

    namespace: str = Field(description="Namespace")
    total_evaluations: int = Field(
        default=0,
        description="Number of distinct evaluations",
    )
    total_rounds: int = Field(
        default=0,
        description="Total number of rounds recorded",
    )
    enrichment_injection_rate: float = Field(
        default=0.0,
        description="Fraction of rounds with enrichment injected (0.0-1.0)",
    )
    avg_round1_coverage: float = Field(
        default=0.0,
        description="Average coverage at round 1",
    )
    avg_final_coverage: float = Field(
        default=0.0,
        description="Average coverage at final round",
    )
    avg_rounds_to_convergence: float = Field(
        default=0.0,
        description="Average number of rounds to reach convergence",
    )
    avg_cost_per_evaluation: float = Field(
        default=0.0,
        description="Average total cost per evaluation",
    )
    ab_breakdown: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-group summary: {group: {metric: value}}",
    )


# ============================================================================
# 📊 Metrics Recorder
# ============================================================================


class EvolutionMetricsRecorder:
    """In-memory evolution metrics collector with aggregation 📊.

    Thread-safe via a ``threading.Lock``. Each metric is keyed by
    ``(namespace, evaluation_id)`` for grouping. Supports real-time
    aggregation and A/B group comparison.

    Usage::

        recorder = EvolutionMetricsRecorder()
        recorder.record(EvolutionRoundMetric(
            namespace="targetmaster",
            evaluation_id="eval-001",
            round_index=0,
            ab_group="treatment",
            enrichment_injected=True,
            enrichment_experience_count=5,
            coverage=0.4,
        ))
        summary = recorder.summarize("targetmaster")
    """

    def __init__(self) -> None:
        """Initialize EvolutionMetricsRecorder 🔧."""
        self._lock = threading.Lock()
        # 📊 Key: (namespace, evaluation_id) → list of round metrics
        self._records: dict[tuple[str, str], list[EvolutionRoundMetric]] = defaultdict(
            list
        )
        logger.debug("📊 EvolutionMetricsRecorder initialized")

    def record(self, metric: EvolutionRoundMetric) -> None:
        """Record a single round metric 📝.

        Args:
            metric: Round metrics snapshot to store.
        """
        key = (metric.namespace, metric.evaluation_id)
        with self._lock:
            self._records[key].append(metric)
        logger.debug(
            "📝 Recorded metric: ns=%s, eval=%s, round=%d, group=%s, enriched=%s",
            metric.namespace,
            metric.evaluation_id,
            metric.round_index,
            metric.ab_group,
            metric.enrichment_injected,
        )

    def summarize(
        self,
        namespace: str,
        ab_group: str | None = None,
    ) -> EvolutionMetricsSummary:
        """Compute aggregated summary for a namespace 📈.

        Args:
            namespace: Namespace to summarize.
            ab_group: Optional filter by A/B group.

        Returns:
            EvolutionMetricsSummary with aggregated metrics.
        """
        with self._lock:
            # 🔍 Collect all records for this namespace
            all_records: list[EvolutionRoundMetric] = []
            for (ns, _eval_id), records in self._records.items():
                if ns == namespace:
                    all_records.extend(records)

        # 🏷️ Filter by ab_group if specified
        if ab_group:
            all_records = [r for r in all_records if r.ab_group == ab_group]

        if not all_records:
            return EvolutionMetricsSummary(namespace=namespace)

        # 📊 Group by evaluation_id
        evals: dict[str, list[EvolutionRoundMetric]] = defaultdict(list)
        for record in all_records:
            evals[record.evaluation_id].append(record)

        # 📈 Compute aggregates
        total_evaluations = len(evals)
        total_rounds = len(all_records)

        enrichment_count = sum(1 for r in all_records if r.enrichment_injected)
        enrichment_rate = enrichment_count / total_rounds if total_rounds > 0 else 0.0

        # 🎯 Round 1 coverage (round_index == 0)
        round1_coverages = [r.coverage for r in all_records if r.round_index == 0]
        avg_round1 = (
            sum(round1_coverages) / len(round1_coverages) if round1_coverages else 0.0
        )

        # 🏁 Final coverage (last round per evaluation)
        final_coverages = []
        total_costs = []
        rounds_per_eval = []
        for eval_records in evals.values():
            sorted_records = sorted(
                eval_records,
                key=lambda r: r.round_index,
            )
            if sorted_records:
                final_coverages.append(sorted_records[-1].coverage)
                total_costs.append(sorted_records[-1].cost_usd)
                rounds_per_eval.append(len(sorted_records))

        avg_final = (
            sum(final_coverages) / len(final_coverages) if final_coverages else 0.0
        )
        avg_cost = sum(total_costs) / len(total_costs) if total_costs else 0.0
        avg_rounds = (
            sum(rounds_per_eval) / len(rounds_per_eval) if rounds_per_eval else 0.0
        )

        # 🔬 A/B breakdown (if no group filter applied)
        ab_breakdown: dict[str, dict[str, float]] = {}
        if not ab_group:
            groups = {r.ab_group for r in all_records}
            for group in groups:
                group_records = [r for r in all_records if r.ab_group == group]
                group_evals: dict[str, list[EvolutionRoundMetric]] = defaultdict(list)
                for r in group_records:
                    group_evals[r.evaluation_id].append(r)

                g_round1 = [r.coverage for r in group_records if r.round_index == 0]
                g_final = []
                g_costs = []
                g_rounds = []
                for eval_recs in group_evals.values():
                    s = sorted(eval_recs, key=lambda r: r.round_index)
                    if s:
                        g_final.append(s[-1].coverage)
                        g_costs.append(s[-1].cost_usd)
                        g_rounds.append(len(s))

                ab_breakdown[group] = {
                    "evaluations": float(len(group_evals)),
                    "rounds": float(len(group_records)),
                    "enrichment_rate": (
                        sum(1 for r in group_records if r.enrichment_injected)
                        / len(group_records)
                        if group_records
                        else 0.0
                    ),
                    "avg_round1_coverage": (
                        sum(g_round1) / len(g_round1) if g_round1 else 0.0
                    ),
                    "avg_final_coverage": (
                        sum(g_final) / len(g_final) if g_final else 0.0
                    ),
                    "avg_cost": (sum(g_costs) / len(g_costs) if g_costs else 0.0),
                    "avg_rounds": (sum(g_rounds) / len(g_rounds) if g_rounds else 0.0),
                }

        return EvolutionMetricsSummary(
            namespace=namespace,
            total_evaluations=total_evaluations,
            total_rounds=total_rounds,
            enrichment_injection_rate=enrichment_rate,
            avg_round1_coverage=avg_round1,
            avg_final_coverage=avg_final,
            avg_rounds_to_convergence=avg_rounds,
            avg_cost_per_evaluation=avg_cost,
            ab_breakdown=ab_breakdown,
        )

    def get_raw_records(
        self,
        namespace: str,
    ) -> list[EvolutionRoundMetric]:
        """Return all raw records for a namespace 📋.

        Args:
            namespace: Namespace to retrieve.

        Returns:
            List of all round metrics for this namespace.
        """
        with self._lock:
            records: list[EvolutionRoundMetric] = []
            for (ns, _eval_id), recs in self._records.items():
                if ns == namespace:
                    records.extend(recs)
            return records

    def clear(self, namespace: str | None = None) -> int:
        """Clear metrics records 🧹.

        Args:
            namespace: If provided, only clear records for this namespace.
                If None, clear all records.

        Returns:
            Number of records cleared.
        """
        with self._lock:
            if namespace is None:
                count = sum(len(v) for v in self._records.values())
                self._records.clear()
                return count

            keys_to_remove = [k for k in self._records if k[0] == namespace]
            count = sum(len(self._records[k]) for k in keys_to_remove)
            for k in keys_to_remove:
                del self._records[k]
            return count
