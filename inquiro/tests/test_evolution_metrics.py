"""Tests for EvolutionMetrics recorder and aggregation 📊.

Verifies:
    - EvolutionRoundMetric construction and validation
    - EvolutionMetricsRecorder recording and summarization
    - A/B group breakdown computation
    - Thread safety of the recorder
    - Edge cases (empty records, single record, etc.)
"""

from __future__ import annotations


from inquiro.evolution.metrics import (
    EvolutionMetricsRecorder,
    EvolutionRoundMetric,
)


# ============================================================================
# 🏗️ Test Helpers
# ============================================================================


def _make_metric(
    namespace: str = "test-ns",
    evaluation_id: str = "eval-001",
    round_index: int = 0,
    ab_group: str = "treatment",
    enrichment_injected: bool = True,
    enrichment_experience_count: int = 3,
    coverage: float = 0.5,
    evidence_count: int = 10,
    cost_usd: float = 0.05,
) -> EvolutionRoundMetric:
    """Build a test metric 🧪."""
    return EvolutionRoundMetric(
        namespace=namespace,
        evaluation_id=evaluation_id,
        round_index=round_index,
        ab_group=ab_group,
        enrichment_injected=enrichment_injected,
        enrichment_experience_count=enrichment_experience_count,
        coverage=coverage,
        evidence_count=evidence_count,
        cost_usd=cost_usd,
    )


# ============================================================================
# 📊 EvolutionRoundMetric Tests
# ============================================================================


class TestEvolutionRoundMetric:
    """Tests for EvolutionRoundMetric model 📊."""

    def test_construction_with_defaults(self) -> None:
        """Metric should construct with sensible defaults."""
        metric = EvolutionRoundMetric(
            namespace="test",
            round_index=0,
        )
        assert metric.ab_group == "treatment"
        assert metric.enrichment_injected is False
        assert metric.enrichment_experience_count == 0
        assert metric.coverage == 0.0

    def test_construction_with_all_fields(self) -> None:
        """Metric should accept all fields."""
        metric = _make_metric()
        assert metric.namespace == "test-ns"
        assert metric.evaluation_id == "eval-001"
        assert metric.round_index == 0
        assert metric.ab_group == "treatment"
        assert metric.enrichment_injected is True
        assert metric.enrichment_experience_count == 3
        assert metric.coverage == 0.5

    def test_recorded_at_auto_populated(self) -> None:
        """Metric should auto-populate recorded_at timestamp."""
        metric = _make_metric()
        assert metric.recorded_at is not None


# ============================================================================
# 📊 EvolutionMetricsRecorder Tests
# ============================================================================


class TestEvolutionMetricsRecorder:
    """Tests for EvolutionMetricsRecorder 📊."""

    def test_empty_summary(self) -> None:
        """Summarize should return empty summary when no records."""
        recorder = EvolutionMetricsRecorder()
        summary = recorder.summarize("test-ns")
        assert summary.total_evaluations == 0
        assert summary.total_rounds == 0
        assert summary.enrichment_injection_rate == 0.0

    def test_record_and_summarize_single(self) -> None:
        """Summarize should work with a single record."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric())
        summary = recorder.summarize("test-ns")
        assert summary.total_evaluations == 1
        assert summary.total_rounds == 1
        assert summary.enrichment_injection_rate == 1.0
        assert summary.avg_round1_coverage == 0.5

    def test_multi_round_single_eval(self) -> None:
        """Summarize should aggregate multiple rounds for one eval."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric(round_index=0, coverage=0.3, cost_usd=0.02))
        recorder.record(_make_metric(round_index=1, coverage=0.6, cost_usd=0.04))
        recorder.record(_make_metric(round_index=2, coverage=0.9, cost_usd=0.06))

        summary = recorder.summarize("test-ns")
        assert summary.total_evaluations == 1
        assert summary.total_rounds == 3
        assert summary.avg_round1_coverage == 0.3
        assert summary.avg_final_coverage == 0.9
        assert summary.avg_rounds_to_convergence == 3.0
        assert summary.avg_cost_per_evaluation == 0.06

    def test_multi_eval(self) -> None:
        """Summarize should aggregate across multiple evaluations."""
        recorder = EvolutionMetricsRecorder()
        # Eval 1: 2 rounds
        recorder.record(
            _make_metric(
                evaluation_id="eval-001",
                round_index=0,
                coverage=0.4,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-001",
                round_index=1,
                coverage=0.8,
            )
        )
        # Eval 2: 3 rounds
        recorder.record(
            _make_metric(
                evaluation_id="eval-002",
                round_index=0,
                coverage=0.5,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-002",
                round_index=1,
                coverage=0.7,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-002",
                round_index=2,
                coverage=0.95,
            )
        )

        summary = recorder.summarize("test-ns")
        assert summary.total_evaluations == 2
        assert summary.total_rounds == 5
        # Round 1 avg: (0.4 + 0.5) / 2 = 0.45
        assert abs(summary.avg_round1_coverage - 0.45) < 0.01
        # Final avg: (0.8 + 0.95) / 2 = 0.875
        assert abs(summary.avg_final_coverage - 0.875) < 0.01
        # Rounds avg: (2 + 3) / 2 = 2.5
        assert abs(summary.avg_rounds_to_convergence - 2.5) < 0.01

    def test_ab_breakdown(self) -> None:
        """Summarize should include A/B group breakdown."""
        recorder = EvolutionMetricsRecorder()
        # Treatment group
        recorder.record(
            _make_metric(
                evaluation_id="eval-t1",
                ab_group="treatment",
                round_index=0,
                coverage=0.6,
                enrichment_injected=True,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-t1",
                ab_group="treatment",
                round_index=1,
                coverage=0.9,
                enrichment_injected=True,
            )
        )
        # Control group
        recorder.record(
            _make_metric(
                evaluation_id="eval-c1",
                ab_group="control",
                round_index=0,
                coverage=0.3,
                enrichment_injected=False,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-c1",
                ab_group="control",
                round_index=1,
                coverage=0.5,
                enrichment_injected=False,
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-c1",
                ab_group="control",
                round_index=2,
                coverage=0.7,
                enrichment_injected=False,
            )
        )

        summary = recorder.summarize("test-ns")
        assert "treatment" in summary.ab_breakdown
        assert "control" in summary.ab_breakdown

        treatment = summary.ab_breakdown["treatment"]
        assert treatment["evaluations"] == 1.0
        assert treatment["enrichment_rate"] == 1.0

        control = summary.ab_breakdown["control"]
        assert control["evaluations"] == 1.0
        assert control["enrichment_rate"] == 0.0

    def test_filter_by_ab_group(self) -> None:
        """Summarize should filter by ab_group when specified."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(
            _make_metric(
                evaluation_id="eval-t1",
                ab_group="treatment",
            )
        )
        recorder.record(
            _make_metric(
                evaluation_id="eval-c1",
                ab_group="control",
            )
        )

        # Filter to treatment only
        summary = recorder.summarize("test-ns", ab_group="treatment")
        assert summary.total_evaluations == 1
        assert summary.total_rounds == 1

        # Filter to control only
        summary = recorder.summarize("test-ns", ab_group="control")
        assert summary.total_evaluations == 1

    def test_namespace_isolation(self) -> None:
        """Summarize should only return records for requested namespace."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric(namespace="ns-a"))
        recorder.record(_make_metric(namespace="ns-b"))

        summary_a = recorder.summarize("ns-a")
        assert summary_a.total_rounds == 1
        summary_b = recorder.summarize("ns-b")
        assert summary_b.total_rounds == 1
        summary_c = recorder.summarize("ns-c")
        assert summary_c.total_rounds == 0

    def test_get_raw_records(self) -> None:
        """get_raw_records should return all records for namespace."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric(namespace="test-ns"))
        recorder.record(_make_metric(namespace="test-ns", round_index=1))
        recorder.record(_make_metric(namespace="other-ns"))

        records = recorder.get_raw_records("test-ns")
        assert len(records) == 2

    def test_clear_all(self) -> None:
        """clear() should remove all records when no namespace specified."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric(namespace="ns-a"))
        recorder.record(_make_metric(namespace="ns-b"))

        count = recorder.clear()
        assert count == 2
        assert recorder.summarize("ns-a").total_rounds == 0
        assert recorder.summarize("ns-b").total_rounds == 0

    def test_clear_by_namespace(self) -> None:
        """clear() should only remove records for specified namespace."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(_make_metric(namespace="ns-a"))
        recorder.record(_make_metric(namespace="ns-b"))

        count = recorder.clear("ns-a")
        assert count == 1
        assert recorder.summarize("ns-a").total_rounds == 0
        assert recorder.summarize("ns-b").total_rounds == 1

    def test_enrichment_rate_partial(self) -> None:
        """Enrichment injection rate should be fractional."""
        recorder = EvolutionMetricsRecorder()
        recorder.record(
            _make_metric(
                round_index=0,
                enrichment_injected=True,
            )
        )
        recorder.record(
            _make_metric(
                round_index=1,
                enrichment_injected=False,
            )
        )

        summary = recorder.summarize("test-ns")
        assert abs(summary.enrichment_injection_rate - 0.5) < 0.01
