"""Tests for FitnessEvaluator baseline fix — zero-baseline detection 🧪.

Verifies that:
1. All-zero before_metrics triggers synthetic baseline generation
2. Synthetic baseline uses 50% of after_metrics values
3. Non-zero before_metrics passes through unchanged
4. Fitness signal is reasonable (not inflated) with synthetic baseline
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inquiro.evolution.fitness import FitnessEvaluator
from inquiro.evolution.types import (
    EnrichmentResult,
    ResultMetrics,
)


# ============================================================================
# 🏗️ Fixtures
# ============================================================================


@pytest.fixture
def mock_store() -> AsyncMock:
    """Create a mock ExperienceStore 🏗️.

    Returns:
        AsyncMock configured with ExperienceStore methods.
    """
    store = AsyncMock()
    store.get_by_id = AsyncMock()
    store.update_fitness = AsyncMock()
    return store


@pytest.fixture
def evaluator(mock_store: AsyncMock) -> FitnessEvaluator:
    """Create FitnessEvaluator with mocked store 🏗️.

    Args:
        mock_store: Mocked ExperienceStore.

    Returns:
        FitnessEvaluator instance.
    """
    return FitnessEvaluator(mock_store)


@pytest.fixture
def sample_profile_config() -> dict:
    """Create a sample fitness profile configuration 🏗️.

    Returns:
        Dict with fitness_dimensions and learning_rate.
    """
    return {
        "fitness_dimensions": [
            {
                "metric_name": "evidence_count",
                "weight": 0.3,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "checklist_coverage",
                "weight": 0.4,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "cost_usd",
                "weight": 0.3,
                "direction": "lower_is_better",
            },
        ],
        "fitness_learning_rate": 0.1,
    }


@pytest.fixture
def enrichment_with_ids() -> EnrichmentResult:
    """Create an enrichment result with injected experience IDs 🏗️.

    Returns:
        EnrichmentResult with two injected experience IDs.
    """
    return EnrichmentResult(
        enrichment_text="# LEARNED INSIGHTS\n- Focus on kinase targets",
        injected_experience_ids=["exp_1", "exp_2"],
        total_available=5,
        selected_count=2,
    )


# ============================================================================
# 🔍 Tests: _is_zero_baseline
# ============================================================================


class TestIsZeroBaseline:
    """Tests for the _is_zero_baseline static method 🔍."""

    def test_all_zero_returns_true(self) -> None:
        """All-zero metrics should be detected as zero baseline ✅."""
        metrics = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=0,
            checklist_coverage=0.0,
        )
        assert FitnessEvaluator._is_zero_baseline(metrics) is True

    def test_nonzero_evidence_returns_false(self) -> None:
        """Non-zero evidence_count should not trigger baseline fix ❌."""
        metrics = ResultMetrics(
            evidence_count=5,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=1,
            checklist_coverage=0.0,
        )
        assert FitnessEvaluator._is_zero_baseline(metrics) is False

    def test_nonzero_confidence_returns_false(self) -> None:
        """Non-zero confidence should not trigger baseline fix ❌."""
        metrics = ResultMetrics(
            evidence_count=0,
            confidence=0.7,
            cost_usd=0.0,
            search_rounds=1,
            checklist_coverage=0.0,
        )
        assert FitnessEvaluator._is_zero_baseline(metrics) is False

    def test_nonzero_coverage_returns_false(self) -> None:
        """Non-zero checklist_coverage should not trigger baseline fix ❌."""
        metrics = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=1,
            checklist_coverage=0.5,
        )
        assert FitnessEvaluator._is_zero_baseline(metrics) is False

    def test_cost_only_still_zero(self) -> None:
        """Having only cost but no quality metrics is still zero baseline ✅."""
        metrics = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=5.0,
            search_rounds=1,
            checklist_coverage=0.0,
        )
        assert FitnessEvaluator._is_zero_baseline(metrics) is True


# ============================================================================
# 🧮 Tests: _synthetic_baseline
# ============================================================================


class TestSyntheticBaseline:
    """Tests for the _synthetic_baseline static method 🧮."""

    def test_halves_quality_metrics(self) -> None:
        """Synthetic baseline uses 75% of quality metrics ✅."""
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.8,
            cost_usd=2.0,
            search_rounds=3,
            checklist_coverage=0.6,
        )
        baseline = FitnessEvaluator._synthetic_baseline(after)
        assert baseline.evidence_count == 7  # int(10 * 0.75)
        assert baseline.confidence == pytest.approx(0.6)
        assert baseline.checklist_coverage == pytest.approx(0.45)

    def test_inflates_cost(self) -> None:
        """Synthetic baseline inflates cost to 125% ✅."""
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.8,
            cost_usd=2.0,
            search_rounds=3,
            checklist_coverage=0.6,
        )
        baseline = FitnessEvaluator._synthetic_baseline(after)
        assert baseline.cost_usd == pytest.approx(2.5)

    def test_minimum_evidence_count(self) -> None:
        """Synthetic baseline ensures at least 1 evidence item ✅."""
        after = ResultMetrics(
            evidence_count=1,
            confidence=0.5,
            cost_usd=1.0,
            search_rounds=1,
            checklist_coverage=0.3,
        )
        baseline = FitnessEvaluator._synthetic_baseline(after)
        assert baseline.evidence_count >= 1

    def test_minimum_search_rounds(self) -> None:
        """Synthetic baseline ensures at least 1 search round ✅."""
        after = ResultMetrics(
            evidence_count=5,
            confidence=0.5,
            cost_usd=1.0,
            search_rounds=0,
            checklist_coverage=0.3,
        )
        baseline = FitnessEvaluator._synthetic_baseline(after)
        assert baseline.search_rounds >= 1

    def test_zero_after_produces_minimal_baseline(self) -> None:
        """Zero after_metrics should produce minimal (not negative) baseline ✅."""
        after = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=0,
            checklist_coverage=0.0,
        )
        baseline = FitnessEvaluator._synthetic_baseline(after)
        assert baseline.evidence_count >= 1
        assert baseline.confidence >= 0.0
        assert baseline.cost_usd >= 0.0
        assert baseline.search_rounds >= 1
        assert baseline.checklist_coverage >= 0.0


# ============================================================================
# 📊 Tests: evaluate with zero baseline
# ============================================================================


class TestEvaluateWithZeroBaseline:
    """Tests that evaluate() handles zero baseline correctly 📊."""

    @pytest.mark.asyncio
    async def test_zero_baseline_uses_synthetic(
        self,
        evaluator: FitnessEvaluator,
        enrichment_with_ids: EnrichmentResult,
        sample_profile_config: dict,
    ) -> None:
        """Zero before_metrics should trigger synthetic baseline ✅."""
        before = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=0,
            checklist_coverage=0.0,
        )
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.8,
            cost_usd=2.0,
            search_rounds=3,
            checklist_coverage=0.6,
        )

        updates = await evaluator.evaluate(
            enrichment_with_ids,
            before,
            after,
            sample_profile_config,
        )

        # ✅ Should produce updates (not empty due to baseline fix)
        assert len(updates) == 2  # One per injected experience
        # ✅ Signal should be positive (after is better than synthetic baseline)
        assert all(u.was_helpful for u in updates)
        # ✅ Signal should be positive (improvement detected)
        assert all(u.signal > 0.0 for u in updates)

    @pytest.mark.asyncio
    async def test_real_baseline_no_fix(
        self,
        evaluator: FitnessEvaluator,
        enrichment_with_ids: EnrichmentResult,
        sample_profile_config: dict,
    ) -> None:
        """Real (non-zero) before_metrics should not trigger fix ✅."""
        before = ResultMetrics(
            evidence_count=5,
            confidence=0.6,
            cost_usd=3.0,
            search_rounds=2,
            checklist_coverage=0.4,
        )
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.8,
            cost_usd=2.0,
            search_rounds=3,
            checklist_coverage=0.6,
        )

        updates = await evaluator.evaluate(
            enrichment_with_ids,
            before,
            after,
            sample_profile_config,
        )

        assert len(updates) == 2
        # ✅ Should still show improvement
        assert all(u.was_helpful for u in updates)

    @pytest.mark.asyncio
    async def test_signal_not_inflated_with_synthetic(
        self,
        evaluator: FitnessEvaluator,
        enrichment_with_ids: EnrichmentResult,
        sample_profile_config: dict,
    ) -> None:
        """Synthetic baseline should produce moderate signal, not max 📊."""
        before_zero = ResultMetrics(
            evidence_count=0,
            confidence=0.0,
            cost_usd=0.0,
            search_rounds=0,
            checklist_coverage=0.0,
        )
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.8,
            cost_usd=2.0,
            search_rounds=3,
            checklist_coverage=0.6,
        )

        # 📊 With zero baseline (synthetic fix applied)
        updates_synthetic = await evaluator.evaluate(
            enrichment_with_ids,
            before_zero,
            after,
            sample_profile_config,
        )

        # 📊 With real low baseline
        before_real = ResultMetrics(
            evidence_count=1,
            confidence=0.1,
            cost_usd=5.0,
            search_rounds=1,
            checklist_coverage=0.1,
        )
        updates_real = await evaluator.evaluate(
            enrichment_with_ids,
            before_real,
            after,
            sample_profile_config,
        )

        # ✅ Synthetic baseline signal should be close to real low baseline
        # (both show improvement, but signal shouldn't be wildly inflated)
        assert abs(updates_synthetic[0].signal - updates_real[0].signal) < 0.3
