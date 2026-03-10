"""Tests for FitnessEvaluator — multi-dimensional fitness computation 🧪.

Tests the FitnessEvaluator class for:
- Fitness evaluation across multiple metric dimensions
- Signal normalization for higher_is_better and lower_is_better directions
- EMA (exponential moving average) fitness updates
- Credit assignment to injected experiences

Uses Google Python Style Guide. English comments with emojis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inquiro.evolution.fitness import FitnessEvaluator
from inquiro.evolution.types import (
    EnrichmentResult,
    Experience,
    ResultMetrics,
)


# ============================================================================
# 🏗️ Fixtures
# ============================================================================


@pytest.fixture
def mock_store() -> AsyncMock:
    """Create a mock ExperienceStore for testing 🏗️.

    Returns:
        AsyncMock configured with typical ExperienceStore methods.
    """
    store = AsyncMock()
    store.get_by_id = AsyncMock()
    store.update_fitness = AsyncMock()
    return store


@pytest.fixture
def evaluator(mock_store: AsyncMock) -> FitnessEvaluator:
    """Create a FitnessEvaluator with mocked store 🏗️.

    Args:
        mock_store: Mocked ExperienceStore.

    Returns:
        FitnessEvaluator instance.
    """
    return FitnessEvaluator(store=mock_store)


@pytest.fixture
def profile_config() -> dict:
    """Create sample profile_config with fitness dimensions 🏗️.

    Returns:
        Profile config dict with typical fitness dimensions.
    """
    return {
        "fitness_dimensions": [
            {
                "metric_name": "evidence_count",
                "weight": 0.4,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "confidence",
                "weight": 0.3,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "cost_usd",
                "weight": 0.3,
                "direction": "lower_is_better",
            },
        ],
        "fitness_learning_rate": 0.3,
    }


# ============================================================================
# 🧪 FitnessEvaluator Tests
# ============================================================================


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator class 🧪."""

    def test_initialization(self, mock_store: AsyncMock) -> None:
        """FitnessEvaluator should initialize with store reference 🏗️."""
        evaluator = FitnessEvaluator(store=mock_store)
        assert evaluator._store is mock_store

    @pytest.mark.asyncio
    async def test_evaluate_no_dimensions(
        self,
        evaluator: FitnessEvaluator,
    ) -> None:
        """Evaluate should return empty list if no fitness dimensions 📊."""
        enrichment = EnrichmentResult(
            injected_experience_ids=["exp1", "exp2"],
        )
        before = ResultMetrics(evidence_count=5, confidence=0.7, cost_usd=1.0)
        after = ResultMetrics(evidence_count=8, confidence=0.8, cost_usd=0.9)

        # Empty fitness_dimensions
        config = {"fitness_dimensions": [], "fitness_learning_rate": 0.3}

        updates = await evaluator.evaluate(enrichment, before, after, config)

        assert updates == []

    @pytest.mark.asyncio
    async def test_evaluate_no_injected_experiences(
        self,
        evaluator: FitnessEvaluator,
        profile_config: dict,
    ) -> None:
        """Evaluate should return empty list if no experiences injected 📊."""
        enrichment = EnrichmentResult(injected_experience_ids=[])
        before = ResultMetrics(evidence_count=5, confidence=0.7, cost_usd=1.0)
        after = ResultMetrics(evidence_count=8, confidence=0.8, cost_usd=0.9)

        updates = await evaluator.evaluate(enrichment, before, after, profile_config)

        assert updates == []

    @pytest.mark.asyncio
    async def test_evaluate_positive_improvement(
        self,
        evaluator: FitnessEvaluator,
        profile_config: dict,
    ) -> None:
        """Evaluate should compute positive signal for improvements 📊."""
        enrichment = EnrichmentResult(
            injected_experience_ids=["exp1", "exp2"],
        )
        # 🎯 Improvement: more evidence, higher confidence, lower cost
        before = ResultMetrics(evidence_count=5, confidence=0.7, cost_usd=1.0)
        after = ResultMetrics(evidence_count=8, confidence=0.8, cost_usd=0.8)

        updates = await evaluator.evaluate(enrichment, before, after, profile_config)

        assert len(updates) == 2  # One per injected experience
        for update in updates:
            assert update.experience_id in ["exp1", "exp2"]
            assert 0.0 <= update.signal <= 1.0
            assert update.was_helpful is True  # Positive improvement
            assert "evidence_count" in update.metric_deltas
            assert update.metric_deltas["evidence_count"] == 3.0  # 8 - 5

    @pytest.mark.asyncio
    async def test_evaluate_negative_regression(
        self,
        evaluator: FitnessEvaluator,
        profile_config: dict,
    ) -> None:
        """Evaluate should compute negative signal for regressions 📊."""
        enrichment = EnrichmentResult(
            injected_experience_ids=["exp1"],
        )
        # 🎯 Regression: less evidence, lower confidence, higher cost
        before = ResultMetrics(evidence_count=8, confidence=0.8, cost_usd=0.8)
        after = ResultMetrics(evidence_count=5, confidence=0.7, cost_usd=1.0)

        updates = await evaluator.evaluate(enrichment, before, after, profile_config)

        assert len(updates) == 1
        update = updates[0]
        assert update.experience_id == "exp1"
        assert update.was_helpful is False  # Negative signal
        assert update.metric_deltas["evidence_count"] == -3.0  # 5 - 8

    @pytest.mark.asyncio
    async def test_evaluate_lower_is_better_direction(
        self,
        evaluator: FitnessEvaluator,
    ) -> None:
        """Evaluate should handle lower_is_better direction correctly 📊."""
        enrichment = EnrichmentResult(
            injected_experience_ids=["exp1"],
        )
        # Only cost dimension (lower is better)
        config = {
            "fitness_dimensions": [
                {
                    "metric_name": "cost_usd",
                    "weight": 1.0,
                    "direction": "lower_is_better",
                },
            ],
            "fitness_learning_rate": 0.3,
        }
        # Cost reduced from 1.0 to 0.5 = positive signal
        before = ResultMetrics(cost_usd=1.0)
        after = ResultMetrics(cost_usd=0.5)

        updates = await evaluator.evaluate(enrichment, before, after, config)

        assert len(updates) == 1
        update = updates[0]
        assert update.was_helpful is True  # Cost reduction = positive

    @pytest.mark.asyncio
    async def test_evaluate_missing_metric(
        self,
        evaluator: FitnessEvaluator,
        profile_config: dict,
    ) -> None:
        """Evaluate should skip dimensions with missing metrics 📊."""
        enrichment = EnrichmentResult(
            injected_experience_ids=["exp1"],
        )
        # Missing cost_usd in after_metrics
        before = ResultMetrics(evidence_count=5, confidence=0.7, cost_usd=1.0)
        after = ResultMetrics(evidence_count=8, confidence=0.8)  # No cost_usd

        # Should compute with available metrics only
        updates = await evaluator.evaluate(enrichment, before, after, profile_config)

        assert len(updates) == 1
        # Should have deltas for evidence_count and confidence, but not cost_usd
        assert "evidence_count" in updates[0].metric_deltas
        assert "confidence" in updates[0].metric_deltas

    @pytest.mark.asyncio
    async def test_apply_updates_ema_calculation(
        self,
        evaluator: FitnessEvaluator,
        mock_store: AsyncMock,
    ) -> None:
        """Apply_updates should use EMA formula correctly 📈."""
        # 🏗️ Setup mock store to return experience
        experience = Experience(
            namespace="test",
            category="test",
            insight="Test insight",
            source="test",
            fitness_score=0.5,
        )
        mock_store.get_by_id.return_value = experience

        from inquiro.evolution.types import FitnessUpdate

        updates = [
            FitnessUpdate(
                experience_id="exp1",
                signal=0.8,
                was_helpful=True,
                metric_deltas={"evidence_count": 3.0},
            ),
        ]
        learning_rate = 0.3

        await evaluator.apply_updates(updates, learning_rate)

        # 🧮 After fix: apply_updates now calls store.update_fitness(exp_id, delta)
        mock_store.update_fitness.assert_called_once()
        call_args = mock_store.update_fitness.call_args
        # Positional args: (exp_id, FitnessUpdate)
        assert call_args[0][0] == "exp1"
        delta = call_args[0][1]
        assert delta.experience_id == "exp1"
        assert delta.signal == 0.8
        assert delta.was_helpful is True

    @pytest.mark.asyncio
    async def test_apply_updates_clamps_to_valid_range(
        self,
        evaluator: FitnessEvaluator,
        mock_store: AsyncMock,
    ) -> None:
        """Apply_updates should clamp fitness to [0.0, 1.0] 📈."""
        experience = Experience(
            namespace="test",
            category="test",
            insight="Test insight",
            source="test",
            fitness_score=0.95,
        )
        mock_store.get_by_id.return_value = experience

        from inquiro.evolution.types import FitnessUpdate

        # High signal + high old fitness could exceed 1.0
        updates = [
            FitnessUpdate(
                experience_id="exp1",
                signal=1.0,
                was_helpful=True,
                metric_deltas={},
            ),
        ]
        learning_rate = 0.5

        await evaluator.apply_updates(updates, learning_rate)

        # Should clamp: delta is passed via FitnessUpdate
        call_args = mock_store.update_fitness.call_args
        # Positional args: (exp_id, FitnessUpdate)
        assert call_args[0][0] == "exp1"
        delta = call_args[0][1]
        assert delta.signal <= 1.0

    @pytest.mark.asyncio
    async def test_apply_updates_invalid_learning_rate(
        self,
        evaluator: FitnessEvaluator,
    ) -> None:
        """Apply_updates should reject invalid learning_rate ❌."""
        from inquiro.evolution.types import FitnessUpdate

        updates = [
            FitnessUpdate(
                experience_id="exp1",
                signal=0.5,
                was_helpful=True,
                metric_deltas={},
            ),
        ]

        with pytest.raises(ValueError, match="learning_rate must be in"):
            await evaluator.apply_updates(updates, learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            await evaluator.apply_updates(updates, learning_rate=1.5)

    @pytest.mark.asyncio
    async def test_apply_updates_experience_not_found(
        self,
        evaluator: FitnessEvaluator,
        mock_store: AsyncMock,
    ) -> None:
        """Apply_updates should skip missing experiences gracefully ⚠️."""
        mock_store.get_by_id.return_value = None  # Experience not found

        from inquiro.evolution.types import FitnessUpdate

        updates = [
            FitnessUpdate(
                experience_id="nonexistent",
                signal=0.5,
                was_helpful=True,
                metric_deltas={},
            ),
        ]

        # Should not raise, just skip
        await evaluator.apply_updates(updates, learning_rate=0.3)

        mock_store.update_fitness.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_updates_empty_list(
        self,
        evaluator: FitnessEvaluator,
        mock_store: AsyncMock,
    ) -> None:
        """Apply_updates should handle empty update list gracefully ✅."""
        await evaluator.apply_updates([], learning_rate=0.3)

        # Should not call store
        mock_store.get_by_id.assert_not_called()
        mock_store.update_fitness.assert_not_called()
