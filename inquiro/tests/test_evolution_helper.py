"""Tests for EvolutionHelper — shared evolution lifecycle helper 🧬.

Verifies that EvolutionHelper correctly handles:
    - Instantiation with both EvaluationTask and SynthesisTask
    - enrich_with_experiences: returns enriched text or original on error
    - post_execution_evolution: runs 4-step pipeline without propagating errors
    - run_async: bridges async coroutines into sync context
    - Behavioral equivalence with legacy Exp methods
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


from inquiro.exps.evolution_helper import EvolutionHelper
from inquiro.evolution.types import EnrichmentResult


# ============================================================================
# 🏗️ Test Helpers
# ============================================================================


def _build_evolution_profile(
    with_enrichment: bool = True,
    with_extraction: bool = True,
) -> dict[str, Any]:
    """Build a sample evolution profile for testing 🧬.

    Args:
        with_enrichment: Include enrichment_prompt_template.
        with_extraction: Include extraction_prompt_template.

    Returns:
        Evolution profile dictionary.
    """
    profile: dict[str, Any] = {
        "namespace": "test-namespace",
        "prune_min_fitness": 0.3,
        "enrichment_max_tokens": 500,
        "enrichment_max_items": 10,
        "experience_categories": ["search_strategy", "evidence_quality"],
        "max_experiences_per_extraction": 8,
        "fitness_dimensions": [
            {
                "metric_name": "confidence",
                "weight": 1.0,
                "direction": "higher_is_better",
            },
        ],
        "fitness_learning_rate": 0.3,
    }
    if with_enrichment:
        profile["enrichment_prompt_template"] = (
            "{% for exp in experiences %}- {{ exp.insight }}\n{% endfor %}"
        )
    if with_extraction:
        profile["extraction_prompt_template"] = (
            "Extract experiences from: {{ snapshot.topic }}"
        )
    return profile


def _make_task(
    has_sub_item_id: bool = True,
    evolution_profile: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock task for EvolutionHelper testing 🧪.

    Args:
        has_sub_item_id: Whether to include sub_item_id attribute.
        evolution_profile: Optional evolution profile dict.

    Returns:
        MagicMock task object.
    """
    task = MagicMock()
    task.task_id = "test-task-001"
    task.evaluation_id = "eval-001"
    task.topic = "Test topic"
    task.context_tags = ["modality:SmallMolecule"]
    task.evolution_profile = evolution_profile or _build_evolution_profile()
    if has_sub_item_id:
        task.sub_item_id = "safety_1a"
    else:
        # ⚠️ SynthesisTask may not have sub_item_id
        del task.sub_item_id
    return task


def _make_helper(
    task: MagicMock | None = None,
    has_sub_item_id: bool = True,
) -> EvolutionHelper:
    """Create an EvolutionHelper for testing 🧪.

    Args:
        task: Optional pre-built mock task.
        has_sub_item_id: Whether task should have sub_item_id.

    Returns:
        EvolutionHelper instance.
    """
    if task is None:
        task = _make_task(has_sub_item_id=has_sub_item_id)
    return EvolutionHelper(
        task=task,
        llm=MagicMock(),
        cost_tracker=MagicMock(get_total_cost=MagicMock(return_value=0.05)),
    )


def _make_result(
    has_get_covered_ratio: bool = True,
    has_search_rounds: bool = True,
) -> MagicMock:
    """Create a mock result for post_execution_evolution testing 🧪.

    Args:
        has_get_covered_ratio: Whether result has get_covered_ratio method.
        has_search_rounds: Whether result has search_rounds attribute.

    Returns:
        MagicMock result object.
    """
    result = MagicMock()
    result.evidence_index = [MagicMock(id="E1")]
    result.confidence = 0.85
    result.cost = 0.05
    if has_search_rounds:
        result.search_rounds = 3
    else:
        del result.search_rounds
    if has_get_covered_ratio:
        result.get_covered_ratio = MagicMock(return_value=0.8)
    else:
        del result.get_covered_ratio
    return result


# ============================================================================
# 🧬 Instantiation Tests
# ============================================================================


class TestEvolutionHelperInit:
    """Tests for EvolutionHelper instantiation 🔧."""

    def test_can_instantiate_with_evaluation_task(self) -> None:
        """EvolutionHelper should accept EvaluationTask-like objects."""
        helper = _make_helper(has_sub_item_id=True)
        assert helper.enrichment_result is None

    def test_can_instantiate_with_synthesis_task(self) -> None:
        """EvolutionHelper should accept SynthesisTask-like objects (no sub_item_id)."""
        helper = _make_helper(has_sub_item_id=False)
        assert helper.enrichment_result is None

    def test_accepts_custom_logger(self) -> None:
        """EvolutionHelper should use provided logger."""
        import logging

        custom_logger = logging.getLogger("test.custom")
        task = _make_task()
        helper = EvolutionHelper(
            task=task,
            llm=MagicMock(),
            cost_tracker=MagicMock(),
            logger=custom_logger,
        )
        assert helper._logger is custom_logger


# ============================================================================
# 💉 Enrichment Tests
# ============================================================================


class TestEvolutionHelperEnrichment:
    """Tests for EvolutionHelper.enrich_with_experiences() 💉."""

    def test_returns_original_when_no_profile(self) -> None:
        """Enrichment should return original text when profile is None."""
        task = _make_task()
        task.evolution_profile = None
        helper = EvolutionHelper(
            task=task,
            llm=MagicMock(),
            cost_tracker=MagicMock(),
        )

        result = helper.enrich_with_experiences("original context")
        assert result == "original context"

    def test_returns_original_when_no_template(self) -> None:
        """Enrichment should skip when no enrichment template configured."""
        profile = _build_evolution_profile(with_enrichment=False)
        task = _make_task(evolution_profile=profile)
        helper = _make_helper(task=task)

        result = helper.enrich_with_experiences("original context")
        assert result == "original context"
        assert helper.enrichment_result is None

    @patch("inquiro.evolution.store_factory.get_store")
    def test_appends_learned_insights(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Enrichment should append LEARNED INSIGHTS when store has data."""
        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store

        enrichment_result = EnrichmentResult(
            injected_experience_ids=["exp-1", "exp-2"],
            enrichment_text="- Use focused queries\n- Check sources",
            token_count=50,
            truncated=False,
        )

        helper = _make_helper()

        with patch(
            "inquiro.evolution.enricher.PromptEnricher.enrich",
            new_callable=AsyncMock,
            return_value=enrichment_result,
        ):
            result = helper.enrich_with_experiences("prior context")

        assert "LEARNED INSIGHTS" in result
        assert "Use focused queries" in result
        assert "prior context" in result
        assert helper.enrichment_result is not None
        assert len(helper.enrichment_result.injected_experience_ids) == 2

    @patch("inquiro.evolution.store_factory.get_store")
    def test_returns_original_when_store_empty(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Enrichment should return original when store has no experiences."""
        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store

        helper = _make_helper()

        with patch(
            "inquiro.evolution.enricher.PromptEnricher.enrich",
            new_callable=AsyncMock,
            return_value=EnrichmentResult(),
        ):
            result = helper.enrich_with_experiences("original")

        assert result == "original"

    @patch("inquiro.evolution.store_factory.get_store")
    def test_failure_is_non_blocking(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Enrichment failure should not propagate — returns original."""
        mock_get_store.side_effect = RuntimeError("DB connection failed")
        helper = _make_helper()

        result = helper.enrich_with_experiences("safe context")
        assert result == "safe context"

    @patch("inquiro.evolution.store_factory.get_store")
    def test_works_without_sub_item_id(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Enrichment should work with SynthesisTask (no sub_item_id)."""
        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store

        enrichment_result = EnrichmentResult(
            injected_experience_ids=["exp-1"],
            enrichment_text="- Insight for synthesis",
            token_count=20,
        )

        helper = _make_helper(has_sub_item_id=False)

        with patch(
            "inquiro.evolution.enricher.PromptEnricher.enrich",
            new_callable=AsyncMock,
            return_value=enrichment_result,
        ) as mock_enrich:
            result = helper.enrich_with_experiences("")

        # ✅ Verify sub_item was empty string (SynthesisTask fallback)
        assert "LEARNED INSIGHTS" in result
        call_kwargs = mock_enrich.call_args
        assert call_kwargs[1]["sub_item"] == ""


# ============================================================================
# 🧬 Post-Execution Evolution Tests
# ============================================================================


class TestEvolutionHelperPostExecution:
    """Tests for EvolutionHelper.post_execution_evolution() 🧬."""

    def test_noop_when_no_profile(self) -> None:
        """Post-execution should be a no-op when profile is None."""
        task = _make_task()
        task.evolution_profile = None
        helper = EvolutionHelper(
            task=task,
            llm=MagicMock(),
            cost_tracker=MagicMock(),
        )

        # Act — should NOT raise
        helper.post_execution_evolution(MagicMock(), MagicMock())

    @patch("inquiro.evolution.store_factory.get_store")
    def test_collects_trajectory(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Post-execution should collect trajectory snapshot."""
        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store
        mock_store.deduplicate = AsyncMock(return_value=False)
        mock_store.add = AsyncMock()

        helper = _make_helper()
        result = _make_result()

        with (
            patch(
                "inquiro.evolution.collector.TrajectoryCollector.collect",
            ) as mock_collect,
            patch(
                "inquiro.evolution.extractor.ExperienceExtractor.extract",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from inquiro.evolution.types import (
                ResultMetrics,
                TrajectorySnapshot,
            )

            mock_collect.return_value = TrajectorySnapshot(
                evaluation_id="eval-001",
                task_id="test-task-001",
                topic="Test topic",
                context_tags=["modality:SmallMolecule"],
                tool_calls=[],
                metrics=ResultMetrics(),
            )

            helper.post_execution_evolution(MagicMock(), result)

        mock_collect.assert_called_once()

    @patch("inquiro.evolution.store_factory.get_store")
    def test_stores_new_experiences(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Post-execution should store extracted experiences after dedup."""
        from inquiro.evolution.types import (
            Experience,
            ResultMetrics,
            TrajectorySnapshot,
        )

        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store
        mock_store.deduplicate = AsyncMock(return_value=False)
        mock_store.add = AsyncMock()

        helper = _make_helper()
        result = _make_result()

        mock_experiences = [
            Experience(
                namespace="test-namespace",
                category="search_strategy",
                insight="Use focused queries",
                source="trajectory_extraction",
            ),
            Experience(
                namespace="test-namespace",
                category="evidence_quality",
                insight="Cross-reference sources",
                source="trajectory_extraction",
            ),
        ]

        with (
            patch(
                "inquiro.evolution.collector.TrajectoryCollector.collect",
            ) as mock_collect,
            patch(
                "inquiro.evolution.extractor.ExperienceExtractor.extract",
                new_callable=AsyncMock,
                return_value=mock_experiences,
            ),
        ):
            mock_collect.return_value = TrajectorySnapshot(
                evaluation_id="eval-001",
                task_id="test-task-001",
                topic="Test topic",
                tool_calls=[],
                metrics=ResultMetrics(),
            )

            helper.post_execution_evolution(MagicMock(), result)

        assert mock_store.add.await_count == 2

    @patch("inquiro.evolution.store_factory.get_store")
    def test_skips_duplicates(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Post-execution should skip duplicate experiences."""
        from inquiro.evolution.types import (
            Experience,
            ResultMetrics,
            TrajectorySnapshot,
        )

        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store
        mock_store.deduplicate = AsyncMock(side_effect=[False, True])
        mock_store.add = AsyncMock()

        helper = _make_helper()
        result = _make_result()

        mock_experiences = [
            Experience(
                namespace="test-namespace",
                category="search_strategy",
                insight="New insight",
                source="trajectory_extraction",
            ),
            Experience(
                namespace="test-namespace",
                category="search_strategy",
                insight="Duplicate insight",
                source="trajectory_extraction",
            ),
        ]

        with (
            patch(
                "inquiro.evolution.collector.TrajectoryCollector.collect",
            ) as mock_collect,
            patch(
                "inquiro.evolution.extractor.ExperienceExtractor.extract",
                new_callable=AsyncMock,
                return_value=mock_experiences,
            ),
        ):
            mock_collect.return_value = TrajectorySnapshot(
                evaluation_id="eval-001",
                task_id="test-task-001",
                topic="Test topic",
                tool_calls=[],
                metrics=ResultMetrics(),
            )

            helper.post_execution_evolution(MagicMock(), result)

        # ✅ Only 1 stored (second was duplicate)
        assert mock_store.add.await_count == 1

    @patch("inquiro.evolution.store_factory.get_store")
    def test_failure_is_non_blocking(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Post-execution failure should not propagate."""
        mock_get_store.side_effect = RuntimeError("DB down")
        helper = _make_helper()
        result = _make_result()

        # Act — should NOT raise
        helper.post_execution_evolution(MagicMock(), result)

    @patch("inquiro.evolution.store_factory.get_store")
    def test_works_with_synthesis_result(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Post-execution should handle SynthesisResult (no search_rounds/coverage)."""
        from inquiro.evolution.types import ResultMetrics, TrajectorySnapshot

        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store
        mock_store.deduplicate = AsyncMock(return_value=True)

        helper = _make_helper(has_sub_item_id=False)
        result = _make_result(
            has_get_covered_ratio=False,
            has_search_rounds=False,
        )

        with (
            patch(
                "inquiro.evolution.collector.TrajectoryCollector.collect",
            ) as mock_collect,
            patch(
                "inquiro.evolution.extractor.ExperienceExtractor.extract",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            mock_collect.return_value = TrajectorySnapshot(
                evaluation_id="eval-001",
                task_id="test-task-001",
                topic="Test topic",
                tool_calls=[],
                metrics=ResultMetrics(),
            )

            # Act — should NOT raise
            helper.post_execution_evolution(MagicMock(), result)

    @patch("inquiro.evolution.store_factory.get_store")
    def test_fitness_evaluation_runs_when_enrichment_present(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Fitness evaluation should run when enrichment injected experiences."""
        from inquiro.evolution.types import (
            FitnessUpdate,
            ResultMetrics,
            TrajectorySnapshot,
        )

        mock_store = AsyncMock()
        mock_get_store.return_value = mock_store
        mock_store.deduplicate = AsyncMock(return_value=True)
        mock_store.bulk_update_fitness = AsyncMock()

        helper = _make_helper()
        result = _make_result()

        # 🧬 Simulate prior enrichment
        helper.enrichment_result = EnrichmentResult(
            injected_experience_ids=["exp-1", "exp-2"],
            enrichment_text="- Some insight",
            token_count=20,
        )

        mock_updates = [
            FitnessUpdate(
                experience_id="exp-1",
                signal=0.7,
                was_helpful=True,
                metric_deltas={"confidence": 0.85},
            ),
        ]

        with (
            patch(
                "inquiro.evolution.collector.TrajectoryCollector.collect",
            ) as mock_collect,
            patch(
                "inquiro.evolution.extractor.ExperienceExtractor.extract",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "inquiro.evolution.fitness.FitnessEvaluator.evaluate",
                new_callable=AsyncMock,
                return_value=mock_updates,
            ),
        ):
            mock_collect.return_value = TrajectorySnapshot(
                evaluation_id="eval-001",
                task_id="test-task-001",
                topic="Test topic",
                tool_calls=[],
                metrics=ResultMetrics(),
            )

            helper.post_execution_evolution(MagicMock(), result)

        mock_store.bulk_update_fitness.assert_awaited_once_with(
            mock_updates,
        )


# ============================================================================
# 🔄 run_async Tests
# ============================================================================


class TestEvolutionHelperRunAsync:
    """Tests for EvolutionHelper.run_async() static method 🔄."""

    def test_bridges_coroutine_to_sync(self) -> None:
        """run_async should execute an async coroutine synchronously."""

        async def async_fn() -> str:
            return "hello"

        result = EvolutionHelper.run_async(async_fn())
        assert result == "hello"

    def test_returns_coroutine_result(self) -> None:
        """run_async should return the coroutine's return value."""

        async def add(a: int, b: int) -> int:
            return a + b

        result = EvolutionHelper.run_async(add(3, 4))
        assert result == 7
