"""Integration tests for the evolution framework wiring 🧬.

Verifies that:
1. Evolution profile flows from API request through to discovery pipeline
2. Experience enrichment produces correct prompt sections
3. Trajectory collection captures expected data
4. End-to-end: enrich → mock run → collect → extract → store
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    EvaluationTask,
)
from inquiro.evolution.collector import TrajectoryCollector
from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.types import (
    EnrichmentResult,
    Experience,
    ResultMetrics,
    TrajectorySnapshot,
)


# ============================================================================
# 🧪 Fixtures
# ============================================================================


@pytest.fixture
def sample_evolution_profile() -> dict[str, Any]:
    """Sample evolution profile config dict 🧬."""
    return {
        "namespace": "targetmaster",
        "version": "v1",
        "extraction_prompt_template": (
            "Analyze the trajectory and extract insights:\n"
            "{{ trajectory_snapshot }}\n"
            "Categories: {{ category_list }}"
        ),
        "experience_categories": [
            "search_strategy",
            "checklist_insight",
            "prompt_pattern",
        ],
        "max_experiences_per_extraction": 5,
        "enrichment_prompt_template": (
            "{% for category, exps in experiences_by_category.items() %}\n"
            "## {{ category }}\n"
            "{% for exp in exps %}\n"
            "- {{ exp.insight }} (fitness: {{ exp.fitness_score }})\n"
            "{% endfor %}\n"
            "{% endfor %}"
        ),
        "enrichment_max_tokens": 300,
        "enrichment_max_items": 5,
        "prune_min_fitness": 0.3,
        "fitness_dimensions": [
            {
                "metric_name": "evidence_count",
                "weight": 0.5,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "confidence",
                "weight": 0.5,
                "direction": "higher_is_better",
            },
        ],
        "fitness_learning_rate": 0.3,
    }


@pytest.fixture
def sample_task_with_evolution(
    sample_evolution_profile: dict[str, Any],
) -> EvaluationTask:
    """Sample EvaluationTask with evolution profile 🔬."""
    return EvaluationTask(
        task_id="test-task-001",
        topic="Test research topic",
        rules="Test evaluation rules",
        checklist=Checklist(
            required=[
                ChecklistItem(id="R1", description="Test required item"),
            ],
        ),
        output_schema={
            "type": "object",
            "required": ["decision"],
            "properties": {"decision": {"type": "string"}},
        },
        evolution_profile=sample_evolution_profile,
        context_tags=["modality:SmallMolecule", "indication:NSCLC"],
        sub_item_id="market_size",
    )


@pytest.fixture
def sample_experiences() -> list[Experience]:
    """Sample experiences for enrichment testing 📝."""
    return [
        Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for clinical evidence first",
            context_tags=["modality:SmallMolecule"],
            fitness_score=0.8,
            source="trajectory_extraction",
        ),
        Experience(
            namespace="targetmaster",
            category="checklist_insight",
            insight="Market size estimates vary widely between sources",
            context_tags=["modality:SmallMolecule"],
            fitness_score=0.7,
            source="human_feedback",
        ),
    ]


# ============================================================================
# 🧪 Test: Evolution Profile Flows Through API → Task
# ============================================================================


class TestEvolutionProfileFlow:
    """Verify evolution_profile flows correctly through the system 🔄."""

    def test_evaluation_task_accepts_evolution_profile(
        self,
        sample_task_with_evolution: EvaluationTask,
    ) -> None:
        """EvaluationTask should accept evolution_profile 🧬."""
        task = sample_task_with_evolution
        assert task.evolution_profile is not None
        assert task.evolution_profile["namespace"] == "targetmaster"
        assert len(task.context_tags) == 2
        assert task.sub_item_id == "market_size"

    def test_evaluation_task_without_evolution(self) -> None:
        """EvaluationTask without evolution should have None 🔒."""
        task = EvaluationTask(
            task_id="test-002",
            topic="Test",
            output_schema={},
        )
        assert task.evolution_profile is None
        assert task.context_tags == []
        assert (
            task.sub_item_id is None
        )  # default per EvaluationTask (Progressive Disclosure)


# ============================================================================
# 🧪 Test: Enrichment Integration
# ============================================================================


class TestEnrichmentIntegration:
    """Verify PromptEnricher produces correct prompt text 💉."""

    @pytest.mark.asyncio
    async def test_enricher_produces_grouped_output(
        self,
        sample_experiences: list[Experience],
        sample_evolution_profile: dict[str, Any],
    ) -> None:
        """Enricher should produce grouped experience text 📝."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=sample_experiences)

        enricher = PromptEnricher(mock_store)
        result = await enricher.enrich(
            task_context_tags=["modality:SmallMolecule"],
            sub_item="market_size",
            profile_config=sample_evolution_profile,
        )

        assert isinstance(result, EnrichmentResult)
        assert len(result.injected_experience_ids) == 2
        assert "PubMed" in result.enrichment_text
        assert "Market size" in result.enrichment_text
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_enricher_empty_store(
        self,
        sample_evolution_profile: dict[str, Any],
    ) -> None:
        """Enricher should handle empty store gracefully 🔒."""
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=[])

        enricher = PromptEnricher(mock_store)
        result = await enricher.enrich(
            task_context_tags=[],
            sub_item="",
            profile_config=sample_evolution_profile,
        )

        assert len(result.injected_experience_ids) == 0
        assert result.enrichment_text == ""


# ============================================================================
# 🧪 Test: Trajectory Collection Integration
# ============================================================================


class TestTrajectoryCollectionIntegration:
    """Verify TrajectoryCollector captures data from mock trajectories 📸."""

    def test_collector_with_context_tags(self) -> None:
        """Collector should pass through context_tags from caller 🏷️."""
        # 🔧 Build mock trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.steps = []

        mock_task = MagicMock()
        mock_task.evaluation_id = "eval-123"
        mock_task.task_id = "task-456"
        mock_task.topic = "Test topic"

        collector = TrajectoryCollector()
        snapshot = collector.collect(
            trajectory=mock_trajectory,
            task=mock_task,
            context_tags=["modality:Antibody", "indication:RA"],
            sub_item_id="immunogenicity_risk",
        )

        assert snapshot.evaluation_id == "eval-123"
        assert snapshot.task_id == "task-456"
        assert snapshot.context_tags == [
            "modality:Antibody",
            "indication:RA",
        ]
        assert snapshot.sub_item_id == "immunogenicity_risk"


# ============================================================================
# 🧪 Test: Extractor Integration
# ============================================================================


class TestExtractorIntegration:
    """Verify ExperienceExtractor works with profile config 🔬."""

    @pytest.mark.asyncio
    async def test_extractor_with_profile_config(
        self,
        sample_evolution_profile: dict[str, Any],
    ) -> None:
        """Extractor should use profile config for extraction 🧬."""
        mock_llm_response = json.dumps(
            [
                {
                    "category": "search_strategy",
                    "insight": "Start with systematic reviews",
                    "context_tags": ["modality:SmallMolecule"],
                    "applicable_sub_items": ["*"],
                }
            ]
        )

        async def mock_llm_fn(prompt: str) -> str:
            return mock_llm_response

        extractor = ExperienceExtractor(llm_fn=mock_llm_fn)
        snapshot = TrajectorySnapshot(
            evaluation_id="eval-001",
            task_id="task-001",
            topic="Test research",
            context_tags=["modality:SmallMolecule"],
            metrics=ResultMetrics(
                evidence_count=10,
                confidence=0.7,
            ),
        )

        experiences = await extractor.extract(
            snapshot=snapshot,
            profile_config=sample_evolution_profile,
        )

        assert len(experiences) == 1
        assert experiences[0].category == "search_strategy"
        assert experiences[0].namespace == "targetmaster"
        assert experiences[0].source == "trajectory_extraction"


# ============================================================================
# 🧪 Test: End-to-End Flow (Mock)
# ============================================================================


class TestEndToEndEvolutionFlow:
    """Verify the complete evolution loop (with mocked components) 🔄."""

    @pytest.mark.asyncio
    async def test_enrich_collect_extract_store(
        self,
        sample_experiences: list[Experience],
        sample_evolution_profile: dict[str, Any],
    ) -> None:
        """Test complete evolution cycle: enrich → collect → extract → store 🔄."""
        # 1. 💉 Enrich — query experiences for prompt injection
        mock_store = AsyncMock()
        mock_store.query = AsyncMock(return_value=sample_experiences)
        mock_store.add = AsyncMock(return_value="new-exp-001")
        mock_store.deduplicate = AsyncMock(return_value=False)

        enricher = PromptEnricher(mock_store)
        enrichment_result = await enricher.enrich(
            task_context_tags=["modality:SmallMolecule"],
            sub_item="market_size",
            profile_config=sample_evolution_profile,
        )

        assert len(enrichment_result.injected_experience_ids) == 2

        # 2. 📸 Collect — capture trajectory data (mock)
        mock_trajectory = MagicMock()
        mock_trajectory.steps = []

        mock_task = MagicMock()
        mock_task.evaluation_id = "eval-e2e"
        mock_task.task_id = "task-e2e"
        mock_task.topic = "E2E test topic"

        collector = TrajectoryCollector()
        snapshot = collector.collect(
            trajectory=mock_trajectory,
            task=mock_task,
            context_tags=["modality:SmallMolecule"],
            sub_item_id="market_size",
        )

        assert snapshot.evaluation_id == "eval-e2e"

        # 3. 🔬 Extract — use LLM to extract new experiences
        async def mock_llm_fn(prompt: str) -> str:
            return json.dumps(
                [
                    {
                        "category": "search_strategy",
                        "insight": "Use Google Scholar for market reports",
                        "context_tags": ["modality:SmallMolecule"],
                    }
                ]
            )

        extractor = ExperienceExtractor(llm_fn=mock_llm_fn)
        new_experiences = await extractor.extract(
            snapshot=snapshot,
            profile_config=sample_evolution_profile,
        )

        assert len(new_experiences) == 1

        # 4. 💾 Store — save new experience (mock)
        for exp in new_experiences:
            is_dup = await mock_store.deduplicate(
                namespace=exp.namespace,
                new_insight=exp.insight,
            )
            if not is_dup:
                exp_id = await mock_store.add(exp)
                assert exp_id is not None

        # ✅ Verify store was called
        mock_store.add.assert_called()
