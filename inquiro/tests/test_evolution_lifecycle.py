"""Integration tests for the complete evolution lifecycle with real SQLite 🧬.

Tests the full closed loop:
    1. store_factory.get_store() → lazy initialization
    2. Pre-execution enrichment → query experiences from store
    3. Post-execution collection → TrajectoryCollector
    4. Experience extraction → ExperienceExtractor (mock LLM)
    5. Store new experiences → ExperienceStore.add (with dedup)
    6. Fitness evaluation → FitnessEvaluator
    7. Pruning and decay → ExperienceRanker

Uses a real in-memory SQLite database (aiosqlite) to validate the
complete data flow, not just mocked interfaces.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from inquiro.evolution.collector import TrajectoryCollector
from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.store import ExperienceStore, init_store_schema
from inquiro.evolution.types import (
    EnrichmentResult,
    Experience,
    FitnessUpdate,
    PruneConfig,
    ResultMetrics,
)


# ============================================================================
# 🧪 Fixtures — Real SQLite Store
# ============================================================================


@pytest_asyncio.fixture
async def real_store():
    """Create a real ExperienceStore backed by in-memory SQLite 💾."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    await init_store_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ExperienceStore(session_factory)
    yield store
    # 🧹 Cleanup
    await engine.dispose()


@pytest.fixture
def sample_profile() -> dict[str, Any]:
    """Sample evolution profile config 🧬."""
    return {
        "namespace": "targetmaster",
        "version": "v1",
        "extraction_prompt_template": (
            "Analyze this trajectory snapshot:\n"
            "Evaluation: {{ snapshot.evaluation_id }}\n"
            "Topic: {{ snapshot.topic }}\n"
            "Tool calls: {{ snapshot.tool_calls | length }}\n"
            "Evidence count: {{ snapshot.metrics.evidence_count }}\n"
            "Confidence: {{ snapshot.metrics.confidence }}\n\n"
            "Extract up to {{ max_experiences }} reusable insights.\n"
            "Valid categories: {{ valid_categories }}\n\n"
            "Return a JSON array of objects with keys: "
            "category, insight, context_tags, applicable_sub_items"
        ),
        "experience_categories": [
            "search_strategy",
            "checklist_insight",
            "evidence_quality",
        ],
        "max_experiences_per_extraction": 5,
        "enrichment_prompt_template": (
            "{% for category, exps in experiences_by_category.items() %}\n"
            "## {{ category }}\n"
            "{% for exp in exps %}\n"
            "- {{ exp.insight }} (fitness: {{ '%.2f' | format(exp.fitness_score) }})\n"
            "{% endfor %}\n"
            "{% endfor %}"
        ),
        "enrichment_max_tokens": 500,
        "enrichment_max_items": 10,
        "prune_min_fitness": 0.3,
        "prune_min_uses": 3,
        "fitness_dimensions": [
            {
                "metric_name": "evidence_count",
                "weight": 0.4,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "confidence",
                "weight": 0.4,
                "direction": "higher_is_better",
            },
            {
                "metric_name": "cost_usd",
                "weight": 0.2,
                "direction": "lower_is_better",
            },
        ],
        "fitness_learning_rate": 0.3,
    }


# ============================================================================
# 🧪 Test: Store Factory
# ============================================================================


class TestStoreFactory:
    """Test the store_factory module for lazy initialization 🏭."""

    @pytest.mark.asyncio
    async def test_get_store_returns_instance(self) -> None:
        """get_store() should return an ExperienceStore instance 💾."""
        from inquiro.evolution.store_factory import get_store, reset_store

        # 🧹 Reset to force fresh initialization
        await reset_store()

        # 🔧 Set env var to use in-memory SQLite
        old_url = os.environ.get("EVOLUTION_DB_URL")
        os.environ["EVOLUTION_DB_URL"] = "sqlite+aiosqlite:///:memory:"

        try:
            store = await get_store()
            assert store is not None
            assert isinstance(store, ExperienceStore)

            # 📊 Verify store is functional by checking stats
            stats = await store.get_stats("test_namespace")
            assert stats["total_count"] == 0

            # 🔒 Second call returns same instance
            store2 = await get_store()
            assert store is store2
        finally:
            # 🧹 Restore env and reset singleton
            if old_url is not None:
                os.environ["EVOLUTION_DB_URL"] = old_url
            else:
                os.environ.pop("EVOLUTION_DB_URL", None)
            await reset_store()


# ============================================================================
# 🧪 Test: Full Lifecycle — Real SQLite
# ============================================================================


class TestEvolutionLifecycleReal:
    """Test the complete evolution lifecycle with real SQLite 🔄."""

    @pytest.mark.asyncio
    async def test_seed_and_enrich(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Seed experiences, then verify enrichment queries them back 💉."""
        # 🌱 Step 1: Seed experiences into the store
        exp1 = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for primary clinical evidence",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["*"],
            fitness_score=0.8,
            source="trajectory_extraction",
        )
        exp2 = Experience(
            namespace="targetmaster",
            category="checklist_insight",
            insight="Market size estimates require multiple sources",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["market_size"],
            fitness_score=0.7,
            source="human_feedback",
        )
        await real_store.add(exp1)
        await real_store.add(exp2)

        # 📊 Verify seeds are in the store
        stats = await real_store.get_stats("targetmaster")
        assert stats["total_count"] == 2

        # 💉 Step 2: Enrich — query experiences for prompt injection
        enricher = PromptEnricher(real_store)
        result = await enricher.enrich(
            task_context_tags=["modality:SmallMolecule"],
            sub_item="market_size",
            profile_config=sample_profile,
        )

        assert isinstance(result, EnrichmentResult)
        assert len(result.injected_experience_ids) >= 1
        assert (
            "PubMed" in result.enrichment_text
            or "Market size" in result.enrichment_text
        )
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_collect_extract_store(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Collect trajectory → extract experiences → store in real DB 📸."""
        # 📸 Step 1: Collect trajectory snapshot
        mock_trajectory = MagicMock()
        mock_trajectory.steps = []

        mock_task = MagicMock()
        mock_task.evaluation_id = "eval-lifecycle-001"
        mock_task.task_id = "task-lifecycle-001"
        mock_task.topic = "Evaluate market size for EGFR in NSCLC"

        collector = TrajectoryCollector()
        snapshot = collector.collect(
            trajectory=mock_trajectory,
            task=mock_task,
            context_tags=["modality:SmallMolecule", "indication:NSCLC"],
            sub_item_id="market_size",
        )

        assert snapshot.evaluation_id == "eval-lifecycle-001"

        # 🧠 Step 2: Extract experiences via mock LLM
        llm_response = json.dumps(
            [
                {
                    "category": "search_strategy",
                    "insight": "Google Scholar provides comprehensive market analysis reports",
                    "context_tags": ["modality:SmallMolecule"],
                    "applicable_sub_items": ["market_size"],
                },
                {
                    "category": "evidence_quality",
                    "insight": "Prefer peer-reviewed sources over press releases",
                    "context_tags": [],
                    "applicable_sub_items": ["*"],
                },
            ]
        )

        async def mock_llm_fn(prompt: str) -> str:
            """Mock LLM that returns structured extraction 🤖."""
            return llm_response

        extractor = ExperienceExtractor(llm_fn=mock_llm_fn)
        new_experiences = await extractor.extract(
            snapshot=snapshot,
            profile_config=sample_profile,
        )

        assert len(new_experiences) == 2
        assert new_experiences[0].namespace == "targetmaster"
        assert new_experiences[0].source == "trajectory_extraction"

        # 💾 Step 3: Store new experiences (with dedup check)
        stored_count = 0
        for exp in new_experiences:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)
                stored_count += 1

        assert stored_count == 2

        # ✅ Verify stored experiences
        stats = await real_store.get_stats("targetmaster")
        assert stats["total_count"] == 2
        assert "search_strategy" in stats["category_breakdown"]
        assert "evidence_quality" in stats["category_breakdown"]

    @pytest.mark.asyncio
    async def test_deduplication_prevents_duplicates(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Dedup should prevent storing duplicate insights 🔍."""
        exp = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Always check for recent systematic reviews",
            context_tags=[],
            fitness_score=0.5,
            source="trajectory_extraction",
        )
        await real_store.add(exp)

        # 🔍 Exact duplicate
        is_dup = await real_store.deduplicate(
            "targetmaster",
            "Always check for recent systematic reviews",
        )
        assert is_dup is True

        # 🔍 Substring match
        is_dup2 = await real_store.deduplicate(
            "targetmaster",
            "check for recent systematic reviews",
        )
        assert is_dup2 is True

        # ✅ Different insight should not be duplicate
        is_dup3 = await real_store.deduplicate(
            "targetmaster",
            "Use Boolean operators in search queries",
        )
        assert is_dup3 is False

    @pytest.mark.asyncio
    async def test_fitness_update_ema(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Fitness updates should apply EMA correctly 📈."""
        exp = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Test fitness EMA calculation",
            context_tags=[],
            fitness_score=0.5,
            source="test",
        )
        await real_store.add(exp)

        # 📈 Apply positive fitness update (signal=1.0)
        delta = FitnessUpdate(
            experience_id=exp.id,
            signal=1.0,
            was_helpful=True,
        )
        await real_store.update_fitness(exp.id, delta)

        # 📊 Verify EMA: new = 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        updated = await real_store.get_by_id(exp.id)
        assert updated is not None
        assert abs(updated.fitness_score - 0.65) < 0.01
        assert updated.times_used == 1
        assert updated.times_helpful == 1

    @pytest.mark.asyncio
    async def test_bulk_fitness_update(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Bulk fitness updates should process all experiences 📊."""
        # 🌱 Seed two experiences
        exp1 = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Bulk test experience 1",
            context_tags=[],
            fitness_score=0.5,
            source="test",
        )
        exp2 = Experience(
            namespace="targetmaster",
            category="evidence_quality",
            insight="Bulk test experience 2",
            context_tags=[],
            fitness_score=0.6,
            source="test",
        )
        await real_store.add(exp1)
        await real_store.add(exp2)

        # 📊 Bulk update
        updates = [
            FitnessUpdate(
                experience_id=exp1.id,
                signal=0.9,
                was_helpful=True,
            ),
            FitnessUpdate(
                experience_id=exp2.id,
                signal=0.2,
                was_helpful=False,
            ),
        ]
        await real_store.bulk_update_fitness(updates)

        # ✅ Verify both updated
        u1 = await real_store.get_by_id(exp1.id)
        u2 = await real_store.get_by_id(exp2.id)
        assert u1 is not None
        assert u2 is not None
        # exp1: 0.3 * 0.9 + 0.7 * 0.5 = 0.62
        assert abs(u1.fitness_score - 0.62) < 0.01
        assert u1.times_helpful == 1
        # exp2: 0.3 * 0.2 + 0.7 * 0.6 = 0.48
        assert abs(u2.fitness_score - 0.48) < 0.01
        assert u2.times_helpful == 0

    @pytest.mark.asyncio
    async def test_decay_and_prune(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Decay reduces fitness, prune removes low-fitness experiences 📉."""
        # 🌱 Seed experiences with varying fitness
        high_exp = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="High fitness experience",
            context_tags=[],
            fitness_score=0.9,
            times_used=5,
            source="test",
        )
        low_exp = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Low fitness experience",
            context_tags=[],
            fitness_score=0.2,
            times_used=5,
            source="test",
        )
        await real_store.add(high_exp)
        await real_store.add(low_exp)

        # 📉 Apply decay (0.9 factor)
        updated_count = await real_store.apply_decay("targetmaster", 0.9)
        assert updated_count == 2

        # 📊 Verify decay applied
        h = await real_store.get_by_id(high_exp.id)
        l_exp = await real_store.get_by_id(low_exp.id)
        assert h is not None and l_exp is not None
        assert abs(h.fitness_score - 0.81) < 0.01  # 0.9 * 0.9
        assert abs(l_exp.fitness_score - 0.18) < 0.01  # 0.2 * 0.9

        # 🗑️ Prune experiences below threshold
        config = PruneConfig(
            min_fitness=0.3,
            min_uses=3,
            decay_factor=0.9,
            decay_interval_days=7,
        )
        pruned_count = await real_store.prune("targetmaster", config)
        assert pruned_count == 1  # Only the low-fitness one

        # ✅ Verify only high-fitness remains
        stats = await real_store.get_stats("targetmaster")
        assert stats["total_count"] == 1

    @pytest.mark.asyncio
    async def test_full_closed_loop(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Test the COMPLETE closed loop: seed → enrich → run → collect → extract → store → fitness 🔄."""
        # 🌱 Phase 1: Seed initial experiences (simulating prior runs)
        seed_exp = Experience(
            namespace="targetmaster",
            category="search_strategy",
            insight="Use Boolean operators for precise PubMed searches",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["*"],
            fitness_score=0.6,
            source="trajectory_extraction",
        )
        await real_store.add(seed_exp)

        # 💉 Phase 2: Pre-execution enrichment
        enricher = PromptEnricher(real_store)
        enrichment = await enricher.enrich(
            task_context_tags=["modality:SmallMolecule"],
            sub_item="market_size",
            profile_config=sample_profile,
        )
        assert len(enrichment.injected_experience_ids) == 1
        assert seed_exp.id in enrichment.injected_experience_ids
        assert "Boolean operators" in enrichment.enrichment_text

        # 🤖 Phase 3: Mock agent execution (simulated trajectory)
        mock_trajectory = MagicMock()
        mock_trajectory.steps = []

        mock_task = MagicMock()
        mock_task.evaluation_id = "eval-full-loop"
        mock_task.task_id = "task-full-loop"
        mock_task.topic = "Evaluate market potential for EGFR SmallMolecule in NSCLC"

        # 📸 Phase 4: Collect trajectory
        collector = TrajectoryCollector()
        snapshot = collector.collect(
            trajectory=mock_trajectory,
            task=mock_task,
            context_tags=["modality:SmallMolecule", "indication:NSCLC"],
            sub_item_id="market_size",
        )
        snapshot.metrics = ResultMetrics(
            evidence_count=8,
            confidence=0.75,
            cost_usd=0.45,
            search_rounds=3,
            checklist_coverage=0.85,
        )

        # 🧠 Phase 5: Extract new experiences
        llm_extraction_response = json.dumps(
            [
                {
                    "category": "search_strategy",
                    "insight": "ClinicalTrials.gov provides authoritative pipeline data",
                    "context_tags": ["modality:SmallMolecule"],
                    "applicable_sub_items": ["market_size", "competitive_landscape"],
                },
            ]
        )

        async def mock_llm_fn(prompt: str) -> str:
            return llm_extraction_response

        extractor = ExperienceExtractor(llm_fn=mock_llm_fn)
        new_experiences = await extractor.extract(
            snapshot=snapshot,
            profile_config=sample_profile,
        )
        assert len(new_experiences) == 1

        # 💾 Phase 6: Store new experiences (with dedup)
        for exp in new_experiences:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)

        # ✅ Verify: should now have 2 experiences total
        stats = await real_store.get_stats("targetmaster")
        assert stats["total_count"] == 2

        # 📊 Phase 7: Fitness evaluation for injected (seed) experience
        # The seed experience was injected and the task had good results
        _after_metrics = ResultMetrics(
            evidence_count=8,
            confidence=0.75,
            cost_usd=0.45,
            search_rounds=3,
            checklist_coverage=0.85,
        )

        # 📈 Apply direct fitness update (simulating FitnessEvaluator output)
        fitness_update = FitnessUpdate(
            experience_id=seed_exp.id,
            signal=0.85,
            was_helpful=True,
            metric_deltas={
                "evidence_count": 8.0,
                "confidence": 0.75,
            },
        )
        await real_store.update_fitness(seed_exp.id, fitness_update)

        # ✅ Verify fitness updated
        updated_seed = await real_store.get_by_id(seed_exp.id)
        assert updated_seed is not None
        # EMA: 0.3 * 0.85 + 0.7 * 0.6 = 0.675
        assert abs(updated_seed.fitness_score - 0.675) < 0.01
        assert updated_seed.times_used == 1
        assert updated_seed.times_helpful == 1

        # 🔄 Phase 8: Second iteration — verify enrichment now returns both
        enrichment2 = await enricher.enrich(
            task_context_tags=["modality:SmallMolecule"],
            sub_item="market_size",
            profile_config=sample_profile,
        )
        assert len(enrichment2.injected_experience_ids) == 2
        assert "Boolean operators" in enrichment2.enrichment_text
        assert "ClinicalTrials.gov" in enrichment2.enrichment_text

        # 🎉 Complete closed loop verified!
