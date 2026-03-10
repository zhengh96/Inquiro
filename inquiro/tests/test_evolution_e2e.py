"""End-to-end tests for the complete evolution lifecycle across multiple iterations 🧬.

Verifies the closed loop:
    Iteration N: enrich(store) → run(agent) → collect(trajectory) → extract(LLM)
                → store(experiences) → fitness(evaluate) → update(store)
    Iteration N+1: experiences from N appear in enrichment, fitness evolves

Tests cover:
- Multi-iteration evolution loop (3 iterations with improving metrics)
- Discovery pipeline integration path (enrichment + post-execution hooks)
- Namespace isolation (experiences don't leak across namespaces)
- Fitness evolution over time (helpful vs unhelpful experiences)
- API request flow (evolution_profile serialization round-trip)

Uses real in-memory SQLite database (aiosqlite) for all persistence tests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from inquiro.evolution.collector import TrajectoryCollector
from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.fitness import FitnessEvaluator
from inquiro.evolution.store import ExperienceStore, init_store_schema
from inquiro.evolution.types import (
    EnrichmentResult,
    Experience,
    ExperienceQuery,
    FitnessUpdate,
    PruneConfig,
    ResultMetrics,
    TrajectorySnapshot,
)


# ============================================================================
# 🧪 Fixtures — Real SQLite Store + Profile Config
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
    """Sample evolution profile config for multi-iteration tests 🧬."""
    return {
        "namespace": "test_e2e",
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
            "- {{ exp.insight }} "
            "(fitness: {{ '%.2f' | format(exp.fitness_score) }})\n"
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


def _make_snapshot(
    eval_id: str,
    task_id: str,
    topic: str,
    evidence_count: int = 5,
    confidence: float = 0.7,
    cost_usd: float = 0.5,
    search_rounds: int = 2,
    checklist_coverage: float = 0.8,
) -> TrajectorySnapshot:
    """Helper to build a TrajectorySnapshot with given metrics 📸.

    Args:
        eval_id: Evaluation ID.
        task_id: Task ID within the evaluation.
        topic: Research topic.
        evidence_count: Number of evidence items.
        confidence: Agent-reported confidence (0.0-1.0).
        cost_usd: Total cost in USD.
        search_rounds: Number of search rounds.
        checklist_coverage: Fraction of checklist items covered.

    Returns:
        TrajectorySnapshot with populated metrics.
    """
    return TrajectorySnapshot(
        evaluation_id=eval_id,
        task_id=task_id,
        topic=topic,
        context_tags=["tag_a", "tag_b"],
        sub_item_id="sub_item_x",
        tool_calls=[],
        metrics=ResultMetrics(
            evidence_count=evidence_count,
            confidence=confidence,
            cost_usd=cost_usd,
            search_rounds=search_rounds,
            checklist_coverage=checklist_coverage,
        ),
    )


def _make_mock_llm_fn(experiences_json: list[dict]) -> Any:
    """Create a mock LLM function returning JSON extraction output 🤖.

    Args:
        experiences_json: List of dicts representing extracted experiences.

    Returns:
        Async callable that returns serialized JSON.
    """
    response_str = json.dumps(experiences_json)

    async def mock_fn(prompt: str) -> str:
        return response_str

    return mock_fn


# ============================================================================
# 🔄 Test: Multi-Iteration Evolution Loop
# ============================================================================


class TestMultiIterationEvolution:
    """Verify that evolution improves across multiple task iterations 🔄."""

    @pytest.mark.asyncio
    async def test_three_iteration_evolution_cycle(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Run 3 iterations and verify experiences accumulate and fitness evolves 🔄.

        Iteration 1: No prior experiences → extract from trajectory → store
        Iteration 2: Enrich with iter 1 data → run → extract more → fitness update
        Iteration 3: Enrich with updated data → verify higher-fitness prioritized
        """
        enricher = PromptEnricher(real_store)
        fitness_evaluator = FitnessEvaluator(real_store)
        namespace = sample_profile["namespace"]

        # ===================================================================
        # 🔄 Iteration 1: No prior experiences
        # ===================================================================

        # 💉 Step 1a: Enrich — should return empty (no experiences yet)
        enrichment_1 = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert len(enrichment_1.injected_experience_ids) == 0
        assert enrichment_1.enrichment_text.strip() == ""

        # 📸 Step 1b: Simulate agent execution + collect trajectory
        snapshot_1 = _make_snapshot(
            eval_id="eval-iter-1",
            task_id="task-iter-1",
            topic="Research topic alpha",
            evidence_count=5,
            confidence=0.65,
            cost_usd=0.40,
        )

        # 🧠 Step 1c: Extract experiences
        extractor_1 = ExperienceExtractor(
            llm_fn=_make_mock_llm_fn(
                [
                    {
                        "category": "search_strategy",
                        "insight": "Use advanced Boolean operators for precision",
                        "context_tags": ["tag_a"],
                        "applicable_sub_items": ["sub_item_x"],
                    },
                    {
                        "category": "evidence_quality",
                        "insight": "Prefer systematic reviews over individual studies",
                        "context_tags": ["tag_a", "tag_b"],
                        "applicable_sub_items": ["*"],
                    },
                ]
            )
        )
        new_exps_1 = await extractor_1.extract(snapshot_1, sample_profile)
        assert len(new_exps_1) == 2

        # 💾 Step 1d: Store experiences (with dedup)
        for exp in new_exps_1:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)

        stats_1 = await real_store.get_stats(namespace)
        assert stats_1["total_count"] == 2

        # ===================================================================
        # 🔄 Iteration 2: Enrich with iter-1 experiences
        # ===================================================================

        # 💉 Step 2a: Enrich — should inject 2 experiences from iter 1
        enrichment_2 = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert len(enrichment_2.injected_experience_ids) == 2
        assert "Boolean operators" in enrichment_2.enrichment_text
        assert "systematic reviews" in enrichment_2.enrichment_text

        # 📸 Step 2b: Simulate improved execution (higher evidence/confidence)
        snapshot_2 = _make_snapshot(
            eval_id="eval-iter-2",
            task_id="task-iter-2",
            topic="Research topic alpha",
            evidence_count=9,
            confidence=0.82,
            cost_usd=0.35,
        )

        # 📊 Step 2c: Fitness evaluation — compare before/after
        before_metrics = ResultMetrics(
            evidence_count=5,
            confidence=0.65,
            cost_usd=0.40,
        )
        after_metrics = ResultMetrics(
            evidence_count=9,
            confidence=0.82,
            cost_usd=0.35,
        )

        updates_2 = await fitness_evaluator.evaluate(
            enrichment_result=enrichment_2,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            profile_config=sample_profile,
        )
        assert len(updates_2) == 2
        # ✅ Positive deltas on all dimensions → helpful
        for u in updates_2:
            assert u.was_helpful is True
            assert u.signal > 0.5

        # 📈 Step 2d: Apply fitness updates
        await fitness_evaluator.apply_updates(
            updates_2,
            learning_rate=sample_profile["fitness_learning_rate"],
        )

        # ✅ Verify fitness scores increased from default 0.5
        for exp_id in enrichment_2.injected_experience_ids:
            exp = await real_store.get_by_id(exp_id)
            assert exp is not None
            assert exp.fitness_score > 0.5
            assert exp.times_used == 1
            assert exp.times_helpful == 1

        # 🧠 Step 2e: Extract new experiences from iter 2
        extractor_2 = ExperienceExtractor(
            llm_fn=_make_mock_llm_fn(
                [
                    {
                        "category": "checklist_insight",
                        "insight": "Cross-reference databases to fill checklist gaps",
                        "context_tags": ["tag_a"],
                        "applicable_sub_items": ["sub_item_x"],
                    },
                ]
            )
        )
        new_exps_2 = await extractor_2.extract(snapshot_2, sample_profile)
        for exp in new_exps_2:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)

        stats_2 = await real_store.get_stats(namespace)
        assert stats_2["total_count"] == 3  # 2 from iter 1 + 1 from iter 2

        # ===================================================================
        # 🔄 Iteration 3: Verify higher-fitness experiences are prioritized
        # ===================================================================

        # 💉 Step 3a: Enrich — should now inject 3 experiences
        enrichment_3 = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert len(enrichment_3.injected_experience_ids) == 3
        assert "Cross-reference" in enrichment_3.enrichment_text

        # 📊 Step 3b: Verify ordering — higher fitness first in store
        all_exps = await real_store.list_by_namespace(namespace)
        assert len(all_exps) == 3
        # 🔍 list_by_namespace returns fitness descending
        fitness_scores = [e.fitness_score for e in all_exps]
        assert fitness_scores == sorted(fitness_scores, reverse=True)
        # ✅ Iter-1 experiences (updated) should have higher fitness than iter-2 (default 0.5)
        assert all_exps[0].fitness_score > 0.5
        assert all_exps[1].fitness_score > 0.5

    @pytest.mark.asyncio
    async def test_iteration_with_declining_metrics_reduces_fitness(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """When metrics decline, fitness of injected experiences should decrease 📉."""
        enricher = PromptEnricher(real_store)
        fitness_evaluator = FitnessEvaluator(real_store)

        # 🌱 Seed an experience
        exp = Experience(
            namespace=sample_profile["namespace"],
            category="search_strategy",
            insight="Always start with broad keyword search",
            context_tags=["tag_a"],
            applicable_sub_items=["*"],
            fitness_score=0.7,
            source="trajectory_extraction",
        )
        await real_store.add(exp)

        # 💉 Enrich to get the experience injected
        enrichment = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert exp.id in enrichment.injected_experience_ids

        # 📉 Simulate declining metrics (worse performance with experience)
        before_metrics = ResultMetrics(
            evidence_count=8,
            confidence=0.80,
            cost_usd=0.30,
        )
        after_metrics = ResultMetrics(
            evidence_count=4,
            confidence=0.50,
            cost_usd=0.60,
        )

        updates = await fitness_evaluator.evaluate(
            enrichment_result=enrichment,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            profile_config=sample_profile,
        )
        assert len(updates) == 1
        assert updates[0].was_helpful is False

        # 📈 Apply the negative update
        await fitness_evaluator.apply_updates(
            updates,
            learning_rate=sample_profile["fitness_learning_rate"],
        )

        # ✅ Verify fitness decreased
        updated = await real_store.get_by_id(exp.id)
        assert updated is not None
        assert updated.fitness_score < 0.7  # Started at 0.7
        assert updated.times_used == 1
        assert updated.times_helpful == 0


# ============================================================================
# 🔬 Test: Discovery Pipeline Evolution Integration Path
# ============================================================================


class TestDiscoveryEvolutionPath:
    """Verify discovery pipeline evolution hooks work with real store 🔬."""

    @pytest.mark.asyncio
    async def test_enrich_with_experiences_queries_store(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """_enrich_with_experiences should query store when profile present 💉."""
        # 🌱 Seed experiences
        exp = Experience(
            namespace=sample_profile["namespace"],
            category="search_strategy",
            insight="Start with high-quality databases first",
            context_tags=["tag_a"],
            applicable_sub_items=["*"],
            fitness_score=0.8,
            source="trajectory_extraction",
        )
        await real_store.add(exp)

        # 💉 Enrich directly via PromptEnricher (same path _enrich_with_experiences uses)
        enricher = PromptEnricher(real_store)
        result = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )

        # ✅ Verify enrichment pulled the seeded experience
        assert len(result.injected_experience_ids) == 1
        assert exp.id in result.injected_experience_ids
        assert "high-quality databases" in result.enrichment_text
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_post_execution_evolution_stores_experiences(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Post-execution pipeline should extract and store experiences 📸."""
        namespace = sample_profile["namespace"]

        # 📸 Collect trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.steps = []

        mock_task = MagicMock()
        mock_task.evaluation_id = "eval-resexp-001"
        mock_task.task_id = "task-resexp-001"
        mock_task.topic = "Test post-execution evolution"

        collector = TrajectoryCollector()
        snapshot = collector.collect(
            trajectory=mock_trajectory,
            task=mock_task,
            context_tags=["tag_a"],
            sub_item_id="sub_item_x",
        )
        snapshot.metrics = ResultMetrics(
            evidence_count=6,
            confidence=0.72,
            cost_usd=0.38,
            search_rounds=2,
            checklist_coverage=0.90,
        )

        # 🧠 Extract via mock LLM
        extractor = ExperienceExtractor(
            llm_fn=_make_mock_llm_fn(
                [
                    {
                        "category": "search_strategy",
                        "insight": "Multi-source triangulation improves confidence",
                        "context_tags": ["tag_a"],
                        "applicable_sub_items": ["sub_item_x"],
                    },
                    {
                        "category": "evidence_quality",
                        "insight": "Check publication year for relevance",
                        "context_tags": [],
                        "applicable_sub_items": ["*"],
                    },
                ]
            )
        )
        new_experiences = await extractor.extract(snapshot, sample_profile)
        assert len(new_experiences) == 2

        # 💾 Store with dedup (same path as _post_execution_evolution)
        stored_count = 0
        for exp in new_experiences:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)
                stored_count += 1

        assert stored_count == 2

        # ✅ Verify stored in DB with correct attributes
        stats = await real_store.get_stats(namespace)
        assert stats["total_count"] == 2
        assert stats["category_breakdown"]["search_strategy"] == 1
        assert stats["category_breakdown"]["evidence_quality"] == 1

        # 🔍 Query back and verify provenance
        all_exps = await real_store.list_by_namespace(namespace)
        for exp in all_exps:
            assert exp.namespace == namespace
            assert exp.source == "trajectory_extraction"
            assert exp.source_evaluation_id == "eval-resexp-001"

    @pytest.mark.asyncio
    async def test_enrichment_empty_when_no_profile(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Enrichment returns empty when no enrichment template configured 🚫."""
        # 🧬 Profile without enrichment template
        profile_no_template = {
            "namespace": "test_e2e",
            "enrichment_prompt_template": "",
            "enrichment_max_tokens": 500,
            "enrichment_max_items": 10,
            "prune_min_fitness": 0.3,
        }

        enricher = PromptEnricher(real_store)
        result = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=profile_no_template,
        )

        # ✅ Should return empty enrichment (template renders empty)
        assert len(result.injected_experience_ids) == 0

    @pytest.mark.asyncio
    async def test_dedup_prevents_duplicate_storage(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Dedup check should prevent storing identical insights twice 🔍."""
        namespace = sample_profile["namespace"]

        # 🌱 Store first experience
        exp1 = Experience(
            namespace=namespace,
            category="search_strategy",
            insight="Use ensemble search for ambiguous terms",
            context_tags=["tag_a"],
            fitness_score=0.5,
            source="trajectory_extraction",
        )
        await real_store.add(exp1)

        # 🔍 Attempt to store exact duplicate
        is_dup = await real_store.deduplicate(
            namespace, "Use ensemble search for ambiguous terms"
        )
        assert is_dup is True

        # 🔍 Attempt to store substring match
        is_dup2 = await real_store.deduplicate(
            namespace, "ensemble search for ambiguous"
        )
        assert is_dup2 is True

        # ✅ Different insight should pass
        is_dup3 = await real_store.deduplicate(
            namespace, "Check source recency before citing"
        )
        assert is_dup3 is False

        # 📊 Only 1 experience in store
        stats = await real_store.get_stats(namespace)
        assert stats["total_count"] == 1


# ============================================================================
# 🔒 Test: Namespace Isolation
# ============================================================================


class TestNamespaceIsolation:
    """Verify experiences are isolated by namespace 🔒."""

    @pytest.mark.asyncio
    async def test_different_namespaces_dont_cross(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """Experiences in namespace A should not appear in namespace B queries 🔒."""
        # 🌱 Seed experiences in two different namespaces
        exp_a = Experience(
            namespace="namespace_alpha",
            category="search_strategy",
            insight="Alpha-specific strategy: use database X",
            context_tags=["shared_tag"],
            applicable_sub_items=["*"],
            fitness_score=0.8,
            source="trajectory_extraction",
        )
        exp_b = Experience(
            namespace="namespace_beta",
            category="search_strategy",
            insight="Beta-specific strategy: use database Y",
            context_tags=["shared_tag"],
            applicable_sub_items=["*"],
            fitness_score=0.9,
            source="trajectory_extraction",
        )
        await real_store.add(exp_a)
        await real_store.add(exp_b)

        # 🔍 Query namespace_alpha — should only see exp_a
        query_alpha = ExperienceQuery(
            namespace="namespace_alpha",
            context_tags=["shared_tag"],
            min_fitness=0.0,
        )
        results_alpha = await real_store.query(query_alpha)
        assert len(results_alpha) == 1
        assert results_alpha[0].id == exp_a.id
        assert "database X" in results_alpha[0].insight

        # 🔍 Query namespace_beta — should only see exp_b
        query_beta = ExperienceQuery(
            namespace="namespace_beta",
            context_tags=["shared_tag"],
            min_fitness=0.0,
        )
        results_beta = await real_store.query(query_beta)
        assert len(results_beta) == 1
        assert results_beta[0].id == exp_b.id
        assert "database Y" in results_beta[0].insight

    @pytest.mark.asyncio
    async def test_enrichment_respects_namespace(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """PromptEnricher should only inject experiences from its own namespace 💉."""
        # 🌱 Seed experiences in different namespaces
        exp_target = Experience(
            namespace="ns_target",
            category="search_strategy",
            insight="Target namespace insight",
            context_tags=["common_tag"],
            applicable_sub_items=["*"],
            fitness_score=0.8,
            source="trajectory_extraction",
        )
        exp_other = Experience(
            namespace="ns_other",
            category="search_strategy",
            insight="Other namespace insight should not appear",
            context_tags=["common_tag"],
            applicable_sub_items=["*"],
            fitness_score=0.9,
            source="trajectory_extraction",
        )
        await real_store.add(exp_target)
        await real_store.add(exp_other)

        enricher = PromptEnricher(real_store)

        # 💉 Enrich for ns_target
        profile = {
            "namespace": "ns_target",
            "enrichment_prompt_template": (
                "{% for category, exps in experiences_by_category.items() %}\n"
                "{% for exp in exps %}"
                "{{ exp.insight }}\n"
                "{% endfor %}{% endfor %}"
            ),
            "enrichment_max_tokens": 500,
            "enrichment_max_items": 10,
            "prune_min_fitness": 0.0,
        }

        result = await enricher.enrich(
            task_context_tags=["common_tag"],
            sub_item="any",
            profile_config=profile,
        )

        # ✅ Only target namespace insight should appear
        assert len(result.injected_experience_ids) == 1
        assert exp_target.id in result.injected_experience_ids
        assert "Target namespace insight" in result.enrichment_text
        assert "Other namespace" not in result.enrichment_text

    @pytest.mark.asyncio
    async def test_stats_are_namespace_scoped(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """get_stats should only count experiences in the specified namespace 📊."""
        # 🌱 Seed 3 in ns_a, 1 in ns_b
        for i in range(3):
            await real_store.add(
                Experience(
                    namespace="ns_a",
                    category="search_strategy",
                    insight=f"ns_a insight {i}",
                    context_tags=[],
                    fitness_score=0.5,
                    source="test",
                )
            )
        await real_store.add(
            Experience(
                namespace="ns_b",
                category="evidence_quality",
                insight="ns_b insight",
                context_tags=[],
                fitness_score=0.6,
                source="test",
            )
        )

        stats_a = await real_store.get_stats("ns_a")
        stats_b = await real_store.get_stats("ns_b")

        assert stats_a["total_count"] == 3
        assert stats_b["total_count"] == 1
        assert "search_strategy" in stats_a["category_breakdown"]
        assert "evidence_quality" in stats_b["category_breakdown"]

    @pytest.mark.asyncio
    async def test_decay_only_affects_own_namespace(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """apply_decay should only decay experiences within the target namespace 📉."""
        exp_a = Experience(
            namespace="ns_decay_a",
            category="search_strategy",
            insight="ns_a decay test",
            context_tags=[],
            fitness_score=0.8,
            source="test",
        )
        exp_b = Experience(
            namespace="ns_decay_b",
            category="search_strategy",
            insight="ns_b should be untouched",
            context_tags=[],
            fitness_score=0.8,
            source="test",
        )
        await real_store.add(exp_a)
        await real_store.add(exp_b)

        # 📉 Decay only ns_decay_a
        await real_store.apply_decay("ns_decay_a", 0.5)

        # ✅ ns_a decayed, ns_b unchanged
        updated_a = await real_store.get_by_id(exp_a.id)
        updated_b = await real_store.get_by_id(exp_b.id)
        assert updated_a is not None
        assert updated_b is not None
        assert abs(updated_a.fitness_score - 0.4) < 0.01  # 0.8 * 0.5
        assert abs(updated_b.fitness_score - 0.8) < 0.01  # unchanged


# ============================================================================
# 📊 Test: Fitness Evolution Over Time
# ============================================================================


class TestFitnessEvolution:
    """Verify fitness scores evolve correctly over multiple uses 📊."""

    @pytest.mark.asyncio
    async def test_helpful_experiences_gain_fitness(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Experiences that improve results should gain fitness over iterations 📈."""
        # 🌱 Start with default fitness
        exp = Experience(
            namespace=sample_profile["namespace"],
            category="search_strategy",
            insight="Experience gaining fitness over time",
            context_tags=["tag_a"],
            applicable_sub_items=["*"],
            fitness_score=0.5,
            source="trajectory_extraction",
        )
        await real_store.add(exp)

        # 📈 Apply 3 consecutive positive fitness updates
        initial_fitness = 0.5
        for i in range(3):
            delta = FitnessUpdate(
                experience_id=exp.id,
                signal=0.9,  # Strong positive signal
                was_helpful=True,
            )
            await real_store.update_fitness(exp.id, delta)

        # ✅ Verify fitness monotonically increased
        updated = await real_store.get_by_id(exp.id)
        assert updated is not None
        assert updated.fitness_score > initial_fitness
        assert updated.times_used == 3
        assert updated.times_helpful == 3

        # 📊 Verify EMA calculation chain:
        # Round 1: 0.3 * 0.9 + 0.7 * 0.5 = 0.62
        # Round 2: 0.3 * 0.9 + 0.7 * 0.62 = 0.704
        # Round 3: 0.3 * 0.9 + 0.7 * 0.704 = 0.7628
        expected = 0.5
        for _ in range(3):
            expected = 0.3 * 0.9 + 0.7 * expected
        assert abs(updated.fitness_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_unhelpful_experiences_lose_fitness(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Experiences that don't help should gradually lose fitness 📉."""
        # 🌱 Start with decent fitness
        exp = Experience(
            namespace=sample_profile["namespace"],
            category="search_strategy",
            insight="Experience losing fitness over time",
            context_tags=["tag_a"],
            applicable_sub_items=["*"],
            fitness_score=0.7,
            source="trajectory_extraction",
        )
        await real_store.add(exp)

        # 📉 Apply 3 consecutive negative fitness updates
        for i in range(3):
            delta = FitnessUpdate(
                experience_id=exp.id,
                signal=0.1,  # Weak signal
                was_helpful=False,
            )
            await real_store.update_fitness(exp.id, delta)

        # ✅ Verify fitness decreased
        updated = await real_store.get_by_id(exp.id)
        assert updated is not None
        assert updated.fitness_score < 0.7
        assert updated.times_used == 3
        assert updated.times_helpful == 0

        # 📊 Verify EMA calculation chain:
        # Round 1: 0.3 * 0.1 + 0.7 * 0.7 = 0.52
        # Round 2: 0.3 * 0.1 + 0.7 * 0.52 = 0.394
        # Round 3: 0.3 * 0.1 + 0.7 * 0.394 = 0.3058
        expected = 0.7
        for _ in range(3):
            expected = 0.3 * 0.1 + 0.7 * expected
        assert abs(updated.fitness_score - expected) < 0.01

    @pytest.mark.asyncio
    async def test_decayed_low_fitness_gets_pruned(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """After decay, low-fitness experiences should be prunable 🗑️."""
        namespace = sample_profile["namespace"]

        # 🌱 Seed: one high-fitness, one borderline
        high_exp = Experience(
            namespace=namespace,
            category="search_strategy",
            insight="High fitness survives pruning",
            context_tags=[],
            fitness_score=0.8,
            times_used=5,
            source="test",
        )
        low_exp = Experience(
            namespace=namespace,
            category="search_strategy",
            insight="Low fitness gets pruned",
            context_tags=[],
            fitness_score=0.35,
            times_used=5,
            source="test",
        )
        await real_store.add(high_exp)
        await real_store.add(low_exp)

        # 📉 Apply decay (0.8 factor) — low_exp drops below 0.3
        await real_store.apply_decay(namespace, 0.8)

        # 🔍 Verify decay applied
        decayed_low = await real_store.get_by_id(low_exp.id)
        assert decayed_low is not None
        assert abs(decayed_low.fitness_score - 0.28) < 0.01  # 0.35 * 0.8

        decayed_high = await real_store.get_by_id(high_exp.id)
        assert decayed_high is not None
        assert abs(decayed_high.fitness_score - 0.64) < 0.01  # 0.8 * 0.8

        # 🗑️ Prune — only low_exp should be removed
        config = PruneConfig(
            min_fitness=0.3,
            min_uses=3,
            decay_factor=0.8,
            decay_interval_days=7,
        )
        pruned_count = await real_store.prune(namespace, config)
        assert pruned_count == 1

        # ✅ Only high_exp survives
        remaining = await real_store.list_by_namespace(namespace)
        assert len(remaining) == 1
        assert remaining[0].id == high_exp.id

    @pytest.mark.asyncio
    async def test_bulk_fitness_update_applies_to_all(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Bulk update should process all experiences correctly 📊."""
        namespace = sample_profile["namespace"]

        # 🌱 Seed multiple experiences
        exp_ids = []
        for i in range(4):
            exp = Experience(
                namespace=namespace,
                category="search_strategy",
                insight=f"Bulk fitness test experience {i}",
                context_tags=[],
                fitness_score=0.5,
                source="test",
            )
            await real_store.add(exp)
            exp_ids.append(exp.id)

        # 📊 Bulk update with mixed signals
        updates = [
            FitnessUpdate(experience_id=exp_ids[0], signal=0.9, was_helpful=True),
            FitnessUpdate(experience_id=exp_ids[1], signal=0.8, was_helpful=True),
            FitnessUpdate(experience_id=exp_ids[2], signal=0.2, was_helpful=False),
            FitnessUpdate(experience_id=exp_ids[3], signal=0.1, was_helpful=False),
        ]
        await real_store.bulk_update_fitness(updates)

        # ✅ Verify each experience updated correctly
        for i, exp_id in enumerate(exp_ids):
            updated = await real_store.get_by_id(exp_id)
            assert updated is not None
            assert updated.times_used == 1

            # EMA: new = 0.3 * signal + 0.7 * 0.5
            expected = 0.3 * updates[i].signal + 0.7 * 0.5
            assert abs(updated.fitness_score - expected) < 0.01

        # ✅ Helpful counter check
        h0 = await real_store.get_by_id(exp_ids[0])
        h2 = await real_store.get_by_id(exp_ids[2])
        assert h0 is not None and h0.times_helpful == 1
        assert h2 is not None and h2.times_helpful == 0

    @pytest.mark.asyncio
    async def test_fitness_evaluator_computes_correct_signal(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """FitnessEvaluator should compute correct signal from metric deltas 🔍."""
        # 🌱 Seed experience and simulate enrichment
        exp = Experience(
            namespace=sample_profile["namespace"],
            category="search_strategy",
            insight="Fitness evaluator signal test",
            context_tags=[],
            fitness_score=0.5,
            source="test",
        )
        await real_store.add(exp)

        enrichment = EnrichmentResult(
            injected_experience_ids=[exp.id],
            enrichment_text="test enrichment",
            token_count=10,
        )

        evaluator = FitnessEvaluator(real_store)

        # 📊 Metrics improved significantly
        before = ResultMetrics(
            evidence_count=3,
            confidence=0.5,
            cost_usd=0.8,
        )
        after = ResultMetrics(
            evidence_count=10,
            confidence=0.9,
            cost_usd=0.4,
        )

        updates = await evaluator.evaluate(
            enrichment_result=enrichment,
            before_metrics=before,
            after_metrics=after,
            profile_config=sample_profile,
        )

        assert len(updates) == 1
        update = updates[0]
        assert update.experience_id == exp.id
        assert update.was_helpful is True
        assert update.signal > 0.5  # Positive improvement
        # 📈 Check metric deltas
        assert update.metric_deltas["evidence_count"] == 7.0  # 10 - 3
        assert abs(update.metric_deltas["confidence"] - 0.4) < 0.01  # 0.9 - 0.5
        assert abs(update.metric_deltas["cost_usd"] - (-0.4)) < 0.01  # 0.4 - 0.8


# ============================================================================
# 📬 Test: API Request Flow
# ============================================================================


class TestEvolutionProfileFlow:
    """Verify evolution_profile flows correctly through API schemas 📬."""

    def test_research_request_preserves_evolution_profile(self) -> None:
        """ResearchRequest should serialize/deserialize evolution_profile 🔄."""
        from inquiro.api.schemas import ResearchRequest, ResearchTaskPayload

        profile = {
            "namespace": "targetmaster",
            "version": "v1",
            "extraction_prompt_template": "Extract insights...",
            "experience_categories": ["search_strategy", "evidence_quality"],
            "enrichment_prompt_template": "{{ experiences }}",
            "enrichment_max_tokens": 500,
            "fitness_dimensions": [
                {
                    "metric_name": "confidence",
                    "weight": 0.5,
                    "direction": "higher_is_better",
                }
            ],
        }

        request = ResearchRequest(
            task_id="test-profile-001",
            task=ResearchTaskPayload(
                topic="Test evolution profile flow",
                output_schema={"type": "object"},
            ),
            evolution_profile=profile,
            context_tags=["tag_x", "tag_y"],
            sub_item_id="sub_1",
        )

        # 🔄 Round-trip serialization
        dumped = request.model_dump()
        restored = ResearchRequest.model_validate(dumped)

        # ✅ Verify profile preserved exactly
        assert restored.evolution_profile is not None
        assert restored.evolution_profile["namespace"] == "targetmaster"
        assert restored.evolution_profile["version"] == "v1"
        assert len(restored.evolution_profile["experience_categories"]) == 2
        assert len(restored.evolution_profile["fitness_dimensions"]) == 1
        assert restored.context_tags == ["tag_x", "tag_y"]
        assert restored.sub_item_id == "sub_1"

    def test_research_request_without_evolution_profile(self) -> None:
        """ResearchRequest should work fine without evolution_profile 🚫."""
        from inquiro.api.schemas import ResearchRequest, ResearchTaskPayload

        request = ResearchRequest(
            task_id="test-no-profile",
            task=ResearchTaskPayload(
                topic="No evolution profile",
                output_schema={"type": "object"},
            ),
        )

        assert request.evolution_profile is None
        assert request.context_tags == []
        assert request.sub_item_id == ""

        # 🔄 Round-trip
        dumped = request.model_dump()
        restored = ResearchRequest.model_validate(dumped)
        assert restored.evolution_profile is None

    def test_synthesize_request_preserves_evolution_profile(self) -> None:
        """SynthesizeRequest should also carry evolution_profile 📊."""
        from inquiro.api.schemas import (
            InputReport,
            SynthesisTaskPayload,
            SynthesizeRequest,
        )

        profile = {
            "namespace": "targetmaster",
            "enrichment_prompt_template": "{{ experiences }}",
        }

        request = SynthesizeRequest(
            task_id="test-synth-profile",
            task=SynthesisTaskPayload(
                objective="Test synthesis evolution",
                input_reports=[
                    InputReport(
                        report_id="r1",
                        label="Report 1",
                        content={"decision": "positive"},
                    ),
                ],
                output_schema={"type": "object"},
            ),
            evolution_profile=profile,
            context_tags=["synth_tag"],
        )

        # 🔄 Round-trip
        dumped = request.model_dump()
        restored = SynthesizeRequest.model_validate(dumped)

        assert restored.evolution_profile is not None
        assert restored.evolution_profile["namespace"] == "targetmaster"
        assert restored.context_tags == ["synth_tag"]

    def test_evolution_profile_json_serialization(self) -> None:
        """Evolution profile should survive JSON serialization 📦."""
        from inquiro.api.schemas import ResearchRequest, ResearchTaskPayload

        profile = {
            "namespace": "targetmaster",
            "fitness_dimensions": [
                {
                    "metric_name": "evidence_count",
                    "weight": 0.4,
                    "direction": "higher_is_better",
                },
            ],
            "nested_config": {
                "key": "value",
                "list_field": [1, 2, 3],
            },
        }

        request = ResearchRequest(
            task_id="test-json-serial",
            task=ResearchTaskPayload(
                topic="JSON serialization test",
                output_schema={"type": "object"},
            ),
            evolution_profile=profile,
        )

        # 🔄 Full JSON round-trip
        json_str = request.model_dump_json()
        restored = ResearchRequest.model_validate_json(json_str)

        assert restored.evolution_profile is not None
        assert restored.evolution_profile["namespace"] == "targetmaster"
        assert restored.evolution_profile["nested_config"]["key"] == "value"
        assert restored.evolution_profile["nested_config"]["list_field"] == [1, 2, 3]


# ============================================================================
# 🔗 Test: End-to-End Closed Loop Integration
# ============================================================================


class TestClosedLoopIntegration:
    """Verify the complete closed loop from enrichment through fitness update 🔗."""

    @pytest.mark.asyncio
    async def test_full_pipeline_seed_enrich_run_extract_fitness(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Complete pipeline: seed → enrich → execute → extract → store → fitness 🔗."""
        namespace = sample_profile["namespace"]
        enricher = PromptEnricher(real_store)
        evaluator = FitnessEvaluator(real_store)

        # 🌱 Phase 1: Seed an initial experience (simulating prior run)
        seed_exp = Experience(
            namespace=namespace,
            category="search_strategy",
            insight="Combine keyword and semantic search approaches",
            context_tags=["tag_a", "tag_b"],
            applicable_sub_items=["*"],
            fitness_score=0.6,
            source="trajectory_extraction",
            source_evaluation_id="eval-prior",
        )
        await real_store.add(seed_exp)

        # 💉 Phase 2: Pre-execution enrichment
        enrichment = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert seed_exp.id in enrichment.injected_experience_ids
        assert "keyword and semantic" in enrichment.enrichment_text

        # 📸 Phase 3: Simulate execution + collect trajectory
        snapshot = _make_snapshot(
            eval_id="eval-pipeline-001",
            task_id="task-pipeline-001",
            topic="Full pipeline test topic",
            evidence_count=12,
            confidence=0.88,
            cost_usd=0.32,
        )

        # 🧠 Phase 4: Extract new experiences
        extractor = ExperienceExtractor(
            llm_fn=_make_mock_llm_fn(
                [
                    {
                        "category": "evidence_quality",
                        "insight": "Triangulate findings across 3+ independent sources",
                        "context_tags": ["tag_a"],
                        "applicable_sub_items": ["sub_item_x"],
                    },
                ]
            )
        )
        new_exps = await extractor.extract(snapshot, sample_profile)
        assert len(new_exps) == 1

        # 💾 Phase 5: Store new experiences
        for exp in new_exps:
            is_dup = await real_store.deduplicate(exp.namespace, exp.insight)
            if not is_dup:
                await real_store.add(exp)

        stats = await real_store.get_stats(namespace)
        assert stats["total_count"] == 2

        # 📊 Phase 6: Fitness evaluation for injected experiences
        before_metrics = ResultMetrics(
            evidence_count=5,
            confidence=0.60,
            cost_usd=0.50,
        )
        after_metrics = ResultMetrics(
            evidence_count=12,
            confidence=0.88,
            cost_usd=0.32,
        )

        updates = await evaluator.evaluate(
            enrichment_result=enrichment,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            profile_config=sample_profile,
        )
        assert len(updates) == 1
        assert updates[0].experience_id == seed_exp.id
        assert updates[0].was_helpful is True

        # 📈 Phase 7: Apply fitness updates
        await evaluator.apply_updates(
            updates, learning_rate=sample_profile["fitness_learning_rate"]
        )

        # ✅ Final verification
        updated_seed = await real_store.get_by_id(seed_exp.id)
        assert updated_seed is not None
        assert updated_seed.fitness_score > 0.6  # Improved from 0.6
        assert updated_seed.times_used == 1
        assert updated_seed.times_helpful == 1

        # 🔄 Phase 8: Second enrichment should show both experiences
        enrichment_2 = await enricher.enrich(
            task_context_tags=["tag_a"],
            sub_item="sub_item_x",
            profile_config=sample_profile,
        )
        assert len(enrichment_2.injected_experience_ids) == 2
        assert "Triangulate" in enrichment_2.enrichment_text

    @pytest.mark.asyncio
    async def test_extractor_handles_invalid_llm_output_gracefully(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """Extractor should return empty list on malformed LLM output 🛡️."""
        snapshot = _make_snapshot(
            eval_id="eval-bad-llm",
            task_id="task-bad-llm",
            topic="Test bad LLM output",
        )

        # 🤖 LLM returns invalid JSON
        async def bad_json_fn(prompt: str) -> str:
            return "This is not valid JSON at all"

        extractor = ExperienceExtractor(llm_fn=bad_json_fn)
        result = await extractor.extract(snapshot, sample_profile)
        assert result == []

        # 🤖 LLM returns JSON object instead of array
        async def wrong_type_fn(prompt: str) -> str:
            return json.dumps({"category": "search_strategy", "insight": "test"})

        extractor2 = ExperienceExtractor(llm_fn=wrong_type_fn)
        result2 = await extractor2.extract(snapshot, sample_profile)
        assert result2 == []

        # 🤖 LLM returns array with invalid category
        async def bad_category_fn(prompt: str) -> str:
            return json.dumps([{"category": "invalid_category", "insight": "test"}])

        extractor3 = ExperienceExtractor(llm_fn=bad_category_fn)
        result3 = await extractor3.extract(snapshot, sample_profile)
        assert result3 == []

    @pytest.mark.asyncio
    async def test_fitness_evaluator_with_no_dimensions(
        self,
        real_store: ExperienceStore,
    ) -> None:
        """FitnessEvaluator returns empty when no fitness dimensions configured 🚫."""
        evaluator = FitnessEvaluator(real_store)

        enrichment = EnrichmentResult(
            injected_experience_ids=["some-id"],
            enrichment_text="test",
        )
        before = ResultMetrics()
        after = ResultMetrics(evidence_count=5)

        # 📊 No fitness_dimensions in config
        updates = await evaluator.evaluate(
            enrichment_result=enrichment,
            before_metrics=before,
            after_metrics=after,
            profile_config={"fitness_dimensions": []},
        )
        assert updates == []

    @pytest.mark.asyncio
    async def test_fitness_evaluator_with_no_injected_experiences(
        self,
        real_store: ExperienceStore,
        sample_profile: dict[str, Any],
    ) -> None:
        """FitnessEvaluator returns empty when no experiences were injected 🚫."""
        evaluator = FitnessEvaluator(real_store)

        # 💉 Empty enrichment — no experiences injected
        enrichment = EnrichmentResult()
        before = ResultMetrics()
        after = ResultMetrics(evidence_count=5)

        updates = await evaluator.evaluate(
            enrichment_result=enrichment,
            before_metrics=before,
            after_metrics=after,
            profile_config=sample_profile,
        )
        assert updates == []
