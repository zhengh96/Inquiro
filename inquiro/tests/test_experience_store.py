"""Tests for ExperienceStore — persistence layer for self-evolution 🧪.

Covers:
- CRUD operations (add, query, get_by_id)
- Namespace isolation (cross-namespace queries must fail)
- Fitness updates (single and bulk, with EMA)
- Pruning (remove low-fitness experiences)
- Deduplication (substring matching)
- Statistics (count, avg fitness, category breakdown)
- Decay (time-based fitness reduction)

Uses in-memory SQLite for fast, isolated tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from inquiro.evolution.store import (
    ExperienceStore,
    drop_store_schema,
    init_store_schema,
)
from inquiro.evolution.types import (
    Experience,
    ExperienceQuery,
    FitnessUpdate,
    PruneConfig,
)


# ============================================================================
# 🧪 Test Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def test_engine() -> AsyncEngine:
    """In-memory SQLite engine for testing 💾.

    Creates a fresh database for each test, ensuring isolation.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    await init_store_schema(engine)
    yield engine
    await drop_store_schema(engine)
    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(
    test_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Session factory for test database 🔧.

    Configured with ``expire_on_commit=False`` to avoid detached instance issues.
    """
    return async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def store(
    session_factory: async_sessionmaker[AsyncSession],
) -> ExperienceStore:
    """ExperienceStore instance for testing 💾."""
    return ExperienceStore(session_factory)


@pytest.fixture
def sample_experience() -> Experience:
    """Sample experience for testing 🧬.

    Uses "targetmaster" namespace with realistic field values.
    """
    return Experience(
        namespace="targetmaster",
        category="search_strategy",
        insight="Use ensemble search for ambiguous protein targets",
        context_tags=["modality:SmallMolecule", "dimension:target_biology"],
        applicable_sub_items=["*"],
        fitness_score=0.5,
        times_used=0,
        times_helpful=0,
        source="trajectory_extraction",
        source_evaluation_id="eval-001",
        source_trajectory_step=3,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_experience_2() -> Experience:
    """Second sample experience with different category 🧬."""
    return Experience(
        namespace="targetmaster",
        category="evidence_validation",
        insight="Prioritize Phase 3 trial data over conference abstracts",
        context_tags=["modality:SmallMolecule", "dimension:clinical_feasibility"],
        applicable_sub_items=["sub-1", "sub-2"],
        fitness_score=0.7,
        times_used=5,
        times_helpful=3,
        source="human_feedback",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_experience_other_namespace() -> Experience:
    """Experience in a different namespace 🧬."""
    return Experience(
        namespace="other_platform",
        category="search_strategy",
        insight="Different platform insight",
        context_tags=["tag:other"],
        fitness_score=0.8,
        source="trajectory_extraction",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# ============================================================================
# ✅ CRUD Operation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_add_experience(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test adding a new experience 💉."""
    exp_id = await store.add(sample_experience)

    assert exp_id == sample_experience.id
    assert exp_id is not None


@pytest.mark.asyncio
async def test_get_by_id(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test retrieving experience by ID 🔍."""
    exp_id = await store.add(sample_experience)

    retrieved = await store.get_by_id(exp_id)

    assert retrieved is not None
    assert retrieved.id == exp_id
    assert retrieved.namespace == "targetmaster"
    assert retrieved.category == "search_strategy"
    assert retrieved.insight == sample_experience.insight
    assert retrieved.fitness_score == 0.5


@pytest.mark.asyncio
async def test_get_by_id_not_found(store: ExperienceStore):
    """Test get_by_id with non-existent ID 🔍."""
    result = await store.get_by_id("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_query_by_namespace(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test querying experiences by namespace 🔍."""
    await store.add(sample_experience)
    await store.add(sample_experience_2)

    query = ExperienceQuery(
        namespace="targetmaster",
        min_fitness=0.0,
        max_results=10,
    )
    results = await store.query(query)

    assert len(results) == 2
    assert all(exp.namespace == "targetmaster" for exp in results)


@pytest.mark.asyncio
async def test_query_by_category(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test querying experiences by category 🔍."""
    await store.add(sample_experience)
    await store.add(sample_experience_2)

    query = ExperienceQuery(
        namespace="targetmaster",
        category="search_strategy",
        min_fitness=0.0,
        max_results=10,
    )
    results = await store.query(query)

    assert len(results) == 1
    assert results[0].category == "search_strategy"


@pytest.mark.asyncio
async def test_query_by_min_fitness(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test filtering by minimum fitness score 📊."""
    await store.add(sample_experience)  # fitness = 0.5
    await store.add(sample_experience_2)  # fitness = 0.7

    query = ExperienceQuery(
        namespace="targetmaster",
        min_fitness=0.6,
        max_results=10,
    )
    results = await store.query(query)

    assert len(results) == 1
    assert results[0].fitness_score >= 0.6


@pytest.mark.asyncio
async def test_query_orders_by_fitness_descending(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test query results are ordered by fitness descending 📊."""
    await store.add(sample_experience)  # fitness = 0.5
    await store.add(sample_experience_2)  # fitness = 0.7

    query = ExperienceQuery(
        namespace="targetmaster",
        min_fitness=0.0,
        max_results=10,
    )
    results = await store.query(query)

    assert len(results) == 2
    assert results[0].fitness_score >= results[1].fitness_score


@pytest.mark.asyncio
async def test_query_respects_max_results(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test max_results limit is enforced 🔢."""
    await store.add(sample_experience)
    await store.add(sample_experience_2)

    query = ExperienceQuery(
        namespace="targetmaster",
        min_fitness=0.0,
        max_results=1,
    )
    results = await store.query(query)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_query_without_namespace_raises_error(store: ExperienceStore):
    """Test query without namespace raises ValueError ❌."""
    query = ExperienceQuery(
        namespace="",  # Empty namespace
        min_fitness=0.0,
        max_results=10,
    )

    with pytest.raises(ValueError, match="must specify a namespace"):
        await store.query(query)


# ============================================================================
# 🔒 Namespace Isolation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_namespace_isolation(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_other_namespace: Experience,
):
    """Test experiences from different namespaces are isolated 🔒."""
    await store.add(sample_experience)
    await store.add(sample_experience_other_namespace)

    # Query targetmaster namespace
    query_tm = ExperienceQuery(
        namespace="targetmaster",
        min_fitness=0.0,
        max_results=10,
    )
    results_tm = await store.query(query_tm)

    # Query other_platform namespace
    query_other = ExperienceQuery(
        namespace="other_platform",
        min_fitness=0.0,
        max_results=10,
    )
    results_other = await store.query(query_other)

    # Each namespace should only see its own experiences
    assert len(results_tm) == 1
    assert results_tm[0].namespace == "targetmaster"

    assert len(results_other) == 1
    assert results_other[0].namespace == "other_platform"


# ============================================================================
# 📈 Fitness Update Tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_fitness_with_ema(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test fitness update uses EMA (exponential moving average) 📈."""
    exp_id = await store.add(sample_experience)

    # Initial fitness = 0.5
    # EMA formula: new_fitness = alpha * signal + (1 - alpha) * old_fitness
    # With alpha = 0.3, signal = 0.8:
    # expected = 0.3 * 0.8 + 0.7 * 0.5 = 0.24 + 0.35 = 0.59

    update = FitnessUpdate(
        experience_id=exp_id,
        signal=0.8,
        was_helpful=True,
        metric_deltas={"confidence": 0.1},
    )

    await store.update_fitness(exp_id, update)

    # Verify updated values
    updated_exp = await store.get_by_id(exp_id)
    assert updated_exp is not None
    assert abs(updated_exp.fitness_score - 0.59) < 0.01
    assert updated_exp.times_used == 1
    assert updated_exp.times_helpful == 1


@pytest.mark.asyncio
async def test_update_fitness_not_helpful(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test fitness update when experience was not helpful 📉."""
    exp_id = await store.add(sample_experience)

    update = FitnessUpdate(
        experience_id=exp_id,
        signal=0.2,
        was_helpful=False,
        metric_deltas={"confidence": -0.1},
    )

    await store.update_fitness(exp_id, update)

    updated_exp = await store.get_by_id(exp_id)
    assert updated_exp is not None
    assert updated_exp.times_used == 1
    assert updated_exp.times_helpful == 0  # Not incremented


@pytest.mark.asyncio
async def test_update_fitness_clamps_to_valid_range(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test fitness score is clamped to [0.0, 1.0] 🔒."""
    sample_experience.fitness_score = 0.1
    exp_id = await store.add(sample_experience)

    # Signal = 0.0 should push fitness down but not below 0.0
    update = FitnessUpdate(
        experience_id=exp_id,
        signal=0.0,
        was_helpful=False,
    )

    await store.update_fitness(exp_id, update)

    updated_exp = await store.get_by_id(exp_id)
    assert updated_exp is not None
    assert 0.0 <= updated_exp.fitness_score <= 1.0


@pytest.mark.asyncio
async def test_bulk_update_fitness(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test bulk fitness updates 📊."""
    exp_id_1 = await store.add(sample_experience)
    exp_id_2 = await store.add(sample_experience_2)

    updates = [
        FitnessUpdate(
            experience_id=exp_id_1,
            signal=0.9,
            was_helpful=True,
        ),
        FitnessUpdate(
            experience_id=exp_id_2,
            signal=0.6,
            was_helpful=False,
        ),
    ]

    await store.bulk_update_fitness(updates)

    # Verify both were updated
    exp_1 = await store.get_by_id(exp_id_1)
    exp_2 = await store.get_by_id(exp_id_2)

    assert exp_1 is not None
    assert exp_1.times_used == 1
    assert exp_1.times_helpful == 1

    assert exp_2 is not None
    assert exp_2.times_used == 6  # Was 5 initially
    assert exp_2.times_helpful == 3  # Not incremented (was_helpful=False)


@pytest.mark.asyncio
async def test_bulk_update_fitness_empty_list(store: ExperienceStore):
    """Test bulk update with empty list is a no-op 📊."""
    # Should not raise any errors
    await store.bulk_update_fitness([])


@pytest.mark.asyncio
async def test_update_fitness_nonexistent_experience_raises_error(
    store: ExperienceStore,
):
    """Test updating non-existent experience raises ValueError ❌."""
    update = FitnessUpdate(
        experience_id="nonexistent-id",
        signal=0.8,
        was_helpful=True,
    )

    with pytest.raises(ValueError, match="not found"):
        await store.update_fitness("nonexistent-id", update)


# ============================================================================
# 🗑️ Pruning Tests
# ============================================================================


@pytest.mark.asyncio
async def test_prune_low_fitness_experiences(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test pruning removes low-fitness experiences 🗑️."""
    # Add two experiences with different fitness scores
    sample_experience.fitness_score = 0.2
    sample_experience.times_used = 10  # Meet min_uses threshold
    await store.add(sample_experience)

    sample_experience_2.fitness_score = 0.8
    await store.add(sample_experience_2)

    # Prune experiences with fitness < 0.5 and times_used >= 5
    config = PruneConfig(
        min_fitness=0.5,
        min_uses=5,
        decay_factor=0.95,
        decay_interval_days=7,
    )
    deleted_count = await store.prune("targetmaster", config)

    assert deleted_count == 1

    # Verify only high-fitness experience remains
    remaining = await store.query(
        ExperienceQuery(
            namespace="targetmaster",
            min_fitness=0.0,
            max_results=10,
        )
    )
    assert len(remaining) == 1
    assert remaining[0].fitness_score == 0.8


@pytest.mark.asyncio
async def test_prune_respects_min_uses_threshold(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test pruning does not remove untested experiences 🛡️."""
    # Low fitness but insufficient uses
    sample_experience.fitness_score = 0.2
    sample_experience.times_used = 2  # Below min_uses threshold
    await store.add(sample_experience)

    config = PruneConfig(
        min_fitness=0.5,
        min_uses=5,
        decay_factor=0.95,
        decay_interval_days=7,
    )
    deleted_count = await store.prune("targetmaster", config)

    # Should not prune (insufficient usage)
    assert deleted_count == 0


@pytest.mark.asyncio
async def test_prune_respects_namespace(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_other_namespace: Experience,
):
    """Test pruning only affects specified namespace 🔒."""
    sample_experience.fitness_score = 0.2
    sample_experience.times_used = 10
    await store.add(sample_experience)

    sample_experience_other_namespace.fitness_score = 0.2
    sample_experience_other_namespace.times_used = 10
    await store.add(sample_experience_other_namespace)

    config = PruneConfig(
        min_fitness=0.5,
        min_uses=5,
        decay_factor=0.95,
        decay_interval_days=7,
    )
    deleted_count = await store.prune("targetmaster", config)

    assert deleted_count == 1

    # Other namespace should be unaffected
    other_remains = await store.get_by_id(sample_experience_other_namespace.id)
    assert other_remains is not None


@pytest.mark.asyncio
async def test_prune_without_namespace_raises_error(store: ExperienceStore):
    """Test pruning without namespace raises ValueError ❌."""
    config = PruneConfig(
        min_fitness=0.5,
        min_uses=5,
        decay_factor=0.95,
        decay_interval_days=7,
    )

    with pytest.raises(ValueError, match="requires a namespace"):
        await store.prune("", config)


# ============================================================================
# 📉 Decay Tests
# ============================================================================


@pytest.mark.asyncio
async def test_apply_decay(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test applying time-based decay to fitness scores 📉."""
    await store.add(sample_experience)  # fitness = 0.5
    await store.add(sample_experience_2)  # fitness = 0.7

    # Apply 10% decay (multiply by 0.9)
    updated_count = await store.apply_decay("targetmaster", 0.9)

    assert updated_count == 2

    # Verify fitness scores were reduced
    exp_1 = await store.get_by_id(sample_experience.id)
    exp_2 = await store.get_by_id(sample_experience_2.id)

    assert exp_1 is not None
    assert abs(exp_1.fitness_score - 0.45) < 0.01  # 0.5 * 0.9

    assert exp_2 is not None
    assert abs(exp_2.fitness_score - 0.63) < 0.01  # 0.7 * 0.9


@pytest.mark.asyncio
async def test_apply_decay_respects_namespace(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_other_namespace: Experience,
):
    """Test decay only affects specified namespace 🔒."""
    await store.add(sample_experience)
    await store.add(sample_experience_other_namespace)

    updated_count = await store.apply_decay("targetmaster", 0.9)

    assert updated_count == 1

    # Other namespace should be unaffected
    other_exp = await store.get_by_id(sample_experience_other_namespace.id)
    assert other_exp is not None
    assert other_exp.fitness_score == 0.8  # Unchanged


@pytest.mark.asyncio
async def test_apply_decay_invalid_factor_raises_error(
    store: ExperienceStore,
):
    """Test invalid decay factor raises ValueError ❌."""
    with pytest.raises(ValueError, match="decay_factor"):
        await store.apply_decay("targetmaster", 1.5)  # > 1.0

    with pytest.raises(ValueError, match="decay_factor"):
        await store.apply_decay("targetmaster", 0.0)  # <= 0.0


@pytest.mark.asyncio
async def test_apply_decay_without_namespace_raises_error(
    store: ExperienceStore,
):
    """Test decay without namespace raises ValueError ❌."""
    with pytest.raises(ValueError, match="requires a namespace"):
        await store.apply_decay("", 0.9)


# ============================================================================
# 📊 Statistics Tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_stats(
    store: ExperienceStore,
    sample_experience: Experience,
    sample_experience_2: Experience,
):
    """Test computing aggregate statistics 📊."""
    await store.add(sample_experience)  # fitness = 0.5, category = search_strategy
    await store.add(
        sample_experience_2
    )  # fitness = 0.7, category = evidence_validation

    stats = await store.get_stats("targetmaster")

    assert stats["total_count"] == 2
    assert abs(stats["avg_fitness"] - 0.6) < 0.01  # (0.5 + 0.7) / 2
    assert stats["category_breakdown"] == {
        "search_strategy": 1,
        "evidence_validation": 1,
    }


@pytest.mark.asyncio
async def test_get_stats_empty_namespace(store: ExperienceStore):
    """Test stats for empty namespace 📊."""
    stats = await store.get_stats("empty_namespace")

    assert stats["total_count"] == 0
    assert stats["avg_fitness"] == 0.0
    assert stats["category_breakdown"] == {}


@pytest.mark.asyncio
async def test_get_stats_without_namespace_raises_error(
    store: ExperienceStore,
):
    """Test stats without namespace raises ValueError ❌."""
    with pytest.raises(ValueError, match="requires a namespace"):
        await store.get_stats("")


# ============================================================================
# 🔍 Deduplication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_deduplicate_exact_match(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test deduplication detects exact duplicate 🔍."""
    await store.add(sample_experience)

    # Try to add same insight
    is_duplicate = await store.deduplicate(
        namespace="targetmaster",
        new_insight=sample_experience.insight,
        threshold=0.9,
    )

    assert is_duplicate is True


@pytest.mark.asyncio
async def test_deduplicate_substring_match(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test deduplication detects substring matches 🔍."""
    await store.add(sample_experience)

    # Try to add substring
    is_duplicate = await store.deduplicate(
        namespace="targetmaster",
        new_insight="Use ensemble search",  # Substring of existing
        threshold=0.9,
    )

    assert is_duplicate is True


@pytest.mark.asyncio
async def test_deduplicate_no_match(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test deduplication allows unique insights 🔍."""
    await store.add(sample_experience)

    # Try to add completely different insight
    is_duplicate = await store.deduplicate(
        namespace="targetmaster",
        new_insight="Completely different insight about biomarkers",
        threshold=0.9,
    )

    assert is_duplicate is False


@pytest.mark.asyncio
async def test_deduplicate_respects_namespace(
    store: ExperienceStore,
    sample_experience: Experience,
):
    """Test deduplication is namespace-isolated 🔒."""
    await store.add(sample_experience)

    # Same insight but different namespace should not be duplicate
    is_duplicate = await store.deduplicate(
        namespace="other_platform",
        new_insight=sample_experience.insight,
        threshold=0.9,
    )

    assert is_duplicate is False


@pytest.mark.asyncio
async def test_deduplicate_without_namespace_raises_error(
    store: ExperienceStore,
):
    """Test deduplicate without namespace raises ValueError ❌."""
    with pytest.raises(ValueError, match="requires a namespace"):
        await store.deduplicate(
            namespace="",
            new_insight="Some insight",
        )
