"""Tests for EvolutionScheduler — background maintenance automation 🧪.

Covers:
- Single maintenance cycle (decay + prune + conflict resolution)
- Decay reduces fitness scores correctly
- Pruning removes low-fitness experiences
- Conflict detection and resolution
- Start/stop lifecycle with graceful shutdown
- MaintenanceResult captures accurate statistics

Uses in-memory SQLite for fast, isolated tests.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from inquiro.evolution.ranker import ExperienceRanker
from inquiro.evolution.scheduler import (
    EvolutionScheduler,
    MaintenanceResult,
    SchedulerConfig,
)
from inquiro.evolution.store import (
    ExperienceStore,
    drop_store_schema,
    init_store_schema,
)
from inquiro.evolution.types import Experience


# ============================================================================
# 🧪 Test Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def test_engine() -> AsyncEngine:
    """In-memory SQLite engine for testing 💾."""
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
    """Session factory for test database 🔧."""
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


@pytest_asyncio.fixture
async def ranker(store: ExperienceStore) -> ExperienceRanker:
    """ExperienceRanker instance for testing 🗑️."""
    return ExperienceRanker(store)


@pytest.fixture
def scheduler_config() -> SchedulerConfig:
    """Default scheduler config for testing ⚙️."""
    return SchedulerConfig(
        namespace="test_ns",
        decay_factor=0.95,
        decay_interval_seconds=1,  # 🧪 Short interval for testing
        prune_min_fitness=0.2,
        prune_min_uses=5,
        enable_conflict_resolution=True,
    )


@pytest_asyncio.fixture
async def scheduler(
    scheduler_config: SchedulerConfig,
    store: ExperienceStore,
    ranker: ExperienceRanker,
) -> EvolutionScheduler:
    """EvolutionScheduler instance for testing 🔄."""
    sched = EvolutionScheduler(
        config=scheduler_config,
        store=store,
        ranker=ranker,
    )
    yield sched
    # 🛑 Ensure scheduler is stopped after each test
    if sched.is_running:
        await sched.stop()


def _make_experience(
    namespace: str = "test_ns",
    category: str = "search_strategy",
    insight: str = "Use ensemble search for ambiguous queries",
    fitness_score: float = 0.5,
    times_used: int = 0,
    times_helpful: int = 0,
    source: str = "trajectory_extraction",
) -> Experience:
    """Create a test experience with sensible defaults 🧬.

    Args:
        namespace: Namespace for data isolation.
        category: Experience category (opaque).
        insight: Learned insight text.
        fitness_score: Initial fitness score.
        times_used: Number of times used.
        times_helpful: Number of times helpful.
        source: Experience source.

    Returns:
        Experience instance ready for store insertion.
    """
    return Experience(
        namespace=namespace,
        category=category,
        insight=insight,
        fitness_score=fitness_score,
        times_used=times_used,
        times_helpful=times_helpful,
        source=source,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# ============================================================================
# ✅ Single Maintenance Cycle Tests
# ============================================================================


@pytest.mark.asyncio
async def test_single_maintenance_cycle(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Run one maintenance cycle — verify decay + prune works end-to-end 🔄."""
    # 🔧 Seed experiences
    # High fitness, low uses — should survive decay and prune
    await store.add(
        _make_experience(fitness_score=0.8, times_used=3),
    )
    # Low fitness, high uses — should be pruned
    await store.add(
        _make_experience(
            insight="Low fitness strategy to be pruned",
            fitness_score=0.15,
            times_used=10,
        ),
    )
    # Medium fitness — should survive but decay
    await store.add(
        _make_experience(
            insight="Medium fitness strategy",
            fitness_score=0.5,
            times_used=2,
        ),
    )

    result = await scheduler.run_maintenance()

    # ✅ Verify maintenance ran all steps
    assert isinstance(result, MaintenanceResult)
    assert result.decayed_count == 3  # All 3 experiences decayed
    assert result.pruned_count == 1  # 1 low-fitness experience pruned
    assert result.errors == []  # No errors
    assert result.timestamp is not None


# ============================================================================
# 📉 Decay Tests
# ============================================================================


@pytest.mark.asyncio
async def test_decay_reduces_fitness(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Verify decay math: new_fitness = old_fitness * decay_factor 📉."""
    exp = _make_experience(fitness_score=0.8)
    exp_id = await store.add(exp)

    result = await scheduler.run_maintenance()

    assert result.decayed_count == 1

    # 📊 Verify decayed fitness: 0.8 * 0.95 = 0.76
    updated = await store.get_by_id(exp_id)
    assert updated is not None
    assert abs(updated.fitness_score - 0.76) < 0.01


@pytest.mark.asyncio
async def test_decay_applied_before_prune(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Verify decay runs before prune — borderline experiences may be pruned 📉."""
    # 🧪 Fitness just above prune threshold (0.2), after 0.95 decay = 0.209
    # Still above threshold, so should NOT be pruned
    exp = _make_experience(fitness_score=0.22, times_used=10)
    exp_id = await store.add(exp)

    result = await scheduler.run_maintenance()

    assert result.decayed_count == 1
    # 📊 After decay: 0.22 * 0.95 = 0.209 — still above 0.2 threshold
    updated = await store.get_by_id(exp_id)
    assert updated is not None
    assert updated.fitness_score > 0.2


# ============================================================================
# 🗑️ Prune Tests
# ============================================================================


@pytest.mark.asyncio
async def test_prune_removes_low_fitness(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Verify pruning removes experiences below min_fitness threshold 🗑️."""
    # ✅ Should survive (high fitness)
    high = _make_experience(
        insight="High fitness strategy",
        fitness_score=0.9,
        times_used=10,
    )
    high_id = await store.add(high)

    # ❌ Should be pruned (low fitness, sufficient uses)
    low = _make_experience(
        insight="Low fitness strategy to remove",
        fitness_score=0.1,
        times_used=10,
    )
    await store.add(low)

    # 🛡️ Should survive (low fitness but insufficient uses)
    untested = _make_experience(
        insight="Low fitness but untested",
        fitness_score=0.1,
        times_used=2,
    )
    untested_id = await store.add(untested)

    result = await scheduler.run_maintenance()

    assert result.pruned_count == 1

    # ✅ High fitness survives
    assert await store.get_by_id(high_id) is not None
    # ✅ Untested survives (below min_uses)
    assert await store.get_by_id(untested_id) is not None


# ============================================================================
# ⚔️ Conflict Resolution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_conflict_resolution(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Seed conflicting experiences — verify resolution keeps higher fitness ⚔️."""
    # 🏆 Higher fitness — should be kept
    winner = _make_experience(
        category="tool_selection",
        insight="Use pubmed for clinical evidence searches",
        fitness_score=0.9,
        times_used=10,
    )
    winner_id = await store.add(winner)

    # 💀 Lower fitness, conflicting — should have fitness reduced
    loser = _make_experience(
        category="tool_selection",
        insight="Avoid pubmed for evidence searches",
        fitness_score=0.4,
        times_used=8,
    )
    loser_id = await store.add(loser)

    result = await scheduler.run_maintenance()

    assert result.conflicts_resolved == 1

    # ✅ Winner still has high fitness (decayed by 0.95)
    winner_updated = await store.get_by_id(winner_id)
    assert winner_updated is not None
    assert winner_updated.fitness_score > 0.5

    # 💀 Loser has reduced fitness (conflict resolution sets signal=0.0)
    loser_updated = await store.get_by_id(loser_id)
    assert loser_updated is not None
    assert loser_updated.fitness_score < 0.4  # Reduced by conflict resolution


@pytest.mark.asyncio
async def test_conflict_resolution_disabled(
    store: ExperienceStore,
    ranker: ExperienceRanker,
):
    """Verify conflict resolution is skipped when disabled ⚙️."""
    config = SchedulerConfig(
        namespace="test_ns",
        decay_interval_seconds=1,
        enable_conflict_resolution=False,
    )
    sched = EvolutionScheduler(config=config, store=store, ranker=ranker)

    # 🔧 Seed conflicting experiences
    await store.add(
        _make_experience(
            category="tool_selection",
            insight="Use pubmed for searches",
            fitness_score=0.9,
        ),
    )
    await store.add(
        _make_experience(
            category="tool_selection",
            insight="Avoid pubmed for searches",
            fitness_score=0.4,
        ),
    )

    result = await sched.run_maintenance()

    # ✅ Conflicts NOT resolved because it's disabled
    assert result.conflicts_resolved == 0


# ============================================================================
# 🔄 Start/Stop Lifecycle Tests
# ============================================================================


@pytest.mark.asyncio
async def test_start_stop_lifecycle(
    store: ExperienceStore,
    ranker: ExperienceRanker,
):
    """Start scheduler, let it run one cycle, stop gracefully 🔄."""
    config = SchedulerConfig(
        namespace="test_ns",
        decay_interval_seconds=1,  # 🧪 1 second interval for fast test
    )
    sched = EvolutionScheduler(config=config, store=store, ranker=ranker)

    # 🔧 Seed one experience so maintenance has work to do
    await store.add(_make_experience(fitness_score=0.8))

    # 🚀 Start the scheduler
    assert not sched.is_running
    await sched.start()
    assert sched.is_running

    # ⏳ Wait for one cycle to complete (interval=1s + processing)
    await asyncio.sleep(1.5)

    # 🛑 Stop gracefully
    await sched.stop()
    assert not sched.is_running

    # ✅ At least one cycle should have run
    assert sched.last_result is not None
    assert sched.last_result.decayed_count >= 0


@pytest.mark.asyncio
async def test_start_idempotent(
    scheduler: EvolutionScheduler,
):
    """Calling start() twice does not create duplicate tasks ✅."""
    await scheduler.start()
    assert scheduler.is_running

    # 🔄 Second call should be a no-op
    await scheduler.start()
    assert scheduler.is_running

    await scheduler.stop()
    assert not scheduler.is_running


@pytest.mark.asyncio
async def test_stop_when_not_running(
    scheduler: EvolutionScheduler,
):
    """Calling stop() when not running is safe ✅."""
    assert not scheduler.is_running
    await scheduler.stop()  # Should not raise
    assert not scheduler.is_running


# ============================================================================
# 📊 MaintenanceResult Tests
# ============================================================================


@pytest.mark.asyncio
async def test_maintenance_result_captures_stats(
    scheduler: EvolutionScheduler,
    store: ExperienceStore,
):
    """Verify MaintenanceResult has correct counts for all operations 📊."""
    # 🔧 Seed diverse experiences
    # ✅ Normal experience — will be decayed only
    await store.add(
        _make_experience(
            insight="Normal strategy",
            fitness_score=0.7,
            times_used=2,
        ),
    )
    # 🗑️ Low fitness, high uses — will be pruned
    await store.add(
        _make_experience(
            insight="Bad strategy to prune",
            fitness_score=0.1,
            times_used=10,
        ),
    )
    # ⚔️ Conflicting pair
    await store.add(
        _make_experience(
            category="data_source",
            insight="Prefer clinicaltrials for trial data",
            fitness_score=0.85,
            times_used=6,
        ),
    )
    await store.add(
        _make_experience(
            category="data_source",
            insight="Avoid clinicaltrials for trial lookups",
            fitness_score=0.35,
            times_used=6,
        ),
    )

    result = await scheduler.run_maintenance()

    # 📊 Verify all counts
    assert result.decayed_count == 4  # All 4 experiences decayed
    assert result.pruned_count == 1  # 1 low-fitness pruned
    assert result.conflicts_resolved == 1  # 1 conflict pair resolved
    assert result.errors == []  # No errors
    assert isinstance(result.timestamp, datetime)


@pytest.mark.asyncio
async def test_maintenance_result_captures_errors(
    store: ExperienceStore,
):
    """Verify errors are captured without stopping subsequent steps 🛡️."""
    config = SchedulerConfig(
        namespace="",  # ❌ Invalid namespace — will cause decay to fail
        decay_interval_seconds=1,
    )
    sched = EvolutionScheduler(config=config, store=store)

    result = await sched.run_maintenance()

    # 🛡️ Errors captured but cycle completed
    assert len(result.errors) > 0
    assert result.timestamp is not None


@pytest.mark.asyncio
async def test_maintenance_on_empty_store(
    scheduler: EvolutionScheduler,
):
    """Verify maintenance on empty store completes without errors 📭."""
    result = await scheduler.run_maintenance()

    assert result.decayed_count == 0
    assert result.pruned_count == 0
    assert result.conflicts_resolved == 0
    assert result.errors == []
