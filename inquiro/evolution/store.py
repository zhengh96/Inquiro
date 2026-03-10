"""Experience Store — PostgreSQL-backed persistence for self-evolution 💾.

Provides async CRUD operations for the Experience model with strict
namespace isolation. All queries MUST specify a namespace to prevent
cross-contamination between different upper-layer platforms.

Key features:
- Dependency injection: Accepts ``async_sessionmaker`` (no module-level global)
- Namespace isolation: All queries filter by namespace
- Fitness updates: Exponential moving average (EMA) with batch support
- Pruning: Remove low-fitness experiences based on configurable thresholds
- Decay: Apply time-based decay to all fitness scores
- Deduplication: Check for similar insights before insertion

Example usage::

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from inquiro.evolution.store import ExperienceStore, init_store_schema

    # Initialize database
    engine = create_async_engine("postgresql+asyncpg://...")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    await init_store_schema(engine)

    # Create store with injected session factory
    store = ExperienceStore(session_factory)

    # Add experience
    exp = Experience(
        namespace="targetmaster",
        category="search_strategy",
        insight="Use ensemble search for ambiguous terms",
        context_tags=["modality:SmallMolecule"],
        fitness_score=0.5,
        source="trajectory_extraction",
    )
    exp_id = await store.add(exp)

    # Query experiences
    query = ExperienceQuery(
        namespace="targetmaster",
        context_tags=["modality:SmallMolecule"],
        min_fitness=0.6,
        max_results=10,
    )
    results = await store.query(query)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    and_,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from inquiro.evolution.types import (
    Experience,
    ExperienceQuery,
    FitnessUpdate,
    PruneConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 🏗️ ORM Base
# ============================================================================


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for evolution schema 🏗️."""

    pass


# ============================================================================
# 💾 ORM Model
# ============================================================================


class ExperienceRecord(Base):
    """Persistent experience record 🧬.

    Maps the domain-layer ``Experience`` Pydantic model to a relational
    table. Uses JSON columns for list fields (context_tags, applicable_sub_items)
    to support both SQLite (dev) and PostgreSQL (prod).

    Indexes:
    - (namespace, category) — Filter by category within namespace
    - (namespace, fitness_score) — Ranking and pruning
    - context_tags — GIN index for PostgreSQL (JSON containment queries)
    """

    __tablename__ = "experiences"

    # 🆔 Identity
    id = Column(String, primary_key=True)
    namespace = Column(String, nullable=False, index=True)

    # 🏷️ Classification (opaque to Inquiro)
    category = Column(String, nullable=False)

    # 📝 Content (opaque to Inquiro)
    insight = Column(Text, nullable=False)
    context_tags = Column(JSON, default=list, nullable=False)
    applicable_sub_items = Column(JSON, default=list, nullable=False)

    # 📊 Fitness
    fitness_score = Column(Float, default=0.5, nullable=False)
    times_used = Column(Integer, default=0, nullable=False)
    times_helpful = Column(Integer, default=0, nullable=False)

    # 📎 Provenance
    source = Column(String, nullable=False)
    source_evaluation_id = Column(String, nullable=True)
    source_trajectory_step = Column(Integer, nullable=True)

    # ⏱️ Timestamps
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


# ============================================================================
# 💾 Store Implementation
# ============================================================================


class ExperienceStore:
    """Async PostgreSQL-backed experience store with namespace isolation 💾.

    All operations require a namespace parameter to enforce data isolation
    between different upper-layer platforms (e.g., "targetmaster").

    Thread-safe when using separate session_factory instances.
    """

    # 🎯 EMA smoothing factor for fitness updates
    EMA_ALPHA = 0.3

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """Initialize store with injected session factory 🔧.

        Args:
            session_factory: SQLAlchemy async session factory for
                database connections. Must be configured with
                ``expire_on_commit=False`` to avoid detached instance issues.
        """
        self._session_factory = session_factory
        logger.debug("💾 ExperienceStore initialized")

    async def add(self, exp: Experience) -> str:
        """Insert a new experience and return its ID 💉.

        Args:
            exp: Experience to insert. The ``id`` field is used as primary key.
                If not provided, the Pydantic default (UUID4) is used.

        Returns:
            The experience ID (same as ``exp.id``).

        Raises:
            sqlalchemy.exc.IntegrityError: If an experience with the same ID
                already exists.
        """
        async with self._session_factory() as session:
            record = ExperienceRecord(
                id=exp.id,
                namespace=exp.namespace,
                category=exp.category,
                insight=exp.insight,
                context_tags=exp.context_tags,
                applicable_sub_items=exp.applicable_sub_items,
                fitness_score=exp.fitness_score,
                times_used=exp.times_used,
                times_helpful=exp.times_helpful,
                source=exp.source,
                source_evaluation_id=exp.source_evaluation_id,
                source_trajectory_step=exp.source_trajectory_step,
                created_at=exp.created_at,
                updated_at=exp.updated_at,
            )
            session.add(record)
            await session.commit()
            logger.info(
                "💉 Added experience %s (namespace=%s, category=%s)",
                exp.id,
                exp.namespace,
                exp.category,
            )
            return exp.id

    async def query(self, q: ExperienceQuery) -> list[Experience]:
        """Query experiences with namespace isolation 🔍.

        Args:
            q: Query specification. The ``namespace`` field is REQUIRED.

        Returns:
            List of matching experiences, ordered by fitness_score descending,
            limited to ``q.max_results``.

        Raises:
            ValueError: If ``q.namespace`` is empty.
        """
        if not q.namespace:
            raise ValueError("Query must specify a namespace")

        async with self._session_factory() as session:
            # 🎯 Build base query with namespace filter
            stmt = select(ExperienceRecord).where(
                ExperienceRecord.namespace == q.namespace
            )

            # 🏷️ Optional category filter
            if q.category:
                stmt = stmt.where(ExperienceRecord.category == q.category)

            # 📊 Fitness threshold filter
            stmt = stmt.where(ExperienceRecord.fitness_score >= q.min_fitness)

            # 📊 Order by fitness descending.
            # ⚠️ Performance note: JSON filtering is done Python-side (not SQL)
            # because json_extract is SQLite-only and jsonb operators are
            # PostgreSQL-only. The 5x fetch multiplier compensates for
            # post-fetch filtering loss. Consider dialect-aware JSON
            # filtering if query volume exceeds ~1000 records/call.
            # Fetch extra rows so Python-side JSON filtering still
            # returns enough results.
            sql_limit = (
                q.max_results * 5 if (q.context_tags or q.sub_item) else q.max_results
            )
            stmt = stmt.order_by(ExperienceRecord.fitness_score.desc()).limit(sql_limit)

            result = await session.execute(stmt)
            records = result.scalars().all()

            # 🔄 Convert ORM records to Pydantic models
            experiences = [self._record_to_experience(rec) for rec in records]

            # 🔍 Python-side JSON filtering (dialect-agnostic; avoids json_extract
            # which is SQLite-only and not available on PostgreSQL).
            if q.context_tags:
                experiences = [
                    e
                    for e in experiences
                    if any(tag in e.context_tags for tag in q.context_tags)
                ]
            if q.sub_item:
                experiences = [
                    e
                    for e in experiences
                    if "*" in e.applicable_sub_items
                    or q.sub_item in e.applicable_sub_items
                ]

            # ✂️ Truncate to requested max after post-filtering
            experiences = experiences[: q.max_results]

            logger.debug(
                "🔍 Query returned %d experiences (namespace=%s, category=%s)",
                len(experiences),
                q.namespace,
                q.category,
            )
            return experiences

    async def update_fitness(
        self,
        exp_id: str,
        delta: FitnessUpdate,
        precomputed_fitness: float | None = None,
    ) -> None:
        """Update fitness score, optionally using a precomputed value 📈.

        When ``precomputed_fitness`` is provided the EMA calculation is
        skipped and the value is used directly (the caller already applied
        the EMA with its own learning_rate).  When it is ``None`` the
        default EMA formula is used:

            new_fitness = EMA_ALPHA * signal + (1 - EMA_ALPHA) * old_fitness

        Also increments ``times_used`` and ``times_helpful`` counters.

        Args:
            exp_id: Experience ID to update.
            delta: Fitness update containing signal strength and helpfulness.
            precomputed_fitness: If provided, use this value directly instead
                of recomputing EMA inside the store.  Must be in [0.0, 1.0].

        Raises:
            ValueError: If experience not found.
        """
        if delta.experience_id != exp_id:
            logger.warning(
                "⚠️ FitnessUpdate.experience_id (%s) does not match exp_id (%s)",
                delta.experience_id,
                exp_id,
            )

        async with self._session_factory() as session:
            # 🔍 Fetch current experience
            result = await session.execute(
                select(ExperienceRecord).where(ExperienceRecord.id == exp_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                raise ValueError(f"Experience not found: {exp_id}")

            old_fitness = record.fitness_score

            if precomputed_fitness is not None:
                # 🎯 Caller already applied EMA — use precomputed value directly
                new_fitness = max(0.0, min(1.0, precomputed_fitness))
            else:
                # 📊 Compute new fitness using EMA
                new_fitness = (
                    self.EMA_ALPHA * delta.signal + (1.0 - self.EMA_ALPHA) * old_fitness
                )
                new_fitness = max(0.0, min(1.0, new_fitness))  # Clamp to [0, 1]

            # 📈 Update counters
            record.fitness_score = new_fitness
            record.times_used += 1
            if delta.was_helpful:
                record.times_helpful += 1
            record.updated_at = datetime.now(timezone.utc)

            await session.commit()

            logger.info(
                "📈 Updated fitness for %s: %.3f -> %.3f (signal=%.3f, helpful=%s)",
                exp_id,
                old_fitness,
                new_fitness,
                delta.signal,
                delta.was_helpful,
            )

    async def bulk_update_fitness(self, updates: list[FitnessUpdate]) -> None:
        """Batch update fitness scores for multiple experiences 📊.

        More efficient than calling ``update_fitness()`` repeatedly.

        Args:
            updates: List of fitness updates to apply.

        Raises:
            ValueError: If any experience is not found (all updates are rolled back).
        """
        if not updates:
            return

        async with self._session_factory() as session:
            for delta in updates:
                # 🔍 Fetch current experience
                result = await session.execute(
                    select(ExperienceRecord).where(
                        ExperienceRecord.id == delta.experience_id
                    )
                )
                record = result.scalar_one_or_none()

                if not record:
                    raise ValueError(f"Experience not found: {delta.experience_id}")

                # 📊 Compute new fitness using EMA
                old_fitness = record.fitness_score
                new_fitness = (
                    self.EMA_ALPHA * delta.signal + (1.0 - self.EMA_ALPHA) * old_fitness
                )
                new_fitness = max(0.0, min(1.0, new_fitness))

                # 📈 Update counters
                record.fitness_score = new_fitness
                record.times_used += 1
                if delta.was_helpful:
                    record.times_helpful += 1
                record.updated_at = datetime.now(timezone.utc)

            await session.commit()
            logger.info("📊 Bulk updated fitness for %d experiences", len(updates))

    async def prune(self, namespace: str, config: PruneConfig) -> int:
        """Remove low-fitness experiences based on pruning config 🗑️.

        Deletes experiences that meet ALL of the following criteria:
        - fitness_score < config.min_fitness
        - times_used >= config.min_uses (avoid deleting untested experiences)
        - namespace matches

        Args:
            namespace: Namespace to prune within.
            config: Pruning configuration from EvolutionProfile.

        Returns:
            Number of experiences deleted.

        Raises:
            ValueError: If namespace is empty.
        """
        if not namespace:
            raise ValueError("Prune requires a namespace")

        async with self._session_factory() as session:
            stmt = delete(ExperienceRecord).where(
                and_(
                    ExperienceRecord.namespace == namespace,
                    ExperienceRecord.fitness_score < config.min_fitness,
                    ExperienceRecord.times_used >= config.min_uses,
                )
            )

            result = await session.execute(stmt)
            await session.commit()

            deleted_count = result.rowcount
            logger.info(
                "🗑️ Pruned %d low-fitness experiences (namespace=%s, min_fitness=%.2f)",
                deleted_count,
                namespace,
                config.min_fitness,
            )
            return deleted_count

    async def apply_decay(self, namespace: str, decay_factor: float) -> int:
        """Apply time-based decay to all fitness scores 📉.

        Multiplies all fitness scores in the namespace by decay_factor.
        This gradually reduces the influence of old experiences over time.

        Args:
            namespace: Namespace to apply decay within.
            decay_factor: Multiplicative factor (e.g., 0.95 for 5% decay).

        Returns:
            Number of experiences updated.

        Raises:
            ValueError: If namespace is empty or decay_factor is invalid.
        """
        if not namespace:
            raise ValueError("Decay requires a namespace")
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}")

        async with self._session_factory() as session:
            stmt = (
                update(ExperienceRecord)
                .where(ExperienceRecord.namespace == namespace)
                .values(
                    fitness_score=ExperienceRecord.fitness_score * decay_factor,
                    updated_at=datetime.now(timezone.utc),
                )
            )

            result = await session.execute(stmt)
            await session.commit()

            updated_count = result.rowcount
            logger.info(
                "📉 Applied decay (%.3f) to %d experiences (namespace=%s)",
                decay_factor,
                updated_count,
                namespace,
            )
            return updated_count

    async def get_stats(self, namespace: str) -> dict:
        """Compute aggregate statistics for a namespace 📊.

        Returns a dictionary with:
        - total_count: Total number of experiences
        - avg_fitness: Average fitness score
        - category_breakdown: Dict[category, count]

        Args:
            namespace: Namespace to compute stats for.

        Returns:
            Statistics dictionary.

        Raises:
            ValueError: If namespace is empty.
        """
        if not namespace:
            raise ValueError("get_stats requires a namespace")

        async with self._session_factory() as session:
            # 🔢 Total count
            count_stmt = (
                select(func.count())
                .select_from(ExperienceRecord)
                .where(ExperienceRecord.namespace == namespace)
            )
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar()

            # 📊 Average fitness
            avg_stmt = select(func.avg(ExperienceRecord.fitness_score)).where(
                ExperienceRecord.namespace == namespace
            )
            avg_result = await session.execute(avg_stmt)
            avg_fitness = avg_result.scalar() or 0.0

            # 🏷️ Category breakdown
            category_stmt = (
                select(
                    ExperienceRecord.category,
                    func.count(ExperienceRecord.id),
                )
                .where(ExperienceRecord.namespace == namespace)
                .group_by(ExperienceRecord.category)
            )
            category_result = await session.execute(category_stmt)
            category_breakdown = {row[0]: row[1] for row in category_result.all()}

            stats = {
                "total_count": total_count,
                "avg_fitness": float(avg_fitness),
                "category_breakdown": category_breakdown,
            }

            logger.debug(
                "📊 Stats for namespace=%s: %d experiences, avg_fitness=%.3f",
                namespace,
                total_count,
                avg_fitness,
            )
            return stats

    async def get_by_id(self, exp_id: str) -> Experience | None:
        """Retrieve a single experience by ID 🔍.

        Args:
            exp_id: Experience ID.

        Returns:
            Experience object, or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(ExperienceRecord).where(ExperienceRecord.id == exp_id)
            )
            record = result.scalar_one_or_none()

            if record:
                return self._record_to_experience(record)
            return None

    async def deduplicate(
        self,
        namespace: str,
        new_insight: str,
        threshold: float = 0.9,
    ) -> bool:
        """Check if a similar experience already exists 🔍.

        Uses simple substring matching for v1. Returns True if a similar
        experience is found (insight should be rejected).

        Args:
            namespace: Namespace to search within.
            new_insight: Insight text to check for duplicates.
            threshold: Similarity threshold (0.9 = 90% match).
                For v1, this is interpreted as: a duplicate exists if
                new_insight is a substring of an existing insight, or vice versa.

        Returns:
            True if a duplicate exists (should reject), False otherwise.

        Raises:
            ValueError: If namespace is empty.
        """
        if not namespace:
            raise ValueError("Deduplicate requires a namespace")

        # 🔍 For v1, use simple substring matching
        # Future versions can use embedding-based similarity
        async with self._session_factory() as session:
            result = await session.execute(
                select(ExperienceRecord.insight).where(
                    ExperienceRecord.namespace == namespace
                )
            )
            existing_insights = [row[0] for row in result.all()]

            # 🔍 Check for substring matches
            new_lower = new_insight.lower().strip()
            for existing in existing_insights:
                existing_lower = existing.lower().strip()

                # Check if one is a substring of the other
                if new_lower in existing_lower or existing_lower in new_lower:
                    logger.info(
                        "🔍 Duplicate detected: '%s' matches existing insight",
                        new_insight[:50],
                    )
                    return True

            return False

    async def list_by_namespace(self, namespace: str) -> list[Experience]:
        """List all experiences in a namespace 📋.

        Returns every experience within the given namespace, ordered by
        fitness_score descending. No filtering is applied.

        Args:
            namespace: Namespace to list experiences for.

        Returns:
            List of all Experience objects in the namespace.

        Raises:
            ValueError: If namespace is empty.
        """
        if not namespace:
            raise ValueError("list_by_namespace requires a namespace")

        async with self._session_factory() as session:
            stmt = (
                select(ExperienceRecord)
                .where(ExperienceRecord.namespace == namespace)
                .order_by(ExperienceRecord.fitness_score.desc())
            )
            result = await session.execute(stmt)
            records = result.scalars().all()

            experiences = [self._record_to_experience(rec) for rec in records]
            logger.debug(
                "📋 Listed %d experiences in namespace=%s",
                len(experiences),
                namespace,
            )
            return experiences

    async def save(self, exp: Experience) -> str:
        """Alias for add() — insert a new experience 💾.

        Provided for API compatibility with components that use
        ``save`` instead of ``add`` (e.g., FeedbackHandler).

        Args:
            exp: Experience to insert.

        Returns:
            The experience ID.
        """
        return await self.add(exp)

    async def update(self, exp: Experience) -> None:
        """Update an existing experience record (full replacement) 🔄.

        Replaces all mutable fields of the experience record. The ``id``
        and ``namespace`` fields are used for lookup and cannot be changed.

        Args:
            exp: Experience with updated fields.

        Raises:
            ValueError: If experience not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(ExperienceRecord).where(ExperienceRecord.id == exp.id)
            )
            record = result.scalar_one_or_none()

            if not record:
                raise ValueError(f"Experience not found: {exp.id}")

            # 🔄 Update all mutable fields
            record.category = exp.category
            record.insight = exp.insight
            record.context_tags = exp.context_tags
            record.applicable_sub_items = exp.applicable_sub_items
            record.fitness_score = exp.fitness_score
            record.times_used = exp.times_used
            record.times_helpful = exp.times_helpful
            record.source = exp.source
            record.source_evaluation_id = exp.source_evaluation_id
            record.source_trajectory_step = exp.source_trajectory_step
            record.updated_at = datetime.now(timezone.utc)

            await session.commit()
            logger.info(
                "🔄 Updated experience %s (namespace=%s)",
                exp.id,
                exp.namespace,
            )

    async def delete(self, exp_id: str) -> None:
        """Delete a single experience by ID 🗑️.

        Args:
            exp_id: Experience ID to delete.

        Raises:
            ValueError: If experience not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(ExperienceRecord).where(ExperienceRecord.id == exp_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                raise ValueError(f"Experience not found: {exp_id}")

            await session.delete(record)
            await session.commit()
            logger.info("🗑️ Deleted experience %s", exp_id)

    # ========================================================================
    # 🔧 Internal Helpers
    # ========================================================================

    def _record_to_experience(self, record: ExperienceRecord) -> Experience:
        """Convert ORM record to Pydantic model 🔄.

        Args:
            record: SQLAlchemy ORM record.

        Returns:
            Pydantic Experience model.
        """
        return Experience(
            id=record.id,
            namespace=record.namespace,
            category=record.category,
            insight=record.insight,
            context_tags=record.context_tags or [],
            applicable_sub_items=record.applicable_sub_items or [],
            fitness_score=record.fitness_score,
            times_used=record.times_used,
            times_helpful=record.times_helpful,
            source=record.source,
            source_evaluation_id=record.source_evaluation_id,
            source_trajectory_step=record.source_trajectory_step,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


# ============================================================================
# 🛠️ Database Initialization
# ============================================================================


async def init_store_schema(engine: AsyncEngine) -> None:
    """Initialize database schema (create tables) 🛠️.

    Creates all tables defined in the ``Base`` metadata.
    Safe to call multiple times (no-op if tables already exist).

    Args:
        engine: SQLAlchemy async engine.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("🛠️ Experience store schema initialized")


async def drop_store_schema(engine: AsyncEngine) -> None:
    """Drop all tables in the experience store schema 🗑️.

    **WARNING:** This deletes all experience data permanently.
    Only use for testing or complete schema resets.

    Args:
        engine: SQLAlchemy async engine.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("🗑️ Experience store schema dropped")
