"""Evolution maintenance scheduler — automated decay, prune, and conflict resolution 🔄.

Provides a background ``asyncio.Task``-based scheduler that periodically
runs maintenance operations on the experience store:

1. **Decay** — Apply multiplicative fitness decay to prevent stale experiences
   from dominating the pool.
2. **Prune** — Remove low-fitness experiences that have been tested enough
   times to confirm they are unhelpful.
3. **Conflict resolution** — Detect and resolve contradictory insights within
   the same category by keeping the higher-fitness experience.

The scheduler is fault-tolerant: individual step failures are logged and
captured in ``MaintenanceResult.errors`` without stopping the cycle.

Example usage::

    from inquiro.evolution.scheduler import EvolutionScheduler, SchedulerConfig

    config = SchedulerConfig(namespace="targetmaster")
    scheduler = EvolutionScheduler(config)

    await scheduler.start()   # 🚀 Begins background loop
    # ... application runs ...
    await scheduler.stop()    # 🛑 Graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from inquiro.evolution.ranker import ExperienceRanker
from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import PruneConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 📋 Configuration and Result Models
# ============================================================================


class SchedulerConfig(BaseModel):
    """Configuration for the evolution maintenance scheduler ⚙️.

    All thresholds and intervals are injected by the upper layer.
    Inquiro does not define domain-specific defaults — only
    infrastructure-level defaults (e.g., interval timing).

    Args:
        namespace: Namespace for data isolation (REQUIRED).
        decay_factor: Multiplicative decay factor per cycle.
        decay_interval_seconds: Seconds between maintenance cycles.
        prune_min_fitness: Minimum fitness score to keep.
        prune_min_uses: Minimum uses before eligible for pruning.
        enable_conflict_resolution: Whether to run conflict detection.
    """

    namespace: str = Field(
        description="Namespace for experience data isolation (REQUIRED)",
    )
    decay_factor: float = Field(
        default=0.95,
        description="Multiplicative decay factor per cycle (e.g., 0.95 = 5% decay)",
        gt=0.0,
        le=1.0,
    )
    decay_interval_seconds: int = Field(
        default=604800,
        description="Seconds between maintenance cycles (default: 7 days)",
        gt=0,
    )
    prune_min_fitness: float = Field(
        default=0.2,
        description="Minimum fitness score to keep (below = prune)",
        ge=0.0,
        le=1.0,
    )
    prune_min_uses: int = Field(
        default=5,
        description="Minimum times_used before eligible for pruning",
        ge=0,
    )
    enable_conflict_resolution: bool = Field(
        default=True,
        description="Whether to run conflict detection and resolution",
    )


class MaintenanceResult(BaseModel):
    """Result of a single maintenance cycle 📊.

    Captures counts from each maintenance step and any errors
    encountered during the cycle.
    """

    decayed_count: int = Field(
        default=0,
        description="Number of experiences that had fitness decayed",
    )
    pruned_count: int = Field(
        default=0,
        description="Number of low-fitness experiences pruned",
    )
    conflicts_resolved: int = Field(
        default=0,
        description="Number of conflicting experience pairs resolved",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages from failed steps",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when maintenance cycle completed",
    )


# ============================================================================
# 🔄 Evolution Scheduler
# ============================================================================


class EvolutionScheduler:
    """Background evolution maintenance scheduler 🔄.

    Runs periodic maintenance cycles using ``asyncio.Task`` to manage
    the experience store health. Each cycle executes:

    1. Fitness decay (always)
    2. Pruning (always)
    3. Conflict resolution (configurable)

    The scheduler is fault-tolerant: individual step failures are
    captured in ``MaintenanceResult.errors`` and do not halt the cycle.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        store: ExperienceStore,
        ranker: ExperienceRanker | None = None,
    ) -> None:
        """Initialize the evolution scheduler 🔧.

        Args:
            config: Scheduler configuration with thresholds and intervals.
            store: ExperienceStore for persistence operations.
            ranker: ExperienceRanker for conflict operations. If None,
                a new ranker is created wrapping the provided store.
        """
        self._config = config
        self._store = store
        self._ranker = ranker or ExperienceRanker(store)
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_result: MaintenanceResult | None = None
        logger.info(
            "🔄 EvolutionScheduler initialized (namespace=%s, interval=%ds)",
            config.namespace,
            config.decay_interval_seconds,
        )

    @property
    def is_running(self) -> bool:
        """Check if the scheduler background loop is active 🔍.

        Returns:
            True if the scheduler is currently running.
        """
        return self._running

    @property
    def last_result(self) -> MaintenanceResult | None:
        """Get the result of the last completed maintenance cycle 📊.

        Returns:
            The last MaintenanceResult, or None if no cycle has completed.
        """
        return self._last_result

    async def start(self) -> None:
        """Start the background maintenance loop 🚀.

        Creates an ``asyncio.Task`` that runs maintenance cycles at
        the configured interval. Safe to call multiple times — if
        already running, logs a warning and returns.
        """
        if self._running:
            logger.warning("⚠️ Scheduler already running, ignoring start()")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "🚀 Scheduler started (namespace=%s, interval=%ds)",
            self._config.namespace,
            self._config.decay_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the background maintenance loop gracefully 🛑.

        Cancels the background task and waits for it to finish.
        Safe to call multiple times — if not running, returns immediately.
        """
        if not self._running:
            logger.debug("🛑 Scheduler not running, ignoring stop()")
            return

        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # ✅ Expected when task is cancelled
            self._task = None

        logger.info("🛑 Scheduler stopped (namespace=%s)", self._config.namespace)

    async def run_maintenance(self) -> MaintenanceResult:
        """Execute a single maintenance cycle 🔧.

        Runs all maintenance steps in sequence:
        1. Fitness decay
        2. Pruning
        3. Conflict resolution (if enabled)

        Each step is wrapped in error handling — failures are logged
        and captured in ``MaintenanceResult.errors`` without stopping
        subsequent steps.

        Returns:
            MaintenanceResult with counts and any errors.
        """
        logger.info(
            "🔧 Starting maintenance cycle (namespace=%s)",
            self._config.namespace,
        )
        result = MaintenanceResult()

        # 📉 Step 1: Apply fitness decay
        result.decayed_count = await self._run_decay(result)

        # 🗑️ Step 2: Prune low-fitness experiences
        result.pruned_count = await self._run_prune(result)

        # ⚔️ Step 3: Conflict resolution (optional)
        if self._config.enable_conflict_resolution:
            result.conflicts_resolved = await self._run_conflict_resolution(
                result,
            )

        result.timestamp = datetime.now(timezone.utc)
        self._last_result = result

        logger.info(
            "✅ Maintenance cycle complete: decayed=%d, pruned=%d, "
            "conflicts_resolved=%d, errors=%d",
            result.decayed_count,
            result.pruned_count,
            result.conflicts_resolved,
            len(result.errors),
        )
        return result

    async def _run_decay(self, result: MaintenanceResult) -> int:
        """Apply fitness decay to all experiences in the namespace 📉.

        Args:
            result: MaintenanceResult to append errors to if decay fails.

        Returns:
            Number of experiences that had fitness decayed.
        """
        try:
            decayed = await self._store.apply_decay(
                self._config.namespace,
                self._config.decay_factor,
            )
            logger.info(
                "📉 Decay applied to %d experiences (factor=%.3f)",
                decayed,
                self._config.decay_factor,
            )
            return decayed
        except Exception as e:
            error_msg = f"Decay failed: {e}"
            logger.error("❌ %s", error_msg)
            result.errors.append(error_msg)
            return 0

    async def _run_prune(self, result: MaintenanceResult) -> int:
        """Prune low-fitness experiences from the namespace 🗑️.

        Args:
            result: MaintenanceResult to append errors to if pruning fails.

        Returns:
            Number of experiences pruned.
        """
        try:
            prune_config = PruneConfig(
                min_fitness=self._config.prune_min_fitness,
                min_uses=self._config.prune_min_uses,
                decay_factor=self._config.decay_factor,
                decay_interval_days=max(
                    1, self._config.decay_interval_seconds // 86400
                ),
            )
            pruned = await self._store.prune(self._config.namespace, prune_config)
            logger.info(
                "🗑️ Pruned %d experiences (min_fitness=%.3f, min_uses=%d)",
                pruned,
                self._config.prune_min_fitness,
                self._config.prune_min_uses,
            )
            return pruned
        except Exception as e:
            error_msg = f"Prune failed: {e}"
            logger.error("❌ %s", error_msg)
            result.errors.append(error_msg)
            return 0

    async def _run_conflict_resolution(self, result: MaintenanceResult) -> int:
        """Detect and resolve conflicting experiences ⚔️.

        Args:
            result: MaintenanceResult to append errors to if resolution fails.

        Returns:
            Number of conflicts resolved.
        """
        try:
            conflicts = await self._ranker.detect_conflicts(
                self._config.namespace,
            )
            if not conflicts:
                logger.debug("✅ No conflicts detected")
                return 0

            resolved = await self._ranker.resolve_conflicts(conflicts)
            logger.info(
                "⚔️ Resolved %d/%d conflicts",
                resolved,
                len(conflicts),
            )
            return resolved
        except Exception as e:
            error_msg = f"Conflict resolution failed: {e}"
            logger.error("❌ %s", error_msg)
            result.errors.append(error_msg)
            return 0

    async def _loop(self) -> None:
        """Background loop that runs maintenance at configured intervals 🔄.

        Sleeps for ``decay_interval_seconds`` between cycles. Catches
        all exceptions to keep the loop alive — only ``CancelledError``
        stops the loop (triggered by ``stop()``).
        """
        logger.info(
            "🔄 Background loop started (interval=%ds)",
            self._config.decay_interval_seconds,
        )

        while self._running:
            try:
                await asyncio.sleep(self._config.decay_interval_seconds)

                if not self._running:
                    break

                await self.run_maintenance()

            except asyncio.CancelledError:
                logger.info("🛑 Background loop cancelled")
                raise
            except Exception as e:
                # 🛡️ Keep the loop alive on unexpected errors
                logger.error("❌ Unexpected error in maintenance loop: %s", e)
