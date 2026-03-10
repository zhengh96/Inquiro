"""Base class for evolution learning mechanisms 🧬.

Every mechanism in the self-evolution system implements this ABC.
The ``UnifiedEvolutionProvider`` orchestrates all enabled mechanisms
through this uniform interface.

Lifecycle per round:
    on_round_start(round_num)
    → produce(snapshot, round_context) → list[Experience]
    → inject(round_context) → str | None
    → on_round_end(round_num, round_record, metrics)
    periodic_maintenance() — called between evaluations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from inquiro.evolution.types import Experience, MechanismType, TrajectorySnapshot

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["BaseMechanism"]


# ============================================================================
# 🧬 BaseMechanism ABC
# ============================================================================


class BaseMechanism(ABC):
    """Base class for a learning mechanism in the evolution system 🧬.

    Subclasses implement ``produce()`` to generate experiences from
    round data, and ``inject()`` to format guidance for prompt injection.
    Optional lifecycle hooks allow mechanisms to react to round
    boundaries and perform background maintenance.

    Attributes:
        enabled: Whether this mechanism is currently active.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        """Initialize BaseMechanism 🔧.

        Args:
            enabled: Whether this mechanism is active. Disabled mechanisms
                skip all lifecycle hooks and return empty results.
        """
        self.enabled = enabled

    @property
    @abstractmethod
    def mechanism_type(self) -> MechanismType:
        """Return the mechanism type identifier 🏷️.

        Returns:
            The MechanismType enum value for this mechanism.
        """
        ...

    @abstractmethod
    async def produce(
        self,
        snapshot: TrajectorySnapshot,
        round_context: dict[str, Any],
    ) -> list[Experience]:
        """Extract experiences from a completed round 📦.

        Args:
            snapshot: Structured execution data from the round.
            round_context: Additional context including round_num,
                gap_items, coverage, sub_item_id, etc.

        Returns:
            List of Experience objects to store. May be empty.
        """
        ...

    @abstractmethod
    def inject(
        self,
        round_context: dict[str, Any],
    ) -> str | None:
        """Format guidance text for prompt injection 💉.

        Args:
            round_context: Context including round_num, gap_items,
                sub_item_id, etc.

        Returns:
            Markdown text to inject into agent prompt, or None
            if this mechanism has nothing to contribute.
        """
        ...

    async def on_round_start(self, round_num: int) -> None:
        """Hook called before a round begins 🟢.

        Override to perform per-round setup (e.g., snapshot state).

        Args:
            round_num: The upcoming round number (1-based).
        """

    async def on_round_end(
        self,
        round_num: int,
        round_record: Any,
        metrics: Any,
    ) -> None:
        """Hook called after a round completes 🔴.

        Override to perform per-round cleanup, metric recording,
        or state updates.

        Args:
            round_num: The completed round number (1-based).
            round_record: Full round record from DiscoveryLoop.
            metrics: Round-level metrics (coverage, cost, etc.).
        """

    async def periodic_maintenance(self) -> None:
        """Background maintenance hook 🔧.

        Called between evaluations (not between rounds). Use for
        expensive operations like pruning, decay, or batch distillation.
        """
