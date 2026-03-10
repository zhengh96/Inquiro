"""Thompson Sampling bandit for MCP tool allocation 🎰.

Maintains per-(context_key, tool_name) Beta distributions that are updated
from live round outcomes.  Zero LLM cost — pure statistical optimisation.

Usage::

    bandit = ToolSelectionBandit(store, namespace="targetmaster")
    bandit.set_static_fallback({"pubmed_search": 0.5, "paper_search": 0.5})

    # During a round — inject priority table into the search prompt
    guidance = bandit.inject(round_context)

    # After the round — update with observed outcomes
    await bandit.on_round_end(round_num, round_record, metrics)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import (
    Experience,
    ExperienceQuery,
    MechanismType,
    ToolStats,
    TrajectorySnapshot,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ToolSelectionBandit"]


# ============================================================================
# 🏷️ Internal helpers
# ============================================================================

_BANDIT_STATE_CATEGORY = "bandit_state"
_BANDIT_STATE_SOURCE = "tool_selection_bandit"

# Stars used for the confidence column in the injected markdown table
_CONFIDENCE_STARS = {0: "☆", 1: "★", 2: "★★", 3: "★★★"}


def _round_bucket(round_number: int) -> str:
    """Map a round number to a coarse bucket label 🗂️.

    Args:
        round_number: 1-based round index.

    Returns:
        'initial' for the first round, 'focused' for subsequent rounds.
    """
    return "initial" if round_number == 1 else "focused"


def _context_key(sub_item_id: str, round_number: int) -> str:
    """Build the context key used to index bandit statistics 🔑.

    Args:
        sub_item_id: Opaque sub-item identifier from the upper layer.
        round_number: 1-based round index.

    Returns:
        Composite key string, e.g. ``"efficacy:initial"``.
    """
    return f"{sub_item_id}:{_round_bucket(round_number)}"


def _star_rating(total_observations: int) -> str:
    """Convert total_observations to a star confidence string ⭐.

    Args:
        total_observations: Number of outcomes recorded so far.

    Returns:
        One of '☆', '★', '★★', '★★★' based on observation count.
    """
    if total_observations < 3:
        return _CONFIDENCE_STARS[0]
    if total_observations < 10:
        return _CONFIDENCE_STARS[1]
    if total_observations < 25:
        return _CONFIDENCE_STARS[2]
    return _CONFIDENCE_STARS[3]


def _priority_label(share: float) -> str:
    """Convert an allocation share to a human-readable priority label 🏷️.

    Args:
        share: Allocation fraction in [0, 1].

    Returns:
        'HIGH', 'MEDIUM', or 'LOW'.
    """
    if share >= 0.40:
        return "HIGH"
    if share >= 0.25:
        return "MEDIUM"
    return "LOW"


# ============================================================================
# 🎰 ToolSelectionBandit
# ============================================================================


class ToolSelectionBandit(BaseMechanism):
    """Contextual Thompson Sampling for MCP tool allocation 🎰.

    Maintains Beta(alpha, beta) distributions per (context_key, tool_name)
    pair.  Zero LLM cost — pure statistical optimisation.

    Context keys are coarse: ``f"{sub_item_id}:{'initial'|'focused'}"``.
    This gives the bandit enough signal to learn per-sub-item patterns
    without over-fragmenting the observation space.

    Attributes:
        enabled: Whether this mechanism is active (inherited).
    """

    def __init__(
        self,
        store: ExperienceStore,
        namespace: str,
        exploration_bonus: float = 0.10,
        confidence_threshold: int = 5,
        decay_factor: float = 0.98,
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize ToolSelectionBandit 🔧.

        Args:
            store: ExperienceStore for persisting bandit state across sessions.
            namespace: Data-isolation namespace (e.g., 'targetmaster').
            exploration_bonus: Minimum aggregate share reserved for exploration.
                Each tool is guaranteed at least ``exploration_bonus / n_tools``
                share, preventing permanent tool exclusion.
            confidence_threshold: Minimum total_observations before the bandit
                trusts its own estimates.  Below this threshold the static
                fallback is used instead.
            decay_factor: Multiplicative decay applied to (alpha-1) and
                (beta-1) before each update to forget stale observations.
            enabled: Whether this mechanism participates in the pipeline.
        """
        super().__init__(enabled=enabled)
        self._store = store
        self._namespace = namespace
        self._exploration_bonus = exploration_bonus
        self._confidence_threshold = confidence_threshold
        self._decay_factor = decay_factor
        # In-memory state: {context_key: {tool_name: ToolStats}}
        self._stats: dict[str, dict[str, ToolStats]] = {}
        self._static_fallback: dict[str, float] | None = None

    # -------------------------------------------------------------------------
    # 🏷️ Identity
    # -------------------------------------------------------------------------

    @property
    def mechanism_type(self) -> MechanismType:
        """Return the mechanism type identifier 🏷️.

        Returns:
            MechanismType.TOOL_SELECTION
        """
        return MechanismType.TOOL_SELECTION

    # -------------------------------------------------------------------------
    # 🎯 Core Bandit Logic
    # -------------------------------------------------------------------------

    def select(
        self,
        available_tools: list[str],
        sub_item_id: str,
        round_number: int,
    ) -> dict[str, float]:
        """Select tool allocation using Thompson Sampling 🎯.

        For each tool, a theta value is sampled from its Beta(alpha, beta)
        distribution.  Samples are normalised to allocations summing to 1.0,
        then an exploration floor is enforced so no tool drops below
        ``exploration_bonus / n_tools``.

        Falls back to the static allocation (if configured) when confidence
        is insufficient for any tool.

        Args:
            available_tools: List of tool names to allocate resources to.
            sub_item_id: Opaque sub-item identifier from the upper layer.
            round_number: 1-based round index.

        Returns:
            Dict mapping tool_name → allocation_percentage in [0, 1].
            Values sum to 1.0.
        """
        if not available_tools:
            return {}

        ctx_key = _context_key(sub_item_id, round_number)
        ctx_stats = self._stats.get(ctx_key, {})

        # 🧊 Cold-start / low-confidence check
        if self._static_fallback is not None:
            for tool in available_tools:
                tool_stats = ctx_stats.get(tool, ToolStats())
                if tool_stats.total_observations < self._confidence_threshold:
                    logger.debug(
                        "🧊 Low confidence for tool '%s' in ctx '%s' "
                        "(obs=%d < threshold=%d) — using static fallback",
                        tool,
                        ctx_key,
                        tool_stats.total_observations,
                        self._confidence_threshold,
                    )
                    return self._normalised_static(available_tools)

        # 🎲 Thompson Sampling: sample theta from Beta(alpha, beta)
        thetas: dict[str, float] = {}
        for tool in available_tools:
            stats = ctx_stats.get(tool, ToolStats())
            thetas[tool] = random.betavariate(stats.alpha, stats.beta)

        # 📊 Normalise to allocations
        total_theta = sum(thetas.values()) or 1.0
        allocations = {t: v / total_theta for t, v in thetas.items()}

        # 🔍 Enforce exploration floor
        n_tools = len(available_tools)
        min_share = self._exploration_bonus / n_tools
        allocations = self._enforce_exploration_floor(allocations, min_share)

        logger.debug(
            "🎯 Thompson Sampling allocation for ctx '%s': %s",
            ctx_key,
            {t: f"{v:.2%}" for t, v in allocations.items()},
        )
        return allocations

    def update(
        self,
        sub_item_id: str,
        round_number: int,
        tool_outcomes: dict[str, bool],
    ) -> None:
        """Update bandit statistics from round outcomes 📊.

        Applies temporal decay to existing alpha/beta values before
        recording the new outcome, so older evidence is down-weighted.

        Args:
            sub_item_id: Opaque sub-item identifier from the upper layer.
            round_number: 1-based round index.
            tool_outcomes: Mapping of tool_name → success flag.
                True  = tool produced evidence that survived EvidencePipeline.
                False = tool produced no usable evidence this round.
        """
        if not tool_outcomes:
            return

        ctx_key = _context_key(sub_item_id, round_number)
        if ctx_key not in self._stats:
            self._stats[ctx_key] = {}

        ctx_stats = self._stats[ctx_key]

        for tool_name, success in tool_outcomes.items():
            if tool_name not in ctx_stats:
                ctx_stats[tool_name] = ToolStats()

            stats = ctx_stats[tool_name]

            # ⏳ Temporal decay: shrink observations toward the prior (alpha=1, beta=1)
            # so stale signal is down-weighted.
            decayed_alpha_excess = (stats.alpha - 1.0) * self._decay_factor
            decayed_beta_excess = (stats.beta - 1.0) * self._decay_factor
            stats.alpha = 1.0 + decayed_alpha_excess
            stats.beta = 1.0 + decayed_beta_excess

            # ✅ Record new observation
            if success:
                stats.alpha += 1.0
            else:
                stats.beta += 1.0
            stats.total_observations += 1

            logger.debug(
                "📊 Updated bandit stats: ctx='%s' tool='%s' "
                "success=%s alpha=%.2f beta=%.2f obs=%d",
                ctx_key,
                tool_name,
                success,
                stats.alpha,
                stats.beta,
                stats.total_observations,
            )

    def set_static_fallback(self, allocations: dict[str, float]) -> None:
        """Set fallback allocations for cold-start scenarios 🧊.

        When any tool lacks sufficient observations, the bandit falls back
        to these allocations instead of trusting unreliable Beta samples.

        Args:
            allocations: Tool-name → target-share mapping (need not sum to 1;
                they are normalised internally when used).
        """
        self._static_fallback = dict(allocations)
        logger.debug(
            "🧊 Static fallback configured with %d tools: %s",
            len(allocations),
            list(allocations.keys()),
        )

    # -------------------------------------------------------------------------
    # 📦 BaseMechanism: produce
    # -------------------------------------------------------------------------

    async def produce(
        self,
        snapshot: TrajectorySnapshot,
        round_context: dict[str, Any],
    ) -> list[Experience]:
        """Serialize bandit state as an Experience for persistence 📦.

        The serialised state is stored with ``category="bandit_state"`` so
        it can be retrieved and reloaded in a future session.

        Args:
            snapshot: Trajectory snapshot from the completed round.
            round_context: Round context dict (round_num, sub_item_id, etc.).

        Returns:
            A single-element list containing the serialised bandit state,
            or an empty list if there is no state to persist.
        """
        if not self.enabled or not self._stats:
            return []

        serialised = self._serialise_stats()
        exp = Experience(
            namespace=self._namespace,
            category=_BANDIT_STATE_CATEGORY,
            insight=serialised,
            context_tags=["mechanism:tool_selection"],
            applicable_sub_items=["*"],
            fitness_score=1.0,  # Always retain bandit state
            source=_BANDIT_STATE_SOURCE,
            source_evaluation_id=snapshot.evaluation_id,
            mechanism_type=MechanismType.TOOL_SELECTION,
        )
        logger.info(
            "📦 Produced bandit state experience (%d ctx keys)",
            len(self._stats),
        )
        return [exp]

    # -------------------------------------------------------------------------
    # 💉 BaseMechanism: inject
    # -------------------------------------------------------------------------

    def inject(self, round_context: dict[str, Any]) -> str | None:
        """Format tool priority table for prompt injection 💉.

        Runs Thompson Sampling to determine current allocations and renders
        them as a markdown table with priority labels and confidence stars.

        Args:
            round_context: Must contain 'available_tools' (list[str]),
                'sub_item_id' (str), and 'round_num' (int).

        Returns:
            Markdown table string, or None if disabled or no tools available.
        """
        if not self.enabled:
            return None

        available_tools: list[str] = round_context.get("available_tools", [])
        sub_item_id: str = round_context.get("sub_item_id", "")
        round_num: int = round_context.get("round_num", 1)

        if not available_tools:
            return None

        allocations = self.select(available_tools, sub_item_id, round_num)
        ctx_key = _context_key(sub_item_id, round_num)
        ctx_stats = self._stats.get(ctx_key, {})

        # 🔽 Sort tools by allocation descending for readability
        sorted_tools = sorted(
            available_tools,
            key=lambda t: allocations.get(t, 0.0),
            reverse=True,
        )

        lines = [
            "## TOOL SELECTION GUIDANCE",
            "| Tool | Priority | Confidence |",
            "|------|----------|------------|",
        ]
        for tool in sorted_tools:
            share = allocations.get(tool, 0.0)
            obs = ctx_stats.get(tool, ToolStats()).total_observations
            label = _priority_label(share)
            stars = _star_rating(obs)
            lines.append(f"| {tool} | {label} ({share:.0%}) | {stars} |")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 🔴 BaseMechanism: on_round_end
    # -------------------------------------------------------------------------

    async def on_round_end(
        self,
        round_num: int,
        round_record: Any,
        metrics: Any,
    ) -> None:
        """Extract tool outcomes and update bandit statistics 📊.

        Parses the round_record for tool call results, builds the
        tool_outcomes dict, and updates the Beta distributions.

        Args:
            round_num: The completed round number (1-based).
            round_record: Full round record from DiscoveryLoop.  Expected to
                expose a ``tool_calls`` attribute that is a list of objects
                with ``tool_name`` and ``success`` fields.
            metrics: Round-level metrics (coverage, cost, etc.).
        """
        if not self.enabled:
            return

        # 🔍 Extract sub_item_id from round_record if available
        sub_item_id: str = getattr(round_record, "sub_item_id", "") or ""

        # 🔧 Parse tool calls from the round record
        tool_calls = getattr(round_record, "tool_calls", None) or []
        if not tool_calls:
            logger.debug(
                "📊 on_round_end round=%d: no tool_calls found in round_record",
                round_num,
            )
            return

        # 🏗️ Aggregate per-tool success: a tool is successful if it produced
        # at least one successful call during the round.
        tool_outcomes: dict[str, bool] = {}
        for call in tool_calls:
            tool_name: str = getattr(call, "tool_name", "") or ""
            success: bool = bool(getattr(call, "success", True))
            if not tool_name:
                continue
            # OR semantics: if any call succeeded, the tool succeeded
            tool_outcomes[tool_name] = tool_outcomes.get(tool_name, False) or success

        if not tool_outcomes:
            return

        self.update(sub_item_id, round_num, tool_outcomes)
        await self._save_to_store()

    # -------------------------------------------------------------------------
    # 💾 Persistence helpers
    # -------------------------------------------------------------------------

    async def _load_from_store(self) -> None:
        """Load persisted bandit state from ExperienceStore 📂.

        Queries for the most recent 'bandit_state' experience and
        deserialises it into self._stats.
        """
        query = ExperienceQuery(
            namespace=self._namespace,
            category=_BANDIT_STATE_CATEGORY,
            min_fitness=0.0,
            max_results=1,
        )
        try:
            results = await self._store.query(query)
        except Exception:  # noqa: BLE001 — store errors must not crash the bandit
            logger.warning(
                "⚠️ Failed to load bandit state from store — starting fresh",
                exc_info=True,
            )
            return

        if not results:
            logger.debug("📂 No persisted bandit state found — starting fresh")
            return

        try:
            raw: dict[str, Any] = json.loads(results[0].insight)
            self._stats = {
                ctx_key: {
                    tool_name: ToolStats.model_validate(stats_dict)
                    for tool_name, stats_dict in tool_map.items()
                }
                for ctx_key, tool_map in raw.items()
            }
            total_entries = sum(len(v) for v in self._stats.values())
            logger.info(
                "📂 Loaded bandit state: %d ctx keys, %d tool entries",
                len(self._stats),
                total_entries,
            )
        except Exception:  # noqa: BLE001 — corrupted state must not crash startup
            logger.warning(
                "⚠️ Corrupted bandit state in store — discarding",
                exc_info=True,
            )
            self._stats = {}

    async def _save_to_store(self) -> None:
        """Persist bandit state to ExperienceStore 💾.

        Serialises self._stats as JSON and stores it as a new Experience
        with category='bandit_state'.  Previous states accumulate in the
        store; only the most recently ranked one is loaded on startup.
        """
        if not self._stats:
            return

        serialised = self._serialise_stats()
        exp = Experience(
            namespace=self._namespace,
            category=_BANDIT_STATE_CATEGORY,
            insight=serialised,
            context_tags=["mechanism:tool_selection"],
            applicable_sub_items=["*"],
            fitness_score=1.0,
            source=_BANDIT_STATE_SOURCE,
            mechanism_type=MechanismType.TOOL_SELECTION,
        )
        try:
            await self._store.add(exp)
            logger.debug(
                "💾 Persisted bandit state (%d ctx keys)",
                len(self._stats),
            )
        except Exception:  # noqa: BLE001 — persistence failure is non-fatal
            logger.warning(
                "⚠️ Failed to persist bandit state — in-memory state preserved",
                exc_info=True,
            )

    # -------------------------------------------------------------------------
    # 🔧 Private helpers
    # -------------------------------------------------------------------------

    def _serialise_stats(self) -> str:
        """Serialise self._stats to a JSON string 🔧.

        Returns:
            JSON string representing the full bandit state.
        """
        raw: dict[str, dict[str, dict[str, Any]]] = {
            ctx_key: {
                tool_name: stats.model_dump()
                for tool_name, stats in tool_map.items()
            }
            for ctx_key, tool_map in self._stats.items()
        }
        return json.dumps(raw)

    def _normalised_static(self, available_tools: list[str]) -> dict[str, float]:
        """Return normalised static fallback allocations 🧊.

        If a tool has no entry in the static fallback, it receives an equal
        share of the remaining budget.

        Args:
            available_tools: Tools that need allocations.

        Returns:
            Dict mapping tool_name → allocation in [0, 1] summing to 1.0.
        """
        if self._static_fallback is None:
            # Uniform distribution as last resort
            share = 1.0 / len(available_tools)
            return {t: share for t in available_tools}

        raw = {t: self._static_fallback.get(t, 0.0) for t in available_tools}
        total = sum(raw.values())
        if total <= 0:
            share = 1.0 / len(available_tools)
            return {t: share for t in available_tools}

        return {t: v / total for t, v in raw.items()}

    @staticmethod
    def _enforce_exploration_floor(
        allocations: dict[str, float],
        min_share: float,
    ) -> dict[str, float]:
        """Enforce a minimum allocation per tool to preserve exploration 🔍.

        Tools below ``min_share`` are raised to ``min_share`` and the excess
        is proportionally subtracted from tools that are above the floor.

        Args:
            allocations: Current allocations (must sum to 1.0).
            min_share: Minimum fraction each tool must receive.

        Returns:
            Adjusted allocations that still sum to 1.0.
        """
        # First pass: identify deficient tools and compute deficit
        deficit = 0.0
        result = dict(allocations)
        for tool, share in result.items():
            if share < min_share:
                deficit += min_share - share
                result[tool] = min_share

        if deficit <= 0.0:
            return result

        # Second pass: proportionally reduce excess from tools above the floor
        excess_tools = {t: v for t, v in result.items() if v > min_share}
        total_excess = sum(excess_tools.values())
        if total_excess <= 0.0:
            # Edge case: all tools are at exactly min_share, re-normalise
            n = len(result)
            return {t: 1.0 / n for t in result}

        scale = (total_excess - deficit) / total_excess
        for tool in excess_tools:
            result[tool] = result[tool] * scale

        return result
