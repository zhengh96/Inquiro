"""Experience pruning, decay, and conflict management 🗑️.

The ExperienceRanker manages experience lifecycle: pruning low-fitness
experiences, applying decay to reduce staleness, detecting contradictions,
and merging similar insights.

This module delegates storage operations to ExperienceStore and implements
domain-agnostic ranking algorithms. Conflict detection and similarity
heuristics are intentionally simple (v1 implementation).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from inquiro.evolution.types import PruneConfig

if TYPE_CHECKING:
    from inquiro.evolution.store import ExperienceStore

# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ExperienceRanker"]


logger = logging.getLogger(__name__)


# ============================================================================
# 🗑️ ExperienceRanker
# ============================================================================


class ExperienceRanker:
    """Experience lifecycle manager — pruning, decay, conflicts, merging 🗑️.

    Manages the health of the experience store by:
    1. Pruning low-fitness experiences that have been tested enough times
    2. Applying decay to reduce staleness over time
    3. Detecting contradictory insights within the same category
    4. Merging highly similar experiences to reduce redundancy

    All operations are domain-agnostic — thresholds and policies come
    from the upper layer's EvolutionProfile.
    """

    def __init__(self, store: ExperienceStore) -> None:
        """Initialize ExperienceRanker 🎯.

        Args:
            store: ExperienceStore for reading and updating experiences.
        """
        self._store = store

    async def prune(self, namespace: str, config: PruneConfig) -> int:
        """Prune low-fitness experiences 🗑️.

        Delegates to store.prune() with the provided PruneConfig.
        Removes experiences that:
        - Have fitness_score < config.min_fitness
        - Have been used at least config.min_uses times

        Args:
            namespace: Namespace to prune (data isolation).
            config: Pruning configuration (min_fitness, min_uses).

        Returns:
            Number of experiences pruned.
        """
        logger.info(
            "Pruning namespace '%s' with min_fitness=%.3f, min_uses=%d 🗑️",
            namespace,
            config.min_fitness,
            config.min_uses,
        )

        pruned_count = await self._store.prune(namespace, config)

        logger.info(
            "Pruned %d experiences from namespace '%s' ✅", pruned_count, namespace
        )
        return pruned_count

    async def apply_decay(self, namespace: str, decay_factor: float) -> int:
        """Apply multiplicative decay to all fitness scores 📉.

        Delegates to store.apply_decay() with the provided decay_factor.
        For each experience: new_fitness = old_fitness * decay_factor

        Args:
            namespace: Namespace to apply decay to.
            decay_factor: Multiplicative decay factor (e.g., 0.95).

        Returns:
            Number of experiences decayed.

        Raises:
            ValueError: If decay_factor is not in (0.0, 1.0].
        """
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}")

        logger.info(
            "Applying decay to namespace '%s' with factor=%.3f 📉",
            namespace,
            decay_factor,
        )

        decayed_count = await self._store.apply_decay(namespace, decay_factor)

        logger.info(
            "Decayed %d experiences in namespace '%s' ✅",
            decayed_count,
            namespace,
        )
        return decayed_count

    async def detect_conflicts(self, namespace: str) -> list[tuple[str, str]]:
        """Detect contradictory insights within same category 🔍.

        V1 implementation: simple substring matching.
        Looks for pairs where one insight says "use X" or "prefer X"
        and another says "avoid X" or "do not use X" within the same
        category.

        Args:
            namespace: Namespace to search for conflicts.

        Returns:
            List of (exp_id_1, exp_id_2) tuples representing conflicts.
        """
        logger.info("Detecting conflicts in namespace '%s' 🔍", namespace)

        # 🔍 Get all experiences in namespace
        all_experiences = await self._store.list_by_namespace(namespace)

        # 📦 Group by category
        by_category: dict[str, list] = {}
        for exp in all_experiences:
            by_category.setdefault(exp.category, []).append(exp)

        conflicts: list[tuple[str, str]] = []

        # 🔎 Check each category for contradictions
        for category, experiences in by_category.items():
            if len(experiences) < 2:
                continue

            logger.debug(
                "Checking category '%s' with %d experiences for conflicts 🔍",
                category,
                len(experiences),
            )

            # 🧪 Simple conflict detection: find "use X" vs "avoid X" patterns
            for i, exp_a in enumerate(experiences):
                for exp_b in experiences[i + 1 :]:
                    if self._are_conflicting(exp_a.insight, exp_b.insight):
                        conflicts.append((exp_a.id, exp_b.id))
                        logger.info(
                            "Conflict detected: %s vs %s in category '%s' ⚠️",
                            exp_a.id[:8],
                            exp_b.id[:8],
                            category,
                        )

        logger.info(
            "Detected %d conflicts in namespace '%s' ✅", len(conflicts), namespace
        )
        return conflicts

    def _are_conflicting(self, insight_a: str, insight_b: str) -> bool:
        """Check if two insights are contradictory 🔍.

        V1 heuristic: look for "use/prefer X" in one and "avoid/don't use X"
        in the other.

        Args:
            insight_a: First insight text.
            insight_b: Second insight text.

        Returns:
            True if insights appear contradictory, False otherwise.
        """
        # 🔤 Normalize to lowercase for matching
        text_a = insight_a.lower()
        text_b = insight_b.lower()

        # 🎯 Extract subjects (simple approach: nouns after action verbs)
        positive_patterns = [
            r"(?:use|prefer|prioritize|favor|choose)\s+(\w+)",
            r"(\w+)\s+(?:is|are)\s+(?:effective|good|recommended)",
        ]
        negative_patterns = [
            r"(?:avoid|skip|don't use|do not use|ignore)\s+(\w+)",
            r"(\w+)\s+(?:is|are)\s+(?:ineffective|bad|not recommended)",
        ]

        # 📝 Extract subjects from each insight
        subjects_a_positive = set()
        subjects_a_negative = set()
        subjects_b_positive = set()
        subjects_b_negative = set()

        for pattern in positive_patterns:
            subjects_a_positive.update(re.findall(pattern, text_a))
            subjects_b_positive.update(re.findall(pattern, text_b))

        for pattern in negative_patterns:
            subjects_a_negative.update(re.findall(pattern, text_a))
            subjects_b_negative.update(re.findall(pattern, text_b))

        # 🔍 Check for contradictions:
        # - A says "use X" and B says "avoid X"
        # - A says "avoid X" and B says "use X"
        conflict_1 = bool(subjects_a_positive & subjects_b_negative)
        conflict_2 = bool(subjects_a_negative & subjects_b_positive)

        return conflict_1 or conflict_2

    async def resolve_conflicts(
        self,
        conflicts: list[tuple[str, str]],
    ) -> int:
        """Resolve conflicts by keeping higher fitness experience 🔧.

        For each conflict pair:
        1. Compare fitness_score
        2. Keep the higher-fitness experience
        3. Mark the lower-fitness experience for pruning (set fitness to 0.0)

        Args:
            conflicts: List of (exp_id_1, exp_id_2) conflict pairs.

        Returns:
            Number of conflicts resolved (experiences marked for pruning).
        """
        if not conflicts:
            logger.debug("No conflicts to resolve, skipping ✅")
            return 0

        logger.info("Resolving %d conflicts 🔧", len(conflicts))

        resolved_count = 0

        for exp_id_1, exp_id_2 in conflicts:
            # 🔍 Get both experiences
            exp_1 = await self._store.get_by_id(exp_id_1)
            exp_2 = await self._store.get_by_id(exp_id_2)

            if exp_1 is None or exp_2 is None:
                logger.warning(
                    "One or both experiences not found: %s, %s ⚠️",
                    exp_id_1[:8],
                    exp_id_2[:8],
                )
                continue

            # 🏆 Determine winner (higher fitness)
            if exp_1.fitness_score >= exp_2.fitness_score:
                winner_id = exp_id_1
                loser_id = exp_id_2
                loser = exp_2
            else:
                winner_id = exp_id_2
                loser_id = exp_id_1
                loser = exp_1

            logger.info(
                "Conflict resolved: keeping %s (fitness=%.3f),"
                " marking %s (fitness=%.3f) for pruning 🗑️",
                winner_id[:8],
                max(exp_1.fitness_score, exp_2.fitness_score),
                loser_id[:8],
                loser.fitness_score,
            )

            # 🗑️ Mark loser for pruning by setting fitness to 0.0
            from inquiro.evolution.types import FitnessUpdate

            loser_delta = FitnessUpdate(
                experience_id=loser_id,
                signal=0.0,
                was_helpful=False,
            )
            await self._store.update_fitness(loser_id, loser_delta)

            resolved_count += 1

        logger.info("Resolved %d conflicts ✅", resolved_count)
        return resolved_count

    async def merge_similar(
        self,
        namespace: str,
        similarity_threshold: float = 0.8,
    ) -> int:
        """Merge highly similar experiences to reduce redundancy 🔗.

        V1 implementation: simple text overlap ratio.
        For experiences with > similarity_threshold text overlap:
        1. Keep the higher-fitness experience
        2. Merge context_tags from both
        3. Delete the lower-fitness experience

        Args:
            namespace: Namespace to search for similar experiences.
            similarity_threshold: Minimum Jaccard similarity to consider merging.

        Returns:
            Number of experiences merged (deleted).

        Raises:
            ValueError: If similarity_threshold is not in [0.0, 1.0].
        """
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0],"
                f" got {similarity_threshold}"
            )

        logger.info(
            "Merging similar experiences in namespace '%s' with threshold=%.3f 🔗",
            namespace,
            similarity_threshold,
        )

        # 🔍 Get all experiences in namespace
        all_experiences = await self._store.list_by_namespace(namespace)

        # 📦 Group by category (only merge within same category)
        by_category: dict[str, list] = {}
        for exp in all_experiences:
            by_category.setdefault(exp.category, []).append(exp)

        merged_count = 0

        # 🔗 Check each category for similar pairs
        for category, experiences in by_category.items():
            if len(experiences) < 2:
                continue

            logger.debug(
                "Checking category '%s' with %d experiences for similarity 🔗",
                category,
                len(experiences),
            )

            # 🧪 Compare all pairs
            for i, exp_a in enumerate(experiences):
                for exp_b in experiences[i + 1 :]:
                    similarity = self._compute_similarity(
                        exp_a.insight,
                        exp_b.insight,
                    )

                    if similarity >= similarity_threshold:
                        logger.info(
                            "Merging similar experiences: %s + %s (similarity=%.3f) 🔗",
                            exp_a.id[:8],
                            exp_b.id[:8],
                            similarity,
                        )

                        # 🏆 Keep higher fitness, delete lower
                        if exp_a.fitness_score >= exp_b.fitness_score:
                            keeper_id = exp_a.id
                            keeper_tags = set(exp_a.context_tags)
                            merger_tags = set(exp_b.context_tags)
                            delete_id = exp_b.id
                        else:
                            keeper_id = exp_b.id
                            keeper_tags = set(exp_b.context_tags)
                            merger_tags = set(exp_a.context_tags)
                            delete_id = exp_a.id

                        # 🔗 Merge context_tags
                        merged_tags = list(keeper_tags | merger_tags)

                        # 📝 Update keeper with merged tags
                        keeper = await self._store.get_by_id(keeper_id)
                        if keeper:
                            keeper.context_tags = merged_tags
                            await self._store.update(keeper)

                        # 🗑️ Delete merger
                        await self._store.delete(delete_id)

                        merged_count += 1

        logger.info(
            "Merged %d similar experiences in namespace '%s' ✅",
            merged_count,
            namespace,
        )
        return merged_count

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts 🔍.

        V1 implementation: word-level Jaccard index.
        similarity = |words_a ∩ words_b| / |words_a ∪ words_b|

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        # 🔤 Tokenize (simple word split)
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a and not words_b:
            return 1.0  # Both empty = identical

        if not words_a or not words_b:
            return 0.0  # One empty = completely different

        # 🧮 Jaccard similarity
        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)
