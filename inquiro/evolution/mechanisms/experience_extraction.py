"""ExperienceExtractionMechanism — wraps ExperienceExtractor + PromptEnricher 🧬.

Thin adapter that implements the ``BaseMechanism`` interface by delegating
to the existing ``ExperienceExtractor`` and ``PromptEnricher`` components.
The mechanism owns the enrichment cache so that ``inject()`` can be called
synchronously after ``on_round_start()`` has fetched the relevant experiences.

Lifecycle per round:
    on_round_start(round_num)          → refresh enrichment cache
    produce(snapshot, round_context)   → delegate to ExperienceExtractor
    inject(round_context)              → return cached enrichment text
"""

from __future__ import annotations

import logging
from typing import Any

from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.types import (
    EnrichmentResult,
    Experience,
    MechanismType,
    TrajectorySnapshot,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ExperienceExtractionMechanism"]


# ============================================================================
# 🧬 Experience Extraction Mechanism
# ============================================================================


class ExperienceExtractionMechanism(BaseMechanism):
    """Wraps ExperienceExtractor + PromptEnricher into BaseMechanism 🧬.

    Delegates trajectory analysis to ``ExperienceExtractor`` and prompt
    enrichment to ``PromptEnricher``. The enrichment result is cached at
    round-start and served synchronously via ``inject()``.

    Attributes:
        enabled: Whether this mechanism is currently active.
    """

    def __init__(
        self,
        extractor: ExperienceExtractor,
        enricher: PromptEnricher,
        store: Any,
        profile_config: dict[str, Any],
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize ExperienceExtractionMechanism 🔧.

        Args:
            extractor: ExperienceExtractor instance for trajectory analysis.
            enricher: PromptEnricher instance for prompt enrichment.
            store: ExperienceStore instance (passed through for direct access
                if needed by subclasses).
            profile_config: Evolution profile configuration dict containing
                extraction and enrichment settings.
            enabled: Whether this mechanism is active.
        """
        super().__init__(enabled=enabled)
        self._extractor = extractor
        self._enricher = enricher
        self._store = store
        self._profile_config = profile_config
        self._enrichment_result: EnrichmentResult | None = None

    @property
    def mechanism_type(self) -> MechanismType:
        """Return EXPERIENCE_EXTRACTION mechanism type 🏷️.

        Returns:
            MechanismType.EXPERIENCE_EXTRACTION
        """
        return MechanismType.EXPERIENCE_EXTRACTION

    async def produce(
        self,
        snapshot: TrajectorySnapshot,
        round_context: dict[str, Any],
    ) -> list[Experience]:
        """Extract experiences from a completed round 📦.

        Delegates to ``ExperienceExtractor.extract()`` and tags all resulting
        experiences with ``mechanism_type = EXPERIENCE_EXTRACTION``.

        Args:
            snapshot: Structured execution data from the round.
            round_context: Additional context including round_num,
                gap_items, coverage, sub_item_id, etc.

        Returns:
            List of Experience objects tagged with this mechanism type.
            Returns empty list if extractor fails or is disabled.
        """
        if not self.enabled:
            return []

        experiences = await self._extractor.extract(snapshot, self._profile_config)

        # 🏷️ Tag all experiences with mechanism provenance
        for exp in experiences:
            exp.mechanism_type = MechanismType.EXPERIENCE_EXTRACTION

        logger.info(
            "🧬 ExperienceExtractionMechanism produced %d experiences "
            "for evaluation_id=%s",
            len(experiences),
            snapshot.evaluation_id,
        )
        return experiences

    def inject(
        self,
        round_context: dict[str, Any],
    ) -> str | None:
        """Return cached enrichment text for prompt injection 💉.

        The cache is populated by ``on_round_start()`` or
        ``prepare_enrichment()``. If the cache is empty, returns None.

        Args:
            round_context: Context including round_num, gap_items,
                sub_item_id, etc. (not used directly — enrichment was
                pre-fetched in on_round_start).

        Returns:
            Markdown enrichment text to inject into agent prompt,
            or None if no enrichment is available.
        """
        if not self.enabled:
            return None
        if not self._enrichment_result:
            return None
        if not self._enrichment_result.enrichment_text:
            return None
        return self._enrichment_result.enrichment_text

    async def on_round_start(self, round_num: int) -> None:
        """Refresh enrichment cache before each round 🟢.

        Reads context tags and sub_item_id from profile_config defaults,
        then delegates to ``prepare_enrichment()`` to populate the cache.

        Args:
            round_num: The upcoming round number (1-based).
        """
        if not self.enabled:
            return

        # 📋 Extract context from profile_config (upper layer may override)
        context_tags: list[str] = self._profile_config.get(
            "default_context_tags", []
        )
        sub_item: str = self._profile_config.get("default_sub_item", "*")

        logger.debug(
            "🟢 ExperienceExtractionMechanism.on_round_start: "
            "round=%d, refreshing enrichment cache",
            round_num,
        )
        await self.prepare_enrichment(context_tags, sub_item)

    async def prepare_enrichment(
        self,
        context_tags: list[str],
        sub_item: str,
    ) -> None:
        """Fetch and cache enrichment for the given context 🎯.

        Calls ``PromptEnricher.enrich()`` with the supplied context and
        stores the result. Call this before ``inject()`` whenever the
        task context changes.

        Args:
            context_tags: Context tags from the task
                (e.g., ["modality:SmallMolecule"]).
            sub_item: Sub-item identifier (e.g., "safety_1a").
        """
        if not self.enabled:
            return

        try:
            self._enrichment_result = await self._enricher.enrich(
                task_context_tags=context_tags,
                sub_item=sub_item,
                profile_config=self._profile_config,
            )
            logger.info(
                "💉 Enrichment cache refreshed: %d experiences, %d tokens, "
                "truncated=%s",
                len(self._enrichment_result.injected_experience_ids),
                self._enrichment_result.token_count,
                self._enrichment_result.truncated,
            )
        except Exception as e:
            logger.warning(
                "⚠️ Failed to refresh enrichment cache: %s", str(e)
            )
            self._enrichment_result = None
