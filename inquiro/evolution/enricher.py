"""Prompt enrichment engine — inject relevant experiences into agent prompts 💉.

Queries the ExperienceStore for relevant experiences, groups them by category,
renders them using a Jinja2 template, and manages token budgets by dropping
low-fitness experiences when necessary.

Workflow:
1. Query ExperienceStore for experiences matching task context
2. Group experiences by category for structured injection
3. Render enrichment template with experiences
4. Truncate if over token budget (drop lowest-fitness first)
5. Return EnrichmentResult with injected IDs and formatted text

The enricher is **domain-agnostic** — it doesn't interpret categories or
insights. The template and token budget come from the upper-layer platform.
"""

from __future__ import annotations

import logging
from typing import Any

from jinja2 import Template

from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import EnrichmentResult, Experience, ExperienceQuery

logger = logging.getLogger(__name__)

# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["PromptEnricher"]


# ============================================================================
# 💉 Prompt Enrichment Engine
# ============================================================================


class PromptEnricher:
    """Prompt enrichment engine — inject relevant experiences 💉.

    Queries ExperienceStore for relevant experiences and injects them
    into agent prompts using Jinja2 templates. Manages token budgets
    by prioritizing high-fitness experiences.

    Attributes:
        store: ExperienceStore instance for querying experiences.
    """

    def __init__(
        self,
        store: ExperienceStore,
    ) -> None:
        """Initialize PromptEnricher 🔧.

        Args:
            store: ExperienceStore instance for querying experiences.
        """
        self.store = store
        logger.info("💉 PromptEnricher initialized")

    async def enrich(
        self,
        task_context_tags: list[str],
        sub_item: str,
        profile_config: dict[str, Any],
    ) -> EnrichmentResult:
        """Enrich a prompt with relevant experiences 🎯.

        Queries the store for experiences matching the task context,
        groups by category, renders using the enrichment template,
        and truncates if over token budget.

        Args:
            task_context_tags: Context tags from the task
                (e.g., ["modality:SmallMolecule"]).
            sub_item: Sub-item identifier (e.g., "safety_1a").
            profile_config: Evolution profile configuration containing:
                - enrichment_prompt_template: Jinja2 template string
                - enrichment_max_tokens: Maximum tokens for enrichment text
                - enrichment_max_items: Maximum experiences to inject
                - namespace: Namespace for data isolation
                - prune_min_fitness: Minimum fitness score threshold

        Returns:
            EnrichmentResult containing injected experience IDs,
            formatted enrichment text, token count, and truncation flag.

        Raises:
            KeyError: If required profile_config keys are missing.
        """
        # 📋 Extract required config fields
        template_str = profile_config["enrichment_prompt_template"]
        max_tokens = profile_config["enrichment_max_tokens"]
        max_items = profile_config["enrichment_max_items"]
        namespace = profile_config["namespace"]
        min_fitness = profile_config["prune_min_fitness"]

        # 🔍 Query ExperienceStore for relevant experiences
        query = ExperienceQuery(
            namespace=namespace,
            context_tags=task_context_tags,
            sub_item=sub_item,
            min_fitness=min_fitness,
            max_results=max_items * 2,  # Query more for better truncation
        )

        experiences = await self.store.query(query)
        logger.debug(
            "🔍 Queried %d experiences for sub_item=%s, tags=%s",
            len(experiences),
            sub_item,
            task_context_tags,
        )

        # 📊 Group experiences by category
        experiences_by_category = self._group_by_category(experiences)

        # 🎨 Render enrichment template
        try:
            template = Template(template_str)
            enrichment_text = template.render(
                experiences_by_category=experiences_by_category,
                experiences=experiences,
                task_context_tags=task_context_tags,
                sub_item=sub_item,
            )
        except Exception as e:
            logger.warning(
                "⚠️ Failed to render enrichment template: %s",
                str(e),
            )
            return EnrichmentResult()

        # 📏 Check token count (simple approximation: len(text) / 4)
        token_count = self._estimate_tokens(enrichment_text)
        truncated = False
        injected_ids = [exp.id for exp in experiences]

        # ✂️ Truncate if over budget
        if token_count > max_tokens:
            logger.debug(
                "✂️ Enrichment exceeds token budget (%d > %d), truncating",
                token_count,
                max_tokens,
            )
            enrichment_text, injected_ids = self._truncate_by_fitness(
                experiences=experiences,
                template_str=template_str,
                max_tokens=max_tokens,
                task_context_tags=task_context_tags,
                sub_item=sub_item,
            )
            token_count = self._estimate_tokens(enrichment_text)
            truncated = True

        logger.info(
            "✅ Enrichment complete: %d experiences, %d tokens, truncated=%s",
            len(injected_ids),
            token_count,
            truncated,
        )

        return EnrichmentResult(
            injected_experience_ids=injected_ids,
            enrichment_text=enrichment_text,
            token_count=token_count,
            truncated=truncated,
        )

    def _group_by_category(
        self,
        experiences: list[Experience],
    ) -> dict[str, list[Experience]]:
        """Group experiences by category 📊.

        Args:
            experiences: List of Experience objects.

        Returns:
            Dictionary mapping category names to lists of experiences.
        """
        groups: dict[str, list[Experience]] = {}
        for exp in experiences:
            if exp.category not in groups:
                groups[exp.category] = []
            groups[exp.category].append(exp)
        return groups

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length 📏.

        Uses a simple heuristic: 1 token ≈ 4 characters.

        Args:
            text: Input text string.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def _truncate_by_fitness(
        self,
        experiences: list[Experience],
        template_str: str,
        max_tokens: int,
        task_context_tags: list[str],
        sub_item: str,
    ) -> tuple[str, list[str]]:
        """Truncate experiences to fit token budget by dropping low-fitness items ✂️.

        Sorts experiences by fitness_score (descending), removes lowest-fitness
        experiences one by one until the rendered text fits within the token budget.

        Args:
            experiences: List of Experience objects to truncate.
            template_str: Jinja2 template string for rendering.
            max_tokens: Maximum allowed tokens.
            task_context_tags: Task context tags for template rendering.
            sub_item: Sub-item identifier for template rendering.

        Returns:
            Tuple of (truncated enrichment text, list of injected experience IDs).
        """
        # Sort by fitness_score descending (highest fitness first)
        sorted_exps = sorted(
            experiences,
            key=lambda e: e.fitness_score,
            reverse=True,
        )

        # Try progressively smaller subsets
        for i in range(len(sorted_exps), 0, -1):
            subset = sorted_exps[:i]
            experiences_by_category = self._group_by_category(subset)

            try:
                template = Template(template_str)
                text = template.render(
                    experiences_by_category=experiences_by_category,
                    experiences=subset,
                    task_context_tags=task_context_tags,
                    sub_item=sub_item,
                )
                token_count = self._estimate_tokens(text)

                if token_count <= max_tokens:
                    logger.debug(
                        "✅ Truncated to %d experiences (%d tokens)",
                        len(subset),
                        token_count,
                    )
                    return text, [exp.id for exp in subset]
            except Exception as e:
                logger.warning(
                    "⚠️ Failed to render during truncation (size=%d): %s",
                    i,
                    str(e),
                )
                continue

        # If even 1 experience exceeds budget, return empty
        logger.warning(
            "⚠️ Could not fit any experiences within token budget (%d tokens)",
            max_tokens,
        )
        return "", []
