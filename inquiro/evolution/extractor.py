"""Experience extraction engine — LLM-powered learning from trajectories 🧠.

Uses LLM to analyze completed task trajectories and extract reusable insights.
The extraction is guided by a template prompt that defines the vocabulary
and categories specific to the upper-layer platform (e.g., TargetMaster).

Workflow:
1. Render extraction prompt with trajectory snapshot data
2. Call LLM to extract structured experiences
3. Parse JSON output into Experience objects
4. Validate categories against profile configuration
5. Return list of Experience objects

The extractor is **domain-agnostic** — it doesn't know what "good evidence"
or "search strategy" means. All domain semantics come from the extraction
prompt template provided by the upper layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from jinja2 import Template

from inquiro.evolution.types import Experience, TrajectorySnapshot

logger = logging.getLogger(__name__)

# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ExperienceExtractor"]


# ============================================================================
# 🧠 Experience Extraction Engine
# ============================================================================


class ExperienceExtractor:
    """LLM-powered experience extraction engine 🧠.

    Analyzes completed task trajectories to extract reusable insights.
    Uses Jinja2 templates to render extraction prompts and LLM to generate
    structured experience data.

    Attributes:
        llm_fn: Async callable that takes a prompt string and returns LLM response.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], Awaitable[str]],
    ) -> None:
        """Initialize ExperienceExtractor 🔧.

        Args:
            llm_fn: Async function that takes a prompt string and returns
                LLM response string. This decouples the extractor from
                any specific LLM provider interface.
        """
        self.llm_fn = llm_fn
        logger.info("🧠 ExperienceExtractor initialized")

    async def extract(
        self,
        snapshot: TrajectorySnapshot,
        profile_config: dict[str, Any],
    ) -> list[Experience]:
        """Extract experiences from a completed trajectory 📸.

        Renders the extraction prompt template using snapshot data,
        calls LLM to extract structured experiences, parses JSON output,
        validates categories, and returns Experience objects.

        Args:
            snapshot: Trajectory snapshot containing task execution data.
            profile_config: Evolution profile configuration containing:
                - extraction_prompt_template: Jinja2 template string
                - experience_categories: List of valid category strings
                - max_experiences_per_extraction: Max experiences to extract
                - namespace: Namespace for data isolation

        Returns:
            List of Experience objects extracted from the trajectory.
            Returns empty list on LLM failure or parsing errors.

        Raises:
            KeyError: If required profile_config keys are missing.
        """
        # 📋 Extract required config fields
        template_str = profile_config["extraction_prompt_template"]
        valid_categories = set(profile_config["experience_categories"])
        max_experiences = profile_config["max_experiences_per_extraction"]
        namespace = profile_config["namespace"]

        # 🎨 Render extraction prompt using Jinja2
        try:
            template = Template(template_str)
            prompt = template.render(
                snapshot=snapshot,
                max_experiences=max_experiences,
                valid_categories=list(valid_categories),
            )
        except Exception as e:
            logger.warning(
                "⚠️ Failed to render extraction prompt: %s",
                str(e),
            )
            return []

        # 🤖 Call LLM to extract experiences
        try:
            logger.info(
                "🤖 Calling LLM for experience extraction: "
                "evaluation_id=%s, prompt_len=%d",
                snapshot.evaluation_id,
                len(prompt),
            )
            llm_response = await self.llm_fn(prompt)
            logger.info(
                "🤖 LLM extraction returned %d chars",
                len(llm_response) if llm_response else 0,
            )
        except Exception as e:
            logger.warning(
                "⚠️ LLM call failed during extraction: %s",
                str(e),
            )
            return []

        # 📦 Parse LLM output as JSON array
        # 🧹 Strip markdown code fences (```json ... ```)
        cleaned = llm_response.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            # Remove closing fence
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            experiences_data = json.loads(cleaned)
            if not isinstance(experiences_data, list):
                logger.warning(
                    "⚠️ LLM output is not a JSON array, got type: %s",
                    type(experiences_data).__name__,
                )
                return []
        except json.JSONDecodeError as e:
            logger.warning(
                "⚠️ Failed to parse LLM output as JSON: %s",
                str(e),
            )
            return []

        # 🔍 Validate and construct Experience objects
        experiences: list[Experience] = []
        for i, exp_data in enumerate(experiences_data):
            if not isinstance(exp_data, dict):
                logger.warning(
                    "⚠️ Experience item %d is not a dict, skipping",
                    i,
                )
                continue

            # Validate category
            category = exp_data.get("category", "")
            if category not in valid_categories:
                logger.warning(
                    "⚠️ Invalid category '%s' in experience %d, skipping",
                    category,
                    i,
                )
                continue

            # ✨ Auto-set fields from context
            exp_data["namespace"] = namespace
            exp_data["source"] = "trajectory_extraction"
            exp_data["source_evaluation_id"] = snapshot.evaluation_id

            # Construct Experience object
            try:
                experience = Experience(**exp_data)
                experiences.append(experience)
            except Exception as e:
                logger.warning(
                    "⚠️ Failed to construct Experience from item %d: %s",
                    i,
                    str(e),
                )
                continue

        logger.info(
            "✅ Extracted %d experiences from evaluation_id=%s",
            len(experiences),
            snapshot.evaluation_id,
        )
        return experiences
