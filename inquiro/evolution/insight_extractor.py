"""Contrastive insight extractor — compare success vs. failure trajectories 🔬.

Uses LLM to analyze pairs of successful and failed trajectory snapshots,
identifying what distinguishes effective search strategies from ineffective ones.
Extracted insights are stored as ``Experience`` objects tagged with
``MechanismType.EXPERIENCE_EXTRACTION``.

Success criteria:
    checklist_coverage >= 0.70 AND confidence >= 0.60

Failure criteria:
    checklist_coverage < 0.50 OR confidence < 0.40

Thresholds are intentionally strict to ensure clear contrastive signal.
Snapshots that fall between success and failure zones are excluded (neutral zone).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from jinja2 import Template

from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import Experience, MechanismType, TrajectorySnapshot

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ContrastiveInsightExtractor"]


# ============================================================================
# 📊 Contrastive Thresholds
# ============================================================================

_SUCCESS_MIN_COVERAGE: float = 0.70
_SUCCESS_MIN_CONFIDENCE: float = 0.60
_FAILURE_MAX_COVERAGE: float = 0.50
_FAILURE_MAX_CONFIDENCE: float = 0.40

# Path to the default contrastive extraction prompt template
_DEFAULT_PROMPT_PATH: Path = (
    Path(__file__).parent.parent / "prompts" / "contrastive_extraction.md"
)


# ============================================================================
# 🔬 Contrastive Insight Extractor
# ============================================================================


class ContrastiveInsightExtractor:
    """Compare successful/failed trajectories to generate high-quality insights 🔬.

    Uses LLM-based contrastive analysis to identify which search strategies,
    tool choices, and reasoning patterns distinguish successful task runs from
    failed ones. Stores results as Experience objects in the ExperienceStore.

    Attributes:
        store: ExperienceStore for persisting extracted insights.
    """

    def __init__(
        self,
        store: ExperienceStore,
        llm_fn: Callable[[str], Awaitable[str]],
        *,
        prompt_template_path: str | os.PathLike | None = None,
    ) -> None:
        """Initialize ContrastiveInsightExtractor 🔧.

        Args:
            store: ExperienceStore instance for persisting insights.
            llm_fn: Async callable that takes a prompt string and returns
                the LLM response string. Decouples extractor from LLM provider.
            prompt_template_path: Optional path to a Jinja2 template file.
                Defaults to ``inquiro/prompts/contrastive_extraction.md``.
        """
        self._store = store
        self._llm_fn = llm_fn

        # 📋 Load prompt template
        template_path = Path(
            prompt_template_path if prompt_template_path else _DEFAULT_PROMPT_PATH
        )
        if template_path.exists():
            self._prompt_template_str = template_path.read_text(encoding="utf-8")
        else:
            logger.warning(
                "⚠️ Contrastive prompt template not found at %s, "
                "using built-in fallback",
                template_path,
            )
            self._prompt_template_str = _FALLBACK_TEMPLATE

        logger.info("🔬 ContrastiveInsightExtractor initialized")

    def classify_snapshot(
        self,
        snapshot: TrajectorySnapshot,
    ) -> str | None:
        """Classify a snapshot as 'success', 'failure', or None (neutral) 🏷️.

        Args:
            snapshot: TrajectorySnapshot to classify.

        Returns:
            'success' if metrics meet success criteria,
            'failure' if metrics meet failure criteria,
            None if in the neutral zone (between thresholds).
        """
        coverage = snapshot.metrics.checklist_coverage
        confidence = snapshot.metrics.confidence

        is_success = (
            coverage >= _SUCCESS_MIN_COVERAGE
            and confidence >= _SUCCESS_MIN_CONFIDENCE
        )
        is_failure = (
            coverage < _FAILURE_MAX_COVERAGE
            or confidence < _FAILURE_MAX_CONFIDENCE
        )

        if is_success:
            return "success"
        if is_failure:
            return "failure"
        return None

    async def extract_contrastive_insights(
        self,
        success_snapshots: list[TrajectorySnapshot],
        failure_snapshots: list[TrajectorySnapshot],
        profile_config: dict[str, Any],
    ) -> list[Experience]:
        """Extract insights by contrasting success and failure trajectories 💡.

        Builds a contrastive prompt from the provided snapshots, calls LLM
        to identify what distinguishes success from failure, and parses the
        structured output into Experience objects.

        Args:
            success_snapshots: Trajectories with high coverage and confidence.
                (checklist_coverage >= 0.70 AND confidence >= 0.60)
            failure_snapshots: Trajectories with low coverage or confidence.
                (checklist_coverage < 0.50 OR confidence < 0.40)
            profile_config: Evolution profile configuration containing:
                - namespace: Namespace for data isolation
                - experience_categories: List of valid category strings
                - max_experiences_per_extraction: Max experiences to return

        Returns:
            List of Experience objects tagged with EXPERIENCE_EXTRACTION.
            Returns empty list if inputs are insufficient or LLM fails.
        """
        if not success_snapshots or not failure_snapshots:
            logger.info(
                "ℹ️ Skipping contrastive extraction: "
                "success=%d, failure=%d (need at least 1 each)",
                len(success_snapshots),
                len(failure_snapshots),
            )
            return []

        namespace = profile_config["namespace"]
        valid_categories = set(profile_config["experience_categories"])
        max_experiences = profile_config["max_experiences_per_extraction"]

        # 🎨 Render contrastive prompt
        prompt = self._render_prompt(
            success_snapshots=success_snapshots,
            failure_snapshots=failure_snapshots,
            valid_categories=list(valid_categories),
            max_experiences=max_experiences,
        )
        if not prompt:
            return []

        # 🤖 Call LLM
        try:
            logger.info(
                "🤖 Calling LLM for contrastive extraction: "
                "success=%d, failure=%d, prompt_len=%d",
                len(success_snapshots),
                len(failure_snapshots),
                len(prompt),
            )
            llm_response = await self._llm_fn(prompt)
        except Exception as e:
            logger.warning(
                "⚠️ LLM call failed during contrastive extraction: %s", str(e)
            )
            return []

        # 📦 Parse and validate experiences
        experiences = self._parse_experiences(
            llm_response=llm_response,
            namespace=namespace,
            valid_categories=valid_categories,
            max_experiences=max_experiences,
            success_snapshots=success_snapshots,
        )

        logger.info(
            "✅ Contrastive extraction produced %d insights "
            "(success=%d, failure=%d)",
            len(experiences),
            len(success_snapshots),
            len(failure_snapshots),
        )
        return experiences

    def _render_prompt(
        self,
        success_snapshots: list[TrajectorySnapshot],
        failure_snapshots: list[TrajectorySnapshot],
        valid_categories: list[str],
        max_experiences: int,
    ) -> str | None:
        """Render the contrastive extraction prompt 🎨.

        Args:
            success_snapshots: Successful trajectory snapshots.
            failure_snapshots: Failed trajectory snapshots.
            valid_categories: List of valid category strings.
            max_experiences: Maximum experiences to request.

        Returns:
            Rendered prompt string, or None on template error.
        """
        # 🔧 Summarize tool call patterns per snapshot
        success_summaries = [
            self._summarize_snapshot(s) for s in success_snapshots
        ]
        failure_summaries = [
            self._summarize_snapshot(s) for s in failure_snapshots
        ]

        try:
            template = Template(self._prompt_template_str)
            return template.render(
                success_summaries=success_summaries,
                failure_summaries=failure_summaries,
                valid_categories=valid_categories,
                max_experiences=max_experiences,
                success_count=len(success_snapshots),
                failure_count=len(failure_snapshots),
            )
        except Exception as e:
            logger.warning(
                "⚠️ Failed to render contrastive prompt template: %s", str(e)
            )
            return None

    def _summarize_snapshot(
        self,
        snapshot: TrajectorySnapshot,
    ) -> dict[str, Any]:
        """Build a concise summary dict from a TrajectorySnapshot 📋.

        Args:
            snapshot: TrajectorySnapshot to summarize.

        Returns:
            Dictionary with topic, metrics, tool_call_counts, and timing.
        """
        # Count tool calls by name
        tool_counts: dict[str, int] = {}
        for tc in snapshot.tool_calls:
            tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1

        # Count failures
        failed_calls = sum(1 for tc in snapshot.tool_calls if not tc.success)

        return {
            "evaluation_id": snapshot.evaluation_id,
            "topic": snapshot.topic,
            "sub_item_id": snapshot.sub_item_id,
            "context_tags": snapshot.context_tags,
            "metrics": {
                "checklist_coverage": snapshot.metrics.checklist_coverage,
                "confidence": snapshot.metrics.confidence,
                "evidence_count": snapshot.metrics.evidence_count,
                "search_rounds": snapshot.metrics.search_rounds,
                "cost_usd": snapshot.metrics.cost_usd,
                "decision": snapshot.metrics.decision,
            },
            "tool_call_counts": tool_counts,
            "failed_tool_calls": failed_calls,
            "wall_time_seconds": snapshot.wall_time_seconds,
        }

    def _parse_experiences(
        self,
        llm_response: str,
        namespace: str,
        valid_categories: set[str],
        max_experiences: int,
        success_snapshots: list[TrajectorySnapshot],
    ) -> list[Experience]:
        """Parse LLM output into validated Experience objects 📦.

        Args:
            llm_response: Raw LLM response string.
            namespace: Namespace for data isolation.
            valid_categories: Set of valid category strings.
            max_experiences: Maximum number of experiences to return.
            success_snapshots: Used to set source_evaluation_id.

        Returns:
            List of validated Experience objects.
        """
        # 🧹 Strip markdown code fences
        cleaned = llm_response.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            experiences_data = json.loads(cleaned)
            if not isinstance(experiences_data, list):
                logger.warning(
                    "⚠️ Contrastive LLM output is not a JSON array, got: %s",
                    type(experiences_data).__name__,
                )
                return []
        except json.JSONDecodeError as e:
            logger.warning(
                "⚠️ Failed to parse contrastive LLM output as JSON: %s", str(e)
            )
            return []

        # 🆔 Use first success snapshot ID as provenance (best available)
        source_eval_id = (
            success_snapshots[0].evaluation_id if success_snapshots else None
        )

        experiences: list[Experience] = []
        for i, exp_data in enumerate(experiences_data[:max_experiences]):
            if not isinstance(exp_data, dict):
                logger.warning(
                    "⚠️ Contrastive experience item %d is not a dict, skipping", i
                )
                continue

            category = exp_data.get("category", "")
            if category not in valid_categories:
                logger.warning(
                    "⚠️ Invalid category '%s' in contrastive experience %d, "
                    "skipping",
                    category,
                    i,
                )
                continue

            # ✨ Auto-set provenance fields
            exp_data["namespace"] = namespace
            exp_data["source"] = "contrastive_extraction"
            exp_data["source_evaluation_id"] = source_eval_id
            exp_data["mechanism_type"] = MechanismType.EXPERIENCE_EXTRACTION

            try:
                experience = Experience(**exp_data)
                experiences.append(experience)
            except Exception as e:
                logger.warning(
                    "⚠️ Failed to construct Experience from contrastive item %d: %s",
                    i,
                    str(e),
                )
                continue

        return experiences


# ============================================================================
# 📝 Fallback Prompt Template (inline, used when file is missing)
# ============================================================================

_FALLBACK_TEMPLATE = """\
You are an expert at analyzing research task trajectories.

Compare the following SUCCESSFUL trajectories (high coverage + confidence)
against FAILED trajectories (low coverage or confidence).

SUCCESSFUL TRAJECTORIES ({{ success_count }}):
{% for s in success_summaries %}
[{{ loop.index }}] Topic: {{ s.topic }}
    Coverage: {{ "%.2f"|format(s.metrics.checklist_coverage) }}, \
Confidence: {{ "%.2f"|format(s.metrics.confidence) }}
    Evidence: {{ s.metrics.evidence_count }}, Rounds: \
{{ s.metrics.search_rounds }}
    Tools: {{ s.tool_call_counts }}
{% endfor %}

FAILED TRAJECTORIES ({{ failure_count }}):
{% for f in failure_summaries %}
[{{ loop.index }}] Topic: {{ f.topic }}
    Coverage: {{ "%.2f"|format(f.metrics.checklist_coverage) }}, \
Confidence: {{ "%.2f"|format(f.metrics.confidence) }}
    Evidence: {{ f.metrics.evidence_count }}, Rounds: \
{{ f.metrics.search_rounds }}
    Tools: {{ f.tool_call_counts }}
{% endfor %}

Extract up to {{ max_experiences }} insights explaining WHY the successful
trajectories outperformed the failed ones.

Valid categories: {{ valid_categories | join(", ") }}

Respond with a JSON array only:
[
  {
    "category": "<one of the valid categories>",
    "insight": "<specific, actionable insight>",
    "context_tags": ["<relevant tags>"],
    "applicable_sub_items": ["*"]
  }
]
"""
