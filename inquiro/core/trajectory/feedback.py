"""Trajectory feedback provider for search optimization 🔄.

Unified facade over QueryTemplateAnalyzer and GapSearchHintAccumulator
that provides formatted hints for prompt injection into SearchExp and
FocusPromptGenerator.

All public methods are zero-blocking: exceptions are caught internally,
and cold-start (empty directory) gracefully returns empty strings.

Design principles:
    - Domain-agnostic: no domain-specific terms
    - Lazy initialization: index built on first call
    - Zero-blocking: all failures return empty defaults
    - Deterministic: no LLM calls, pure data aggregation

Thread safety: NOT thread-safe. Designed for single-threaded async use
within DiscoveryLoop. The lazy initialization uses a simple boolean flag
without locking. If multi-threaded access is needed in the future, add
a threading.Lock around _ensure_initialized().
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from inquiro.core.trajectory.gap_hints import (
    GapSearchHint,
    GapSearchHintAccumulator,
)
from inquiro.core.trajectory.index import TrajectoryIndex
from inquiro.core.trajectory.query_analyzer import QueryTemplateAnalyzer

logger = logging.getLogger(__name__)


# ============================================================================
# 📊 Structured feedback result
# ============================================================================


@dataclass
class FeedbackResult:
    """Structured feedback data from trajectory analysis 📊.

    Provides both formatted text and structured data for downstream
    consumers that need machine-readable hints.

    Attributes:
        system_prompt_hints: Formatted Markdown for system prompt.
        gap_hints: Matched GapSearchHint objects.
    """

    system_prompt_hints: str = ""
    gap_hints: list[GapSearchHint] = field(default_factory=list)


# ============================================================================
# 🔄 TrajectoryFeedbackProvider
# ============================================================================


class TrajectoryFeedbackProvider:
    """Unified facade for trajectory-based search optimization 🔄.

    Wraps QueryTemplateAnalyzer and GapSearchHintAccumulator behind
    a simple API that returns formatted Markdown strings for prompt
    injection.  Handles lazy initialization, cold-start, and all
    error conditions internally.

    Example::

        provider = TrajectoryFeedbackProvider("/path/to/trajectories")
        hints = provider.get_system_prompt_hints(task_id="task-001")
        focus = provider.get_focus_hints(["missing biomarker data"])

    Attributes:
        _trajectory_dir: Directory containing JSONL trajectory files.
        _max_templates: Maximum number of query templates to return.
        _max_hints: Maximum number of gap hints to return.
    """

    def __init__(
        self,
        trajectory_dir: str,
        *,
        max_templates: int = 5,
        max_hints: int = 3,
    ) -> None:
        """Initialize the feedback provider 🔧.

        Args:
            trajectory_dir: Path to directory with trajectory JSONL files.
            max_templates: Maximum query templates to include in hints.
            max_hints: Maximum gap search hints to include.
        """
        self._trajectory_dir = trajectory_dir
        self._max_templates = max_templates
        self._max_hints = max_hints

        # 🔧 Lazy-initialized components
        self._index: TrajectoryIndex | None = None
        self._query_analyzer: QueryTemplateAnalyzer | None = None
        self._gap_accumulator: GapSearchHintAccumulator | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy-initialize index and analyzers on first use 🔧.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return self._index is not None

        self._initialized = True

        try:
            if not os.path.isdir(self._trajectory_dir):
                logger.debug(
                    "📂 Trajectory dir does not exist: %s",
                    self._trajectory_dir,
                )
                return False

            # 🏗️ Build SQLite index from JSONL files
            db_path = os.path.join(self._trajectory_dir, ".trajectory_feedback.db")
            self._index = TrajectoryIndex(db_path)
            indexed = self._index.index_from_directory(
                self._trajectory_dir,
            )

            if not indexed and self._index.count() == 0:
                logger.debug(
                    "📊 No trajectory data found in %s",
                    self._trajectory_dir,
                )
                return False

            # 🔧 Initialize analyzers
            self._query_analyzer = QueryTemplateAnalyzer(self._index)
            self._gap_accumulator = GapSearchHintAccumulator(self._index)

            logger.info(
                "✅ Trajectory feedback initialized: %d indexed, %d total in %s",
                len(indexed),
                self._index.count(),
                self._trajectory_dir,
            )
            return True

        except Exception as exc:
            logger.warning(
                "⚠️ Trajectory feedback initialization failed: %s",
                exc,
            )
            self._index = None
            return False

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    def get_system_prompt_hints(self, task_id: str | None = None) -> str:
        """Get formatted historical query templates for system prompt 📊.

        Returns Markdown-formatted section with effective query patterns
        from past evaluations, suitable for injection into the SearchAgent
        system prompt.

        Cold start (no data) returns empty string.  All exceptions are
        caught internally to guarantee zero blocking.

        Args:
            task_id: Optional task_id filter to scope analysis.

        Returns:
            Markdown string with historical patterns, or empty string.
        """
        try:
            if not self._ensure_initialized():
                return ""
            if self._query_analyzer is None:
                return ""

            records = self._query_analyzer.analyze(
                task_id=task_id,
                limit=self._max_templates,
            )
            if not records:
                return ""

            # 📝 Format as Markdown section
            lines: list[str] = [
                "## Historical Search Patterns",
                "",
                "The following query patterns have been effective in past evaluations:",
                "",
            ]

            for rec in records:
                yield_pct = int(rec.yield_rate * 100)
                lines.append(
                    f"- `{rec.template}` (yield: {yield_pct}%, {rec.usage_count} uses)"
                )

            lines.append("")
            lines.append("Consider adapting these patterns to your current search.")

            return "\n".join(lines)

        except Exception as exc:
            logger.warning(
                "⚠️ Failed to generate system prompt hints: %s",
                exc,
            )
            return ""

    def get_focus_hints(self, gap_descriptions: list[str]) -> str:
        """Get formatted gap-closing strategies for focus prompt 🎯.

        Returns Markdown-formatted section with historically effective
        strategies for closing similar gaps, suitable for injection into
        the user prompt focus section.

        Cold start or no matches returns empty string.  All exceptions
        are caught internally.

        Args:
            gap_descriptions: List of gap description strings to match.

        Returns:
            Markdown string with gap-closing strategies, or empty string.
        """
        try:
            if not self._ensure_initialized():
                return ""
            if self._gap_accumulator is None:
                return ""

            hints = self._gap_accumulator.get_hints_for_gaps(
                gap_descriptions=gap_descriptions,
                top_k=self._max_hints,
            )
            if not hints:
                return ""

            return self._format_gap_hints(hints)

        except Exception as exc:
            logger.warning(
                "⚠️ Failed to generate focus hints: %s",
                exc,
            )
            return ""

    def get_feedback(self, gap_descriptions: list[str]) -> FeedbackResult:
        """Get structured feedback with both text and data 📊.

        Combines system prompt hints and gap hints into a single
        result object with both formatted text and structured data
        for downstream consumers.

        Args:
            gap_descriptions: List of gap description strings.

        Returns:
            FeedbackResult with formatted text and raw hints.
        """
        try:
            if not self._ensure_initialized():
                return FeedbackResult()
            if self._gap_accumulator is None:
                return FeedbackResult()

            hints = self._gap_accumulator.get_hints_for_gaps(
                gap_descriptions=gap_descriptions,
                top_k=self._max_hints,
            )

            return FeedbackResult(
                system_prompt_hints=self.get_system_prompt_hints(),
                gap_hints=hints,
            )

        except Exception as exc:
            logger.warning(
                "⚠️ Failed to generate structured feedback: %s",
                exc,
            )
            return FeedbackResult()

    # ====================================================================
    # 🔧 Internal formatting helpers
    # ====================================================================

    @staticmethod
    def _format_gap_hints(hints: list[GapSearchHint]) -> str:
        """Format gap hints as Markdown section 📝.

        Args:
            hints: List of GapSearchHint objects.

        Returns:
            Markdown string with gap-closing strategy details.
        """
        lines: list[str] = [
            "### Learned Gap-Closing Strategies",
            "",
            "For similar gaps, these approaches were effective:",
            "",
        ]

        for hint in hints:
            lines.append(f"**Gap pattern**: {hint.gap_pattern}")

            if hint.effective_queries:
                queries_str = ", ".join(f'"{q}"' for q in hint.effective_queries[:5])
                lines.append(f"- Recommended queries: [{queries_str}]")

            if hint.recommended_tools:
                tools_str = ", ".join(hint.recommended_tools[:5])
                lines.append(f"- Recommended tools: [{tools_str}]")

            delta_pct = int(hint.avg_coverage_delta * 100)
            lines.append(
                f"- Historical success: closed "
                f"{hint.success_count} times, "
                f"avg +{delta_pct}% coverage"
            )
            lines.append("")

        return "\n".join(lines)
