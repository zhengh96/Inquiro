"""Gap Search Hints extraction from trajectory data 🎯.

Analyzes focus prompts and their outcomes across Discovery trajectories
to build a catalog of reusable gap-closing search strategies.

When a focus prompt leads to measurable coverage improvement in the
subsequent round, the associated queries and tools are captured as
effective hints for future runs (Phase 3 O3).

Design principles:
    - Reads from TrajectoryIndex (SQLite), never parses JSONL directly
    - Domain-agnostic: generalizes gap descriptions into patterns
    - Accumulates across multiple trajectories for statistical strength
    - Exports to YAML for TargetMaster catalog integration
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import yaml
from pydantic import BaseModel, Field

from inquiro.core.trajectory.index import TrajectoryIndex

logger = logging.getLogger(__name__)


# ============================================================================
# 📊 Data models
# ============================================================================


class GapSearchHint(BaseModel):
    """Reusable search strategy for closing a specific gap type 🎯.

    Attributes:
        gap_pattern: Generalized gap description pattern.
        effective_queries: Query templates that helped close this gap.
        success_count: Number of times this pattern closed a gap.
        avg_coverage_delta: Average coverage improvement per success.
        recommended_tools: MCP tools that worked best for this gap.
    """

    gap_pattern: str = Field(description="Generalized gap description pattern")
    effective_queries: list[str] = Field(
        default_factory=list,
        description="Query templates that helped close this gap",
    )
    success_count: int = Field(
        default=0,
        description="Times this pattern successfully closed a gap",
    )
    avg_coverage_delta: float = Field(
        default=0.0,
        description="Average coverage improvement per success",
    )
    recommended_tools: list[str] = Field(
        default_factory=list,
        description="MCP tools that worked best for this gap",
    )


# ============================================================================
# 🔧 Pattern generalization helpers
# ============================================================================


def _generalize_gap(gap_text: str) -> str:
    """Generalize a specific gap description into a reusable pattern 🔄.

    Strips numbering prefixes, trailing identifiers, and normalizes
    whitespace to produce a canonical gap pattern string.

    Args:
        gap_text: Raw gap description from trajectory data.

    Returns:
        Normalized pattern string suitable for grouping.
    """
    # ✅ Strip leading numbering (e.g., "1.", "1)", "1 -")
    pattern = re.sub(r"^\d+[\.\)\-\s]+", "", gap_text.strip())
    # ✅ Normalize whitespace
    pattern = re.sub(r"\s+", " ", pattern).strip()
    # ✅ Lowercase for canonical comparison
    pattern = pattern.lower()
    return pattern


def _generalize_query(query_text: str) -> str:
    """Generalize a specific query into a reusable template 🔄.

    Replaces domain-specific identifiers with placeholders while
    preserving the structural search pattern.

    Args:
        query_text: Raw query text from trajectory data.

    Returns:
        Generalized query template with {topic} placeholders.
    """
    # ✅ Replace site-specific filters with generic form
    template = re.sub(r"site:\S+", "site:{domain}", query_text.strip())
    # ✅ Normalize whitespace
    template = re.sub(r"\s+", " ", template).strip()
    return template


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Compute keyword overlap ratio between two texts 📊.

    Uses simple word-level Jaccard similarity for relevance
    matching between gap descriptions and hint patterns.

    Args:
        text_a: First text to compare.
        text_b: Second text to compare.

    Returns:
        Jaccard similarity score (0.0 to 1.0).
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ============================================================================
# 🎯 GapSearchHintAccumulator
# ============================================================================


class GapSearchHintAccumulator:
    """Accumulates effective gap-closing strategies from trajectory data 🎯.

    Analyzes focus prompts and their outcomes across Discovery trajectories
    to build a catalog of reusable search strategies. A focus prompt is
    considered "effective" when the subsequent round shows coverage
    improvement.

    Example::

        index = TrajectoryIndex("/path/to/index.db")
        accumulator = GapSearchHintAccumulator(index)
        hints = accumulator.accumulate(task_id="my-task")
        accumulator.export_yaml(hints, "/path/to/hints.yaml")

    Attributes:
        index: The TrajectoryIndex to read data from.
    """

    def __init__(self, index: TrajectoryIndex) -> None:
        """Initialize the accumulator with a trajectory index 🔧.

        Args:
            index: TrajectoryIndex instance to query for data.
        """
        self.index = index

    def analyze_trajectory(self, trajectory_id: str) -> list[GapSearchHint]:
        """Extract hints from a single trajectory's gap phases 🔍.

        Finds consecutive round pairs where coverage improved after a
        focus prompt, then extracts the successful queries and tools
        from the improved round.

        Args:
            trajectory_id: Unique trajectory identifier to analyze.

        Returns:
            List of GapSearchHint objects extracted from this trajectory.
        """
        rounds = self._get_rounds_with_details(trajectory_id)
        if len(rounds) < 2:
            return []

        hints: list[GapSearchHint] = []

        for i in range(len(rounds) - 1):
            current_round = rounds[i]
            next_round = rounds[i + 1]

            # 🎯 Check if focus prompt existed in current round
            focus_prompt_text = current_round.get("focus_prompt_text", "")
            if not focus_prompt_text:
                continue

            # 📊 Check for coverage improvement
            current_coverage = current_round.get("coverage_ratio", 0.0)
            next_coverage = next_round.get("coverage_ratio", 0.0)
            coverage_delta = next_coverage - current_coverage

            if coverage_delta <= 0:
                continue

            # ✅ Effective focus prompt found! Extract the hint.
            target_gaps = current_round.get("target_gaps", [])
            next_queries = next_round.get("queries", [])
            next_tools = next_round.get("tools", [])

            # 🔄 Generalize gap patterns
            gap_patterns = (
                [_generalize_gap(g) for g in target_gaps]
                if target_gaps
                else [_generalize_gap(focus_prompt_text)]
            )

            # 🔄 Generalize query templates
            query_templates = list(
                dict.fromkeys(_generalize_query(q) for q in next_queries if q)
            )

            # 🔄 Deduplicate tools
            unique_tools = list(dict.fromkeys(t for t in next_tools if t))

            for gap_pattern in gap_patterns:
                if not gap_pattern:
                    continue
                hints.append(
                    GapSearchHint(
                        gap_pattern=gap_pattern,
                        effective_queries=query_templates,
                        success_count=1,
                        avg_coverage_delta=coverage_delta,
                        recommended_tools=unique_tools,
                    )
                )

        logger.info(
            "🎯 Extracted %d hints from trajectory %s",
            len(hints),
            trajectory_id,
        )
        return hints

    def accumulate(self, *, task_id: str | None = None) -> list[GapSearchHint]:
        """Accumulate hints across multiple trajectories 📊.

        Analyzes all matching trajectories, merges hints by gap
        pattern, and returns aggregated results ordered by
        success_count descending.

        Args:
            task_id: Optional filter to restrict to a specific task.

        Returns:
            Aggregated hints ordered by success_count DESC.
        """
        # 🔍 Find all relevant trajectories
        trajectories = self.index.list_trajectories(
            task_id=task_id,
            status="completed",
            limit=1000,
        )

        if not trajectories:
            return []

        # 🎯 Collect all individual hints
        all_hints: list[GapSearchHint] = []
        for traj in trajectories:
            traj_hints = self.analyze_trajectory(traj.trajectory_id)
            all_hints.extend(traj_hints)

        if not all_hints:
            return []

        # 📊 Merge hints by gap_pattern
        return self._merge_hints(all_hints)

    def export_yaml(self, hints: list[GapSearchHint], output_path: str) -> None:
        """Export hints to YAML file for TargetMaster catalog integration 📄.

        Args:
            hints: List of GapSearchHint to export.
            output_path: Filesystem path for the output YAML file.
        """
        data: dict[str, Any] = {
            "hints": [
                {
                    "gap_pattern": h.gap_pattern,
                    "effective_queries": h.effective_queries,
                    "recommended_tools": h.recommended_tools,
                    "success_count": h.success_count,
                    "avg_coverage_delta": round(h.avg_coverage_delta, 4),
                }
                for h in hints
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info("📄 Exported %d hints to %s", len(hints), output_path)

    def get_hints_for_gaps(
        self,
        gap_descriptions: list[str],
        top_k: int = 5,
    ) -> list[GapSearchHint]:
        """Get most relevant hints for given gap descriptions 🔍.

        Uses simple keyword overlap (Jaccard similarity) to rank
        accumulated hints by relevance to the provided gaps.

        Args:
            gap_descriptions: List of gap description strings to match.
            top_k: Maximum number of hints to return.

        Returns:
            Most relevant hints sorted by relevance score.
        """
        # 🎯 First accumulate all available hints
        all_hints = self.accumulate()
        if not all_hints:
            return []

        if not gap_descriptions:
            return all_hints[:top_k]

        # 📊 Score each hint against all gap descriptions
        scored: list[tuple[float, GapSearchHint]] = []
        for hint in all_hints:
            max_score = 0.0
            for gap_desc in gap_descriptions:
                score = _keyword_overlap(
                    hint.gap_pattern,
                    gap_desc.lower(),
                )
                max_score = max(max_score, score)
            scored.append((max_score, hint))

        # ✅ Sort by relevance descending, then by success_count
        scored.sort(key=lambda x: (x[0], x[1].success_count), reverse=True)

        return [hint for _, hint in scored[:top_k]]

    # ====================================================================
    # 🔧 Internal helpers
    # ====================================================================

    def _get_rounds_with_details(self, trajectory_id: str) -> list[dict[str, Any]]:
        """Fetch round data with queries and tools from the index 🔧.

        Queries the SQLite database for round records, focus prompt
        details (including target_gaps from the JSONL), and per-round
        query/tool data.

        Args:
            trajectory_id: Trajectory to query.

        Returns:
            List of dicts with round data, queries, tools, and gaps.
        """
        result: list[dict[str, Any]] = []

        with self.index._connect() as conn:
            # 📊 Fetch round records
            round_rows = conn.execute(
                "SELECT round_number, coverage_ratio, focus_prompt_text "
                "FROM rounds "
                "WHERE trajectory_id = ? "
                "ORDER BY round_number",
                (trajectory_id,),
            ).fetchall()

            for row in round_rows:
                round_number = row["round_number"]

                # 🔍 Fetch queries for this round
                query_rows = conn.execute(
                    "SELECT query_text, mcp_tool "
                    "FROM queries "
                    "WHERE trajectory_id = ? AND round_number = ?",
                    (trajectory_id, round_number),
                ).fetchall()
                queries = [r["query_text"] for r in query_rows]
                tools = list(
                    dict.fromkeys(r["mcp_tool"] for r in query_rows if r["mcp_tool"])
                )

                # 🎯 Extract target gaps from focus prompt text
                # Target gaps are stored in the JSONL but not in SQLite,
                # so we parse them from the JSONL via the index's raw data.
                # For now, use the focus prompt text itself as the gap.
                target_gaps = self._extract_target_gaps(
                    trajectory_id, round_number, conn
                )

                result.append(
                    {
                        "round_number": round_number,
                        "coverage_ratio": row["coverage_ratio"],
                        "focus_prompt_text": row["focus_prompt_text"],
                        "target_gaps": target_gaps,
                        "queries": queries,
                        "tools": tools,
                    }
                )

        return result

    def _extract_target_gaps(
        self,
        trajectory_id: str,
        round_number: int,
        conn: Any,
    ) -> list[str]:
        """Extract target gap descriptions for a round 🎯.

        Looks up the JSONL source file and parses the target_gaps
        from the focus_prompt record for the specified round.

        Args:
            trajectory_id: Trajectory identifier.
            round_number: Round number to extract gaps from.
            conn: Active SQLite connection.

        Returns:
            List of target gap description strings.
        """
        import json
        import os

        # 🔍 Get the JSONL path for this trajectory
        traj_row = conn.execute(
            "SELECT jsonl_path FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
        if not traj_row:
            return []

        jsonl_path = traj_row["jsonl_path"]
        if not os.path.isfile(jsonl_path):
            return []

        # 📄 Parse target_gaps from the specific round
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        record.get("type") == "round"
                        and record.get("round_number") == round_number
                    ):
                        focus_prompt = (
                            record.get("gap_phase", {}).get("focus_prompt") or {}
                        )
                        return focus_prompt.get("target_gaps", [])
        except OSError:
            logger.warning(
                "🎯 Could not read JSONL for target_gaps: %s",
                jsonl_path,
            )
        return []

    @staticmethod
    def _merge_hints(
        hints: list[GapSearchHint],
    ) -> list[GapSearchHint]:
        """Merge individual hints by gap_pattern 📊.

        Aggregates success counts, coverage deltas, queries, and
        tools across all hints sharing the same gap pattern.

        Args:
            hints: Raw hints from individual trajectory analysis.

        Returns:
            Merged hints ordered by success_count DESC.
        """
        # 📊 Group by gap_pattern
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "queries": [],
                "tools": [],
                "success_count": 0,
                "coverage_deltas": [],
            }
        )

        for hint in hints:
            group = groups[hint.gap_pattern]
            group["queries"].extend(hint.effective_queries)
            group["tools"].extend(hint.recommended_tools)
            group["success_count"] += hint.success_count
            group["coverage_deltas"].append(hint.avg_coverage_delta)

        # ✅ Build merged hints
        merged: list[GapSearchHint] = []
        for pattern, data in groups.items():
            # 🔄 Deduplicate queries and tools while preserving order
            unique_queries = list(dict.fromkeys(data["queries"]))
            unique_tools = list(dict.fromkeys(data["tools"]))

            avg_delta = (
                sum(data["coverage_deltas"]) / len(data["coverage_deltas"])
                if data["coverage_deltas"]
                else 0.0
            )

            merged.append(
                GapSearchHint(
                    gap_pattern=pattern,
                    effective_queries=unique_queries,
                    success_count=data["success_count"],
                    avg_coverage_delta=avg_delta,
                    recommended_tools=unique_tools,
                )
            )

        # 📊 Sort by success_count descending
        merged.sort(key=lambda h: h.success_count, reverse=True)
        return merged
