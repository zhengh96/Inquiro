"""Query template analysis for search optimization 📊.

Extracts generalized templates from historical search queries,
computes effectiveness metrics per template, and ranks them by
evidence yield rate. Provides top-K recommendations for prompt
injection into SearchAgent.

This is Phase 3 O2 (Query Template Optimization), building on
the TrajectoryIndex (O1) foundation.

Design principles:
    - Domain-agnostic: no domain-specific terms in template logic
    - Deterministic: template extraction uses regex, not LLM
    - Composable: takes TrajectoryIndex, returns Pydantic models
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from inquiro.core.trajectory.index import TrajectoryIndex

logger = logging.getLogger(__name__)

# ============================================================================
# 📊 Result models
# ============================================================================

_MAX_EXAMPLE_QUERIES = 3


class TemplateEffectivenessRecord(BaseModel):
    """Effectiveness metrics for a query template 📊.

    Tracks how well a generalized query pattern performs across
    historical executions, including yield, cost, and success.

    Attributes:
        template: Generalized template pattern.
        usage_count: Number of times this template was used.
        avg_result_count: Average raw results per execution.
        avg_cost_usd: Average cost per execution.
        yield_rate: Ratio of cleaned evidence to total queries.
        success_rate: Fraction of executions producing > 0 results.
        example_queries: Up to 3 concrete query instantiations.
    """

    template: str = Field(description="Generalized template pattern")
    usage_count: int = Field(description="Number of times used")
    avg_result_count: float = Field(description="Average raw results per execution")
    avg_cost_usd: float = Field(description="Average cost per execution")
    yield_rate: float = Field(description="Cleaned evidence / queries ratio")
    success_rate: float = Field(description="Fraction of executions with > 0 results")
    example_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 example query instantiations",
    )


# ============================================================================
# 🔧 Template extraction patterns
# ============================================================================

# 🎯 Regex for quoted strings (double or single quotes)
_QUOTED_RE = re.compile(r'"[^"]+"|\'[^\']+\'')

# 🎯 Regex for standalone numbers (e.g., phase 3, top 10)
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?!\w)")

# 🎯 Regex for site: operators (preserved as-is in templates)
_SITE_RE = re.compile(r"site:\S+")

# 🎯 Regex for filetype: operators (preserved as-is)
_FILETYPE_RE = re.compile(r"filetype:\S+")


def _extract_template(query_text: str) -> str:
    """Convert a concrete query into a generalized template 🔧.

    Applies the following transformations in order:
    1. Extract and preserve site:/filetype: operators
    2. Replace quoted strings with {quoted_term}
    3. Replace remaining topic words with {topic}
    4. Replace standalone numbers with {number}
    5. Normalize whitespace

    Args:
        query_text: The raw query string.

    Returns:
        A generalized template string.
    """
    if not query_text or not query_text.strip():
        return "{topic}"

    text = query_text.strip()

    # ✅ Step 1: Extract preserved operators (site:, filetype:)
    preserved_operators: list[str] = []
    for pattern in (_SITE_RE, _FILETYPE_RE):
        for match in pattern.finditer(text):
            preserved_operators.append(match.group())
        text = pattern.sub("", text)

    # ✅ Step 2: Replace quoted strings with {quoted_term}
    _has_quoted = bool(_QUOTED_RE.search(text))
    text = _QUOTED_RE.sub("{quoted_term}", text)

    # ✅ Step 3: Replace standalone numbers with {number}
    text = _NUMBER_RE.sub("{number}", text)

    # ✅ Step 4: Collapse remaining non-placeholder words into {topic}
    # Split into tokens, identify placeholder tokens vs topic words
    tokens = text.split()
    result_tokens: list[str] = []
    topic_run = False

    for token in tokens:
        is_placeholder = "{quoted_term}" in token or "{number}" in token
        if is_placeholder:
            if topic_run:
                result_tokens.append("{topic}")
                topic_run = False
            result_tokens.append(token)
        else:
            # 🎯 Check if token is a common search modifier to keep
            lower = token.lower()
            if lower in _SEARCH_MODIFIERS:
                if topic_run:
                    result_tokens.append("{topic}")
                    topic_run = False
                result_tokens.append(lower)
            else:
                topic_run = True

    if topic_run:
        result_tokens.append("{topic}")

    # ✅ Step 5: Re-attach preserved operators
    result_tokens.extend(preserved_operators)

    # ✅ Step 6: Normalize whitespace and dedup adjacent {topic}
    template = " ".join(result_tokens)
    template = re.sub(r"(\{topic\}\s*)+", "{topic} ", template).strip()

    # 🔧 Handle edge case: empty template
    if not template:
        return "{topic}"

    return template


# 🎯 Common search modifiers to preserve verbatim in templates
_SEARCH_MODIFIERS = frozenset(
    {
        "and",
        "or",
        "not",
        "clinical",
        "trials",
        "review",
        "systematic",
        "meta-analysis",
        "randomized",
        "controlled",
        "phase",
        "study",
        "studies",
        "efficacy",
        "safety",
        "mechanism",
        "vs",
        "versus",
        "compared",
        "recent",
        "latest",
        "new",
        "pdf",
        "abstract",
        "full-text",
    }
)


# ============================================================================
# 📊 QueryTemplateAnalyzer
# ============================================================================


class QueryTemplateAnalyzer:
    """Analyzes query templates from trajectory data for optimization 📊.

    Extracts patterns from historical search queries, computes
    effectiveness metrics, and ranks templates by their evidence
    yield rate. Used for SearchAgent prompt injection.

    Example::

        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze(task_id="task-001")
        top = analyzer.get_top_templates(top_k=5)

    Attributes:
        _index: The backing TrajectoryIndex instance.
    """

    def __init__(self, index: TrajectoryIndex) -> None:
        """Initialize the analyzer with a TrajectoryIndex 🔧.

        Args:
            index: TrajectoryIndex instance to query data from.
        """
        self._index = index

    def extract_templates(
        self,
        queries: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group raw queries into template categories 🔧.

        Templates are created by:
        1. Replacing quoted strings with {quoted_term}
        2. Replacing topic-specific words with {topic}
        3. Replacing standalone numbers with {number}
        4. Keeping site: operators and search modifiers intact

        Args:
            queries: List of query dicts with keys: query_text,
                result_count, cost_usd.

        Returns:
            Dict mapping template string to list of matching
            query records.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for query in queries:
            query_text = query.get("query_text", "")
            template = _extract_template(query_text)
            groups[template].append(query)

        logger.debug(
            "📊 Extracted %d templates from %d queries",
            len(groups),
            len(queries),
        )
        return dict(groups)

    def analyze(
        self,
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[TemplateEffectivenessRecord]:
        """Analyze and rank query templates by effectiveness 📊.

        Fetches query data from the index, groups by template,
        computes per-template metrics, and returns ranked list
        ordered by yield_rate DESC.

        Args:
            task_id: Optional filter to restrict analysis to a
                specific task's trajectories.
            limit: Maximum number of template records to return.

        Returns:
            Ranked list of TemplateEffectivenessRecord ordered by
            yield_rate descending, then by success_rate descending.
        """
        # 🎯 Fetch raw query + round data from the index
        queries = self._fetch_queries(task_id=task_id)
        if not queries:
            logger.info("📊 No queries found for analysis")
            return []

        # 🎯 Fetch round-level cleaned evidence for yield calculation
        round_evidence = self._fetch_round_evidence(task_id=task_id)

        # 🎯 Group queries by template
        groups = self.extract_templates(queries)

        # 🎯 Compute metrics per template group
        records: list[TemplateEffectivenessRecord] = []
        for template, group_queries in groups.items():
            usage_count = len(group_queries)

            # 📊 Average result count
            result_counts = [q.get("result_count", 0) for q in group_queries]
            avg_result_count = (
                sum(result_counts) / usage_count if usage_count > 0 else 0.0
            )

            # 💰 Average cost
            costs = [q.get("cost_usd", 0.0) for q in group_queries]
            avg_cost = sum(costs) / usage_count if usage_count > 0 else 0.0

            # ✅ Success rate: fraction of queries with > 0 results
            successful = sum(1 for c in result_counts if c > 0)
            success_rate = successful / usage_count if usage_count > 0 else 0.0

            # 📊 Yield rate: cleaned evidence / total queries in rounds
            # where this template was used
            yield_rate = self._compute_yield_rate(group_queries, round_evidence)

            # 📝 Example queries (up to 3 unique)
            seen: set[str] = set()
            examples: list[str] = []
            for q in group_queries:
                qt = q.get("query_text", "")
                if qt and qt not in seen:
                    seen.add(qt)
                    examples.append(qt)
                    if len(examples) >= _MAX_EXAMPLE_QUERIES:
                        break

            records.append(
                TemplateEffectivenessRecord(
                    template=template,
                    usage_count=usage_count,
                    avg_result_count=round(avg_result_count, 2),
                    avg_cost_usd=round(avg_cost, 4),
                    yield_rate=round(yield_rate, 4),
                    success_rate=round(success_rate, 4),
                    example_queries=examples,
                )
            )

        # 🎯 Sort by yield_rate DESC, then success_rate DESC
        records.sort(
            key=lambda r: (r.yield_rate, r.success_rate),
            reverse=True,
        )

        logger.info(
            "📊 Analyzed %d templates from %d queries",
            len(records),
            len(queries),
        )
        return records[:limit]

    def get_top_templates(
        self,
        *,
        task_id: str | None = None,
        top_k: int = 10,
    ) -> list[str]:
        """Get top-K most effective template strings for prompt injection 🚀.

        Convenience method that returns just the template strings,
        suitable for direct injection into SearchAgent prompts.

        Args:
            task_id: Optional filter by task_id.
            top_k: Number of top templates to return.

        Returns:
            List of template strings ordered by effectiveness.
        """
        records = self.analyze(task_id=task_id, limit=top_k)
        return [r.template for r in records]

    # ====================================================================
    # 🔧 Internal helpers
    # ====================================================================

    def _fetch_queries(
        self,
        *,
        task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch raw query records from the index database 🔧.

        Args:
            task_id: Optional filter by task_id.

        Returns:
            List of query dicts with keys: query_text, result_count,
            cost_usd, trajectory_id, round_number.
        """
        with self._index._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT q.query_text, q.result_count, q.cost_usd, "
                    "q.trajectory_id, q.round_number "
                    "FROM queries q "
                    "JOIN trajectories t "
                    "ON q.trajectory_id = t.trajectory_id "
                    "WHERE t.task_id = ?",
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT query_text, result_count, cost_usd, "
                    "trajectory_id, round_number "
                    "FROM queries",
                ).fetchall()

            return [
                {
                    "query_text": r["query_text"],
                    "result_count": r["result_count"],
                    "cost_usd": r["cost_usd"],
                    "trajectory_id": r["trajectory_id"],
                    "round_number": r["round_number"],
                }
                for r in rows
            ]

    def _fetch_round_evidence(
        self,
        *,
        task_id: str | None = None,
    ) -> dict[tuple[str, int], int]:
        """Fetch cleaned evidence counts per (trajectory_id, round) 🔧.

        Args:
            task_id: Optional filter by task_id.

        Returns:
            Dict mapping (trajectory_id, round_number) to
            cleaned_evidence count.
        """
        with self._index._connect() as conn:
            if task_id:
                rows = conn.execute(
                    "SELECT r.trajectory_id, r.round_number, "
                    "r.cleaned_evidence "
                    "FROM rounds r "
                    "JOIN trajectories t "
                    "ON r.trajectory_id = t.trajectory_id "
                    "WHERE t.task_id = ?",
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT trajectory_id, round_number, cleaned_evidence FROM rounds",
                ).fetchall()

            return {
                (r["trajectory_id"], r["round_number"]): r["cleaned_evidence"]
                for r in rows
            }

    @staticmethod
    def _compute_yield_rate(
        group_queries: list[dict[str, Any]],
        round_evidence: dict[tuple[str, int], int],
    ) -> float:
        """Compute yield rate for a template group 📊.

        Yield rate = total cleaned evidence in rounds where the
        template was used / total number of queries in that group.

        Args:
            group_queries: List of query records for one template.
            round_evidence: Mapping of (trajectory_id, round) to
                cleaned evidence count.

        Returns:
            Yield rate as a float ratio.
        """
        if not group_queries:
            return 0.0

        # 🎯 Collect unique (trajectory_id, round_number) pairs
        round_keys: set[tuple[str, int]] = set()
        for q in group_queries:
            key = (q["trajectory_id"], q["round_number"])
            round_keys.add(key)

        # 📊 Sum cleaned evidence across those rounds
        total_evidence = sum(round_evidence.get(key, 0) for key in round_keys)

        total_queries = len(group_queries)
        return total_evidence / total_queries if total_queries > 0 else 0.0
