"""SharedEvidencePool — thread-safe cross-task evidence reuse 🔄.

Enables evidence sharing across sub-items within the same evaluation
group (e.g., a dimension in TargetMaster).  Uses content-hash
deduplication to avoid redundant evidence.

Architecture position::

    TargetMaster Orchestrator
        └── creates ONE SharedEvidencePool per evaluation group
                ├── passes to DiscoveryLoop(sub_item_A)
                ├── passes to DiscoveryLoop(sub_item_B)
                └── passes to DiscoveryLoop(sub_item_C)

Key design decisions:
    - Domain-agnostic: no pharma/dimension/sub-item terminology.
    - Thread-safe: all mutations guarded by ``threading.Lock``.
    - Content-hash dedup: URL + summary hash prevents duplicates.
    - Keyword-based relevance: lightweight TF matching for retrieval.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from inquiro.core.canonical_hash import canonical_evidence_hash
from inquiro.core.types import Evidence

logger = logging.getLogger(__name__)


class SharedEvidencePool:
    """Thread-safe shared evidence pool for cross-task evidence reuse 🔄.

    Enables evidence sharing across multiple evaluation tasks.
    Uses content-hash deduplication to avoid redundant evidence.

    Attributes:
        _lock: Threading lock for safe concurrent access.
        _evidence: Internal storage mapping content_hash -> Evidence.
        _dedup_count: Counter for rejected duplicate additions.
    """

    def __init__(self) -> None:
        """Initialize empty evidence pool 🔧."""
        self._lock = threading.Lock()
        self._evidence: dict[str, Evidence] = {}
        self._dedup_count: int = 0
        self._source_counts: dict[str, int] = {}

        logger.info("🔄 SharedEvidencePool initialized (empty)")

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    def add(self, evidence: list[Evidence]) -> int:
        """Add evidence items to the pool, deduplicating by content hash 📥.

        Each evidence item is hashed by URL + summary.  Items with
        matching hashes are silently skipped (dedup).

        Args:
            evidence: List of Evidence items to add.

        Returns:
            Number of newly added (non-duplicate) items.
        """
        if not evidence:
            return 0

        newly_added = 0
        with self._lock:
            for item in evidence:
                content_hash = self._compute_hash(item)
                if content_hash not in self._evidence:
                    self._evidence[content_hash] = item
                    newly_added += 1

                    # 📊 Track source distribution
                    source = item.source or "unknown"
                    self._source_counts[source] = self._source_counts.get(source, 0) + 1
                else:
                    self._dedup_count += 1

        if newly_added > 0:
            logger.info(
                "📥 SharedEvidencePool: added %d new items "
                "(skipped %d duplicates), total=%d",
                newly_added,
                len(evidence) - newly_added,
                len(self._evidence),
            )

        return newly_added

    def get_relevant(
        self,
        checklist_items: list[str],
        limit: int = 50,
    ) -> list[Evidence]:
        """Retrieve evidence relevant to given checklist items 🎯.

        Uses simple keyword matching for relevance scoring.
        Each evidence summary is scored against all checklist item
        descriptions.  Returns the top ``limit`` items by score.

        Args:
            checklist_items: Checklist item description strings
                to match against.
            limit: Maximum number of evidence items to return.

        Returns:
            List of Evidence items sorted by relevance (best first),
            up to ``limit`` items.
        """
        if not checklist_items:
            return self.get_all()[:limit]

        with self._lock:
            all_items = list(self._evidence.values())

        if not all_items:
            return []

        # 🔍 Extract keywords from checklist items
        keywords = self._extract_keywords(checklist_items)

        if not keywords:
            return all_items[:limit]

        # 📊 Score each evidence item by keyword overlap
        scored: list[tuple[float, Evidence]] = []
        for item in all_items:
            score = self._relevance_score(item, keywords)
            scored.append((score, item))

        # 🏆 Sort by descending score, then take top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def get_all(self) -> list[Evidence]:
        """Get all evidence items in the pool 📋.

        Returns:
            List of all stored Evidence items (unordered).
        """
        with self._lock:
            return list(self._evidence.values())

    @property
    def size(self) -> int:
        """Current number of unique evidence items in the pool 📊."""
        with self._lock:
            return len(self._evidence)

    def get_stats(self) -> dict[str, Any]:
        """Return pool statistics 📊.

        Returns:
            Dictionary with:
                - total: Current evidence count.
                - dedup_rejected: Total duplicates rejected.
                - by_source: Evidence count per source.
        """
        with self._lock:
            return {
                "total": len(self._evidence),
                "dedup_rejected": self._dedup_count,
                "by_source": dict(self._source_counts),
            }

    # ====================================================================
    # 🔧 Internal methods
    # ====================================================================

    @staticmethod
    def _compute_hash(evidence: Evidence) -> str:
        """Compute canonical content hash for deduplication 🔑.

        Delegates to ``canonical_evidence_hash`` for a unified
        hash algorithm across the codebase.

        Args:
            evidence: Evidence item to hash.

        Returns:
            64-character hex digest string.
        """
        return canonical_evidence_hash(evidence.url, evidence.summary or "")

    @staticmethod
    def _extract_keywords(descriptions: list[str]) -> set[str]:
        """Extract meaningful keywords from description texts 🔍.

        Tokenizes, lowercases, and filters short/common words.

        Args:
            descriptions: Text descriptions to extract keywords from.

        Returns:
            Set of lowercase keyword strings.
        """
        # ⚠️ Simple stop words to filter out
        stop_words = frozenset(
            {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "is",
                "it",
                "as",
                "be",
                "this",
                "that",
                "are",
                "was",
                "were",
                "has",
                "have",
                "had",
                "not",
                "no",
                "if",
                "can",
                "will",
                "do",
                "does",
                "did",
                "all",
                "any",
                "each",
                "been",
            }
        )

        keywords: set[str] = set()
        for desc in descriptions:
            # 🔧 Tokenize: split on non-alphanumeric chars
            tokens = re.findall(r"[a-zA-Z0-9]+", desc.lower())
            for token in tokens:
                # 🔍 Skip short tokens and stop words
                if len(token) >= 3 and token not in stop_words:
                    keywords.add(token)

        return keywords

    @staticmethod
    def _relevance_score(
        evidence: Evidence,
        keywords: set[str],
    ) -> float:
        """Score evidence relevance against keyword set 📊.

        Simple keyword overlap count.  Checks both the summary and
        the query fields for keyword matches.

        Args:
            evidence: Evidence item to score.
            keywords: Set of lowercase keywords.

        Returns:
            Relevance score (higher = more relevant).
        """
        # 🔧 Combine searchable text fields
        text = (f"{evidence.summary} {evidence.query}").lower()

        score = 0.0
        for kw in keywords:
            if kw in text:
                score += 1.0

        return score
