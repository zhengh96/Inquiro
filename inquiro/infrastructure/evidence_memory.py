"""Inquiro EvidenceMemory -- cross-task evidence reuse within a session 🧠.

Provides a lightweight, session-scoped evidence cache that allows
subsequent research tasks to query and reuse evidence collected by
prior tasks, reducing duplicate MCP searches.

Thread-safe for use across concurrent tasks.

Usage::

    memory = EvidenceMemory(max_capacity=500)

    # Store evidence from completed task
    memory.store("task_001", [
        {"id": "E1", "url": "https://...", "summary": "..."},
    ])

    # Query from a new task
    relevant = memory.query(keywords=["market", "size"], limit=5)
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StoredEvidence(BaseModel):
    """Evidence item stored in memory with metadata 📋.

    Attributes:
        evidence_id: Original evidence ID (e.g., "E1").
        source_task_id: Task that originally collected this evidence.
        source: Data source name (e.g., "perplexity", "pubmed").
        url: URL of the evidence (used for dedup).
        summary: Evidence summary text.
        keywords: Extracted keywords for search matching.
        quality_label: Quality assessment (high/medium/low).
    """

    evidence_id: str = Field(description="Original evidence ID")
    source_task_id: str = Field(description="Task that collected this evidence")
    source: str = Field(default="", description="Data source name")
    url: str = Field(default="", description="URL for dedup")
    summary: str = Field(default="", description="Evidence summary")
    keywords: list[str] = Field(default_factory=list, description="Search keywords")
    quality_label: str = Field(default="medium", description="Quality tier")


class EvidenceMemory:
    """Session-scoped evidence memory for cross-task reuse 🧠.

    Stores evidence from completed research tasks and allows
    subsequent tasks to query for relevant prior evidence.

    Thread-safe: all mutable state protected by ``threading.Lock``.

    Dedup strategy:
        - URL-based: same URL -> same evidence, skip storage.
        - Summary-based: evidence with identical summary[:200] -> skip.

    Example::

        memory = EvidenceMemory(max_capacity=500)

        # Store evidence from completed task
        memory.store("task_001", [evidence1, evidence2])

        # Query from a new task
        relevant = memory.query(keywords=["market", "size"], limit=5)
    """

    DEFAULT_MAX_CAPACITY: int = 1000

    def __init__(self, max_capacity: int = DEFAULT_MAX_CAPACITY) -> None:
        """Initialize EvidenceMemory 🔧.

        Args:
            max_capacity: Maximum number of evidence items to store.
                When exceeded, oldest items are evicted (FIFO).
        """
        self._lock = threading.Lock()
        self._storage: list[StoredEvidence] = []
        self._url_index: set[str] = set()
        self._summary_index: set[str] = set()
        self._max_capacity = max(1, max_capacity)

    @property
    def size(self) -> int:
        """Current number of stored evidence items 📊."""
        with self._lock:
            return len(self._storage)

    def store(
        self,
        task_id: str,
        evidence_list: list[dict[str, Any]],
    ) -> int:
        """Store evidence from a completed task 💾.

        Deduplicates by URL and summary prefix. Evicts oldest items
        when capacity is exceeded.

        Args:
            task_id: ID of the task that collected this evidence.
            evidence_list: List of evidence dicts with fields:
                id, source, url, summary, quality_label, etc.

        Returns:
            Number of new evidence items actually stored (after dedup).
        """
        stored_count = 0
        with self._lock:
            for ev in evidence_list:
                url = ev.get("url", "")
                summary = ev.get("summary", "")
                summary_key = summary[:200]

                # 🔍 URL-based dedup
                if url and url in self._url_index:
                    continue

                # 🔍 Summary-based dedup
                if summary_key and summary_key in self._summary_index:
                    continue

                # 📝 Extract keywords from summary
                keywords = self._extract_keywords(summary)

                item = StoredEvidence(
                    evidence_id=ev.get("id", ""),
                    source_task_id=task_id,
                    source=ev.get("source", ""),
                    url=url,
                    summary=summary,
                    keywords=keywords,
                    quality_label=ev.get("quality_label", "medium"),
                )

                self._storage.append(item)
                if url:
                    self._url_index.add(url)
                if summary_key:
                    self._summary_index.add(summary_key)
                stored_count += 1

            # 🧹 Evict oldest if over capacity
            if len(self._storage) > self._max_capacity:
                evict_count = len(self._storage) - self._max_capacity
                evicted = self._storage[:evict_count]
                self._storage = self._storage[evict_count:]
                # ✨ Clean up indexes for evicted items
                for item in evicted:
                    if item.url:
                        self._url_index.discard(item.url)
                    key = item.summary[:200]
                    if key:
                        self._summary_index.discard(key)

        logger.info(
            "💾 Stored %d/%d evidence items from task %s (total: %d)",
            stored_count,
            len(evidence_list),
            task_id,
            self.size,
        )
        return stored_count

    def query(
        self,
        keywords: list[str],
        limit: int = 10,
        exclude_task_id: str | None = None,
    ) -> list[StoredEvidence]:
        """Query stored evidence by keyword matching 🔍.

        Returns evidence items whose keywords or summary contain
        any of the provided keywords. Results ordered by relevance
        (number of keyword matches).

        Args:
            keywords: Search keywords to match against.
            limit: Maximum number of results to return.
            exclude_task_id: Exclude evidence from this task (to
                avoid circular reuse of own evidence).

        Returns:
            List of matching StoredEvidence items, sorted by
            relevance (highest match count first).
        """
        if not keywords:
            return []

        lower_keywords = [k.lower() for k in keywords]

        with self._lock:
            candidates: list[tuple[int, StoredEvidence]] = []

            for item in self._storage:
                if exclude_task_id and item.source_task_id == exclude_task_id:
                    continue

                # 🎯 Score by keyword matches
                score = 0
                item_text = " ".join(item.keywords + [item.summary.lower()])
                for kw in lower_keywords:
                    if kw in item_text:
                        score += 1

                if score > 0:
                    candidates.append((score, item))

        # 📊 Sort by relevance (highest score first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in candidates[:limit]]

    def get_all(self) -> list[StoredEvidence]:
        """Get all stored evidence items 📋.

        Returns:
            Copy of all stored evidence items.
        """
        with self._lock:
            return list(self._storage)

    def clear(self) -> None:
        """Clear all stored evidence 🧹."""
        with self._lock:
            self._storage.clear()
            self._url_index.clear()
            self._summary_index.clear()

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract simple keywords from text 🔑.

        Splits on whitespace, lowercases, and filters short words
        and common stop words.

        Args:
            text: Input text to extract keywords from.

        Returns:
            List of lowercase keyword strings (length >= 4).
        """
        if not text:
            return []
        words = text.lower().split()
        # ✨ Filter out short words and common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "are",
            "was",
            "were",
            "been",
            "have",
            "has",
            "had",
            "not",
            "but",
            "can",
            "will",
            "its",
            "also",
        }
        return [w for w in words if len(w) >= 4 and w not in stop_words]
