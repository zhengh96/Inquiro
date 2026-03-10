"""LRU + TTL cache for completed evaluation results 🗄️.

Provides bounded cache with automatic eviction to prevent memory
growth in long-running service deployments. Combines LRU (Least
Recently Used) eviction with TTL (Time-To-Live) expiry.

Architecture:
    EvalTaskRunner → CompletedResultsCache (LRU + TTL)
        → OrderedDict (maintains access order for LRU)
        → Timestamp tracking (for TTL expiry)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


class CompletedResultsCache:
    """Bounded cache with LRU eviction and TTL expiry 📦.

    Thread-safe cache that automatically limits memory usage by:
    1. Evicting least-recently-used entries when max_size is reached (LRU).
    2. Expiring entries older than ttl_seconds (TTL).

    Uses OrderedDict to maintain insertion/access order for efficient
    LRU eviction, and stores timestamps for TTL expiry checks.

    Attributes:
        _cache: OrderedDict mapping task_id to (result, timestamp) tuples.
        _lock: Reentrant lock protecting concurrent access.
        _max_size: Maximum number of cached results (evict LRU when full).
        _ttl_seconds: Time-to-live in seconds (expire old entries).

    Example:
        >>> cache = CompletedResultsCache(max_size=1000, ttl_seconds=3600)
        >>> cache.put("task-1", {"status": "completed", "result": {...}})
        >>> result = cache.get("task-1")  # Returns result if not expired
        >>> cache.evict_expired()  # Clean up expired entries
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600) -> None:
        """Initialize bounded cache with LRU + TTL 🏗️.

        Args:
            max_size: Maximum number of cached results. When exceeded,
                least-recently-used entries are evicted. Default 1000.
            ttl_seconds: Time-to-live in seconds. Entries older than
                this are considered expired and return None on get().
                Default 3600 (1 hour). Set to 0 to disable TTL.
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        # 🔒 RLock allows nested calls (e.g., put() calling _evict_lru())
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def put(self, task_id: str, result: Any) -> None:
        """Store result with current timestamp 📝.

        Thread-safe: acquires _lock for the duration of the operation.
        If cache is at max_size, evicts the least-recently-used entry
        before inserting the new one.

        Args:
            task_id: Unique task identifier (cache key).
            result: Result object to cache (typically dict with
                "status", "result", "error" fields).
        """
        with self._lock:
            # 🗑️ Remove existing entry if present (to update timestamp)
            if task_id in self._cache:
                del self._cache[task_id]

            # 🔥 Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            # 📝 Insert with current timestamp
            timestamp = time.time()
            self._cache[task_id] = (result, timestamp)

    def get(self, task_id: str) -> Any | None:
        """Retrieve cached result if not expired 🔍.

        Thread-safe: acquires _lock for the duration of the operation.
        On access, moves the entry to the end of the OrderedDict (LRU).

        Args:
            task_id: Unique task identifier to look up.

        Returns:
            Cached result object if found and not expired.
            None if task not found or entry has exceeded TTL.
        """
        with self._lock:
            entry = self._cache.get(task_id)
            if entry is None:
                return None

            result, timestamp = entry

            # ⏰ Check TTL expiry (if TTL enabled)
            if self._ttl_seconds > 0:
                age = time.time() - timestamp
                if age > self._ttl_seconds:
                    # 🗑️ Expired — remove and return None
                    del self._cache[task_id]
                    return None

            # 🔄 Move to end (mark as recently used for LRU)
            self._cache.move_to_end(task_id)
            return result

    def remove(self, task_id: str) -> bool:
        """Remove entry from cache 🗑️.

        Thread-safe: acquires _lock for the deletion.

        Args:
            task_id: Unique task identifier to remove.

        Returns:
            True if entry existed and was removed.
            False if entry was not found.
        """
        with self._lock:
            if task_id in self._cache:
                del self._cache[task_id]
                return True
            return False

    def size(self) -> int:
        """Return the current number of cached entries 📊.

        Thread-safe: acquires _lock for the read.

        Returns:
            Count of entries currently in cache.
        """
        with self._lock:
            return len(self._cache)

    def evict_expired(self) -> int:
        """Remove all expired entries based on TTL ⏰.

        Thread-safe: acquires _lock for the scan and deletions.
        Call this periodically (e.g., every 5 minutes) to prevent
        memory buildup from expired entries that aren't being accessed.

        Returns:
            Number of entries evicted due to TTL expiry.
        """
        if self._ttl_seconds <= 0:
            # 🚫 TTL disabled, nothing to evict
            return 0

        with self._lock:
            now = time.time()
            expired_keys = []

            for task_id, (_, timestamp) in self._cache.items():
                age = now - timestamp
                if age > self._ttl_seconds:
                    expired_keys.append(task_id)

            # 🗑️ Remove expired entries
            for task_id in expired_keys:
                del self._cache[task_id]

            return len(expired_keys)

    def clear(self) -> None:
        """Remove all entries from the cache 🧹.

        Thread-safe: acquires _lock for the clear operation.
        Useful for testing or administrative cleanup.
        """
        with self._lock:
            self._cache.clear()

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry (internal) 🔥.

        Must be called with _lock already held. Removes the first
        entry from the OrderedDict (oldest in access order).
        """
        # ⚠️ popitem(last=False) removes the first (oldest) entry
        if self._cache:
            evicted_task_id, _ = self._cache.popitem(last=False)
            # 📝 Optional: log eviction for monitoring
            # logging.debug("🔥 LRU evicted task: %s", evicted_task_id)

    # ====================================================================
    # 🎭 Dict-like interface for partial compatibility (optional)
    # ====================================================================

    def __contains__(self, task_id: str) -> bool:
        """Support 'in' operator with TTL-aware check ✅.

        Checks presence AND TTL expiry for consistent semantics
        with get(). Does NOT affect LRU order.

        Args:
            task_id: Task identifier to check.

        Returns:
            True if task is in cache and not expired.
        """
        with self._lock:
            if task_id not in self._cache:
                return False
            if self._ttl_seconds > 0:
                _, timestamp = self._cache[task_id]
                if (time.monotonic() - timestamp) > self._ttl_seconds:
                    return False
            return True

    def __len__(self) -> int:
        """Support len(cache) for size queries 📊.

        Returns:
            Number of entries currently in cache.
        """
        return self.size()
