"""Tests for MCPResponseCache — LRU + TTL in-memory cache 🧪.

Tests cover:
- Basic get/put/get cycle (cache hit)
- Cold cache miss
- TTL expiry
- LRU eviction when max_size is reached
- Cache key normalization (whitespace/case insensitivity)
- Per-server TTL overrides
- Server-scoped invalidation
- Statistics counters (hits, misses, hit_rate)
- Thread safety under concurrent access
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from inquiro.infrastructure.mcp_response_cache import MCPResponseCache


# ============================================================================
# 🔧 Fixtures
# ============================================================================


@pytest.fixture()
def cache() -> MCPResponseCache:
    """Return a fresh MCPResponseCache with default settings."""
    return MCPResponseCache(max_size=10, default_ttl_seconds=60)


# ============================================================================
# 🟢 Basic hit / miss
# ============================================================================


class TestCacheHitAndMiss:
    """Tests for basic get/put cache operations 🟢."""

    def test_cache_hit_returns_stored(self, cache: MCPResponseCache) -> None:
        """Stored response is returned on subsequent get 🟢."""
        # Arrange
        response = "Search results for IL-6"

        # Act
        cache.put("pubmed", "search", {"query": "IL-6"}, response)
        result = cache.get("pubmed", "search", {"query": "IL-6"})

        # Assert
        assert result == response

    def test_cache_miss_returns_none(self, cache: MCPResponseCache) -> None:
        """Cold cache returns None for unknown keys 🔍."""
        # Act
        result = cache.get("pubmed", "search", {"query": "unknown"})

        # Assert
        assert result is None

    def test_cache_different_servers_are_independent(
        self, cache: MCPResponseCache
    ) -> None:
        """Entries for different servers do not collide ✅."""
        # Arrange
        cache.put("pubmed", "search", {"query": "IL-6"}, "pubmed_result")
        cache.put("perplexity", "search", {"query": "IL-6"}, "perplexity_result")

        # Act
        r1 = cache.get("pubmed", "search", {"query": "IL-6"})
        r2 = cache.get("perplexity", "search", {"query": "IL-6"})

        # Assert
        assert r1 == "pubmed_result"
        assert r2 == "perplexity_result"

    def test_cache_different_tools_are_independent(
        self, cache: MCPResponseCache
    ) -> None:
        """Entries for different tools on same server do not collide ✅."""
        # Arrange
        cache.put("pubmed", "search", {"query": "q"}, "result_search")
        cache.put("pubmed", "fetch", {"query": "q"}, "result_fetch")

        # Act
        r1 = cache.get("pubmed", "search", {"query": "q"})
        r2 = cache.get("pubmed", "fetch", {"query": "q"})

        # Assert
        assert r1 == "result_search"
        assert r2 == "result_fetch"

    def test_put_updates_existing_entry(self, cache: MCPResponseCache) -> None:
        """Subsequent put() with same key overwrites the stored value 🔄."""
        # Arrange
        cache.put("pubmed", "search", {"query": "q"}, "old_result")

        # Act
        cache.put("pubmed", "search", {"query": "q"}, "new_result")
        result = cache.get("pubmed", "search", {"query": "q"})

        # Assert
        assert result == "new_result"


# ============================================================================
# ⏰ TTL expiry
# ============================================================================


class TestCacheTTLExpiry:
    """Tests for TTL-based expiry behavior ⏰."""

    def test_cache_ttl_expiry(self) -> None:
        """Entry is expired and returns None after TTL elapses ⏰."""
        # Arrange
        cache = MCPResponseCache(max_size=10, default_ttl_seconds=1)
        cache.put("pubmed", "search", {"query": "q"}, "result")

        # Act — advance time past TTL
        with patch("inquiro.infrastructure.mcp_response_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2  # 2s past TTL
            result = cache.get("pubmed", "search", {"query": "q"})

        # Assert
        assert result is None

    def test_cache_not_expired_before_ttl(self) -> None:
        """Entry is still available just before TTL elapses ✅."""
        # Arrange
        cache = MCPResponseCache(max_size=10, default_ttl_seconds=60)
        cache.put("pubmed", "search", {"query": "q"}, "result")

        # Act — within TTL window
        result = cache.get("pubmed", "search", {"query": "q"})

        # Assert
        assert result == "result"

    def test_cache_server_specific_ttl(self) -> None:
        """Per-server TTL overrides the default TTL 🕐."""
        # Arrange — pubmed gets 10s TTL, perplexity uses the 1s default
        cache = MCPResponseCache(
            max_size=10,
            default_ttl_seconds=1,
            server_ttl_overrides={"pubmed": 10},
        )
        cache.put("pubmed", "search", {"query": "q"}, "pubmed_result")
        cache.put("perplexity", "search", {"query": "q"}, "perplexity_result")

        # Simulate 2 seconds passing (beyond default 1s TTL)
        original_time = time.time

        def fake_time() -> float:
            return original_time() + 2.0

        with patch("inquiro.infrastructure.mcp_response_cache.time") as mock_time:
            mock_time.time.return_value = original_time() + 2.0

            # pubmed (10s TTL) should still be alive
            pubmed_result = cache.get("pubmed", "search", {"query": "q"})
            # perplexity (1s default TTL) should be expired
            perplexity_result = cache.get("perplexity", "search", {"query": "q"})

        # Assert
        assert pubmed_result == "pubmed_result"
        assert perplexity_result is None

    def test_evict_expired_removes_stale_entries(self) -> None:
        """evict_expired() cleans up entries not accessed after TTL ⏰."""
        # Arrange
        cache = MCPResponseCache(max_size=10, default_ttl_seconds=1)
        cache.put("pubmed", "search", {"query": "q1"}, "r1")
        cache.put("pubmed", "search", {"query": "q2"}, "r2")

        with patch("inquiro.infrastructure.mcp_response_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2.0
            evicted = cache.evict_expired()

        # Assert
        assert evicted == 2
        assert cache.stats()["size"] == 0


# ============================================================================
# 🔥 LRU eviction
# ============================================================================


class TestCacheLRUEviction:
    """Tests for LRU eviction when max_size is reached 🔥."""

    def test_cache_lru_eviction(self) -> None:
        """Oldest entry is evicted when max_size is reached 🔥."""
        # Arrange — cache with max 3 entries
        cache = MCPResponseCache(max_size=3, default_ttl_seconds=3600)
        cache.put("srv", "tool", {"q": "a"}, "result_a")
        cache.put("srv", "tool", {"q": "b"}, "result_b")
        cache.put("srv", "tool", {"q": "c"}, "result_c")

        # Act — inserting a 4th entry should evict the LRU (first inserted)
        cache.put("srv", "tool", {"q": "d"}, "result_d")

        # Assert — "a" was LRU and should be evicted
        assert cache.get("srv", "tool", {"q": "a"}) is None
        assert cache.get("srv", "tool", {"q": "b"}) == "result_b"
        assert cache.get("srv", "tool", {"q": "c"}) == "result_c"
        assert cache.get("srv", "tool", {"q": "d"}) == "result_d"

    def test_get_refreshes_lru_order(self) -> None:
        """Accessing an entry moves it to the front of LRU order 🔄."""
        # Arrange — 3-entry cache
        cache = MCPResponseCache(max_size=3, default_ttl_seconds=3600)
        cache.put("srv", "tool", {"q": "a"}, "result_a")
        cache.put("srv", "tool", {"q": "b"}, "result_b")
        cache.put("srv", "tool", {"q": "c"}, "result_c")

        # Act — access "a" to make it recently used
        cache.get("srv", "tool", {"q": "a"})

        # Insert 4th entry — "b" should be evicted (oldest un-accessed)
        cache.put("srv", "tool", {"q": "d"}, "result_d")

        # Assert
        assert cache.get("srv", "tool", {"q": "a"}) == "result_a"
        assert cache.get("srv", "tool", {"q": "b"}) is None  # evicted
        assert cache.get("srv", "tool", {"q": "c"}) == "result_c"
        assert cache.get("srv", "tool", {"q": "d"}) == "result_d"

    def test_cache_size_stays_within_max(self) -> None:
        """Cache size never exceeds max_size 📏."""
        # Arrange
        cache = MCPResponseCache(max_size=5, default_ttl_seconds=3600)

        # Act — insert 20 entries
        for i in range(20):
            cache.put("srv", "tool", {"q": str(i)}, f"result_{i}")

        # Assert
        assert cache.stats()["size"] <= 5


# ============================================================================
# 🔑 Key normalization
# ============================================================================


class TestCacheKeyNormalization:
    """Tests for deterministic key normalization behavior 🔑."""

    def test_cache_key_normalization_whitespace(self, cache: MCPResponseCache) -> None:
        """Leading/trailing whitespace in values maps to same key ✅."""
        # Arrange
        cache.put("srv", "tool", {"query": "  IL-6  "}, "result")

        # Act — different whitespace, same canonical value
        result = cache.get("srv", "tool", {"query": "IL-6"})

        # Assert
        assert result == "result"

    def test_cache_key_normalization_case(self, cache: MCPResponseCache) -> None:
        """Different string casing maps to same key ✅."""
        # Arrange
        cache.put("srv", "tool", {"query": "IL-6"}, "result")

        # Act — uppercase variant
        result = cache.get("srv", "tool", {"query": "il-6"})

        # Assert
        assert result == "result"

    def test_cache_key_normalization_dict_ordering(
        self, cache: MCPResponseCache
    ) -> None:
        """Different dict key ordering maps to same cache key ✅."""
        # Arrange
        cache.put("srv", "tool", {"b": "2", "a": "1"}, "result")

        # Act — reversed ordering
        result = cache.get("srv", "tool", {"a": "1", "b": "2"})

        # Assert
        assert result == "result"

    def test_cache_key_normalization_combined(self, cache: MCPResponseCache) -> None:
        """Combination of whitespace, case, and ordering normalizes ✅."""
        # Arrange
        cache.put(
            "srv",
            "tool",
            {"b": "VALUE B", "a": "  Value A  "},
            "result",
        )

        # Act — normalized form
        result = cache.get(
            "srv",
            "tool",
            {"a": "value a", "b": "value b"},
        )

        # Assert
        assert result == "result"

    def test_different_args_produce_different_keys(
        self, cache: MCPResponseCache
    ) -> None:
        """Genuinely different args do NOT collide 🚫."""
        # Arrange
        cache.put("srv", "tool", {"query": "IL-6"}, "result_il6")
        cache.put("srv", "tool", {"query": "TNF-α"}, "result_tnf")

        # Act
        r1 = cache.get("srv", "tool", {"query": "IL-6"})
        r2 = cache.get("srv", "tool", {"query": "TNF-α"})

        # Assert
        assert r1 == "result_il6"
        assert r2 == "result_tnf"


# ============================================================================
# 🗑️ Invalidation
# ============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation behavior 🗑️."""

    def test_cache_invalidate_server(self, cache: MCPResponseCache) -> None:
        """invalidate(server_name) removes only that server's entries 🗑️."""
        # Arrange
        cache.put("pubmed", "search", {"q": "1"}, "pubmed_r1")
        cache.put("pubmed", "search", {"q": "2"}, "pubmed_r2")
        cache.put("perplexity", "search", {"q": "1"}, "perplexity_r1")

        # Act
        removed = cache.invalidate("pubmed")

        # Assert
        assert removed == 2
        assert cache.get("pubmed", "search", {"q": "1"}) is None
        assert cache.get("pubmed", "search", {"q": "2"}) is None
        # perplexity entries survive
        assert cache.get("perplexity", "search", {"q": "1"}) == "perplexity_r1"

    def test_cache_invalidate_all(self, cache: MCPResponseCache) -> None:
        """invalidate(None) clears all entries 🗑️."""
        # Arrange
        cache.put("pubmed", "search", {"q": "1"}, "r1")
        cache.put("perplexity", "search", {"q": "1"}, "r2")

        # Act
        removed = cache.invalidate()

        # Assert
        assert removed == 2
        assert cache.stats()["size"] == 0

    def test_cache_invalidate_nonexistent_server(self, cache: MCPResponseCache) -> None:
        """invalidate() for an unknown server returns 0 and does not error ✅."""
        # Arrange
        cache.put("pubmed", "search", {"q": "1"}, "r1")

        # Act
        removed = cache.invalidate("no_such_server")

        # Assert
        assert removed == 0
        assert cache.stats()["size"] == 1


# ============================================================================
# 📊 Statistics
# ============================================================================


class TestCacheStats:
    """Tests for cache statistics counters 📊."""

    def test_cache_stats_initial_state(self, cache: MCPResponseCache) -> None:
        """Fresh cache has zero hits and misses 📊."""
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cache_stats_hit_and_miss(self, cache: MCPResponseCache) -> None:
        """Hit and miss counters are accurate 📊."""
        # Arrange
        cache.put("pubmed", "search", {"q": "q"}, "result")

        # Act — 1 hit, 1 miss
        cache.get("pubmed", "search", {"q": "q"})
        cache.get("pubmed", "search", {"q": "other"})

        # Assert
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_stats_size_reflects_entries(self, cache: MCPResponseCache) -> None:
        """stats()['size'] matches actual number of cached entries 📏."""
        # Act
        cache.put("s", "t", {"q": "1"}, "r1")
        cache.put("s", "t", {"q": "2"}, "r2")
        cache.put("s", "t", {"q": "3"}, "r3")

        # Assert
        assert cache.stats()["size"] == 3

    def test_cache_stats_ttl_expiry_counts_as_miss(self) -> None:
        """Expired entry lookup increments misses counter ⏰."""
        # Arrange
        cache = MCPResponseCache(max_size=10, default_ttl_seconds=1)
        cache.put("s", "t", {"q": "q"}, "result")

        with patch("inquiro.infrastructure.mcp_response_cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2.0
            cache.get("s", "t", {"q": "q"})

        # Assert
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_stats_max_size_reported(self, cache: MCPResponseCache) -> None:
        """stats()['max_size'] reflects the configured maximum 📋."""
        stats = cache.stats()
        assert stats["max_size"] == 10


# ============================================================================
# 🔒 Thread safety
# ============================================================================


class TestCacheThreadSafety:
    """Tests for concurrent access thread safety 🔒."""

    def test_cache_thread_safety(self) -> None:
        """Concurrent get/put from multiple threads does not corrupt state 🔒."""
        # Arrange
        cache = MCPResponseCache(max_size=100, default_ttl_seconds=3600)
        errors: list[Exception] = []
        iterations = 50

        def writer(thread_id: int) -> None:
            try:
                for i in range(iterations):
                    cache.put(
                        "srv",
                        "tool",
                        {"q": f"thread-{thread_id}-{i}"},
                        f"result-{thread_id}-{i}",
                    )
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        def reader(thread_id: int) -> None:
            try:
                for i in range(iterations):
                    cache.get("srv", "tool", {"q": f"thread-{thread_id}-{i}"})
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(exc)

        # Act — 4 writer threads + 4 reader threads concurrently
        threads = []
        for t in range(4):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))

        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10)

        # Assert
        assert errors == [], f"Thread errors: {errors}"
        # Cache should remain consistent (size <= max_size)
        assert cache.stats()["size"] <= 100

    def test_cache_concurrent_invalidate_and_put(self) -> None:
        """Concurrent invalidate() and put() do not cause deadlocks 🔒."""
        # Arrange
        cache = MCPResponseCache(max_size=200, default_ttl_seconds=3600)
        errors: list[Exception] = []
        stop_event = threading.Event()

        def continuous_writer() -> None:
            i = 0
            while not stop_event.is_set():
                try:
                    cache.put("srv", "tool", {"q": str(i)}, "result")
                    i += 1
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(exc)
                    break

        def continuous_invalidator() -> None:
            while not stop_event.is_set():
                try:
                    cache.invalidate("srv")
                except Exception as exc:  # pylint: disable=broad-except
                    errors.append(exc)
                    break

        # Act
        writer_thread = threading.Thread(target=continuous_writer)
        invalidator_thread = threading.Thread(target=continuous_invalidator)
        writer_thread.start()
        invalidator_thread.start()

        time.sleep(0.2)
        stop_event.set()

        writer_thread.join(timeout=5)
        invalidator_thread.join(timeout=5)

        # Assert
        assert errors == [], f"Thread errors: {errors}"
