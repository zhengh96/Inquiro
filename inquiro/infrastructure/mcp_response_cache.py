"""In-memory LRU + TTL cache for MCP tool call responses 🗄️.

Caches responses from MCP tool calls to avoid duplicate external API
calls across rounds or tasks. Cache key is derived from a deterministic
combination of server_name, tool_name, and canonicalized args.

Architecture:
    MCPToolWrapper.execute() → MCPResponseCache.get() (cache hit → return early)
    MCPToolWrapper.execute() → ... actual MCP call ... → MCPResponseCache.put()

Thread-safe via threading.RLock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)


class MCPResponseCache:
    """In-memory LRU + TTL cache for MCP tool call responses 🗄️.

    Caches responses from MCP tool calls to avoid duplicate external
    API calls. Cache key is (server_name, tool_name, canonical_args).

    Combines two eviction policies:
    1. LRU (Least Recently Used): evicts oldest-accessed entry when
       max_size is reached.
    2. TTL (Time-To-Live): entries older than ttl_seconds are expired
       and return None on get().

    Per-server TTL overrides allow different cache durations for
    servers with different data freshness requirements (e.g., live
    clinical trial data may need a shorter TTL than static literature).

    An internal auxiliary map (key → server_name) enables efficient
    server-scoped invalidation without a secondary full-scan.

    Thread-safe via threading.RLock for all public operations.

    Attributes:
        _cache: OrderedDict mapping SHA-256 key → (response, timestamp).
        _key_server: Auxiliary map of cache key → server_name.
        _lock: Reentrant lock protecting all mutable state.
        _max_size: Maximum number of cached entries.
        _default_ttl: Default TTL in seconds.
        _server_ttl: Per-server TTL override map.
        _hits: Count of cache hits since creation.
        _misses: Count of cache misses since creation.

    Example:
        >>> cache = MCPResponseCache(
        ...     max_size=500,
        ...     default_ttl_seconds=1800,
        ...     server_ttl_overrides={"pubmed": 3600},
        ... )
        >>> cache.put("pubmed", "search", {"query": "IL-6"}, "<results>")
        >>> response = cache.get("pubmed", "search", {"query": "IL-6"})
        >>> stats = cache.stats()  # {"hits": 1, "misses": 0, ...}
    """

    def __init__(
        self,
        max_size: int = 500,
        default_ttl_seconds: int = 1800,
        server_ttl_overrides: dict[str, int] | None = None,
    ) -> None:
        """Initialize MCPResponseCache with LRU + TTL policy 🔧.

        Args:
            max_size: Maximum number of cached responses. When the cache
                reaches this limit, the least-recently-used entry is
                evicted before inserting a new one. Default 500.
            default_ttl_seconds: Default time-to-live in seconds for
                cached responses. Applies to all servers without an
                explicit override. Default 1800 (30 minutes).
            server_ttl_overrides: Optional mapping of server_name to
                per-server TTL in seconds. Takes precedence over
                default_ttl_seconds for named servers. Example::

                    {"pubmed": 3600, "clinicaltrials": 7200}
        """
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        # 🗺️ Auxiliary map for efficient server-scoped invalidation
        self._key_server: dict[str, str] = {}
        # 🔒 RLock allows reentrant calls (e.g., put() → _evict_lru())
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._server_ttl: dict[str, int] = dict(server_ttl_overrides or {})

        # 📊 Statistics counters
        self._hits: int = 0
        self._misses: int = 0

        logger.debug(
            "🗄️ MCPResponseCache created: max_size=%d, "
            "default_ttl=%ds, server_overrides=%s",
            max_size,
            default_ttl_seconds,
            list(self._server_ttl.keys()),
        )

    def get(
        self,
        server_name: str,
        tool_name: str,
        args: dict,
    ) -> str | None:
        """Look up cached response. Returns None on miss 🔍.

        On a cache hit, moves the entry to the end of the LRU order
        (marks it as recently used). TTL is checked against the
        per-server override (or default TTL).

        Args:
            server_name: MCP server name (e.g., "perplexity").
            tool_name: Tool name on the server (e.g., "perplexity_search").
            args: Tool argument dict (will be canonicalized for key lookup).

        Returns:
            Cached response string if found and not expired, or None
            on cache miss or TTL expiry.
        """
        key = self._make_key(server_name, tool_name, args)
        ttl = self._server_ttl.get(server_name, self._default_ttl)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            response, timestamp = entry

            # ⏰ Check TTL expiry
            age = time.time() - timestamp
            if age > ttl:
                # 🗑️ Expired — evict and report miss
                del self._cache[key]
                self._key_server.pop(key, None)
                self._misses += 1
                logger.debug(
                    "⏰ MCP cache expired: %s/%s (age=%.0fs > ttl=%ds)",
                    server_name,
                    tool_name,
                    age,
                    ttl,
                )
                return None

            # 🔄 Move to end to mark as recently used (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(
                "🗄️ MCP cache hit: %s/%s (age=%.0fs)",
                server_name,
                tool_name,
                age,
            )
            return response

    def put(
        self,
        server_name: str,
        tool_name: str,
        args: dict,
        response: str,
    ) -> None:
        """Store response in cache 📦.

        If an entry already exists for the key, it is updated (timestamp
        refreshed and moved to the end of LRU order). If the cache is at
        capacity, the least-recently-used entry is evicted first.

        Args:
            server_name: MCP server name.
            tool_name: Tool name on the server.
            args: Tool argument dict (will be canonicalized for key).
            response: Successful MCP response string to cache.
        """
        key = self._make_key(server_name, tool_name, args)

        with self._lock:
            # 🗑️ Remove existing entry so we can refresh timestamp + LRU
            if key in self._cache:
                del self._cache[key]
                # 🗺️ No need to touch _key_server — same key, same server

            # 🔥 Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            # 📝 Insert at end (= most recently used)
            self._cache[key] = (response, time.time())
            # 🗺️ Record server ownership for server-scoped invalidation
            self._key_server[key] = server_name

        logger.debug(
            "📦 MCP cache stored: %s/%s (cache_size=%d)",
            server_name,
            tool_name,
            len(self._cache),
        )

    def _make_key(
        self,
        server_name: str,
        tool_name: str,
        args: dict,
    ) -> str:
        """Create deterministic cache key from args 🔑.

        Normalizes args by sorting keys, lowercasing string values,
        and stripping whitespace to maximize cache hits across calls
        with equivalent semantics but different formatting.

        The key is a SHA-256 hex digest of the canonical string
        ``"<server>|<tool>|<canonical_json>"``, keeping key length
        constant regardless of argument size.

        Args:
            server_name: MCP server name.
            tool_name: Tool name on the server.
            args: Raw argument dict from the tool call.

        Returns:
            Hex-encoded SHA-256 cache key string.
        """
        canonical_args = self._canonicalize_args(args)
        raw = f"{server_name}|{tool_name}|{canonical_args}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _canonicalize_args(self, args: dict) -> str:
        """Serialize args to a normalized, order-independent form 🔧.

        Normalization rules applied recursively:
        - Dict keys: sorted alphabetically.
        - String values: stripped of leading/trailing whitespace and
          lowercased (for query strings that differ only in case).
        - Numeric and bool values: preserved as-is.
        - Nested dicts and lists: normalized recursively.

        Args:
            args: Argument dict to normalize.

        Returns:
            Compact JSON string with normalized values.
        """

        def _normalize(obj: object) -> object:
            if isinstance(obj, dict):
                return {k: _normalize(v) for k, v in sorted(obj.items())}
            if isinstance(obj, list):
                return [_normalize(item) for item in obj]
            if isinstance(obj, str):
                return obj.strip().lower()
            return obj

        return json.dumps(_normalize(args), separators=(",", ":"))

    def invalidate(self, server_name: str | None = None) -> int:
        """Remove entries. If server_name given, only that server 🗑️.

        Uses the internal key→server auxiliary map for O(n_server)
        complexity rather than scanning all cache entries.

        Args:
            server_name: If provided, only entries belonging to this
                MCP server are removed. If None, all entries are cleared.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            if server_name is None:
                count = len(self._cache)
                self._cache.clear()
                self._key_server.clear()
                logger.info("🗑️ MCP cache invalidated all %d entries", count)
                return count

            # 🔍 Find all keys belonging to this server via auxiliary map
            to_delete = [
                k for k, sname in self._key_server.items() if sname == server_name
            ]
            for k in to_delete:
                self._cache.pop(k, None)
                del self._key_server[k]

            logger.info(
                "🗑️ MCP cache invalidated %d entries for server '%s'",
                len(to_delete),
                server_name,
            )
            return len(to_delete)

    def stats(self) -> dict:
        """Return cache statistics (hits, misses, size, hit_rate) 📊.

        Returns:
            Dict with the following keys:
            - ``hits``: Total cache hits since creation.
            - ``misses``: Total cache misses since creation.
            - ``size``: Current number of cached entries.
            - ``max_size``: Configured maximum cache size.
            - ``hit_rate``: Float in [0.0, 1.0]; 0.0 if no lookups yet.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
                "hit_rate": round(hit_rate, 4),
            }

    def evict_expired(self) -> int:
        """Remove all expired entries based on per-server TTL ⏰.

        Performs a full scan of the cache. Call this periodically to
        reclaim memory from entries that are never re-accessed after
        they expire (i.e., entries not naturally evicted by TTL during
        get() calls).

        Returns:
            Number of entries removed due to TTL expiry.
        """
        with self._lock:
            now = time.time()
            expired: list[str] = []

            for key, (_, timestamp) in list(self._cache.items()):
                # 🔍 Resolve per-server TTL via auxiliary map
                sname = self._key_server.get(key)
                ttl = (
                    self._server_ttl.get(sname, self._default_ttl)
                    if sname is not None
                    else self._default_ttl
                )
                if (now - timestamp) > ttl:
                    expired.append(key)

            for key in expired:
                del self._cache[key]
                self._key_server.pop(key, None)

            if expired:
                logger.debug(
                    "⏰ MCP cache evicted %d expired entries",
                    len(expired),
                )
            return len(expired)

    # -- Internal helpers ---------------------------------------------------

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry (internal) 🔥.

        Must be called with _lock already held. Removes the first
        entry in the OrderedDict (the entry least recently accessed).
        """
        if self._cache:
            evicted_key, _ = self._cache.popitem(last=False)
            self._key_server.pop(evicted_key, None)
            logger.debug("🔥 MCP cache LRU evicted key=%s...", evicted_key[:8])
