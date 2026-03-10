"""Tool effectiveness tracking for recommendation and analysis 📊.

Records each tool call outcome (evidence yield, latency) and computes
per-tool success rates. Provides ranked recommendations so the agent
(or orchestrator) can prefer tools with higher historical effectiveness.

Thread-safe via ``threading.RLock`` — safe for concurrent agent usage.

Usage::

    tracker = ToolEffectivenessTracker()
    tracker.record(
        tool_name="mcp__bohrium__search",
        yielded_evidence=True,
        latency_ms=142,
        domain="clinical",
    )
    recs = tracker.get_recommendations(limit=5, min_calls=3)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class _ToolCallRecord:
    """Single tool call outcome 📝.

    Attributes:
        timestamp: ISO 8601 timestamp of the call.
        yielded_evidence: Whether the call produced useful evidence.
        latency_ms: Call duration in milliseconds.
        domain: Optional domain tag for domain-specific filtering.
    """

    timestamp: str
    yielded_evidence: bool
    latency_ms: float
    domain: str | None = None


class ToolEffectivenessTracker:
    """Thread-safe tracker for tool call outcomes 📊.

    Uses ``threading.RLock`` (reentrant) so methods can safely
    call each other without deadlocking.

    Accumulates per-tool call records and computes success rates,
    average latency, and ranked recommendations.

    Attributes:
        _records: Internal mapping of tool_name to call records.
        _lock: Threading lock for thread-safe access.
    """

    def __init__(self) -> None:
        """Initialize tracker with empty state 🔧."""
        self._records: dict[str, list[_ToolCallRecord]] = {}
        self._lock = threading.RLock()  # 🔒 Reentrant to avoid deadlock

    def record(
        self,
        tool_name: str,
        yielded_evidence: bool,
        latency_ms: float,
        domain: str | None = None,
    ) -> None:
        """Record a single tool call outcome 📝.

        Args:
            tool_name: Fully qualified tool name
                (e.g., "mcp__bohrium__search").
            yielded_evidence: True if the call produced useful
                evidence.
            latency_ms: Call duration in milliseconds.
            domain: Optional domain tag for domain-specific
                analysis.
        """
        entry = _ToolCallRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            yielded_evidence=yielded_evidence,
            latency_ms=latency_ms,
            domain=domain,
        )
        with self._lock:
            if tool_name not in self._records:
                self._records[tool_name] = []
            self._records[tool_name].append(entry)

        logger.debug(
            "📊 Recorded tool call: %s evidence=%s latency=%dms",
            tool_name,
            yielded_evidence,
            latency_ms,
        )

    def get_tool_stats(
        self,
        tool_name: str,
        domain: str | None = None,
    ) -> dict:
        """Compute statistics for a single tool 📈.

        Args:
            tool_name: Fully qualified tool name.
            domain: Optional domain filter. When provided, only
                records matching this domain are included.

        Returns:
            Dict with keys: total_calls, successful_calls,
            success_rate, avg_latency_ms.
        """
        with self._lock:
            records = list(self._records.get(tool_name, []))

        # 🔍 Apply domain filter if specified
        if domain is not None:
            records = [r for r in records if r.domain == domain]

        if not records:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        total = len(records)
        successes = sum(1 for r in records if r.yielded_evidence)
        avg_latency = sum(r.latency_ms for r in records) / total

        return {
            "total_calls": total,
            "successful_calls": successes,
            "success_rate": successes / total,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def get_recommendations(
        self,
        limit: int = 5,
        min_calls: int = 1,
        domain: str | None = None,
    ) -> list[dict]:
        """Return tools ranked by success rate (descending) 🏆.

        Args:
            limit: Maximum number of tools to return.
            min_calls: Minimum call count to be included
                in recommendations.
            domain: Optional domain filter. When provided, only
                records from this domain are used for statistics.

        Returns:
            List of dicts with keys: tool_name, success_rate,
            total_calls, avg_latency_ms.
        """
        with self._lock:
            tool_names = list(self._records.keys())

        results: list[dict] = []
        for name in tool_names:
            stats = self.get_tool_stats(name, domain=domain)
            if stats["total_calls"] >= min_calls:
                results.append(
                    {
                        "tool_name": name,
                        "success_rate": stats["success_rate"],
                        "total_calls": stats["total_calls"],
                        "avg_latency_ms": stats["avg_latency_ms"],
                    }
                )

        # 🏆 Sort by success_rate descending, then by latency ascending
        results.sort(
            key=lambda r: (-r["success_rate"], r["avg_latency_ms"]),
        )

        return results[:limit]

    def summary(self) -> dict:
        """Generate aggregate summary across all tools 📊.

        Returns:
            Dict with total_tools_tracked, total_calls,
            overall_success_rate, avg_latency_ms.
        """
        with self._lock:
            all_records = [r for records in self._records.values() for r in records]
            num_tools = len(self._records)

        if not all_records:
            return {
                "total_tools_tracked": 0,
                "total_calls": 0,
                "overall_success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        total = len(all_records)
        successes = sum(1 for r in all_records if r.yielded_evidence)
        avg_latency = sum(r.latency_ms for r in all_records) / total

        return {
            "total_tools_tracked": num_tools,
            "total_calls": total,
            "overall_success_rate": successes / total,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def reset(self) -> None:
        """Clear all tracked data 🗑️."""
        with self._lock:
            self._records.clear()
        logger.info("📊 Tool effectiveness tracker reset")
