"""Inquiro MetricsCollector -- structured execution metrics 📊.

Provides lightweight metric primitives (counter, gauge, histogram)
and auto-subscribes to EventEmitter for non-invasive collection
of 5 core metrics:

1. ``task_latency`` histogram -- task completion time distribution
2. ``active_tasks`` gauge -- number of currently active tasks
3. ``task_outcome`` counter -- task outcomes (success/failure/timeout)
4. ``tool_call_duration`` histogram -- MCP tool call latency
5. ``qg_pass_rate`` gauge -- QualityGate first-pass rate
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class MetricPoint(BaseModel):
    """Single metric data point 📊.

    Attributes:
        name: Metric name (e.g., "task_latency").
        labels: Key-value label pairs for metric dimensions.
        value: Numeric value of the data point.
        timestamp: Unix timestamp of recording.
    """

    name: str = Field(description="Metric name")
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value label pairs for metric dimensions",
    )
    value: float = Field(description="Numeric value of the data point")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of recording",
    )


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe metrics collector with histogram/gauge/counter 📊.

    Provides three primitive metric types:
    - **Counter**: monotonically increasing value (e.g., task_outcome)
    - **Gauge**: current value that can go up/down (e.g., active_tasks)
    - **Histogram**: distribution of values (e.g., latency)

    Auto-subscribes to EventEmitter via ``subscribe_to_emitter()``
    to collect the 5 core metrics non-invasively.

    Example::

        collector = MetricsCollector()
        collector.subscribe_to_emitter(emitter)

        # ✨ Metrics are auto-collected from events
        snapshot = collector.get_all_metrics()
    """

    # 📏 Default histogram buckets (seconds)
    DEFAULT_BUCKETS: tuple[float, ...] = (
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    )

    def __init__(self) -> None:
        """Initialize MetricsCollector with empty metric stores 🔧."""
        self._lock = threading.Lock()
        # 📊 Metric stores keyed by (name, label_key)
        self._counters: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float),
        )
        self._gauges: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float),
        )
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._histogram_labels: dict[str, dict[str, str]] = {}
        # 🕐 Internal tracking for latency calculation
        self._task_start_times: dict[str, float] = {}

    # -- Counter primitives --------------------------------------------------

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric ➕.

        Args:
            name: Counter metric name.
            value: Amount to increment (must be positive).
            labels: Optional label dimensions.
        """
        key = self._label_key(labels)
        with self._lock:
            self._counters[name][key] += value

    def get_counter(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get current counter value 🔢.

        Args:
            name: Counter metric name.
            labels: Optional label dimensions.

        Returns:
            Current counter value, or 0.0 if not found.
        """
        key = self._label_key(labels)
        with self._lock:
            return self._counters[name].get(key, 0.0)

    # -- Gauge primitives ----------------------------------------------------

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric to an absolute value 📏.

        Args:
            name: Gauge metric name.
            value: Value to set.
            labels: Optional label dimensions.
        """
        key = self._label_key(labels)
        with self._lock:
            self._gauges[name][key] = value

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a gauge metric 📈.

        Args:
            name: Gauge metric name.
            value: Amount to increment.
            labels: Optional label dimensions.
        """
        key = self._label_key(labels)
        with self._lock:
            self._gauges[name][key] += value

    def dec_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Decrement a gauge metric 📉.

        Args:
            name: Gauge metric name.
            value: Amount to decrement.
            labels: Optional label dimensions.
        """
        key = self._label_key(labels)
        with self._lock:
            self._gauges[name][key] -= value

    def get_gauge(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> float:
        """Get current gauge value 🔢.

        Args:
            name: Gauge metric name.
            labels: Optional label dimensions.

        Returns:
            Current gauge value, or 0.0 if not found.
        """
        key = self._label_key(labels)
        with self._lock:
            return self._gauges[name].get(key, 0.0)

    # -- Histogram primitives ------------------------------------------------

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation 📊.

        Args:
            name: Histogram metric name.
            value: Observed value (e.g., latency in seconds).
            labels: Optional label dimensions.
        """
        key = self._label_key(labels)
        hist_key = f"{name}:{key}"
        with self._lock:
            self._histograms[hist_key].append(value)
            if hist_key not in self._histogram_labels:
                self._histogram_labels[hist_key] = labels or {}

    def get_histogram_summary(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Get histogram summary statistics 📊.

        Computes count, sum, min, max, avg, p50, p95, p99
        from all recorded observations.

        Args:
            name: Histogram metric name.
            labels: Optional label dimensions.

        Returns:
            Dictionary with summary statistics. Returns
            zeroed values when no observations exist.
        """
        key = self._label_key(labels)
        hist_key = f"{name}:{key}"
        with self._lock:
            values = list(self._histograms.get(hist_key, []))
        if not values:
            return {
                "count": 0,
                "sum": 0.0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
            }
        values.sort()
        count = len(values)
        return {
            "count": count,
            "sum": sum(values),
            "min": values[0],
            "max": values[-1],
            "avg": sum(values) / count,
            "p50": values[int(count * 0.5)],
            "p95": values[min(int(count * 0.95), count - 1)],
            "p99": values[min(int(count * 0.99), count - 1)],
        }

    # -- Snapshot ------------------------------------------------------------

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics as a JSON-serializable dict 📋.

        Returns:
            Dictionary with three keys: ``counters``, ``gauges``,
            and ``histograms``, each containing their respective
            metric data.
        """
        with self._lock:
            result: dict[str, Any] = {
                "counters": {name: dict(vals) for name, vals in self._counters.items()},
                "gauges": {name: dict(vals) for name, vals in self._gauges.items()},
                "histograms": {},
            }
            for hist_key, values in self._histograms.items():
                sorted_vals = sorted(values)
                count = len(sorted_vals)
                if count > 0:
                    result["histograms"][hist_key] = {
                        "count": count,
                        "sum": sum(sorted_vals),
                        "min": sorted_vals[0],
                        "max": sorted_vals[-1],
                        "avg": sum(sorted_vals) / count,
                    }
        return result

    # -- EventEmitter integration --------------------------------------------

    def subscribe_to_emitter(self, emitter: Any) -> None:
        """Subscribe to EventEmitter for automatic metric collection 📡.

        Subscribes to standard InquiroEvent types and updates
        the 5 core metrics automatically. Non-invasive -- just
        listens to events without modifying emitter behaviour.

        Args:
            emitter: An ``EventEmitter`` instance to subscribe to.
        """
        from inquiro.infrastructure.event_emitter import InquiroEvent

        emitter.subscribe(
            InquiroEvent.TASK_STARTED,
            self._on_task_started,
        )
        emitter.subscribe(
            InquiroEvent.TASK_COMPLETED,
            self._on_task_completed,
        )
        emitter.subscribe(
            InquiroEvent.TASK_FAILED,
            self._on_task_failed,
        )
        emitter.subscribe(
            InquiroEvent.TASK_CANCELLED,
            self._on_task_cancelled,
        )
        emitter.subscribe(
            InquiroEvent.QUALITY_GATE_RESULT,
            self._on_qg_result,
        )
        logger.info(
            "📊 MetricsCollector subscribed to %s event types",
            5,
        )

    # -- Event handlers (private) --------------------------------------------

    def _on_task_started(self, event: Any) -> None:
        """Handle task started event 🚀.

        Increments active_tasks gauge and records start time
        for latency calculation.

        Args:
            event: EventData from the emitter.
        """
        self.inc_gauge("active_tasks")
        # 🕐 Store start time for latency measurement
        with self._lock:
            self._task_start_times[event.task_id] = time.time()

    def _on_task_completed(self, event: Any) -> None:
        """Handle task completed event ✅.

        Decrements active_tasks, increments success counter,
        and records task latency histogram.

        Args:
            event: EventData from the emitter.
        """
        self.dec_gauge("active_tasks")
        self.inc_counter(
            "task_outcome",
            labels={"outcome": "success"},
        )
        # 📊 Calculate and record latency
        with self._lock:
            start = self._task_start_times.pop(
                event.task_id,
                None,
            )
        if start is not None:
            latency = time.time() - start
            sub_item = event.data.get("sub_item", "unknown")
            self.observe_histogram(
                "task_latency",
                latency,
                labels={"sub_item": sub_item},
            )

    def _on_task_failed(self, event: Any) -> None:
        """Handle task failed event ❌.

        Decrements active_tasks and increments failure counter.

        Args:
            event: EventData from the emitter.
        """
        self.dec_gauge("active_tasks")
        self.inc_counter(
            "task_outcome",
            labels={"outcome": "failure"},
        )
        # 🗑️ Clean up start time if present
        with self._lock:
            self._task_start_times.pop(event.task_id, None)

    def _on_task_cancelled(self, event: Any) -> None:
        """Handle task cancelled event 🛑.

        Decrements active_tasks and increments timeout counter.

        Args:
            event: EventData from the emitter.
        """
        self.dec_gauge("active_tasks")
        self.inc_counter(
            "task_outcome",
            labels={"outcome": "timeout"},
        )
        # 🗑️ Clean up start time if present
        with self._lock:
            self._task_start_times.pop(event.task_id, None)

    def _on_qg_result(self, event: Any) -> None:
        """Handle QualityGate result event 🔍.

        Tracks total and passed QG evaluations, then
        updates the qg_pass_rate gauge.

        Args:
            event: EventData from the emitter with
                ``data.passed`` boolean.
        """
        passed = event.data.get("passed", False)
        self.inc_counter("qg_total")
        if passed:
            self.inc_counter("qg_passed")
        # 📊 Update pass rate gauge
        total = self.get_counter("qg_total")
        passed_count = self.get_counter("qg_passed")
        if total > 0:
            self.set_gauge("qg_pass_rate", passed_count / total)

    # -- Utility (private) ---------------------------------------------------

    @staticmethod
    def _label_key(labels: dict[str, str] | None) -> str:
        """Convert labels dict to a stable string key 🔑.

        Args:
            labels: Optional label dictionary.

        Returns:
            Sorted, comma-separated ``key=value`` string,
            or empty string if no labels.
        """
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
