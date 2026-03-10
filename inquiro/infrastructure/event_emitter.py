"""Inquiro EventEmitter — async event system with SSE support 📡.

Provides a thread-safe event bus that allows components to emit
structured events consumed by SSE endpoints for real-time progress.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import threading
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class InquiroEvent(str, enum.Enum):
    """Standard event types emitted during research/synthesis lifecycle 🎯.

    Attributes:
        TASK_STARTED: A research or synthesis task has begun.
        ROUND_COMPLETED: One search-reason round finished.
        QUALITY_GATE_RESULT: QualityGate validation completed.
        QUALITY_GATE_RETRY: Hard failure triggered a retry.
        ADDITIONAL_RESEARCH_REQUESTED: SynthesisAgent triggered deep dive.
        SYNTHESIS_STARTED: A synthesis task has begun.
        TASK_COMPLETED: Task finished successfully.
        TASK_FAILED: Task terminated with an error.
        COST_WARNING: Cost approaching budget limit.
        TASK_CANCELLED: Task was cancelled via CancellationToken.
        PHASE_CHANGED: Exp lifecycle phase transitioned.
        DISCOVERY_STARTED: Discovery loop has begun.
        DISCOVERY_ROUND_STARTED: A discovery round has started.
        DISCOVERY_ROUND_COMPLETED: A discovery round has finished.
        DISCOVERY_COVERAGE_UPDATED: Coverage progression updated.
        DISCOVERY_CONVERGED: Discovery loop reached convergence.
        DISCOVERY_COMPLETED: Discovery pipeline finished.
        DISCOVERY_SEARCH_WARNING: Search section produced partial errors.
    """

    TASK_STARTED = "task_started"
    ROUND_COMPLETED = "round_completed"
    QUALITY_GATE_RESULT = "quality_gate_result"
    QUALITY_GATE_RETRY = "quality_gate_retry"
    ADDITIONAL_RESEARCH_REQUESTED = "additional_research_requested"
    SYNTHESIS_STARTED = "synthesis_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    COST_WARNING = "cost_warning"
    TASK_CANCELLED = "task_cancelled"
    PHASE_CHANGED = "phase_changed"

    # 📡 Discovery pipeline SSE events
    DISCOVERY_STARTED = "discovery_started"
    DISCOVERY_ROUND_STARTED = "discovery_round_started"
    DISCOVERY_ROUND_COMPLETED = "discovery_round_completed"
    DISCOVERY_COVERAGE_UPDATED = "discovery_coverage_updated"
    DISCOVERY_CONVERGED = "discovery_converged"
    DISCOVERY_COMPLETED = "discovery_completed"
    DISCOVERY_SEARCH_WARNING = "discovery_search_warning"


class EventData(BaseModel):
    """Structured event payload 📋.

    Attributes:
        task_id: ID of the task that produced this event.
        event_type: The InquiroEvent type string.
        timestamp: UTC ISO-8601 timestamp of emission.
        data: Arbitrary event-specific payload.
    """

    task_id: str
    event_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data: dict[str, Any] = Field(default_factory=dict)


# -- Callback type alias ----------------------------------------------------

EventCallback = Callable[[EventData], None]


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """SSE-compatible async event emitter for real-time progress 📡.

    Thread-safe for use across concurrent tasks. Supports both
    callback-based subscriptions and ``AsyncGenerator``-based
    streaming for SSE endpoints.

    Event history is bounded to prevent unbounded memory growth
    in long-running services. When the history limit is reached,
    oldest events are evicted (FIFO).

    Example (callback)::

        emitter = EventEmitter()
        emitter.subscribe(InquiroEvent.TASK_COMPLETED, my_handler)
        emitter.emit(InquiroEvent.TASK_STARTED, "task_123", {"attempt": 1})

    Example (SSE streaming)::

        async for event in emitter.get_event_stream("task_123"):
            yield f"data: {event.model_dump_json()}\\n\\n"
    """

    # 📏 Default maximum history size (prevents memory leaks)
    DEFAULT_MAX_HISTORY: int = 10_000

    def __init__(
        self,
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        """Initialize EventEmitter 🔧.

        Args:
            max_history: Maximum number of events retained in history.
                Oldest events are evicted when limit is reached.
                Defaults to 10,000.
        """
        # 🎯 Callback subscriptions: event_type → list of callbacks
        self._callbacks: dict[str, list[EventCallback]] = {}
        # 📬 Async queues for SSE subscribers (per task_id)
        self._queues: dict[str, list[asyncio.Queue[EventData | None]]] = {}
        # 📜 Event history for late-joining subscribers
        self._history: list[EventData] = []
        self._max_history = max(1, max_history)
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    # -- Emit ----------------------------------------------------------------

    def emit(
        self,
        event_type: str | InquiroEvent,
        task_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an event to all subscribers 📡.

        Thread-safe. Dispatches to both callback subscribers and
        async queue subscribers (for SSE).

        Args:
            event_type: Event type string or ``InquiroEvent`` enum.
            task_id: ID of the task producing the event.
            data: Event-specific payload dictionary.
        """
        # 🔄 Normalize enum to string
        event_type_str = (
            event_type.value if isinstance(event_type, InquiroEvent) else event_type
        )

        event = EventData(
            task_id=task_id,
            event_type=event_type_str,
            data=data or {},
        )

        with self._lock:
            # 📜 Record in history (with FIFO eviction)
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            # 🎯 Dispatch to callback subscribers
            callbacks = list(self._callbacks.get(event_type_str, []))

            # 📬 Dispatch to SSE queue subscribers
            queues = list(self._queues.get(task_id, []))

        # ✨ Fire callbacks outside lock to avoid deadlocks
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                self._logger.warning(
                    "⚠️ Callback error for %s: %s",
                    event_type_str,
                    exc,
                )

        # 📬 Put event into all SSE queues for this task_id
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._logger.warning(
                    "⚠️ SSE queue full for task '%s'",
                    task_id,
                )

    # -- Callback subscriptions ----------------------------------------------

    def subscribe(
        self,
        event_type: str | InquiroEvent,
        callback: EventCallback,
    ) -> None:
        """Register a callback for a specific event type 📬.

        Args:
            event_type: Event type to listen for.
            callback: ``fn(EventData) -> None`` called on each event.
        """
        event_type_str = (
            event_type.value if isinstance(event_type, InquiroEvent) else event_type
        )
        with self._lock:
            if event_type_str not in self._callbacks:
                self._callbacks[event_type_str] = []
            self._callbacks[event_type_str].append(callback)

    def unsubscribe(
        self,
        event_type: str | InquiroEvent,
        callback: EventCallback,
    ) -> None:
        """Remove a previously registered callback 🗑️.

        Args:
            event_type: Event type the callback was registered for.
            callback: The callback to remove.
        """
        event_type_str = (
            event_type.value if isinstance(event_type, InquiroEvent) else event_type
        )
        with self._lock:
            callbacks = self._callbacks.get(event_type_str, [])
            try:
                callbacks.remove(callback)
            except ValueError:
                pass  # ⚠️ Callback not found — ignore

    # -- SSE streaming -------------------------------------------------------

    async def get_event_stream(
        self,
        task_id: str,
        include_history: bool = True,
    ) -> AsyncGenerator[EventData, None]:
        """Yield events for a task as an async generator (SSE) 🌐.

        Creates an ``asyncio.Queue`` under the hood, replays history
        for the task if ``include_history`` is ``True``, then yields
        new events until a sentinel ``None`` is received.

        Args:
            task_id: Filter events for this task ID.
            include_history: Whether to replay past events for the task.

        Yields:
            EventData instances for the specified task.
        """
        queue: asyncio.Queue[EventData | None] = asyncio.Queue()

        with self._lock:
            # 🏗️ Register the queue for this task
            if task_id not in self._queues:
                self._queues[task_id] = []
            self._queues[task_id].append(queue)

            # 📜 Replay history if requested
            if include_history:
                history = [e for e in self._history if e.task_id == task_id]
            else:
                history = []

        # ✨ Yield historical events first
        for event in history:
            yield event

        # 🔄 Yield live events until sentinel
        try:
            while True:
                event = await queue.get()
                if event is None:
                    # 🛑 Sentinel received — stream done
                    break
                yield event
        finally:
            # 🗑️ Cleanup: remove queue from subscribers
            with self._lock:
                task_queues = self._queues.get(task_id, [])
                try:
                    task_queues.remove(queue)
                except ValueError:
                    pass
                if not task_queues and task_id in self._queues:
                    del self._queues[task_id]

    def close_stream(self, task_id: str) -> None:
        """Signal end-of-stream for all SSE subscribers of a task 🛑.

        Pushes a ``None`` sentinel to every queue subscribed to
        *task_id*, causing ``get_event_stream`` generators to exit.

        Args:
            task_id: The task whose streams should be closed.
        """
        with self._lock:
            queues = list(self._queues.get(task_id, []))

        for queue in queues:
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                self._logger.warning(
                    "⚠️ Could not push sentinel to queue for task '%s'",
                    task_id,
                )

    # -- History / utility ---------------------------------------------------

    def get_history(
        self,
        task_id: str | None = None,
        event_type: str | InquiroEvent | None = None,
    ) -> list[EventData]:
        """Return recorded event history, optionally filtered 📜.

        Args:
            task_id: If given, only return events for this task.
            event_type: If given, only return events of this type.

        Returns:
            List of matching ``EventData`` instances.
        """
        event_type_str = (
            event_type.value if isinstance(event_type, InquiroEvent) else event_type
        )

        with self._lock:
            result = list(self._history)

        # 🔍 Apply filters
        if task_id is not None:
            result = [e for e in result if e.task_id == task_id]
        if event_type_str is not None:
            result = [e for e in result if e.event_type == event_type_str]

        return result

    def clear_history(self, task_id: str | None = None) -> None:
        """Clear event history 🗑️.

        Args:
            task_id: If given, only clear history for this task.
                Otherwise clears all history.
        """
        with self._lock:
            if task_id is None:
                self._history.clear()
            else:
                self._history = [e for e in self._history if e.task_id != task_id]
