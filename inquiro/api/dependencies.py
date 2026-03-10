"""FastAPI dependency injection for Inquiro service 🔧.

Provides injectable dependencies for route handlers:
- get_task_runner: Returns the service-level EvalTaskRunner
- get_event_emitter: Creates a new EventEmitter per task
- get_task_store: Returns the in-memory task store

Dependencies follow the FastAPI Depends() pattern for clean
separation of concerns and easy testing via dependency overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Request

from inquiro.api.task_store import TaskStore

if TYPE_CHECKING:
    from inquiro.infrastructure.event_emitter import EventEmitter


# ✨ In-memory task store (Phase 0: TaskStore, production: PostgreSQL)
_task_store = TaskStore()


def get_task_store() -> TaskStore:
    """Get the in-memory task store 📦.

    Returns:
        TaskStore mapping task_id to TaskState objects.
        Phase 0 uses in-memory TaskStore; production will use PostgreSQL.
    """
    return _task_store


def get_task_runner(request: Request) -> Any:
    """Get the service-level EvalTaskRunner 🎯.

    Retrieves the EvalTaskRunner instance stored in app.state
    during startup. The runner manages shared resources (MCP pool,
    LLM pool) and dispatches tasks to DiscoveryLoop/SynthesisExp.

    Args:
        request: FastAPI request (provides access to app.state)

    Returns:
        EvalTaskRunner instance

    Raises:
        RuntimeError: If the service has not been initialized
    """
    runner = getattr(request.app.state, "task_runner", None)
    if runner is None:
        raise RuntimeError(
            "EvalTaskRunner not initialized. "
            "Ensure the app lifespan startup completed successfully."
        )
    return runner


def get_event_emitter() -> "EventEmitter":
    """Create a fresh EventEmitter for a new task 📡.

    Each task gets its own EventEmitter to isolate SSE streams.
    Subscribers can attach to receive real-time progress events.

    Returns:
        New EventEmitter instance
    """
    # 🔄 Lazy import to avoid circular dependency
    from inquiro.infrastructure.event_emitter import EventEmitter

    return EventEmitter()


def get_active_task_count(request: Request) -> int:
    """Get the count of currently active (running) tasks 📊.

    Uses the runner's public API instead of accessing the private
    ``_active_tasks`` dict directly.

    Args:
        request: FastAPI request (provides access to app.state)

    Returns:
        Number of tasks with status 'running'
    """
    runner = getattr(request.app.state, "task_runner", None)
    if runner is None:
        return 0
    return runner.get_active_task_count()
