"""Thread-safe task store for managing evaluation task state 🔒.

Provides atomic operations for task state management with RLock
protection for concurrent access from multiple request handlers.

Architecture:
    FastAPI handlers → TaskStore (thread-safe dict wrapper)
        → In-memory dict (Phase 0)
        → PostgreSQL (production, future)
"""

from __future__ import annotations

import threading
from typing import Any


class TaskStore:
    """Thread-safe wrapper around task state dictionary 🔐.

    Provides atomic operations for task state management with RLock
    protection for concurrent access. In Phase 0, backs onto an
    in-memory dict; in production, will delegate to PostgreSQL.

    Uses RLock (reentrant lock) to allow the same thread to acquire
    the lock multiple times safely (e.g., update_status calling get).

    Attributes:
        _store: Internal dict mapping task_id to task state objects.
        _lock: Reentrant lock protecting concurrent access to _store.
    """

    def __init__(self) -> None:
        """Initialize empty task store with reentrant lock 🏗️."""
        self._store: dict[str, Any] = {}
        # 🔒 RLock allows same thread to acquire multiple times (for nested calls)
        self._lock = threading.RLock()

    def put(self, task_id: str, data: Any) -> None:
        """Store or update task state atomically 📝.

        Thread-safe: acquires _lock for the duration of the write.

        Args:
            task_id: Unique task identifier (e.g., UUID).
            data: Task state object (typically TaskState from schemas.py).
        """
        with self._lock:
            self._store[task_id] = data

    def get(self, task_id: str) -> Any | None:
        """Retrieve task state by ID 🔍.

        Thread-safe: acquires _lock for the duration of the read.

        Args:
            task_id: Unique task identifier.

        Returns:
            Task state object if found, None otherwise.
        """
        with self._lock:
            return self._store.get(task_id)

    def update_status(self, task_id: str, status: str) -> bool:
        """Update task status field atomically 🔄.

        Thread-safe: acquires _lock for the duration of read + write.

        Args:
            task_id: Unique task identifier.
            status: New status value (e.g., "running", "completed", "failed").

        Returns:
            True if task exists and status was updated.
            False if task not found.
        """
        with self._lock:
            task_state = self._store.get(task_id)
            if task_state is None:
                return False
            # 📝 Assumes task_state has a 'status' attribute or dict key
            if hasattr(task_state, "status"):
                task_state.status = status
            elif isinstance(task_state, dict):
                task_state["status"] = status
            return True

    def remove(self, task_id: str) -> bool:
        """Remove task state atomically 🗑️.

        Thread-safe: acquires _lock for the duration of the delete.

        Args:
            task_id: Unique task identifier to remove.

        Returns:
            True if task existed and was removed.
            False if task was not found.
        """
        with self._lock:
            if task_id in self._store:
                del self._store[task_id]
                return True
            return False

    def list_tasks(self) -> list[str]:
        """List all task IDs currently in the store 📋.

        Thread-safe: acquires _lock and creates a snapshot of keys.

        Returns:
            List of task ID strings. Returns empty list if store is empty.
        """
        with self._lock:
            return list(self._store.keys())

    def size(self) -> int:
        """Return the number of tasks in the store 📊.

        Thread-safe: acquires _lock for the read.

        Returns:
            Count of tasks currently stored.
        """
        with self._lock:
            return len(self._store)

    # ====================================================================
    # 🎭 Dict-like interface for backward compatibility
    # ====================================================================

    def __getitem__(self, task_id: str) -> Any:
        """Support bracket notation: task_store[task_id] 🔍.

        Raises KeyError if task not found (standard dict behavior).

        Args:
            task_id: Unique task identifier.

        Returns:
            Task state object.

        Raises:
            KeyError: If task_id not found in store.
        """
        with self._lock:
            if task_id not in self._store:
                raise KeyError(f"Task {task_id} not found in store")
            return self._store[task_id]

    def __setitem__(self, task_id: str, data: Any) -> None:
        """Support bracket notation: task_store[task_id] = data 📝.

        Equivalent to put(task_id, data).

        Args:
            task_id: Unique task identifier.
            data: Task state object to store.
        """
        self.put(task_id, data)

    def __contains__(self, task_id: str) -> bool:
        """Support 'in' operator: task_id in task_store ✅.

        Thread-safe: acquires _lock for the check.

        Args:
            task_id: Unique task identifier to check.

        Returns:
            True if task exists in store, False otherwise.
        """
        with self._lock:
            return task_id in self._store
