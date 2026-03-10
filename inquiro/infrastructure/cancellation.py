"""Inquiro CancellationToken — thread-safe cooperative cancellation 🛑.

Provides a lightweight cancellation signal mechanism that allows
graceful termination of long-running research and synthesis tasks.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone


class CancelledError(Exception):
    """Raised when an operation is cancelled via CancellationToken ❌."""

    def __init__(self, reason: str = "Operation cancelled"):
        self.reason = reason
        super().__init__(reason)


class CancellationToken:
    """Thread-safe cancellation signal for cooperative task termination 🛑.

    Used by EvalTaskRunner to propagate cancellation requests into
    running SearchExp / SynthesisExp instances. Agents check
    ``is_cancelled`` at the start of each step to exit gracefully.

    Example::

        token = CancellationToken()
        # In the runner:
        token.cancel(reason="User requested cancellation")
        # In the agent step:
        token.check()  # raises CancelledError

    Attributes:
        reason: Human-readable cancellation reason (set after cancel).
    """

    def __init__(self) -> None:
        """Initialize a non-cancelled token 🔧."""
        self._cancelled = False
        self._reason: str | None = None
        self._cancelled_at: datetime | None = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []

    # -- Query ---------------------------------------------------------------

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if ``cancel()`` has been called 🔍.

        Returns:
            Current cancellation state.
        """
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """Return the cancellation reason, or ``None`` if not cancelled 📝.

        Returns:
            Reason string or None.
        """
        return self._reason

    @property
    def cancelled_at(self) -> datetime | None:
        """Return the UTC timestamp of cancellation, or ``None`` ⏱️.

        Returns:
            Datetime of cancellation or None.
        """
        return self._cancelled_at

    # -- Mutate --------------------------------------------------------------

    def cancel(self, reason: str = "Operation cancelled") -> None:
        """Signal cancellation to all observers 🛑.

        Thread-safe. Idempotent — calling ``cancel()`` multiple times
        has no additional effect.

        Args:
            reason: Human-readable explanation for why we cancelled.
        """
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            self._reason = reason
            self._cancelled_at = datetime.now(timezone.utc)
            # Fire registered callbacks ✨
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(reason)
            except Exception:
                # ⚠️ Swallow callback errors to avoid disrupting cancellation
                pass

    def check(self) -> None:
        """Raise ``CancelledError`` if cancellation was requested ✅.

        Convenience method for cooperative cancellation checkpoints.

        Raises:
            CancelledError: If the token has been cancelled.
        """
        if self._cancelled:
            raise CancelledError(self._reason or "Operation cancelled")

    # -- Callbacks -----------------------------------------------------------

    def on_cancel(self, callback: Callable) -> None:
        """Register a callback invoked when ``cancel()`` is called 📬.

        If the token is already cancelled the callback fires immediately.

        Args:
            callback: ``fn(reason: str) -> None`` invoked on cancellation.
        """
        with self._lock:
            if self._cancelled:
                # Already cancelled — fire immediately 🚀
                callback(self._reason)
                return
            self._callbacks.append(callback)

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        state = "cancelled" if self._cancelled else "active"
        return f"<CancellationToken state={state}>"

    def __bool__(self) -> bool:
        """Return ``True`` when the token is **not** cancelled.

        Allows idiomatic ``if token:`` checks.

        Returns:
            ``False`` if cancelled, ``True`` otherwise.
        """
        return not self._cancelled
