"""Inquiro CircuitBreaker — per-MCP-server fault isolation 🔌.

Implements the standard CLOSED → OPEN → HALF_OPEN state machine
to prevent cascading failures when MCP servers become unhealthy.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class CircuitState(str, enum.Enum):
    """Circuit breaker state 🔌.

    Attributes:
        CLOSED: Normal operation — requests pass through.
        OPEN: Circuit tripped — requests are rejected immediately.
        HALF_OPEN: Recovery probe — limited requests allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerConfig(BaseModel):
    """Configuration for a single CircuitBreaker instance ⚙️.

    Attributes:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait in OPEN before probing.
        half_open_max_calls: Max probe calls in HALF_OPEN.
        recovery_success_threshold: Successes needed to close again.
    """

    failure_threshold: int = Field(
        default=3, description="Consecutive failures before opening circuit"
    )
    recovery_timeout: float = Field(
        default=60.0, description="Seconds to wait in OPEN before probing"
    )
    half_open_max_calls: int = Field(
        default=3, description="Max probe calls allowed in HALF_OPEN state"
    )
    recovery_success_threshold: int = Field(
        default=1, description="Successes needed in HALF_OPEN to close circuit"
    )


class CircuitOpenError(Exception):
    """Raised when a call is attempted on an open circuit ❌."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        super().__init__(
            f"Circuit breaker for '{server_name}' is OPEN — "
            f"requests are blocked until recovery timeout elapses"
        )


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Per-MCP-server circuit breaker with state machine 🔌.

    States::

        CLOSED  ──(failure_threshold reached)──▶  OPEN
        OPEN    ──(recovery_timeout elapsed)───▶  HALF_OPEN
        HALF_OPEN ──(success)──────────────────▶  CLOSED
        HALF_OPEN ──(failure)──────────────────▶  OPEN

    Example::

        cb = CircuitBreaker("opentargets")
        if cb.can_execute():
            try:
                result = mcp_call(...)
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # use fallback or skip ⚠️

    Attributes:
        server_name: Name of the MCP server this breaker guards.
    """

    def __init__(
        self,
        server_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize CircuitBreaker 🔧.

        Args:
            server_name: MCP server name for identification.
            config: Breaker configuration. Uses defaults if ``None``.
        """
        self.server_name = server_name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._half_open_calls: int = 0
        self._last_failure_time: float | None = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{server_name}]")

    # -- Query ---------------------------------------------------------------

    def get_state(self) -> CircuitState:
        """Return the current circuit state 🔍.

        Performs a passive timeout check: if OPEN and the recovery
        timeout has elapsed the state transitions to HALF_OPEN.

        Returns:
            Current CircuitState.
        """
        with self._lock:
            # 🔄 Passive transition: OPEN → HALF_OPEN after timeout
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time is not None
                    and (time.monotonic() - self._last_failure_time)
                    >= self._config.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    self._logger.info(
                        "⏰ Recovery timeout elapsed — transitioning to HALF_OPEN"
                    )
            return self._state

    def can_execute(self) -> bool:
        """Check whether the circuit allows a new request 🔍.

        Returns:
            ``True`` if execution is allowed.
        """
        state = self.get_state()
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            # ✅ Allow probe calls up to the configured max
            with self._lock:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        # 🔴 OPEN — block execution
        return False

    # -- Recording -----------------------------------------------------------

    def record_success(self) -> None:
        """Record a successful MCP call ✅.

        In HALF_OPEN state, once ``recovery_success_threshold``
        successes are recorded the circuit transitions to CLOSED.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.recovery_success_threshold:
                    # 🟢 Recovery confirmed — close the circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._last_failure_time = None
                    self._logger.info("🟢 Recovery successful — circuit CLOSED")
            elif self._state == CircuitState.CLOSED:
                # ✨ Reset failure count on success in CLOSED state
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed MCP call ❌.

        Increments the failure counter. When ``failure_threshold``
        is reached the circuit transitions to OPEN.
        In HALF_OPEN state, any failure re-opens the circuit.
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # 🔴 Any failure in HALF_OPEN re-opens the circuit
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()
                self._success_count = 0
                self._half_open_calls = 0
                self._logger.warning("🔴 Failure in HALF_OPEN — circuit re-opened")
            else:
                # 📊 Track consecutive failures in CLOSED state
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._logger.warning(
                        "🔴 Failure threshold reached (%s) — circuit OPEN",
                        self._failure_count,
                    )

    # -- Convenience ---------------------------------------------------------

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *func* through the circuit breaker 🚀.

        Checks the circuit before calling, records success/failure
        automatically, and re-raises the original exception on failure.

        Args:
            func: The callable to execute.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            CircuitOpenError: If the circuit is OPEN.
            Exception: Re-raised from *func* on failure.
        """
        if not self.can_execute():
            raise CircuitOpenError(self.server_name)
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state 🔄.

        Clears all counters and timestamps.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._logger.info("🔄 Circuit manually reset to CLOSED")

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<CircuitBreaker server={self.server_name!r} state={self._state.value}>"


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """Manages per-MCP-server circuit breakers 🗂️.

    Lazily creates a ``CircuitBreaker`` for each server on first
    access and exposes aggregated state queries.

    Example::

        registry = CircuitBreakerRegistry()
        cb = registry.get_breaker("opentargets")
        states = registry.get_all_states()

    Attributes:
        default_config: Default config applied to new breakers.
    """

    def __init__(
        self,
        default_config: CircuitBreakerConfig | None = None,
        per_server_configs: dict[str, CircuitBreakerConfig] | None = None,
    ) -> None:
        """Initialize CircuitBreakerRegistry 🔧.

        Args:
            default_config: Fallback config for servers without a
                custom configuration.
            per_server_configs: Optional per-server config overrides.
        """
        self.default_config = default_config or CircuitBreakerConfig()
        self._per_server_configs = per_server_configs or {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    def get_breaker(self, server_name: str) -> CircuitBreaker:
        """Get or create the circuit breaker for *server_name* 🔧.

        Thread-safe. Creates a new breaker with the appropriate config
        on first access.

        Args:
            server_name: MCP server name.

        Returns:
            CircuitBreaker instance for the server.
        """
        with self._lock:
            if server_name not in self._breakers:
                # 🏗️ Lazy creation with per-server or default config
                config = self._per_server_configs.get(server_name, self.default_config)
                self._breakers[server_name] = CircuitBreaker(server_name, config)
                self._logger.debug(
                    "✨ Created breaker for '%s'",
                    server_name,
                )
            return self._breakers[server_name]

    def get_all_states(self) -> dict[str, CircuitState]:
        """Return a snapshot of all circuit breaker states 📊.

        Returns:
            Mapping of ``{server_name: CircuitState}``.
        """
        with self._lock:
            return {
                name: breaker.get_state() for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reset all breakers to CLOSED state 🔄."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            self._logger.info("🔄 All circuit breakers reset")

    def reset(self, server_name: str) -> None:
        """Reset a single server's breaker to CLOSED 🔄.

        Args:
            server_name: MCP server name.
        """
        with self._lock:
            if server_name in self._breakers:
                self._breakers[server_name].reset()
