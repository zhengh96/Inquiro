"""Tests for Inquiro CircuitBreaker 🧪.

Tests the per-MCP-server fault isolation system:
- State transitions: CLOSED → OPEN → HALF_OPEN → CLOSED
- Failure threshold triggering
- Recovery timeout behavior
- Half-open probe calls
- Success resets failure count
"""

from __future__ import annotations

import time


from inquiro.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


# ============================================================
# 🔒 CLOSED State Tests
# ============================================================


class TestClosedState:
    """Tests for CircuitBreaker in CLOSED (healthy) state 🔒."""

    def test_initial_state_is_closed(self) -> None:
        """CircuitBreaker should start in CLOSED state."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=3),
        )
        assert cb.get_state() == CircuitState.CLOSED

    def test_can_execute_when_closed(self) -> None:
        """Execution should be allowed in CLOSED state."""
        cb = CircuitBreaker("test-server")
        assert cb.can_execute() is True

    def test_success_stays_closed(self) -> None:
        """Recording success in CLOSED state should stay CLOSED."""
        cb = CircuitBreaker("test-server")
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_single_failure_stays_closed(self) -> None:
        """A single failure below threshold should stay CLOSED."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=3),
        )
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 1

    def test_failures_below_threshold_stay_closed(self) -> None:
        """Failures below threshold should keep circuit CLOSED."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=3),
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 2


# ============================================================
# 🔴 CLOSED → OPEN Transition Tests
# ============================================================


class TestClosedToOpenTransition:
    """Tests for CircuitBreaker CLOSED → OPEN transition 🔴."""

    def test_threshold_failures_opens_circuit(self) -> None:
        """Reaching failure threshold should open the circuit."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=3),
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

    def test_cannot_execute_when_open(self) -> None:
        """Execution should be blocked in OPEN state."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0),
        )
        cb.record_failure()
        assert cb.can_execute() is False

    def test_last_failure_time_recorded(self) -> None:
        """Opening circuit should record the last failure timestamp."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1),
        )
        before = time.monotonic()
        cb.record_failure()
        after = time.monotonic()
        assert cb._last_failure_time is not None
        assert before <= cb._last_failure_time <= after


# ============================================================
# 🟡 OPEN → HALF_OPEN Transition Tests
# ============================================================


class TestOpenToHalfOpenTransition:
    """Tests for CircuitBreaker OPEN → HALF_OPEN transition 🟡."""

    def test_recovery_timeout_transitions_to_half_open(self) -> None:
        """After recovery timeout, should transition to HALF_OPEN."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1),
        )
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # ⏰ Wait for recovery timeout
        time.sleep(0.15)
        assert cb.get_state() == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

    def test_before_timeout_stays_open(self) -> None:
        """Before recovery timeout, should stay OPEN."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0),
        )
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_can_execute_in_half_open(self) -> None:
        """Execution should be allowed in HALF_OPEN state (probe call)."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
        )
        cb.record_failure()
        time.sleep(0.06)
        # 🟡 Should be in HALF_OPEN now
        assert cb.can_execute() is True


# ============================================================
# 🟢 HALF_OPEN → CLOSED Transition Tests
# ============================================================


class TestHalfOpenToClosedTransition:
    """Tests for CircuitBreaker HALF_OPEN → CLOSED transition 🟢."""

    def test_success_in_half_open_closes_circuit(self) -> None:
        """Success in HALF_OPEN state should close the circuit."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(
                failure_threshold=1,
                recovery_timeout=0.05,
                recovery_success_threshold=1,
            ),
        )
        cb.record_failure()
        time.sleep(0.06)
        # 🟡 Transition to HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN
        cb.record_success()
        # 🟢 Should be CLOSED now
        assert cb.get_state() == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_failure_in_half_open_reopens_circuit(self) -> None:
        """Failure in HALF_OPEN state should reopen the circuit."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
        )
        cb.record_failure()
        time.sleep(0.06)
        assert cb.get_state() == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN


# ============================================================
# 🔄 Full Lifecycle Tests
# ============================================================


class TestFullLifecycle:
    """Tests for CircuitBreaker full state transition lifecycle 🔄."""

    def test_full_cycle_closed_open_halfopen_closed(self) -> None:
        """Test complete lifecycle: CLOSED → OPEN → HALF_OPEN → CLOSED."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1),
        )
        # ✅ Step 1: CLOSED → OPEN
        assert cb.get_state() == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        # ✅ Step 2: OPEN → HALF_OPEN (after timeout)
        time.sleep(0.15)
        assert cb.get_state() == CircuitState.HALF_OPEN

        # ✅ Step 3: HALF_OPEN → CLOSED (on success)
        cb.record_success()
        assert cb.get_state() == CircuitState.CLOSED

    def test_repeated_open_close_cycles(self) -> None:
        """CircuitBreaker should handle multiple open/close cycles."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
        )
        for _ in range(3):
            # 🔴 Open
            cb.record_failure()
            assert cb.get_state() == CircuitState.OPEN
            # ⏰ Wait for recovery
            time.sleep(0.06)
            assert cb.get_state() == CircuitState.HALF_OPEN
            # 🟢 Close
            cb.record_success()
            assert cb.get_state() == CircuitState.CLOSED

    def test_success_resets_failure_count(self) -> None:
        """A success should reset the failure counter to zero."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=3),
        )
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2
        cb.record_success()
        assert cb._failure_count == 0
        # 📊 Two more failures should NOT open (count was reset)
        cb.record_failure()
        cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED

    def test_custom_failure_threshold(self) -> None:
        """Custom failure_threshold should be respected."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=5),
        )
        for _ in range(4):
            cb.record_failure()
        assert cb.get_state() == CircuitState.CLOSED

    def test_custom_recovery_timeout(self) -> None:
        """Custom recovery_timeout should be respected."""
        cb = CircuitBreaker(
            "test-server",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05),
        )
        cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.get_state() == CircuitState.HALF_OPEN
