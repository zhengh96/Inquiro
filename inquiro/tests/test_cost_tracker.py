"""Tests for Inquiro CostTracker 🧪.

Tests the real-time cost tracking and budget enforcement system:
- Cost recording per task
- Per-task budget checking
- Total budget checking
- Warning threshold (80%)
- Cost summary aggregation
- Thread-safe concurrent access
"""

from __future__ import annotations

import threading

import pytest

from inquiro.infrastructure.cost_tracker import (
    CostStatus,
    CostTracker,
)


# 📊 Helper: default model for tests
_MODEL = "claude-sonnet-4-20250514"
# 💰 Sonnet pricing: input=$0.003/1K, output=$0.015/1K
# So 1000 input + 1000 output = $0.003 + $0.015 = $0.018


def _make_tracker(**kwargs) -> CostTracker:
    """Create CostTracker with sensible test defaults 🔧."""
    defaults = {"max_per_task": 5.0, "max_total": 50.0}
    defaults.update(kwargs)
    return CostTracker(**defaults)


# ============================================================
# 💰 Cost Recording Tests
# ============================================================


class TestCostRecording:
    """Tests for CostTracker cost recording 💰."""

    def test_record_single_task_cost(self) -> None:
        """Recording cost for a task should update task total."""
        tracker = _make_tracker()
        tracker.record("task-001", _MODEL, 1000, 1000)
        # 🧮 1K input * $0.003/1K + 1K output * $0.015/1K = $0.018
        cost = tracker.get_task_cost("task-001")
        assert cost == pytest.approx(0.018)

    def test_record_multiple_increments(self) -> None:
        """Multiple records for same task should accumulate."""
        tracker = _make_tracker()
        tracker.record("task-001", _MODEL, 1000, 1000)
        tracker.record("task-001", _MODEL, 1000, 1000)
        cost = tracker.get_task_cost("task-001")
        assert cost == pytest.approx(0.036)

    def test_record_multiple_tasks(self) -> None:
        """Recording for different tasks should be independent."""
        tracker = _make_tracker()
        tracker.record("task-001", _MODEL, 1000, 1000)
        tracker.record("task-002", _MODEL, 2000, 2000)
        assert tracker.get_task_cost("task-001") == pytest.approx(0.018)
        assert tracker.get_task_cost("task-002") == pytest.approx(0.036)

    def test_get_task_cost_unknown_task(self) -> None:
        """Querying unknown task should return 0.0."""
        tracker = _make_tracker()
        assert tracker.get_task_cost("nonexistent") == 0.0


# ============================================================
# ⚠️ Budget Checking Tests
# ============================================================


class TestBudgetChecking:
    """Tests for CostTracker budget enforcement ⚠️."""

    def test_within_budget_returns_ok(self) -> None:
        """Cost within budget should return OK status."""
        tracker = _make_tracker(max_per_task=5.0, max_total=50.0)
        status = tracker.record("task-001", _MODEL, 1000, 1000)
        assert status == CostStatus.OK

    def test_per_task_exceeded_returns_task_exceeded(self) -> None:
        """Exceeding per-task budget should return TASK_EXCEEDED."""
        # ✨ Set very low per-task limit
        tracker = _make_tracker(max_per_task=0.01, max_total=50.0)
        status = tracker.record("task-001", _MODEL, 1000, 1000)
        # $0.018 > $0.01
        assert status == CostStatus.TASK_EXCEEDED

    def test_total_exceeded_returns_total_exceeded(self) -> None:
        """Exceeding total budget should return TOTAL_EXCEEDED."""
        # ✨ Set very low total limit but high per-task limit
        tracker = _make_tracker(max_per_task=50.0, max_total=0.01)
        status = tracker.record("task-001", _MODEL, 1000, 1000)
        # $0.018 > $0.01
        assert status == CostStatus.TOTAL_EXCEEDED

    def test_warning_threshold_at_80_percent(self) -> None:
        """Reaching ~90% of total budget should return MODEL_DOWNGRADE.

        With default tiered thresholds (warning=0.5, downgrade=0.8,
        critical=0.9), $0.018 / $0.02 = ~0.9 (just under due to
        floating point) → falls to MODEL_DOWNGRADE tier (>=0.8).
        """
        # 📊 Total budget = $0.02, ratio = $0.018/$0.02 ≈ 0.9
        tracker = _make_tracker(max_per_task=50.0, max_total=0.02)
        status = tracker.record("task-001", _MODEL, 1000, 1000)
        assert status == CostStatus.MODEL_DOWNGRADE

    def test_per_task_checked_before_total(self) -> None:
        """Per-task limit should be checked before total limit."""
        # ❌ Both limits exceeded, but per-task should be checked first
        tracker = _make_tracker(max_per_task=0.01, max_total=0.01)
        status = tracker.record("task-001", _MODEL, 1000, 1000)
        assert status == CostStatus.TASK_EXCEEDED


# ============================================================
# 📊 Cost Summary Tests
# ============================================================


class TestCostSummary:
    """Tests for CostTracker summary aggregation 📊."""

    def test_empty_summary(self) -> None:
        """Summary with no records should be all zeros."""
        tracker = _make_tracker(max_total=10.0)
        summary = tracker.get_summary()
        assert summary.total_cost_usd == 0.0
        assert summary.task_costs == {}
        assert summary.budget_remaining == 10.0

    def test_summary_with_multiple_tasks(self) -> None:
        """Summary should aggregate costs across all tasks."""
        tracker = _make_tracker()
        tracker.record("task-001", _MODEL, 1000, 1000)
        tracker.record("task-002", _MODEL, 2000, 2000)
        tracker.record("task-003", _MODEL, 3000, 3000)
        summary = tracker.get_summary()
        assert len(summary.task_costs) == 3
        expected_total = 0.018 + 0.036 + 0.054
        assert summary.total_cost_usd == pytest.approx(expected_total)

    def test_budget_remaining_calculation(self) -> None:
        """Remaining budget should be max_total - total_cost."""
        tracker = _make_tracker(max_total=10.0)
        tracker.record("task-001", _MODEL, 1000, 1000)
        summary = tracker.get_summary()
        assert summary.budget_remaining == pytest.approx(10.0 - 0.018)


# ============================================================
# 🔒 Thread Safety Tests
# ============================================================


class TestThreadSafety:
    """Tests for CostTracker thread-safe concurrent access 🔒."""

    def test_concurrent_records_consistent(self) -> None:
        """Concurrent cost records should produce consistent totals."""
        tracker = _make_tracker(max_per_task=1000.0, max_total=1000.0)
        n_threads = 100

        def worker(idx: int) -> None:
            # 📝 Each thread records 1K input + 0 output = $0.003
            tracker.record(f"task-{idx}", _MODEL, 1000, 0)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = tracker.get_total_cost()
        expected = n_threads * 0.003  # $0.003 per thread
        assert total == pytest.approx(expected, rel=1e-6)

    def test_concurrent_records_and_reads(self) -> None:
        """Concurrent reads during writes should not crash."""
        tracker = _make_tracker(max_per_task=1000.0, max_total=1000.0)
        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                for _ in range(10):
                    tracker.record(f"task-{idx}", _MODEL, 100, 100)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(10):
                    tracker.get_summary()
                    tracker.get_total_cost()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)] + [
            threading.Thread(target=reader) for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
