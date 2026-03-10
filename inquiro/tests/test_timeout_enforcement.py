"""Tests for wall-clock timeout enforcement in EvalTaskRunner 🧪.

Verifies that the runner correctly provides timeout infrastructure:
default timeout values, build_timeout_result helpers, and
build_synthesis_timeout_result helpers.

Tests:
- Default timeout value (600s) used when not configured
- _build_timeout_result and _build_synthesis_timeout_result helpers
"""

from __future__ import annotations

from unittest.mock import MagicMock

from inquiro.core.runner import EvalTaskRunner, _DEFAULT_TIMEOUT_SECONDS
from inquiro.core.types import (
    CostGuardConfig,
    Decision,
    EvaluationResult,
    SynthesisResult,
)


# ============================================================
# ⚙️ Default timeout value tests
# ============================================================


class TestDefaultTimeoutValue:
    """Default timeout is 1200s when not configured ⚙️."""

    def test_default_timeout_seconds_constant(self) -> None:
        """Module-level default should be 1200.0 seconds."""
        assert _DEFAULT_TIMEOUT_SECONDS == 1200.0

    def test_cost_guard_default_timeout(self) -> None:
        """CostGuardConfig default timeout_seconds should be 1200.0."""
        config = CostGuardConfig()
        assert config.timeout_seconds == 1200.0

    def test_resolve_timeout_with_configured_value(self) -> None:
        """_resolve_timeout should return configured timeout value."""
        config = CostGuardConfig(timeout_seconds=120.0)
        result = EvalTaskRunner._resolve_timeout(config)
        assert result == 120.0

    def test_resolve_timeout_with_default(self) -> None:
        """_resolve_timeout should return default when not configured."""
        config = CostGuardConfig()
        result = EvalTaskRunner._resolve_timeout(config)
        assert result == 1200.0

    def test_resolve_timeout_fallback_for_missing_attr(self) -> None:
        """_resolve_timeout should fallback if attr missing entirely."""
        # 🔧 Simulate an object without timeout_seconds
        mock_guard = MagicMock(spec=[])
        result = EvalTaskRunner._resolve_timeout(mock_guard)
        assert result == _DEFAULT_TIMEOUT_SECONDS


# ============================================================
# 🏗️ Build timeout result tests
# ============================================================


class TestBuildTimeoutResult:
    """Tests for _build_timeout_result and _build_synthesis_timeout_result 🏗️."""

    def test_build_research_timeout_result_structure(self) -> None:
        """Research timeout result should be valid EvaluationResult."""
        result = EvalTaskRunner._build_timeout_result(
            "task-123",
            300.0,
        )
        assert isinstance(result, EvaluationResult)
        assert result.task_id == "task-123"
        assert result.decision == Decision.NEGATIVE
        assert result.confidence == 0.0
        assert result.search_rounds == 0
        assert "300" in result.gaps_remaining[0]

    def test_build_synthesis_timeout_result_structure(self) -> None:
        """Synthesis timeout result should be valid SynthesisResult."""
        result = EvalTaskRunner._build_synthesis_timeout_result(
            "task-456",
            900.0,
        )
        assert isinstance(result, SynthesisResult)
        assert result.task_id == "task-456"
        assert result.decision == Decision.NEGATIVE
        assert result.confidence == 0.0
        assert result.source_reports == []
        assert "900" in result.gaps_remaining[0]
