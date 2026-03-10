"""Tests for Inquiro startup connectivity gate ✅."""

from __future__ import annotations

import pytest

from inquiro.api.app import _assert_required_startup_connectivity
from inquiro.api.schemas import PreflightResponse, ServiceCheckResult


def _build_preflight(
    *,
    llm: list[tuple[str, str]],
    mcp: list[tuple[str, str]],
) -> PreflightResponse:
    """Build minimal PreflightResponse fixture helper 🧪."""
    return PreflightResponse(
        status="all_healthy",
        llm_checks=[
            ServiceCheckResult(name=name, status=status) for name, status in llm
        ],
        mcp_checks=[
            ServiceCheckResult(name=name, status=status) for name, status in mcp
        ],
    )


def test_startup_gate_passes_when_all_required_services_up() -> None:
    """All six required services up should pass ✅."""
    result = _build_preflight(
        llm=[
            ("claude-bedrock", "up"),
            ("gpt-5", "up"),
            ("gemini-3-pro", "up"),
        ],
        mcp=[
            ("bohrium", "up"),
            ("perplexity", "up"),
            ("brave", "up"),
        ],
    )

    _assert_required_startup_connectivity(result)


def test_startup_gate_fails_when_required_service_missing() -> None:
    """Missing any required service should fail startup ❌."""
    result = _build_preflight(
        llm=[
            ("claude-bedrock", "up"),
            ("gpt-5", "up"),
            # gemini-3-pro missing
        ],
        mcp=[
            ("bohrium", "up"),
            ("perplexity", "up"),
            ("brave", "up"),
        ],
    )

    with pytest.raises(RuntimeError, match="gemini-3-pro"):
        _assert_required_startup_connectivity(result)


def test_startup_gate_fails_when_required_service_down() -> None:
    """Down status on required service should fail startup ❌."""
    result = _build_preflight(
        llm=[
            ("claude-bedrock", "up"),
            ("gpt-5", "down"),
            ("gemini-3-pro", "up"),
        ],
        mcp=[
            ("bohrium", "up"),
            ("perplexity", "up"),
            ("brave", "up"),
        ],
    )

    with pytest.raises(RuntimeError, match="LLM\\[gpt-5\\]=down"):
        _assert_required_startup_connectivity(result)
