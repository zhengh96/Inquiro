"""Tests for Inquiro preflight connectivity check system 🧪.

Tests cover:
- preflight_check() with various MCP and LLM combinations
- Individual MCP server probe logic (stdio, HTTP, error handling)
- Individual LLM provider probe logic
- Overall status computation (all_healthy, degraded, all_down)
- GET /api/v1/preflight endpoint via AsyncClient
- Integration with app lifespan
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inquiro.api.preflight import (
    _check_llm_providers,
    _check_mcp_servers,
    _compute_overall_status,
    _probe_llm_provider,
    _probe_mcp_server,
    preflight_check,
)
from inquiro.api.schemas import PreflightResponse, ServiceCheckResult


# ============================================================
# 📊 _compute_overall_status Tests
# ============================================================


class TestComputeOverallStatus:
    """Tests for overall status derivation 📊."""

    def test_empty_checks_returns_all_healthy(self) -> None:
        """No checks at all should report all_healthy ✅."""
        assert _compute_overall_status([]) == "all_healthy"

    def test_all_up_returns_all_healthy(self) -> None:
        """All services up should report all_healthy ✅."""
        checks = [
            ServiceCheckResult(name="a", status="up"),
            ServiceCheckResult(name="b", status="up"),
        ]
        assert _compute_overall_status(checks) == "all_healthy"

    def test_all_down_returns_all_down(self) -> None:
        """All services down should report all_down ❌."""
        checks = [
            ServiceCheckResult(name="a", status="down", error="timeout"),
            ServiceCheckResult(name="b", status="down", error="refused"),
        ]
        assert _compute_overall_status(checks) == "all_down"

    def test_mixed_returns_degraded(self) -> None:
        """Some up and some down should report degraded ⚠️."""
        checks = [
            ServiceCheckResult(name="a", status="up"),
            ServiceCheckResult(name="b", status="down", error="timeout"),
        ]
        assert _compute_overall_status(checks) == "degraded"

    def test_single_up_returns_all_healthy(self) -> None:
        """Single service up should report all_healthy ✅."""
        checks = [
            ServiceCheckResult(name="a", status="up"),
        ]
        assert _compute_overall_status(checks) == "all_healthy"

    def test_single_down_returns_all_down(self) -> None:
        """Single service down should report all_down ❌."""
        checks = [
            ServiceCheckResult(name="a", status="down", error="error"),
        ]
        assert _compute_overall_status(checks) == "all_down"


# ============================================================
# 🔌 _check_mcp_servers Tests
# ============================================================


class TestCheckMCPServers:
    """Tests for MCP server probe orchestration 🔌."""

    def test_none_pool_returns_empty(self) -> None:
        """No MCP pool should return empty results ✅."""
        results = _check_mcp_servers(None)
        assert results == []

    def test_no_enabled_servers_returns_empty(self) -> None:
        """Pool with no enabled servers returns empty ✅."""
        mock_pool = MagicMock()
        mock_pool.get_enabled_servers.return_value = []
        results = _check_mcp_servers(mock_pool)
        assert results == []

    @patch("inquiro.api.preflight._probe_mcp_server")
    def test_probes_each_enabled_server(self, mock_probe: MagicMock) -> None:
        """Should probe every enabled server ✅."""
        mock_pool = MagicMock()
        mock_pool.get_enabled_servers.return_value = ["server_a", "server_b"]
        mock_probe.return_value = ServiceCheckResult(
            name="server_a", status="up", latency_ms=50.0
        )

        results = _check_mcp_servers(mock_pool)
        assert len(results) == 2
        assert mock_probe.call_count == 2

    @patch("inquiro.api.preflight._probe_mcp_server")
    def test_mixed_up_and_down(self, mock_probe: MagicMock) -> None:
        """Should handle mix of up and down servers ⚠️."""
        mock_pool = MagicMock()
        mock_pool.get_enabled_servers.return_value = ["healthy", "broken"]
        mock_probe.side_effect = [
            ServiceCheckResult(name="healthy", status="up", latency_ms=30.0),
            ServiceCheckResult(
                name="broken",
                status="down",
                error="connection refused",
            ),
        ]

        results = _check_mcp_servers(mock_pool)
        assert len(results) == 2
        assert results[0].status == "up"
        assert results[1].status == "down"


# ============================================================
# 🔌 _probe_mcp_server Tests
# ============================================================


class TestProbeMCPServer:
    """Tests for individual MCP server probing 🔌."""

    def test_server_config_not_found(self) -> None:
        """Unknown server should return down status ❌."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.side_effect = RuntimeError("not found")
        result = _probe_mcp_server(mock_pool, "unknown")
        assert result.status == "down"
        assert "not found" in result.error

    def test_unsupported_transport(self) -> None:
        """Unsupported transport should return down ❌."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.return_value = {
            "transport": "grpc",
            "timeout_seconds": 5,
        }
        result = _probe_mcp_server(mock_pool, "grpc_server")
        assert result.status == "down"
        assert "Unsupported transport" in result.error

    @patch("inquiro.api.preflight._probe_stdio")
    def test_stdio_success(self, mock_probe_stdio: MagicMock) -> None:
        """Successful stdio probe should return up ✅."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.return_value = {
            "transport": "stdio",
            "command": "echo",
            "timeout_seconds": 5,
        }
        mock_probe_stdio.return_value = None  # No exception = success

        result = _probe_mcp_server(mock_pool, "stdio_server")
        assert result.status == "up"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    @patch("inquiro.api.preflight._probe_stdio")
    def test_stdio_failure(self, mock_probe_stdio: MagicMock) -> None:
        """Failed stdio probe should return down ❌."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.return_value = {
            "transport": "stdio",
            "command": "nonexistent",
            "timeout_seconds": 5,
        }
        mock_probe_stdio.side_effect = RuntimeError("command not found")

        result = _probe_mcp_server(mock_pool, "bad_server")
        assert result.status == "down"
        assert "command not found" in result.error

    @patch("inquiro.api.preflight._probe_http")
    def test_http_success(self, mock_probe_http: MagicMock) -> None:
        """Successful HTTP probe should return up ✅."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.return_value = {
            "transport": "http",
            "endpoint": "http://localhost:9000",
            "timeout_seconds": 5,
        }
        mock_probe_http.return_value = None

        result = _probe_mcp_server(mock_pool, "http_server")
        assert result.status == "up"

    @patch("inquiro.api.preflight._probe_http")
    def test_http_failure(self, mock_probe_http: MagicMock) -> None:
        """Failed HTTP probe should return down ❌."""
        mock_pool = MagicMock()
        mock_pool.get_server_config.return_value = {
            "transport": "http",
            "endpoint": "http://localhost:9999",
            "timeout_seconds": 5,
        }
        mock_probe_http.side_effect = RuntimeError("connection refused")

        result = _probe_mcp_server(mock_pool, "bad_http")
        assert result.status == "down"
        assert "connection refused" in result.error


# ============================================================
# 🤖 _check_llm_providers Tests
# ============================================================


class TestCheckLLMProviders:
    """Tests for LLM provider probe orchestration 🤖."""

    def test_none_pool_returns_empty(self) -> None:
        """No LLM pool should return empty results ✅."""
        results = _check_llm_providers(None)
        assert results == []

    def test_no_default_model_returns_empty(self) -> None:
        """Pool with no default model should return empty ✅."""
        mock_pool = MagicMock()
        mock_pool.default_model = ""
        results = _check_llm_providers(mock_pool)
        assert results == []

    @patch("inquiro.api.preflight._probe_llm_provider")
    def test_probes_default_model(self, mock_probe: MagicMock) -> None:
        """Should probe the default model ✅."""
        mock_pool = MagicMock()
        mock_pool.default_model = "claude-sonnet"
        mock_probe.return_value = ServiceCheckResult(
            name="claude-sonnet", status="up", latency_ms=200.0
        )

        results = _check_llm_providers(mock_pool)
        assert len(results) == 1
        assert results[0].name == "claude-sonnet"
        assert results[0].status == "up"
        mock_probe.assert_called_once_with(mock_pool, "claude-sonnet")

    @patch("inquiro.api.preflight._probe_llm_provider")
    def test_llm_down(self, mock_probe: MagicMock) -> None:
        """Down LLM should be reported correctly ❌."""
        mock_pool = MagicMock()
        mock_pool.default_model = "broken-model"
        mock_probe.return_value = ServiceCheckResult(
            name="broken-model",
            status="down",
            error="API key invalid",
        )

        results = _check_llm_providers(mock_pool)
        assert len(results) == 1
        assert results[0].status == "down"

    @patch("inquiro.api.preflight._probe_llm_provider")
    def test_skips_unconfigured_optional_provider(self, mock_probe: MagicMock) -> None:
        """Providers with unresolved secrets should be skipped ⏭️."""
        mock_pool = MagicMock()
        mock_pool.get_available_models.return_value = [
            "claude-bedrock",
            "claude-sonnet",
        ]
        mock_pool._providers = {
            "claude-bedrock": {
                "provider": "bedrock",
                "aws_access_key_id": "AKIA_TEST",
                "aws_secret_access_key": "SECRET_TEST",
            },
            "claude-sonnet": {
                "provider": "anthropic",
                "api_key": "${ANTHROPIC_API_KEY}",
            },
        }
        mock_probe.return_value = ServiceCheckResult(
            name="claude-bedrock", status="up", latency_ms=150.0
        )

        results = _check_llm_providers(mock_pool)
        assert len(results) == 1
        assert results[0].name == "claude-bedrock"
        mock_probe.assert_called_once_with(mock_pool, "claude-bedrock")


# ============================================================
# 🤖 _probe_llm_provider Tests
# ============================================================


class TestProbeLLMProvider:
    """Tests for individual LLM provider probing 🤖."""

    def test_successful_ping(self) -> None:
        """Successful LLM call should return up ✅."""
        mock_pool = MagicMock()
        mock_llm = MagicMock()
        mock_llm._call.return_value = MagicMock(content="pong")
        mock_pool.get_llm.return_value = mock_llm

        result = _probe_llm_provider(mock_pool, "test-model")
        assert result.status == "up"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0
        mock_llm._call.assert_called_once()
        # 🔍 Verify minimal request
        call_args = mock_llm._call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["content"] == "ping"

    def test_llm_get_fails(self) -> None:
        """Failure to get LLM instance should return down ❌."""
        mock_pool = MagicMock()
        mock_pool.get_llm.side_effect = RuntimeError("No provider config")

        result = _probe_llm_provider(mock_pool, "missing-model")
        assert result.status == "down"
        assert "No provider config" in result.error

    def test_llm_call_fails(self) -> None:
        """LLM call failure should return down ❌."""
        mock_pool = MagicMock()
        mock_llm = MagicMock()
        mock_llm._call.side_effect = ConnectionError("timeout")
        mock_pool.get_llm.return_value = mock_llm

        result = _probe_llm_provider(mock_pool, "timeout-model")
        assert result.status == "down"
        assert "timeout" in result.error


# ============================================================
# 🔍 preflight_check() Integration Tests
# ============================================================


class TestPreflightCheck:
    """Tests for the top-level preflight_check function 🔍."""

    @pytest.mark.asyncio
    async def test_all_healthy(self) -> None:
        """All services up should return all_healthy ✅."""
        mock_mcp = MagicMock()
        mock_mcp.get_enabled_servers.return_value = ["server_a"]
        mock_mcp.get_server_config.return_value = {
            "transport": "stdio",
            "command": "echo",
            "timeout_seconds": 5,
        }

        mock_llm = MagicMock()
        mock_llm.default_model = "test-model"
        mock_llm_instance = MagicMock()
        mock_llm_instance._call.return_value = MagicMock(content="pong")
        mock_llm.get_llm.return_value = mock_llm_instance

        with patch("inquiro.api.preflight._probe_stdio") as mock_stdio:
            mock_stdio.return_value = None
            result = await preflight_check(mcp_pool=mock_mcp, llm_pool=mock_llm)

        assert result.status == "all_healthy"
        assert len(result.mcp_checks) == 1
        assert len(result.llm_checks) == 1
        assert result.mcp_checks[0].status == "up"
        assert result.llm_checks[0].status == "up"

    @pytest.mark.asyncio
    async def test_all_down(self) -> None:
        """All services down should return all_down ❌."""
        mock_mcp = MagicMock()
        mock_mcp.get_enabled_servers.return_value = ["broken"]
        mock_mcp.get_server_config.return_value = {
            "transport": "stdio",
            "command": "nonexistent",
            "timeout_seconds": 5,
        }

        mock_llm = MagicMock()
        mock_llm.default_model = "bad-model"
        mock_llm.get_llm.side_effect = RuntimeError("no key")

        with patch("inquiro.api.preflight._probe_stdio") as mock_stdio:
            mock_stdio.side_effect = RuntimeError("not found")
            result = await preflight_check(mcp_pool=mock_mcp, llm_pool=mock_llm)

        assert result.status == "all_down"
        assert result.mcp_checks[0].status == "down"
        assert result.llm_checks[0].status == "down"

    @pytest.mark.asyncio
    async def test_degraded(self) -> None:
        """Mixed results should return degraded ⚠️."""
        mock_mcp = MagicMock()
        mock_mcp.get_enabled_servers.return_value = ["up_server"]
        mock_mcp.get_server_config.return_value = {
            "transport": "stdio",
            "command": "echo",
            "timeout_seconds": 5,
        }

        mock_llm = MagicMock()
        mock_llm.default_model = "bad-model"
        mock_llm.get_llm.side_effect = RuntimeError("no key")

        with patch("inquiro.api.preflight._probe_stdio") as mock_stdio:
            mock_stdio.return_value = None
            result = await preflight_check(mcp_pool=mock_mcp, llm_pool=mock_llm)

        assert result.status == "degraded"
        assert result.mcp_checks[0].status == "up"
        assert result.llm_checks[0].status == "down"

    @pytest.mark.asyncio
    async def test_no_pools(self) -> None:
        """No pools should return all_healthy (vacuous) ✅."""
        result = await preflight_check(mcp_pool=None, llm_pool=None)
        assert result.status == "all_healthy"
        assert result.mcp_checks == []
        assert result.llm_checks == []

    @pytest.mark.asyncio
    async def test_response_has_timestamp(self) -> None:
        """Response should include a UTC timestamp 🕐."""
        result = await preflight_check(mcp_pool=None, llm_pool=None)
        assert result.timestamp is not None


# ============================================================
# 🌐 GET /api/v1/preflight Endpoint Tests
# ============================================================


class TestPreflightEndpoint:
    """Tests for the /api/v1/preflight API endpoint 🌐."""

    @pytest.mark.asyncio
    async def test_preflight_returns_200(self, async_client) -> None:
        """Preflight endpoint should return 200 ✅."""
        with patch(
            "inquiro.api.preflight.preflight_check",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = PreflightResponse(
                status="all_healthy",
                mcp_checks=[],
                llm_checks=[],
            )
            response = await async_client.get("/api/v1/preflight")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_preflight_returns_correct_structure(self, async_client) -> None:
        """Preflight response should match PreflightResponse schema ✅."""
        with patch(
            "inquiro.api.preflight.preflight_check",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = PreflightResponse(
                status="degraded",
                mcp_checks=[
                    ServiceCheckResult(
                        name="perplexity",
                        status="up",
                        latency_ms=45.2,
                    ),
                ],
                llm_checks=[
                    ServiceCheckResult(
                        name="claude-sonnet",
                        status="down",
                        error="API key invalid",
                    ),
                ],
            )
            response = await async_client.get("/api/v1/preflight")

        data = response.json()
        assert data["status"] == "degraded"
        assert len(data["mcp_checks"]) == 1
        assert data["mcp_checks"][0]["name"] == "perplexity"
        assert data["mcp_checks"][0]["status"] == "up"
        assert len(data["llm_checks"]) == 1
        assert data["llm_checks"][0]["name"] == "claude-sonnet"
        assert data["llm_checks"][0]["status"] == "down"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_preflight_all_healthy(self, async_client) -> None:
        """All services up should report all_healthy ✅."""
        with patch(
            "inquiro.api.preflight.preflight_check",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = PreflightResponse(
                status="all_healthy",
                mcp_checks=[
                    ServiceCheckResult(
                        name="server_a",
                        status="up",
                        latency_ms=30.0,
                    ),
                ],
                llm_checks=[
                    ServiceCheckResult(
                        name="test-model",
                        status="up",
                        latency_ms=200.0,
                    ),
                ],
            )
            response = await async_client.get("/api/v1/preflight")

        data = response.json()
        assert data["status"] == "all_healthy"


# ============================================================
# 📦 Pydantic Model Tests
# ============================================================


class TestPreflightModels:
    """Tests for preflight Pydantic response models 📦."""

    def test_service_check_result_up(self) -> None:
        """ServiceCheckResult with up status ✅."""
        result = ServiceCheckResult(
            name="test-server",
            status="up",
            latency_ms=42.5,
        )
        assert result.name == "test-server"
        assert result.status == "up"
        assert result.latency_ms == 42.5
        assert result.error is None

    def test_service_check_result_down(self) -> None:
        """ServiceCheckResult with down status ❌."""
        result = ServiceCheckResult(
            name="broken-server",
            status="down",
            error="connection refused",
        )
        assert result.status == "down"
        assert result.error == "connection refused"
        assert result.latency_ms is None

    def test_preflight_response_serialization(self) -> None:
        """PreflightResponse should serialize to valid JSON 📦."""
        response = PreflightResponse(
            status="degraded",
            mcp_checks=[
                ServiceCheckResult(name="s1", status="up", latency_ms=10.0),
                ServiceCheckResult(
                    name="s2",
                    status="down",
                    error="timeout",
                ),
            ],
            llm_checks=[
                ServiceCheckResult(
                    name="model-a",
                    status="up",
                    latency_ms=150.0,
                ),
            ],
        )
        data = response.model_dump()
        assert data["status"] == "degraded"
        assert len(data["mcp_checks"]) == 2
        assert len(data["llm_checks"]) == 1
        assert "timestamp" in data

    def test_preflight_response_json_round_trip(self) -> None:
        """PreflightResponse should survive JSON round-trip 🔄."""
        original = PreflightResponse(
            status="all_healthy",
            mcp_checks=[
                ServiceCheckResult(name="s1", status="up", latency_ms=5.0),
            ],
            llm_checks=[],
        )
        json_str = original.model_dump_json()
        restored = PreflightResponse.model_validate_json(json_str)
        assert restored.status == original.status
        assert len(restored.mcp_checks) == 1
        assert restored.mcp_checks[0].name == "s1"
