"""Tests for Inquiro infrastructure pools and config loader 🧪.

Tests cover:
- ConfigLoader: YAML loading, env var interpolation, error handling
- LLMProviderPool: initialization, get_llm, fallback, errors
- MCPConnectionPool: initialization, get_tools, server filtering,
  get_health, call_tool
- MCPToolWrapper: execution, spec generation, error handling,
  circuit breaker integration
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from inquiro.infrastructure.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from inquiro.infrastructure.config_loader import (
    ConfigLoadError,
    ConfigLoader,
    _interpolate_env_vars,
)
from inquiro.infrastructure.llm_pool import LLMProviderError, LLMProviderPool
from inquiro.infrastructure.mcp_pool import MCPConnectionPool, MCPPoolError
from inquiro.tools.mcp_tool_wrapper import MCPToolWrapper


# ============================================================================
# 📋 ConfigLoader Tests
# ============================================================================


class TestInterpolateEnvVars:
    """Tests for env var interpolation helper 🔍."""

    def test_simple_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve a simple ${VAR} placeholder ✅."""
        monkeypatch.setenv("TEST_KEY", "secret123")
        assert _interpolate_env_vars("${TEST_KEY}") == "secret123"

    def test_env_var_with_default(self) -> None:
        """Use default value when env var is not set ✅."""
        # ✨ Ensure var is unset
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        result = _interpolate_env_vars("${NONEXISTENT_VAR_12345:-fallback}")
        assert result == "fallback"

    def test_env_var_missing_no_default(self) -> None:
        """Keep placeholder when env var is missing and no default ⚠️."""
        os.environ.pop("MISSING_VAR_99999", None)
        result = _interpolate_env_vars("${MISSING_VAR_99999}")
        assert result == "${MISSING_VAR_99999}"

    def test_nested_dict_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recursively resolve env vars in nested dicts ✅."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        data = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
            }
        }
        result = _interpolate_env_vars(data)
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == "5432"

    def test_list_interpolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve env vars in list elements ✅."""
        monkeypatch.setenv("ITEM_A", "alpha")
        data = ["${ITEM_A}", "static", "${ITEM_A}"]
        result = _interpolate_env_vars(data)
        assert result == ["alpha", "static", "alpha"]

    def test_non_string_passthrough(self) -> None:
        """Non-string values pass through unchanged ✅."""
        assert _interpolate_env_vars(42) == 42
        assert _interpolate_env_vars(True) is True
        assert _interpolate_env_vars(None) is None

    def test_mixed_text_and_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolve env var embedded in a larger string ✅."""
        monkeypatch.setenv("API_HOST", "api.example.com")
        result = _interpolate_env_vars("https://${API_HOST}/v1")
        assert result == "https://api.example.com/v1"

    def test_alias_fallback_for_openai_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback to OPENAI_API_BASE when OPENAI_BASE_URL is unset 🔄."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_BASE", "https://legacy-gateway.example/v1")
        result = _interpolate_env_vars("${OPENAI_BASE_URL}")
        assert result == "https://legacy-gateway.example/v1"

    def test_canonical_openai_base_url_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Canonical OPENAI_BASE_URL wins when both vars are set ✅."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://canonical-gateway.example/v1")
        monkeypatch.setenv("OPENAI_API_BASE", "https://legacy-gateway.example/v1")
        result = _interpolate_env_vars("${OPENAI_BASE_URL}")
        assert result == "https://canonical-gateway.example/v1"

    def test_alias_fallback_for_brave_search_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback to BRAVE_API_KEY when BRAVE_SEARCH_API_KEY is unset 🔄."""
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "brave-key-legacy")
        result = _interpolate_env_vars("${BRAVE_SEARCH_API_KEY}")
        assert result == "brave-key-legacy"


class TestConfigLoader:
    """Tests for ConfigLoader class 📋."""

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        """Load a simple YAML config file ✅."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "service.yaml").write_text(yaml.dump({"service": {"port": 8100}}))
        loader = ConfigLoader(config_dir)
        config = loader.get_service_config()
        assert config["service"]["port"] == 8100

    def test_env_var_in_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resolve ${ENV_VAR} in loaded YAML ✅."""
        monkeypatch.setenv("MY_API_KEY", "key-abc-123")
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "llm_providers.yaml").write_text(
            yaml.dump(
                {
                    "providers": {
                        "test-model": {
                            "api_key": "${MY_API_KEY}",
                            "model_id": "test",
                        }
                    }
                }
            )
        )
        loader = ConfigLoader(config_dir)
        config = loader.get_llm_config()
        assert config["providers"]["test-model"]["api_key"] == "key-abc-123"

    def test_missing_directory_raises(self) -> None:
        """Raise ConfigLoadError for nonexistent directory ❌."""
        with pytest.raises(ConfigLoadError, match="does not exist"):
            ConfigLoader("/nonexistent/path/12345")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raise ConfigLoadError for missing YAML file ❌."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        loader = ConfigLoader(config_dir)
        with pytest.raises(ConfigLoadError, match="not found"):
            loader.get_mcp_config()

    def test_caching(self, tmp_path: Path) -> None:
        """Second load returns cached result ✅."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "service.yaml").write_text(yaml.dump({"key": "value"}))
        loader = ConfigLoader(config_dir)
        first = loader.get_service_config()
        second = loader.get_service_config()
        assert first is second  # Same object from cache

    def test_reload_clears_cache(self, tmp_path: Path) -> None:
        """reload() clears the cache ✅."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "service.yaml").write_text(yaml.dump({"key": "v1"}))
        loader = ConfigLoader(config_dir)
        _first = loader.get_service_config()

        # ✏️ Modify the file
        (config_dir / "service.yaml").write_text(yaml.dump({"key": "v2"}))
        loader.reload()
        second = loader.get_service_config()
        assert second["key"] == "v2"

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Raise ConfigLoadError for malformed YAML ❌."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "service.yaml").write_text("{{invalid yaml::")
        loader = ConfigLoader(config_dir)
        with pytest.raises(ConfigLoadError, match="Failed to parse"):
            loader.get_service_config()


# ============================================================================
# 🤖 LLMProviderPool Tests
# ============================================================================


class TestLLMProviderPool:
    """Tests for LLMProviderPool class 🤖."""

    def test_init_empty(self) -> None:
        """Initialize with no config (stub mode) ✅."""
        pool = LLMProviderPool()
        assert pool.get_available_models() == []

    def test_init_with_providers(self) -> None:
        """Initialize with provider config ✅."""
        pool = LLMProviderPool(
            config={
                "providers": {
                    "model-a": {"provider": "openai", "api_key": "k"},
                    "model-b": {"provider": "anthropic", "api_key": "k"},
                },
                "default_model": "model-a",
            }
        )
        assert pool.default_model == "model-a"
        assert sorted(pool.get_available_models()) == ["model-a", "model-b"]

    def test_get_llm_no_model_no_default_raises(self) -> None:
        """Raise LLMProviderError when no model specified and no default ❌."""
        pool = LLMProviderPool(config={"providers": {}})
        with pytest.raises(LLMProviderError, match="No model specified"):
            pool.get_llm()

    def test_get_llm_unknown_model_raises(self) -> None:
        """Raise LLMProviderError for unknown model without default ❌."""
        pool = LLMProviderPool(
            config={
                "providers": {},
                "default_model": "",
            }
        )
        with pytest.raises(LLMProviderError, match="No provider configuration"):
            pool.get_llm("nonexistent")

    @patch("inquiro.infrastructure.llm_pool.create_llm")
    def test_get_llm_creates_and_caches(self, mock_create: MagicMock) -> None:
        """get_llm creates instance on first call and caches it ✅."""
        mock_llm = MagicMock()
        mock_create.return_value = mock_llm

        pool = LLMProviderPool(
            config={
                "providers": {
                    "test-model": {
                        "provider": "openai",
                        "model_id": "gpt-4",
                        "api_key": "test-key",
                    },
                },
                "default_model": "test-model",
            }
        )

        # 🔧 First call creates
        result1 = pool.get_llm("test-model")
        assert result1 is mock_llm
        assert mock_create.call_count == 1

        # 🔄 Second call returns cached
        result2 = pool.get_llm("test-model")
        assert result2 is mock_llm
        assert mock_create.call_count == 1  # No new creation

    @patch("inquiro.infrastructure.llm_pool.create_llm")
    def test_get_llm_fallback_to_default(self, mock_create: MagicMock) -> None:
        """Fall back to default model when requested model not found ✅."""
        mock_llm = MagicMock()
        mock_create.return_value = mock_llm

        pool = LLMProviderPool(
            config={
                "providers": {
                    "default-model": {
                        "provider": "openai",
                        "model_id": "gpt-4",
                        "api_key": "test-key",
                    },
                },
                "default_model": "default-model",
            }
        )

        result = pool.get_llm("unknown-model")
        assert result is mock_llm

    def test_close(self) -> None:
        """close() clears cached instances ✅."""
        pool = LLMProviderPool()
        pool._instances["test"] = MagicMock()
        pool.close()
        assert len(pool._instances) == 0


# ============================================================================
# 🔌 MCPConnectionPool Tests
# ============================================================================


class TestMCPConnectionPool:
    """Tests for MCPConnectionPool class 🔌."""

    @pytest.fixture()
    def sample_mcp_config(self) -> dict[str, Any]:
        """Provide sample MCP config for tests 📋."""
        return {
            "mcp_servers": {
                "server_a": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "echo",
                    "args": [],
                    "timeout_seconds": 30,
                    "circuit_breaker": {
                        "failure_threshold": 3,
                        "timeout_seconds": 60,
                    },
                    "tools": [
                        {
                            "name": "tool_one",
                            "enabled": True,
                            "description": "First test tool",
                        },
                        {
                            "name": "tool_two",
                            "enabled": True,
                            "description": "Second test tool",
                        },
                        {
                            "name": "tool_disabled",
                            "enabled": False,
                            "description": "Disabled tool",
                        },
                    ],
                },
                "server_b": {
                    "enabled": True,
                    "transport": "http",
                    "endpoint": "http://localhost:9000",
                    "timeout_seconds": 20,
                    "tools": [
                        {
                            "name": "tool_three",
                            "enabled": True,
                            "description": "Third test tool",
                        },
                    ],
                },
                "server_disabled": {
                    "enabled": False,
                    "transport": "stdio",
                    "command": "echo",
                    "tools": [
                        {
                            "name": "tool_never",
                            "enabled": True,
                            "description": "Never loaded",
                        },
                    ],
                },
            }
        }

    def test_init_empty(self) -> None:
        """Initialize with no config (stub mode) ✅."""
        pool = MCPConnectionPool()
        assert len(pool.servers) == 0

    def test_init_with_config(self, sample_mcp_config: dict[str, Any]) -> None:
        """Initialize with server config ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        assert len(pool.servers) == 3

    @pytest.mark.asyncio
    async def test_initialize(self, sample_mcp_config: dict[str, Any]) -> None:
        """initialize() sets initialized flag ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        await pool.initialize()
        assert pool._initialized is True

    def test_get_tools_all_enabled(self, sample_mcp_config: dict[str, Any]) -> None:
        """get_tools() with no filter returns all enabled tools ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools()
        # ✨ server_a: 2 enabled, server_b: 1 enabled, server_disabled: 0
        assert len(registry) == 3
        names = registry.get_tool_names()
        assert "mcp__server_a__tool_one" in names
        assert "mcp__server_a__tool_two" in names
        assert "mcp__server_b__tool_three" in names

    def test_get_tools_filtered(self, sample_mcp_config: dict[str, Any]) -> None:
        """get_tools() with specific servers only returns their tools ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools(["server_a"])
        assert len(registry) == 2
        names = registry.get_tool_names()
        assert "mcp__server_a__tool_one" in names
        assert "mcp__server_b__tool_three" not in names

    def test_get_tools_disabled_server_skipped(
        self, sample_mcp_config: dict[str, Any]
    ) -> None:
        """Disabled servers are skipped even if explicitly requested ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools(["server_disabled"])
        assert len(registry) == 0

    def test_get_tools_disabled_tool_skipped(
        self, sample_mcp_config: dict[str, Any]
    ) -> None:
        """Disabled tools within enabled servers are skipped ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools(["server_a"])
        names = registry.get_tool_names()
        assert "mcp__server_a__tool_disabled" not in names

    def test_get_tools_unknown_server_warning(
        self, sample_mcp_config: dict[str, Any]
    ) -> None:
        """Unknown server names produce empty results (no crash) ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools(["nonexistent_server"])
        assert len(registry) == 0

    def test_get_server_config(self, sample_mcp_config: dict[str, Any]) -> None:
        """get_server_config returns correct config ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        config = pool.get_server_config("server_a")
        assert config["transport"] == "stdio"

    def test_get_server_config_not_found(
        self, sample_mcp_config: dict[str, Any]
    ) -> None:
        """get_server_config raises for unknown server ❌."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        with pytest.raises(MCPPoolError, match="not found"):
            pool.get_server_config("nonexistent")

    def test_get_enabled_servers(self, sample_mcp_config: dict[str, Any]) -> None:
        """get_enabled_servers returns only enabled ones ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        enabled = pool.get_enabled_servers()
        assert "server_a" in enabled
        assert "server_b" in enabled
        assert "server_disabled" not in enabled

    def test_circuit_breaker_created(self, sample_mcp_config: dict[str, Any]) -> None:
        """Circuit breakers are created from server configs ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        cb = pool.get_circuit_breaker("server_a")
        assert cb.server_name == "server_a"

    @pytest.mark.asyncio
    async def test_close(self, sample_mcp_config: dict[str, Any]) -> None:
        """close() clears caches and resets state ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        await pool.initialize()
        # 🔧 Force populate cache
        pool.get_tools()
        await pool.close()
        assert pool._initialized is False
        assert len(pool._tool_cache) == 0

    def test_mcp_tool_markers(self, sample_mcp_config: dict[str, Any]) -> None:
        """MCPToolWrapper instances have correct MCP markers ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools(["server_a"])
        mcp_tools = registry.get_mcp_tools()
        assert len(mcp_tools) == 2
        for tool in mcp_tools:
            assert getattr(tool, "_is_mcp_tool", False) is True
            assert getattr(tool, "_mcp_server", None) == "server_a"

    def test_tools_by_server(self, sample_mcp_config: dict[str, Any]) -> None:
        """ToolRegistry.get_tools_by_server works with MCPToolWrapper ✅."""
        pool = MCPConnectionPool(config=sample_mcp_config)
        registry = pool.get_tools()  # All enabled
        server_a_tools = registry.get_tools_by_server("server_a")
        assert len(server_a_tools) == 2
        server_b_tools = registry.get_tools_by_server("server_b")
        assert len(server_b_tools) == 1


# ============================================================================
# 🔧 MCPToolWrapper Tests
# ============================================================================


class TestMCPToolWrapper:
    """Tests for MCPToolWrapper class 🔧."""

    def _make_wrapper(self) -> MCPToolWrapper:
        """Create a test wrapper instance 🏗️."""
        return MCPToolWrapper(
            server_name="test_server",
            tool_name="test_tool",
            tool_description="A test MCP tool for testing",
            server_config={
                "transport": "stdio",
                "command": "echo",
                "timeout_seconds": 30,
            },
        )

    def test_name_format(self) -> None:
        """Tool name follows mcp__<server>__<tool> convention ✅."""
        wrapper = self._make_wrapper()
        assert wrapper.name == "mcp__test_server__test_tool"

    def test_mcp_markers(self) -> None:
        """Tool has correct MCP identification markers ✅."""
        wrapper = self._make_wrapper()
        assert wrapper._is_mcp_tool is True
        assert wrapper._mcp_server == "test_server"

    def test_execute_returns_tuple(self) -> None:
        """execute() returns (observation, info) tuple ✅."""
        wrapper = self._make_wrapper()
        session = MagicMock()
        observation, info = wrapper.execute(session, '{"query": "test"}')
        assert isinstance(observation, str)
        assert isinstance(info, dict)
        assert info["server"] == "test_server"
        assert info["tool"] == "test_tool"
        assert info["success"] is True

    def test_execute_with_invalid_json(self) -> None:
        """execute() handles non-JSON args gracefully ✅."""
        wrapper = self._make_wrapper()
        session = MagicMock()
        observation, info = wrapper.execute(session, "plain text query")
        assert isinstance(observation, str)
        assert info["success"] is True

    def test_execute_with_empty_args(self) -> None:
        """execute() handles empty args ✅."""
        wrapper = self._make_wrapper()
        session = MagicMock()
        observation, info = wrapper.execute(session, "")
        assert info["success"] is True

    def test_get_tool_spec(self) -> None:
        """get_tool_spec() returns valid ToolSpec ✅."""
        wrapper = self._make_wrapper()
        spec = wrapper.get_tool_spec()
        assert spec.type == "function"
        assert spec.function.name == "mcp__test_server__test_tool"
        assert "test MCP tool" in spec.function.description

    def test_repr(self) -> None:
        """__repr__ is informative ✅."""
        wrapper = self._make_wrapper()
        r = repr(wrapper)
        assert "MCPToolWrapper" in r
        assert "test_server" in r

    def test_execute_with_pool_delegates(self) -> None:
        """execute() delegates to pool.call_tool() when pool is set ✅."""
        mock_pool = MagicMock()
        mock_pool.call_tool.return_value = "search result from MCP"
        mock_cb = MagicMock()
        mock_cb.can_execute.return_value = True

        wrapper = MCPToolWrapper(
            server_name="test_server",
            tool_name="test_tool",
            tool_description="Test tool",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            circuit_breaker=mock_cb,
        )
        session = MagicMock()
        observation, info = wrapper.execute(session, '{"query": "test"}')

        assert observation == "search result from MCP"
        assert info["success"] is True
        mock_pool.call_tool.assert_called_once_with(
            "test_server", "test_tool", {"query": "test"}
        )
        mock_cb.can_execute.assert_called_once()
        mock_cb.record_success.assert_called_once()

    def test_execute_circuit_open_rejects(self) -> None:
        """execute() returns error when circuit breaker is open 🔴."""
        mock_pool = MagicMock()
        mock_cb = MagicMock()
        mock_cb.can_execute.return_value = False

        wrapper = MCPToolWrapper(
            server_name="test_server",
            tool_name="test_tool",
            tool_description="Test tool",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            circuit_breaker=mock_cb,
        )
        session = MagicMock()
        observation, info = wrapper.execute(session, '{"query": "test"}')

        assert info["success"] is False
        assert info["error"] == "circuit_open"
        assert "OPEN" in observation
        # 🔍 Pool should NOT be called when circuit is open
        mock_pool.call_tool.assert_not_called()

    def test_execute_pool_error_records_failure(self) -> None:
        """execute() records failure on CB when pool raises ❌."""
        mock_pool = MagicMock()
        mock_pool.call_tool.side_effect = RuntimeError("timeout")
        mock_cb = MagicMock()
        mock_cb.can_execute.return_value = True

        wrapper = MCPToolWrapper(
            server_name="test_server",
            tool_name="test_tool",
            tool_description="Test tool",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            circuit_breaker=mock_cb,
        )
        session = MagicMock()
        observation, info = wrapper.execute(session, '{"query": "test"}')

        assert info["success"] is False
        assert "timeout" in info["error"]
        mock_cb.record_failure.assert_called_once()
        mock_cb.record_success.assert_not_called()

    def test_execute_no_pool_no_cb_standalone(self) -> None:
        """execute() works in standalone mode without pool or CB ✅."""
        wrapper = MCPToolWrapper(
            server_name="standalone",
            tool_name="search",
            tool_description="Standalone tool",
            server_config={"transport": "stdio"},
        )
        session = MagicMock()
        observation, info = wrapper.execute(session, '{"q": "hello"}')

        assert info["success"] is True
        assert "standalone/search" in observation
        assert wrapper.pool is None
        assert wrapper.circuit_breaker is None

    def test_unwrap_params_wrapper(self) -> None:
        """execute() unwraps {"params": {actual args}} 🔄."""
        mock_pool = MagicMock()
        mock_pool.call_tool.return_value = "result"
        wrapper = MCPToolWrapper(
            server_name="bohrium",
            tool_name="search",
            tool_description="Search papers",
            server_config={"transport": "stdio"},
            pool=mock_pool,
        )
        session = MagicMock()
        wrapper.execute(
            session,
            '{"params": {"query": "TL1A obesity"}}',
        )
        mock_pool.call_tool.assert_called_once_with(
            "bohrium",
            "search",
            {"query": "TL1A obesity"},
        )

    def test_alias_limit_to_page_size(self) -> None:
        """execute() maps 'limit' → 'page_size' when schema known 🔄."""
        mock_pool = MagicMock()
        mock_pool.call_tool.return_value = "result"
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "page_size": {"type": "integer"},
            },
            "required": ["query"],
        }
        wrapper = MCPToolWrapper(
            server_name="bohrium",
            tool_name="search",
            tool_description="Search papers",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            input_schema=schema,
        )
        session = MagicMock()
        wrapper.execute(
            session,
            '{"query": "test", "limit": 10}',
        )
        mock_pool.call_tool.assert_called_once_with(
            "bohrium",
            "search",
            {"query": "test", "page_size": 10},
        )

    def test_strip_unknown_fields(self) -> None:
        """execute() strips fields not in input_schema 🗑️."""
        mock_pool = MagicMock()
        mock_pool.call_tool.return_value = "result"
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        wrapper = MCPToolWrapper(
            server_name="bohrium",
            tool_name="search",
            tool_description="Search papers",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            input_schema=schema,
        )
        session = MagicMock()
        wrapper.execute(
            session,
            '{"query": "test", "bogus_field": 42}',
        )
        mock_pool.call_tool.assert_called_once_with(
            "bohrium",
            "search",
            {"query": "test"},
        )

    def test_alias_output_format_to_response_format(self) -> None:
        """execute() maps 'output_format' → 'response_format' 🔄."""
        mock_pool = MagicMock()
        mock_pool.call_tool.return_value = "result"
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "response_format": {"type": "string"},
            },
            "required": ["query"],
        }
        wrapper = MCPToolWrapper(
            server_name="bohrium",
            tool_name="search",
            tool_description="Search papers",
            server_config={"transport": "stdio"},
            pool=mock_pool,
            input_schema=schema,
        )
        session = MagicMock()
        wrapper.execute(
            session,
            '{"query": "test", "output_format": "json"}',
        )
        mock_pool.call_tool.assert_called_once_with(
            "bohrium",
            "search",
            {"query": "test", "response_format": "json"},
        )


# ============================================================================
# ❤️ MCPConnectionPool.get_health() Tests
# ============================================================================


class TestMCPConnectionPoolHealth:
    """Tests for MCPConnectionPool.get_health() ❤️."""

    @pytest.fixture()
    def health_config(self) -> dict[str, Any]:
        """Provide config with multiple servers for health tests 📋."""
        return {
            "mcp_servers": {
                "healthy_server": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "echo",
                    "tools": [
                        {"name": "t1", "description": "Tool 1"},
                    ],
                },
                "degraded_server": {
                    "enabled": True,
                    "transport": "http",
                    "endpoint": "http://localhost:9000",
                    "tools": [
                        {"name": "t2", "description": "Tool 2"},
                    ],
                },
                "disabled_server": {
                    "enabled": False,
                    "transport": "stdio",
                    "command": "echo",
                    "tools": [],
                },
            }
        }

    def test_get_health_all_closed(self, health_config: dict[str, Any]) -> None:
        """All enabled servers report 'connected' by default ✅."""
        pool = MCPConnectionPool(config=health_config)
        health = pool.get_health()

        assert "healthy_server" in health
        assert "degraded_server" in health
        # 🔍 Disabled servers should NOT appear
        assert "disabled_server" not in health
        assert health["healthy_server"] == "connected"
        assert health["degraded_server"] == "connected"

    def test_get_health_open_circuit(self, health_config: dict[str, Any]) -> None:
        """Server with open circuit reports 'disconnected' 🔴."""
        pool = MCPConnectionPool(config=health_config)
        # 🔧 Force circuit breaker to OPEN
        cb = pool.get_circuit_breaker("healthy_server")
        for _ in range(3):
            cb.record_failure()
        assert cb.get_state() == CircuitState.OPEN

        health = pool.get_health()
        assert health["healthy_server"] == "disconnected"

    def test_get_health_half_open_circuit(self, health_config: dict[str, Any]) -> None:
        """Server with half-open circuit reports 'degraded' ⚠️."""
        pool = MCPConnectionPool(config=health_config)
        # 🔧 Use a CB with instant recovery for testing
        fast_cb_config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.0)
        cb_registry = CircuitBreakerRegistry(
            per_server_configs={
                "healthy_server": fast_cb_config,
            }
        )
        pool.cb_registry = cb_registry
        cb = pool.get_circuit_breaker("healthy_server")
        cb.record_failure()  # 🔴 Opens the circuit
        # ⏰ With recovery_timeout=0, get_state() transitions to HALF_OPEN
        assert cb.get_state() == CircuitState.HALF_OPEN

        health = pool.get_health()
        assert health["healthy_server"] == "degraded"

    def test_get_health_empty_pool(self) -> None:
        """Empty pool returns empty health dict ✅."""
        pool = MCPConnectionPool()
        health = pool.get_health()
        assert health == {}


# ============================================================================
# 🔧 MCPConnectionPool.call_tool() Tests
# ============================================================================


class TestMCPConnectionPoolCallTool:
    """Tests for MCPConnectionPool.call_tool() 🔧."""

    @pytest.fixture()
    def call_config(self) -> dict[str, Any]:
        """Provide config for call_tool tests 📋."""
        return {
            "mcp_servers": {
                "stdio_server": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "echo",
                    "args": [],
                    "timeout_seconds": 5,
                    "tools": [
                        {"name": "search", "description": "Search"},
                    ],
                },
                "http_server": {
                    "enabled": True,
                    "transport": "http",
                    "endpoint": "http://localhost:9999",
                    "timeout_seconds": 5,
                    "tools": [
                        {"name": "fetch", "description": "Fetch"},
                    ],
                },
                "disabled_server": {
                    "enabled": False,
                    "transport": "stdio",
                    "command": "echo",
                    "tools": [],
                },
            }
        }

    def test_call_tool_unknown_server_raises(self, call_config: dict[str, Any]) -> None:
        """call_tool() raises MCPPoolError for unknown server ❌."""
        pool = MCPConnectionPool(config=call_config)
        with pytest.raises(MCPPoolError, match="not found"):
            pool.call_tool("nonexistent", "search", {})

    def test_call_tool_disabled_server_raises(
        self, call_config: dict[str, Any]
    ) -> None:
        """call_tool() raises MCPPoolError for disabled server ❌."""
        pool = MCPConnectionPool(config=call_config)
        with pytest.raises(MCPPoolError, match="disabled"):
            pool.call_tool("disabled_server", "search", {})

    def test_call_tool_unsupported_transport_raises(
        self, call_config: dict[str, Any]
    ) -> None:
        """call_tool() raises for unsupported transport ❌."""
        pool = MCPConnectionPool(config=call_config)
        # 🔧 Hack transport to trigger error
        pool.servers["stdio_server"]["transport"] = "grpc"
        with pytest.raises(MCPPoolError, match="Unsupported transport"):
            pool.call_tool("stdio_server", "search", {})

    @patch("subprocess.Popen")
    def test_call_tool_stdio_success(
        self,
        mock_popen: MagicMock,
        call_config: dict[str, Any],
    ) -> None:
        """call_tool() via stdio returns extracted text ✅."""
        import json
        from io import StringIO

        # 🔧 Build mock responses: init response + tools/call response
        init_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        call_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "content": [{"type": "text", "text": "Evidence found"}],
                },
            }
        )

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = StringIO(f"{init_resp}\n{call_resp}\n")
        mock_proc.stderr = StringIO("")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        pool = MCPConnectionPool(config=call_config)
        result = pool.call_tool("stdio_server", "search", {"query": "test"})

        assert result == "Evidence found"
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_call_tool_stdio_timeout(
        self,
        mock_popen: MagicMock,
        call_config: dict[str, Any],
    ) -> None:
        """call_tool() via stdio raises on subprocess timeout ❌."""
        import subprocess as _subprocess

        mock_popen.side_effect = _subprocess.TimeoutExpired(cmd="echo", timeout=5)

        pool = MCPConnectionPool(config=call_config)
        with pytest.raises(MCPPoolError, match="timed out"):
            pool.call_tool("stdio_server", "search", {"query": "test"})

    @patch("subprocess.Popen")
    def test_call_tool_stdio_nonzero_exit(
        self,
        mock_popen: MagicMock,
        call_config: dict[str, Any],
    ) -> None:
        """call_tool() via stdio raises on init RPC error ❌."""
        import json
        from io import StringIO

        # 🔧 Return error on initialize
        error_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "Bad request"},
            }
        )

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = StringIO(f"{error_resp}\n")
        mock_proc.stderr = StringIO("")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        pool = MCPConnectionPool(config=call_config)
        with pytest.raises(MCPPoolError, match="initialize failed"):
            pool.call_tool("stdio_server", "search", {"query": "test"})

    @patch("subprocess.Popen")
    def test_call_tool_stdio_rpc_error(
        self,
        mock_popen: MagicMock,
        call_config: dict[str, Any],
    ) -> None:
        """call_tool() via stdio raises on JSON-RPC error response ❌."""
        import json
        from io import StringIO

        init_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        error_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "error": {"code": -32601, "message": "Method not found"},
            }
        )

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = StringIO(f"{init_resp}\n{error_resp}\n")
        mock_proc.stderr = StringIO("")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        pool = MCPConnectionPool(config=call_config)
        with pytest.raises(MCPPoolError, match="MCP error"):
            pool.call_tool("stdio_server", "search", {"query": "test"})

    def test_call_tool_stdio_no_command(
        self,
        call_config: dict[str, Any],
    ) -> None:
        """call_tool() raises when no command configured ❌."""
        pool = MCPConnectionPool(config=call_config)
        pool.servers["stdio_server"]["command"] = ""

        with pytest.raises(MCPPoolError, match="No command"):
            pool.call_tool("stdio_server", "search", {"query": "test"})

    def test_call_tool_http_no_endpoint(self, call_config: dict[str, Any]) -> None:
        """call_tool() raises when no HTTP endpoint configured ❌."""
        pool = MCPConnectionPool(config=call_config)
        pool.servers["http_server"]["endpoint"] = ""

        with pytest.raises(MCPPoolError, match="No endpoint"):
            pool.call_tool("http_server", "fetch", {"url": "http://x"})


# ============================================================================
# 🔌 MCPToolWrapper + Pool Integration Tests
# ============================================================================


class TestMCPToolWrapperPoolIntegration:
    """Integration tests for MCPToolWrapper wired to MCPConnectionPool 🔌."""

    @pytest.fixture()
    def pool_config(self) -> dict[str, Any]:
        """Provide integration test config 📋."""
        return {
            "mcp_servers": {
                "int_server": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "echo",
                    "args": [],
                    "timeout_seconds": 5,
                    "circuit_breaker": {
                        "failure_threshold": 2,
                        "timeout_seconds": 60,
                    },
                    "tools": [
                        {
                            "name": "int_tool",
                            "description": "Integration test tool",
                        },
                    ],
                },
            }
        }

    def test_wrappers_have_pool_and_cb(self, pool_config: dict[str, Any]) -> None:
        """Wrappers created by pool have pool and CB references ✅."""
        pool = MCPConnectionPool(config=pool_config)
        registry = pool.get_tools(["int_server"])
        tools = registry.get_all_tools()

        assert len(tools) == 1
        wrapper = tools[0]
        assert isinstance(wrapper, MCPToolWrapper)
        assert wrapper.pool is pool
        assert wrapper.circuit_breaker is not None
        assert wrapper.circuit_breaker.server_name == "int_server"

    @patch("subprocess.Popen")
    def test_end_to_end_execute_through_pool(
        self,
        mock_popen: MagicMock,
        pool_config: dict[str, Any],
    ) -> None:
        """Full execute() flow: wrapper → CB → pool → stdio ✅."""
        import json
        from io import StringIO

        init_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        call_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "content": [{"type": "text", "text": "Found 3 results"}],
                },
            }
        )

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = StringIO(f"{init_resp}\n{call_resp}\n")
        mock_proc.stderr = StringIO("")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        pool = MCPConnectionPool(config=pool_config)
        registry = pool.get_tools(["int_server"])
        wrapper = registry.get_all_tools()[0]
        session = MagicMock()

        observation, info = wrapper.execute(session, '{"query": "evidence"}')

        assert observation == "Found 3 results"
        assert info["success"] is True
        assert info["server"] == "int_server"
        mock_popen.assert_called_once()

    @patch("subprocess.Popen")
    def test_circuit_breaker_opens_after_failures(
        self,
        mock_popen: MagicMock,
        pool_config: dict[str, Any],
    ) -> None:
        """CB opens after threshold failures, subsequent calls rejected 🔴."""
        mock_popen.side_effect = RuntimeError("connection refused")

        pool = MCPConnectionPool(config=pool_config)
        registry = pool.get_tools(["int_server"])
        wrapper = registry.get_all_tools()[0]
        session = MagicMock()

        # 🔧 Trigger 2 failures (threshold = 2)
        for _ in range(2):
            obs, info = wrapper.execute(session, '{"q": "test"}')
            assert info["success"] is False

        # 🔴 Circuit should now be OPEN
        cb = pool.get_circuit_breaker("int_server")
        assert cb.get_state() == CircuitState.OPEN

        # ⚡ Next call should be rejected by circuit breaker
        obs, info = wrapper.execute(session, '{"q": "test"}')
        assert info["success"] is False
        assert info["error"] == "circuit_open"


# ============================================================================
# 📖 Bohrium MCP Server Config Tests
# ============================================================================


class TestBohriumMCPConfig:
    """Tests for Bohrium MCP Server configuration and integration 📖."""

    @pytest.fixture()
    def bohrium_config(self) -> dict[str, Any]:
        """Provide Bohrium server config matching mcp_servers.yaml 📋."""
        return {
            "mcp_servers": {
                "bohrium": {
                    "enabled": True,
                    "transport": "stdio",
                    "command": "python",
                    "args": ["reference/PharmMaster/mcp-servers/bohrium/server.py"],
                    "env": {
                        "BOHRIUM_API_KEY": "test_key_12345",
                    },
                    "timeout_seconds": 120,
                    "max_retries": 2,
                    "circuit_breaker": {
                        "failure_threshold": 3,
                        "timeout_seconds": 120,
                        "recovery_timeout": 60,
                        "success_threshold": 1,
                        "half_open_max_calls": 2,
                    },
                    "tools": [
                        {
                            "name": "search_papers",
                            "enabled": True,
                            "description": ("Search academic papers by keyword"),
                        },
                        {
                            "name": "create_search_session",
                            "enabled": True,
                            "description": ("Create AI-powered search session"),
                        },
                        {
                            "name": "get_session_papers",
                            "enabled": True,
                            "description": ("Get papers from search session"),
                        },
                        {
                            "name": "get_ai_summary",
                            "enabled": True,
                            "description": ("Get AI-generated literature summary"),
                        },
                        {
                            "name": "ask_followup",
                            "enabled": True,
                            "description": ("Ask follow-up in search session"),
                        },
                    ],
                }
            }
        }

    def test_bohrium_pool_creates_five_tools(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """MCPConnectionPool creates 5 tools for bohrium server ✅."""
        pool = MCPConnectionPool(config=bohrium_config)
        registry = pool.get_tools(["bohrium"])
        tools = registry.get_all_tools()

        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {
            "mcp__bohrium__search_papers",
            "mcp__bohrium__create_search_session",
            "mcp__bohrium__get_session_papers",
            "mcp__bohrium__get_ai_summary",
            "mcp__bohrium__ask_followup",
        }

    def test_bohrium_tools_have_descriptions(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """All bohrium tools have non-empty tool_description ✅."""
        pool = MCPConnectionPool(config=bohrium_config)
        registry = pool.get_tools(["bohrium"])
        tools = registry.get_all_tools()

        for tool in tools:
            assert hasattr(tool, "tool_description")
            assert tool.tool_description, f"Tool {tool.name} has empty description"

    def test_bohrium_tools_have_mcp_markers(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """All bohrium tools have correct MCP identification 🏷️."""
        pool = MCPConnectionPool(config=bohrium_config)
        registry = pool.get_tools(["bohrium"])
        tools = registry.get_all_tools()

        for tool in tools:
            assert tool._is_mcp_tool is True
            assert tool._mcp_server == "bohrium"

    def test_bohrium_server_in_enabled_list(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """Bohrium appears in enabled servers list ✅."""
        pool = MCPConnectionPool(config=bohrium_config)
        enabled = pool.get_enabled_servers()
        assert "bohrium" in enabled

    def test_bohrium_circuit_breaker_configured(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """Bohrium has circuit breaker with correct config 🔌."""
        pool = MCPConnectionPool(config=bohrium_config)
        cb = pool.get_circuit_breaker("bohrium")
        assert cb is not None
        assert cb.get_state() == CircuitState.CLOSED

    def test_bohrium_env_var_resolved_in_subprocess(
        self, bohrium_config: dict[str, Any]
    ) -> None:
        """Resolved env var is passed to subprocess env ✅."""
        pool = MCPConnectionPool(config=bohrium_config)
        server_config = pool.get_server_config("bohrium")
        env_vars = server_config.get("env", {})

        # 🔍 The key should be resolved (not start with ${)
        api_key = env_vars.get("BOHRIUM_API_KEY", "")
        assert api_key == "test_key_12345"
        assert not api_key.startswith("${")

    @patch("subprocess.Popen")
    def test_bohrium_search_papers_execute(
        self,
        mock_popen: MagicMock,
        bohrium_config: dict[str, Any],
    ) -> None:
        """Bohrium search_papers executes through MCP protocol ✅."""
        import json
        from io import StringIO

        init_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        )
        call_resp = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Found 3 papers for: EGFR inhibitor\n"
                                "Title: Paper 1\nDOI: 10.1234/test"
                            ),
                        }
                    ],
                },
            }
        )

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = StringIO(f"{init_resp}\n{call_resp}\n")
        mock_proc.stderr = StringIO("")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        pool = MCPConnectionPool(config=bohrium_config)
        registry = pool.get_tools(["bohrium"])

        # 🔍 Find search_papers tool
        search_tool = None
        for tool in registry.get_all_tools():
            if tool.name == "mcp__bohrium__search_papers":
                search_tool = tool
                break

        assert search_tool is not None
        session = MagicMock()
        observation, info = search_tool.execute(
            session,
            '{"query": "EGFR inhibitor resistance"}',
        )

        assert "Found 3 papers" in observation
        assert info["success"] is True
        assert info["server"] == "bohrium"
        assert info["tool"] == "search_papers"


# ============================================================================
# 🔄 MCPConnectionPool.reload_config() Tests
# ============================================================================


class TestMCPConnectionPoolReload:
    """Tests for MCPConnectionPool.reload_config() hot-reload 🔄."""

    @pytest.fixture()
    def initial_config(self) -> dict[str, Any]:
        """Provide initial MCP config for reload tests 📋."""
        return {
            "mcp_servers": {
                "alpha": {
                    "enabled": True,
                    "transport": "stdio",
                    "persistent": False,
                    "command": "echo",
                    "args": [],
                    "timeout_seconds": 10,
                    "tools": [
                        {"name": "t1", "enabled": True, "description": "T1"},
                    ],
                },
            }
        }

    @pytest.fixture()
    def updated_config(self) -> dict[str, Any]:
        """Provide updated MCP config after reload 📋."""
        return {
            "mcp_servers": {
                "alpha": {
                    "enabled": True,
                    "transport": "stdio",
                    "persistent": False,
                    "command": "echo",
                    "args": ["--v2"],
                    "timeout_seconds": 20,
                    "tools": [
                        {"name": "t1", "enabled": True, "description": "T1 v2"},
                        {"name": "t2", "enabled": True, "description": "T2"},
                    ],
                },
                "beta": {
                    "enabled": True,
                    "transport": "http",
                    "persistent": False,
                    "endpoint": "http://localhost:8080",
                    "timeout_seconds": 15,
                    "tools": [
                        {"name": "t3", "enabled": True, "description": "T3"},
                    ],
                },
            }
        }

    @pytest.mark.asyncio
    async def test_reload_config_updates_servers(
        self,
        initial_config: dict[str, Any],
        updated_config: dict[str, Any],
    ) -> None:
        """reload_config() replaces servers with new config ✅."""
        pool = MCPConnectionPool(config=initial_config)
        await pool.initialize()

        # ✅ Verify initial state
        assert "alpha" in pool.servers
        assert "beta" not in pool.servers
        assert len(pool.servers) == 1

        # 🔄 Reload with updated config
        status = await pool.reload_config(updated_config)

        # ✅ Verify servers updated
        assert "alpha" in pool.servers
        assert "beta" in pool.servers
        assert len(pool.servers) == 2

        # ✅ Verify alpha config updated (new timeout)
        assert pool.servers["alpha"]["timeout_seconds"] == 20

        # ✅ Verify status report includes both servers
        assert "alpha" in status
        assert "beta" in status

    @pytest.mark.asyncio
    async def test_reload_config_no_attribute_error(
        self,
        initial_config: dict[str, Any],
    ) -> None:
        """reload_config() does not raise AttributeError on self.config 🐛."""
        pool = MCPConnectionPool(config=initial_config)
        await pool.initialize()

        # 🔄 Reload with same config — should not raise AttributeError
        status = await pool.reload_config(initial_config)
        assert isinstance(status, dict)
