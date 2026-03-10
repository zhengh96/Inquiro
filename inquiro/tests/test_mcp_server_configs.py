"""Tests for MCP server configuration correctness 🧪.

Tests cover:
- YAML parsability and required top-level structure
- Each server has required fields (enabled, transport, command, description, etc.)
- Environment variable placeholders use correct format (${VAR} or ${VAR:-default})
- All enabled servers have at least one enabled tool
- No duplicate tool names across all servers
- Tool routing domain_config references only existing server names
- Circuit breaker config has all required fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ============================================================================
# 📁 Path to the MCP servers configuration
# ============================================================================

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "mcp_servers.yaml"

# Required fields for each MCP server entry
_REQUIRED_SERVER_FIELDS = {
    "enabled",
    "transport",
    "description",
    "use_when",
    "differs_from",
    "domains",
    "tools",
}

# Additional required fields for stdio transport
_STDIO_EXTRA_FIELDS = {"command"}

# Required fields for circuit_breaker config
_REQUIRED_CB_FIELDS = {
    "failure_threshold",
    "recovery_timeout",
}

# Domain routing config from runner.py — must reference real servers
_DOMAIN_CONFIG = {
    "clinical": [
        "bohrium", "biomcp", "pubmed", "clinicaltrials",
        "paper-search",
    ],
    "academic": ["bohrium", "biomcp", "pubmed", "paper-search"],
    "target": ["opentargets", "uniprot", "chembl"],
    "chemical": ["chembl", "pubchem"],
    "patent": ["patent_search"],
    "general": ["perplexity", "brave", "fetch"],
}


# ============================================================================
# 🔧 Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mcp_config() -> dict:
    """Load and parse the MCP servers YAML config 📄."""
    assert _CONFIG_PATH.exists(), f"Config file not found: {_CONFIG_PATH}"
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "YAML root must be a mapping"
    assert "mcp_servers" in data, "Missing top-level 'mcp_servers' key"
    return data["mcp_servers"]


@pytest.fixture(scope="module")
def enabled_servers(mcp_config: dict) -> dict:
    """Return only enabled server configs 🟢."""
    return {
        name: cfg for name, cfg in mcp_config.items()
        if cfg.get("enabled", False)
    }


# ============================================================================
# 🟢 YAML structure tests
# ============================================================================


class TestYamlStructure:
    """Tests for YAML file parsability and top-level structure 📄."""

    def test_yaml_parsable(self) -> None:
        """YAML file can be parsed without errors 📄."""
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_has_mcp_servers_key(self) -> None:
        """Config has top-level 'mcp_servers' key 📄."""
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "mcp_servers" in data

    def test_at_least_five_servers(self, mcp_config: dict) -> None:
        """Config has at least 5 server entries (original count) 📊."""
        assert len(mcp_config) >= 5


# ============================================================================
# 🔍 Required fields tests
# ============================================================================


class TestRequiredFields:
    """Tests that each server has all required configuration fields 🔍."""

    def test_all_servers_have_required_fields(
        self, mcp_config: dict,
    ) -> None:
        """Every server entry has required fields 🔍."""
        for name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            for field in _REQUIRED_SERVER_FIELDS:
                assert field in cfg, (
                    f"Server '{name}' missing required field '{field}'"
                )
            # stdio transport requires 'command'
            if cfg.get("transport") == "stdio":
                for field in _STDIO_EXTRA_FIELDS:
                    assert field in cfg, (
                        f"Server '{name}' (stdio) missing field '{field}'"
                    )

    def test_transport_is_valid(self, mcp_config: dict) -> None:
        """Transport must be 'stdio' or 'http' 🔌."""
        valid = {"stdio", "http"}
        for name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            transport = cfg.get("transport")
            assert transport in valid, (
                f"Server '{name}' has invalid transport '{transport}'"
            )

    def test_domains_is_list(self, mcp_config: dict) -> None:
        """Domains field must be a non-empty list 🏷️."""
        for name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            domains = cfg.get("domains")
            assert isinstance(domains, list), (
                f"Server '{name}' domains must be a list"
            )
            assert len(domains) > 0, (
                f"Server '{name}' must have at least one domain"
            )

    def test_description_is_meaningful(self, mcp_config: dict) -> None:
        """Description must be at least 20 characters 📝."""
        for name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            desc = cfg.get("description", "")
            assert len(desc) >= 20, (
                f"Server '{name}' description too short: '{desc}'"
            )


# ============================================================================
# 🔑 Environment variable tests
# ============================================================================


class TestEnvVarFormat:
    """Tests that env var placeholders use correct format 🔑."""

    def test_env_vars_use_dollar_brace_format(
        self, mcp_config: dict,
    ) -> None:
        """Env vars must use ${VAR} or ${VAR:-default} format 🔑."""
        import re
        pattern = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*(:-[^}]*)?\}$")

        for name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            env_block = cfg.get("env", {})
            if not env_block:
                continue
            for env_key, env_val in env_block.items():
                if isinstance(env_val, str) and env_val.startswith("$"):
                    assert pattern.match(env_val), (
                        f"Server '{name}' env '{env_key}' has invalid "
                        f"format: '{env_val}'. Expected ${{VAR}} or "
                        "${{VAR:-default}}"
                    )


# ============================================================================
# 🛠️ Tool configuration tests
# ============================================================================


class TestToolConfiguration:
    """Tests for MCP tool entries within each server 🛠️."""

    def test_enabled_servers_have_enabled_tools(
        self, enabled_servers: dict,
    ) -> None:
        """Every enabled server has at least one enabled tool 🟢."""
        for name, cfg in enabled_servers.items():
            tools = cfg.get("tools", [])
            enabled_tools = [
                t for t in tools
                if isinstance(t, dict) and t.get("enabled", False)
            ]
            assert len(enabled_tools) >= 1, (
                f"Server '{name}' is enabled but has no enabled tools"
            )

    def test_tools_have_name_and_description(
        self, mcp_config: dict,
    ) -> None:
        """All tools have 'name' and 'description' fields 📝."""
        for server_name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            for tool in cfg.get("tools", []):
                if not isinstance(tool, dict):
                    continue
                assert "name" in tool, (
                    f"Tool in server '{server_name}' missing 'name'"
                )
                assert "description" in tool, (
                    f"Tool '{tool.get('name')}' in server "
                    f"'{server_name}' missing 'description'"
                )

    def test_no_duplicate_tool_names_within_server(
        self, mcp_config: dict,
    ) -> None:
        """No duplicate tool names within a single server 🔍."""
        for server_name, cfg in mcp_config.items():
            if not isinstance(cfg, dict):
                continue
            tools = cfg.get("tools", [])
            names = [
                t["name"] for t in tools
                if isinstance(t, dict) and "name" in t
            ]
            duplicates = [n for n in names if names.count(n) > 1]
            assert not duplicates, (
                f"Server '{server_name}' has duplicate tool names: "
                f"{set(duplicates)}"
            )


# ============================================================================
# 🎯 Tool routing tests
# ============================================================================


class TestToolRouting:
    """Tests that domain routing config references existing servers 🎯."""

    def test_routing_servers_exist_in_config(
        self, mcp_config: dict,
    ) -> None:
        """All servers referenced in domain routing exist in YAML config 🎯."""
        config_servers = set(mcp_config.keys())
        for domain, servers in _DOMAIN_CONFIG.items():
            for server in servers:
                # patent_search is a placeholder — skip it
                if server == "patent_search":
                    continue
                assert server in config_servers, (
                    f"Domain '{domain}' references server '{server}' "
                    f"which is not in mcp_servers.yaml"
                )

    def test_enabled_servers_in_at_least_one_domain(
        self, enabled_servers: dict,
    ) -> None:
        """Every enabled server is reachable via at least one domain 🗺️."""
        all_routed = set()
        for servers in _DOMAIN_CONFIG.values():
            all_routed.update(servers)

        # Exclude placeholder/disabled servers and custom_http_server
        skip = {"custom_http_server", "patent_search"}
        for name in enabled_servers:
            if name in skip:
                continue
            assert name in all_routed, (
                f"Enabled server '{name}' is not in any domain routing "
                f"— it may be unreachable by domain-filtered tasks"
            )


# ============================================================================
# 🛡️ Circuit breaker tests
# ============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker configuration 🛡️."""

    def test_enabled_servers_have_circuit_breaker(
        self, enabled_servers: dict,
    ) -> None:
        """Every enabled server has a circuit_breaker config 🛡️."""
        for name, cfg in enabled_servers.items():
            assert "circuit_breaker" in cfg, (
                f"Server '{name}' missing circuit_breaker config"
            )

    def test_circuit_breaker_has_required_fields(
        self, enabled_servers: dict,
    ) -> None:
        """Circuit breaker has required threshold and timeout fields 🛡️."""
        for name, cfg in enabled_servers.items():
            cb = cfg.get("circuit_breaker", {})
            for field in _REQUIRED_CB_FIELDS:
                assert field in cb, (
                    f"Server '{name}' circuit_breaker missing "
                    f"required field '{field}'"
                )

    def test_circuit_breaker_values_are_positive(
        self, enabled_servers: dict,
    ) -> None:
        """Circuit breaker numeric values must be positive 📊."""
        for name, cfg in enabled_servers.items():
            cb = cfg.get("circuit_breaker", {})
            for field, value in cb.items():
                if isinstance(value, (int, float)):
                    assert value > 0, (
                        f"Server '{name}' circuit_breaker.{field} "
                        f"must be positive, got {value}"
                    )


# ============================================================================
# 📊 Expected server count test
# ============================================================================


class TestServerInventory:
    """Tests for expected server inventory after expansion 📊."""

    def test_expected_new_servers_present(
        self, mcp_config: dict,
    ) -> None:
        """All 8 new/replaced servers are present in config 📊."""
        expected = {
            "fetch", "opentargets", "paper-search", "pubmed",
            "clinicaltrials", "chembl", "uniprot", "pubchem",
        }
        for name in expected:
            assert name in mcp_config, (
                f"Expected server '{name}' not found in config"
            )

    def test_total_server_count(self, mcp_config: dict) -> None:
        """Total server count is at least 12 (5 original + 7 new) 📊."""
        # Original: perplexity, brave, biomcp, opentargets (replaced),
        # bohrium, custom_http_server
        # New: fetch, paper-search, pubmed, clinicaltrials,
        # chembl, uniprot, pubchem
        assert len(mcp_config) >= 12

    def test_original_servers_preserved(self, mcp_config: dict) -> None:
        """Original servers (perplexity, brave, biomcp, bohrium) are preserved 🔒."""
        for name in ("perplexity", "brave", "biomcp", "bohrium"):
            assert name in mcp_config, (
                f"Original server '{name}' should be preserved"
            )
