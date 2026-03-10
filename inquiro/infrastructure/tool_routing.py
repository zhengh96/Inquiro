"""ToolRoutingStrategy -- domain-based MCP tool filtering 🎯.

Dynamically filters tools from a ToolRegistry based on task domain
tags, reducing prompt token usage by excluding irrelevant MCP servers.

Built-in tools (finish, reflect, think, etc.) are always included
regardless of domain. When no domain is specified or the domain is
unknown, all tools are returned as a fallback.

The strategy is stateless and thread-safe: it holds only an immutable
domain-to-server mapping and performs read-only filtering.

Example::

    strategy = ToolRoutingStrategy({
        "clinical": ["bohrium", "biomcp"],
        "patent": ["patent_search"],
        "general": ["perplexity", "brave"],
    })
    filtered = strategy.filter_tools(registry, domain="clinical")
    # => [bohrium tools, biomcp tools, finish, reflect, think, ...]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evomaster.agent.tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


class ToolRoutingStrategy:
    """Domain-based tool routing for MCP tool filtering 🎯.

    Maps domain names to lists of MCP server names. When filtering,
    only MCP tools belonging to the matched servers are included.
    Built-in (non-MCP) tools are always preserved.

    Attributes:
        _domain_config: Immutable mapping of domain names to server
            name lists. Set once at init, never mutated.
    """

    def __init__(
        self,
        domain_config: dict[str, list[str]],
    ) -> None:
        """Initialize ToolRoutingStrategy 🔧.

        Args:
            domain_config: Mapping of domain names to lists of MCP
                server names. Example:
                ``{"clinical": ["bohrium", "biomcp"]}``.
        """
        # 🔒 Store as a frozen copy to ensure thread safety
        self._domain_config: dict[str, list[str]] = {
            k: list(v) for k, v in domain_config.items()
        }

    def filter_tools(
        self,
        registry: ToolRegistry,
        domain: str | list[str] | None = None,
    ) -> list[BaseTool]:
        """Filter tools from registry based on domain 🔍.

        Built-in tools (non-MCP, i.e. ``_is_mcp_tool`` is False or
        absent) are always included. MCP tools are filtered by their
        ``_mcp_server`` attribute against the allowed server set for
        the given domain(s).

        Fallback behavior: if ``domain`` is None, empty, or not found
        in the config, all tools are returned unfiltered.

        Args:
            registry: ToolRegistry containing all available tools.
            domain: Single domain string, list of domain strings for
                union filtering, or None for no filtering.

        Returns:
            Filtered list of BaseTool instances.
        """
        all_tools: list[Any] = registry.get_all_tools()

        # 🎯 Resolve allowed server set from domain(s)
        allowed_servers = self._resolve_servers(domain)

        if allowed_servers is None:
            # ✨ No filtering — return all tools
            logger.debug(
                "🔧 Tool routing: no domain filter, returning all %d tools",
                len(all_tools),
            )
            return all_tools

        # 🔍 Filter: keep built-ins + MCP tools on allowed servers
        filtered: list[Any] = []
        for tool in all_tools:
            is_mcp = getattr(tool, "_is_mcp_tool", False)
            if not is_mcp:
                # ✅ Built-in tool — always included
                filtered.append(tool)
            else:
                server = getattr(tool, "_mcp_server", None)
                if server in allowed_servers:
                    filtered.append(tool)

        logger.info(
            "🔧 Tool routing: domain=%s, servers=%s, filtered %d/%d tools",
            domain,
            sorted(allowed_servers),
            len(filtered),
            len(all_tools),
        )

        return filtered

    def get_servers_for_domain(
        self,
        domain: str,
    ) -> list[str]:
        """Get MCP server names configured for a domain 📋.

        Args:
            domain: Domain name to look up.

        Returns:
            List of server names, or empty list if domain not found.
        """
        return list(self._domain_config.get(domain, []))

    def _resolve_servers(
        self,
        domain: str | list[str] | None,
    ) -> set[str] | None:
        """Resolve domain(s) to a set of allowed server names 🔍.

        Args:
            domain: Single domain string, list of domain strings,
                or None.

        Returns:
            Set of allowed server names, or None if no filtering
            should be applied (fallback to all tools).
        """
        if domain is None:
            return None

        # 📝 Normalize to list
        domains = [domain] if isinstance(domain, str) else domain

        # 🔧 Collect union of servers from all domains
        servers: set[str] = set()
        for d in domains:
            if d in self._domain_config:
                servers.update(self._domain_config[d])

        # ✨ If no domain matched, return None (fallback to all)
        if not servers:
            logger.debug(
                "🔧 Tool routing: domain(s) %s not in config, "
                "falling back to all tools",
                domains,
            )
            return None

        return servers
