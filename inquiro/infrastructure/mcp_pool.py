"""Inquiro MCPConnectionPool — service-level MCP server management 🔌.

Manages MCP server configurations and provides filtered ToolRegistry
instances for specific tasks. Each MCP server tool is wrapped as an
EvoMaster BaseTool via MCPToolWrapper.

Supports two connection modes per server (S4-7):
- **Persistent** (``persistent: true``, default): Long-lived connection
  via PersistentMCPConnection with auto-reconnect and metrics.
- **Per-call** (``persistent: false``): Legacy stateless spawn-per-call
  mode for backward compatibility.

Provides:
- ``get_tools()``: Build a ToolRegistry with MCPToolWrapper instances.
- ``call_tool()``: Execute a single tool call via MCP transport.
- ``get_health()``: Report per-server health from circuit breakers.
- ``get_metrics()``: Per-server connection metrics (queue_wait_ms, etc.).

Example::

    pool = MCPConnectionPool(
        config=config_loader.get_mcp_config(),
        cb_registry=CircuitBreakerRegistry(),
    )
    await pool.initialize()  # Establishes persistent connections
    tools = pool.get_tools(["perplexity", "opentargets"])
    health = pool.get_health()  # {"perplexity": "connected", ...}
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import threading
from typing import Any

from evomaster.agent.tools.base import ToolRegistry

from inquiro.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)
from inquiro.infrastructure.mcp_response_cache import MCPResponseCache
from inquiro.infrastructure.persistent_connection import (
    ConnectionState,
    MCPEventLoop,
    PersistentMCPConnection,
)
from inquiro.tools.mcp_tool_wrapper import MCPToolWrapper

logger = logging.getLogger(__name__)

# 🗺️ Map CircuitState → health status string for API responses
_STATE_TO_HEALTH: dict[CircuitState, str] = {
    CircuitState.CLOSED: "connected",
    CircuitState.HALF_OPEN: "degraded",
    CircuitState.OPEN: "disconnected",
}


class MCPPoolError(Exception):
    """Raised when MCP pool operations fail ❌."""


class MCPConnectionPool:
    """Service-level MCP connection pool 🔌.

    Manages MCP server configurations and provides filtered
    ToolRegistry instances for specific tasks.

    Supports two modes per server (controlled by ``persistent`` config key):
    - **Persistent** (default): Long-lived connections via
      PersistentMCPConnection with auto-reconnect and metrics.
    - **Per-call** (``persistent: false``): Legacy stateless
      spawn-per-call mode for backward compatibility.

    Thread-safe: mutable state protected by a lock.

    Attributes:
        servers: Mapping of server_name -> server config.
        cb_registry: Circuit breaker registry for fault isolation.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cb_registry: CircuitBreakerRegistry | None = None,
        response_cache: MCPResponseCache | None = None,
    ) -> None:
        """Initialize MCPConnectionPool 🔧.

        Args:
            config: MCP configuration dict from ``mcp_servers.yaml``
                with structure::

                    {
                        "mcp_servers": {
                            "<server_name>": {
                                "enabled": true,
                                "transport": "stdio" | "http",
                                "persistent": true,  # default
                                "command": "uvx",
                                "args": [...],
                                "timeout_seconds": 30,
                                "circuit_breaker": {...},
                                "tools": [
                                    {"name": "...", "description": "..."},
                                ],
                            },
                            ...
                        }
                    }

                If None, pool starts empty (stub mode).
            cb_registry: Circuit breaker registry. Creates a new one
                if not provided.
            response_cache: Optional MCPResponseCache for deduplicating
                identical MCP calls across rounds. When None (default),
                no caching is performed and all existing behavior is
                unchanged.
        """
        self._config = config or {}
        # 🔍 Support both top-level keys
        self.servers: dict[str, dict[str, Any]] = self._config.get(
            "mcp_servers", self._config.get("servers", {})
        )
        self.cb_registry = cb_registry or self._build_cb_registry()
        self._response_cache = response_cache
        self._tool_cache: dict[str, list[MCPToolWrapper]] = {}
        self._lock = threading.Lock()
        self._initialized = False

        # 🔌 Persistent connections (S4-1 + S4-2)
        self._mcp_loop: MCPEventLoop | None = None
        self._persistent_conns: dict[str, PersistentMCPConnection] = {}

        logger.info(
            "🔌 MCPConnectionPool created with %d server(s)",
            len(self.servers),
        )

    async def initialize(self) -> None:
        """Perform async initialization and establish persistent connections 🚀.

        Starts the shared MCP event loop and connects to all enabled
        servers that have ``persistent: true`` (default). Servers with
        ``persistent: false`` use per-call spawn mode.
        """
        self._initialized = True
        enabled_servers = [
            (name, cfg)
            for name, cfg in self.servers.items()
            if cfg.get("enabled", False)
        ]

        # 🔄 Start shared MCP event loop (S4-2)
        persistent_servers = [
            (name, cfg) for name, cfg in enabled_servers if cfg.get("persistent", True)
        ]

        if persistent_servers:
            self._mcp_loop = MCPEventLoop()
            self._mcp_loop.start()

            # 🔌 Establish persistent connections (S4-1)
            for server_name, server_config in persistent_servers:
                max_reconnect = server_config.get("max_reconnect_attempts", 5)
                conn = PersistentMCPConnection(
                    server_name=server_name,
                    server_config=server_config,
                    mcp_loop=self._mcp_loop,
                    max_reconnect_attempts=max_reconnect,
                )
                if conn.connect(timeout=30.0):
                    self._persistent_conns[server_name] = conn
                else:
                    logger.warning(
                        "⚠️ Failed to establish persistent connection "
                        "to '%s', falling back to per-call mode",
                        server_name,
                    )

        per_call_count = len(enabled_servers) - len(self._persistent_conns)
        logger.info(
            "✅ MCPConnectionPool initialized: "
            "%d persistent + %d per-call = %d/%d servers",
            len(self._persistent_conns),
            per_call_count,
            len(enabled_servers),
            len(self.servers),
        )

    def get_tools(self, mcp_servers: list[str] | None = None) -> ToolRegistry:
        """Build a ToolRegistry with tools from specified MCP servers 🔧.

        Args:
            mcp_servers: List of MCP server names to include. If None,
                includes tools from ALL enabled servers. If an explicit
                list (even empty), only those servers are included.

        Returns:
            ToolRegistry populated with MCPToolWrapper instances.
        """
        registry = ToolRegistry()
        # ⚠️ Explicit None → all servers; empty list → no servers
        target_servers = (
            list(self.servers.keys()) if mcp_servers is None else mcp_servers
        )

        for server_name in target_servers:
            server_config = self.servers.get(server_name)
            if server_config is None:
                logger.warning("⚠️ MCP server '%s' not found in config", server_name)
                continue
            if not server_config.get("enabled", False):
                logger.debug("⏭️ Skipping disabled MCP server: %s", server_name)
                continue

            # 🏗️ Build tool wrappers for this server
            wrappers = self._get_server_tools(server_name, server_config)
            for wrapper in wrappers:
                registry.register(wrapper)

        logger.info(
            "🔧 Built ToolRegistry with %d tools from %d server(s)",
            len(registry),
            len(target_servers),
        )
        return registry

    def get_server_config(self, server_name: str) -> dict[str, Any]:
        """Get configuration for a specific MCP server 📋.

        Args:
            server_name: MCP server name.

        Returns:
            Server configuration dict.

        Raises:
            MCPPoolError: If server not found.
        """
        config = self.servers.get(server_name)
        if config is None:
            raise MCPPoolError(
                f"MCP server '{server_name}' not found. "
                f"Available: {list(self.servers.keys())}"
            )
        return config

    def get_enabled_servers(self) -> list[str]:
        """Return names of all enabled MCP servers 📋.

        Returns:
            Sorted list of enabled server names.
        """
        return sorted(
            name for name, cfg in self.servers.items() if cfg.get("enabled", False)
        )

    def get_circuit_breaker(self, server_name: str) -> CircuitBreaker:
        """Get the circuit breaker for a specific server 🔌.

        Args:
            server_name: MCP server name.

        Returns:
            CircuitBreaker instance for the server.
        """
        return self.cb_registry.get_breaker(server_name)

    def get_health(self) -> dict[str, str]:
        """Report per-server health based on connection and circuit state ❤️.

        For persistent connections, reports based on ConnectionState.
        For per-call servers, reports based on circuit breaker state.

        Status values:
        - ``"connected"``: persistent READY or circuit CLOSED.
        - ``"degraded"``: persistent RECONNECTING or circuit HALF_OPEN.
        - ``"disconnected"``: persistent FAILED/CLOSED or circuit OPEN.
        - ``"connecting"``: persistent connection establishing.

        Returns:
            Dict mapping enabled server names to health status strings.
        """
        # 🔄 Map persistent ConnectionState → health string
        _conn_state_health: dict[ConnectionState, str] = {
            ConnectionState.INIT: "connecting",
            ConnectionState.CONNECTING: "connecting",
            ConnectionState.READY: "connected",
            ConnectionState.FAILED: "disconnected",
            ConnectionState.RECONNECTING: "degraded",
            ConnectionState.CLOSED: "disconnected",
        }

        health: dict[str, str] = {}
        for server_name in self.get_enabled_servers():
            # 🔌 Check persistent connection first
            if server_name in self._persistent_conns:
                conn_state = self._persistent_conns[server_name].state
                health[server_name] = _conn_state_health.get(conn_state, "unknown")
            else:
                # 📡 Fall back to circuit breaker state
                cb = self.cb_registry.get_breaker(server_name)
                state = cb.get_state()
                health[server_name] = _STATE_TO_HEALTH.get(state, "unknown")
        return health

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a single MCP tool call via the configured transport 🚀.

        Dispatches to persistent connection first; on any failure,
        automatically falls back to per-call subprocess mode (Task 8).
        For non-persistent servers, goes directly to per-call.

        Records queue_wait_ms metric for each call (S4-6).

        Args:
            server_name: MCP server name.
            tool_name: Tool name on the server.
            arguments: Parsed argument dict to pass to the tool.

        Returns:
            Text observation from the MCP server.

        Raises:
            MCPPoolError: If the server is not found, disabled, or
                the transport type is unsupported.
            Exception: If both persistent and per-call modes fail,
                the per-call exception is re-raised.
        """
        # 🔍 Validate server
        server_config = self.servers.get(server_name)
        if server_config is None:
            raise MCPPoolError(
                f"MCP server '{server_name}' not found. "
                f"Available: {list(self.servers.keys())}"
            )
        if not server_config.get("enabled", False):
            raise MCPPoolError(f"MCP server '{server_name}' is disabled")

        timeout = server_config.get("timeout_seconds", 30)

        # 🔌 Route to persistent connection if available (S4-4)
        if server_name in self._persistent_conns:
            try:
                return self._persistent_conns[server_name].call_tool(
                    tool_name,
                    arguments,
                    timeout=timeout,
                )
            except Exception as exc:
                # 🔄 Task 8: On ANY persistent failure, fall back to
                # per-call subprocess mode. Catches both
                # MCPConnectionError and unexpected exceptions.
                logger.warning(
                    "⚠️ Persistent call failed for '%s/%s', "
                    "falling back to per-call: %s",
                    server_name,
                    tool_name,
                    exc,
                )

        # 📡 Per-call mode (fallback or direct for non-persistent)
        return self._call_tool_per_call(
            server_name,
            server_config,
            tool_name,
            arguments,
            timeout,
        )

    def _call_tool_per_call(
        self,
        server_name: str,
        server_config: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        """Execute an MCP tool call via per-call subprocess spawn 📡.

        Dispatches to the appropriate transport handler (_call_stdio
        or _call_http) based on server configuration. Used as the
        fallback when persistent connections fail (Task 8).

        Args:
            server_name: MCP server name.
            server_config: Server configuration dict.
            tool_name: Tool name to invoke.
            arguments: Arguments dict forwarded to the tool.
            timeout: Call timeout in seconds.

        Returns:
            Text observation from the MCP server.

        Raises:
            MCPPoolError: On subprocess failure, invalid response,
                or unsupported transport type.
        """
        transport = server_config.get("transport", "stdio")

        if transport == "stdio":
            return self._call_stdio(
                server_name,
                server_config,
                tool_name,
                arguments,
                timeout,
            )
        elif transport == "http":
            return self._call_http(
                server_name,
                server_config,
                tool_name,
                arguments,
                timeout,
            )
        else:
            raise MCPPoolError(
                f"Unsupported transport '{transport}' for "
                f"server '{server_name}'. "
                f"Supported: stdio, http"
            )

    def get_metrics(self) -> dict[str, Any]:
        """Return per-server connection metrics snapshot 📊 (S4-6).

        Returns:
            Dict mapping server names to their metrics, including
            queue_wait_ms, call counts, and connection states.
        """
        metrics: dict[str, Any] = {}
        for name, conn in self._persistent_conns.items():
            metrics[name] = conn.get_metrics()
        return metrics

    async def close(self) -> None:
        """Clean up resources — persistent connections and caches 🧹.

        Closes all persistent connections, stops the MCP event loop,
        resets circuit breakers, and clears caches.
        """
        # 🔌 Close persistent connections
        for name, conn in list(self._persistent_conns.items()):
            try:
                conn.close(timeout=5.0)
            except Exception as exc:
                logger.warning(
                    "⚠️ Error closing persistent connection '%s': %s",
                    name,
                    exc,
                )
        self._persistent_conns.clear()

        # 🛑 Stop MCP event loop
        if self._mcp_loop is not None:
            self._mcp_loop.stop()
            self._mcp_loop = None

        self.cb_registry.reset_all()
        with self._lock:
            self._tool_cache.clear()
        self._initialized = False
        logger.info("🧹 MCPConnectionPool closed")

    async def cleanup(self) -> None:
        """Alias for close() — matches app.py usage 🧹."""
        await self.close()

    # -- S8-3: Dynamic tool discovery -----------------------------------------

    def get_dynamic_tool_defs(
        self,
        server_name: str,
    ) -> list[dict[str, Any]]:
        """Return tool definitions from dynamic discovery 🔍 (S8-3).

        Retrieves tool defs captured by PersistentMCPConnection during
        connect(). These include real input_schema from the MCP server.

        Args:
            server_name: MCP server name.

        Returns:
            List of tool definition dicts with name, description,
            and optional input_schema. Empty list if server not
            persistent or not connected.
        """
        conn = self._persistent_conns.get(server_name)
        if conn is not None:
            return conn.get_tool_defs()
        return []

    # -- S8-4: Health check + reload ------------------------------------------

    def health_check_all(self) -> dict[str, dict[str, Any]]:
        """Active health probe for all persistent connections 🩺 (S8-4).

        Runs list_tools() on each persistent connection to verify
        responsiveness and update tool defs.

        Returns:
            Dict mapping server names to health check results.
        """
        results: dict[str, dict[str, Any]] = {}
        for name, conn in self._persistent_conns.items():
            results[name] = conn.health_check()
        return results

    async def reload_config(
        self,
        new_config: dict[str, Any],
    ) -> dict[str, str]:
        """Hot-reload MCP configuration without full restart 🔄 (S8-4).

        Gracefully closes all existing connections, replaces config,
        and re-initializes. Returns per-server status report.

        Args:
            new_config: New MCP servers config dict (same format
                as constructor's config parameter).

        Returns:
            Dict mapping server names to reload status strings.
        """
        logger.info("🔄 Reloading MCP configuration...")

        # 🛑 Step 1: Close existing connections
        await self.close()

        # 🔧 Step 2: Replace config and re-derive servers
        self._config = new_config
        self.servers = self._config.get("mcp_servers", self._config.get("servers", {}))

        # 🚀 Step 3: Re-initialize
        await self.initialize()

        # 📊 Step 4: Report status
        status: dict[str, str] = {}
        for name, conn in self._persistent_conns.items():
            status[name] = conn.state.value
        # 📋 Include per-call servers
        for name, cfg in self.servers.items():
            if not cfg.get("enabled", True):
                status[name] = "disabled"
            elif name not in status:
                status[name] = "per-call"

        logger.info(
            "✅ MCP config reloaded: %d servers",
            len(status),
        )
        return status

    # -- Internal helpers ---------------------------------------------------

    def _get_server_tools(
        self,
        server_name: str,
        server_config: dict[str, Any],
    ) -> list[MCPToolWrapper]:
        """Get or create MCPToolWrapper instances for a server 🏗️.

        Results are cached per server name. The cache is invalidated
        by ``close()``. Each wrapper receives a back-reference to this
        pool and the server's circuit breaker so ``execute()`` can
        delegate properly.

        Args:
            server_name: MCP server name.
            server_config: Server configuration dict.

        Returns:
            List of MCPToolWrapper instances for the server.
        """
        with self._lock:
            if server_name in self._tool_cache:
                return self._tool_cache[server_name]

        wrappers: list[MCPToolWrapper] = []
        tools_config = server_config.get("tools", [])
        # 🔌 Get the circuit breaker for this server
        cb = self.cb_registry.get_breaker(server_name)

        for tool_def in tools_config:
            if not tool_def.get("enabled", True):
                continue
            tool_name = tool_def.get("name", "")
            if not tool_name:
                continue

            wrapper = MCPToolWrapper(
                server_name=server_name,
                tool_name=tool_name,
                tool_description=tool_def.get("description", ""),
                server_config=server_config,
                pool=self,
                circuit_breaker=cb,
                response_cache=self._response_cache,
            )
            wrappers.append(wrapper)

        with self._lock:
            self._tool_cache[server_name] = wrappers

        logger.debug(
            "🔧 Created %d tool wrappers for server '%s'",
            len(wrappers),
            server_name,
        )
        return wrappers

    def _call_stdio(
        self,
        server_name: str,
        server_config: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        """Execute an MCP tool call via stdio transport 📡.

        Spawns the configured command as a subprocess with full MCP
        protocol handshake: initialize → initialized → tools/call.
        Uses Popen for bidirectional communication.

        Args:
            server_name: MCP server name (for logging).
            server_config: Server configuration with ``command``
                and ``args`` keys.
            tool_name: Tool name to invoke.
            arguments: Arguments dict forwarded to the tool.
            timeout: Subprocess timeout in seconds.

        Returns:
            Text content extracted from the MCP response.

        Raises:
            MCPPoolError: On subprocess failure or invalid response.
        """
        import os as _os

        command = server_config.get("command", "")
        cmd_args = server_config.get("args", [])
        if not command:
            raise MCPPoolError(
                f"No command configured for stdio server '{server_name}'"
            )

        full_cmd = [command] + list(cmd_args)

        # 🔧 Build subprocess environment with server-specific env vars
        # Subprocess inherits parent process env (os.environ.copy()),
        # then we overlay any resolved env vars from the server config.
        # Unresolved ${...} placeholders are skipped with a warning.
        env = _os.environ.copy()
        server_env = server_config.get("env", {})
        for k, v in server_env.items():
            if not isinstance(v, str):
                continue
            if v.startswith("${"):
                logger.warning(
                    "⚠️ Env var '%s' unresolved ('%s') for server '%s' "
                    "— check .env file or process environment",
                    k,
                    v,
                    server_name,
                )
                continue
            env[k] = v

        logger.debug(
            "📡 stdio call: %s tool=%s cmd=%s",
            server_name,
            tool_name,
            full_cmd,
        )

        proc = None
        try:
            proc = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            # 🤝 Step 1: Send MCP initialize request
            init_request = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "inquiro",
                            "version": "0.1.0",
                        },
                    },
                }
            )
            self._stdio_send(proc, init_request)

            # 📨 Read initialize response
            init_response = self._stdio_read(proc, timeout)
            if init_response and "error" in init_response:
                raise MCPPoolError(
                    f"MCP initialize failed for '{server_name}': "
                    f"{init_response['error']}"
                )
            logger.debug(
                "🤝 MCP initialized: %s",
                server_name,
            )

            # 📣 Step 2: Send initialized notification (no id, no response)
            notification = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            )
            self._stdio_send(proc, notification)

            # 🚀 Step 3: Send tools/call request
            call_request = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                }
            )
            self._stdio_send(proc, call_request)

            # 📨 Read tools/call response
            call_response = self._stdio_read(proc, timeout)

            if not call_response:
                return f"[MCP:{server_name}/{tool_name}] Empty response"

            if "error" in call_response:
                err = call_response["error"]
                raise MCPPoolError(
                    f"MCP error from '{server_name}/{tool_name}': "
                    f"{err.get('message', err)}"
                )

            # ✨ Extract content from MCP response
            rpc_result = call_response.get("result", {})
            content_list = rpc_result.get("content", [])
            texts = [
                item.get("text", "")
                for item in content_list
                if item.get("type") == "text"
            ]
            return "\n".join(texts) if texts else json.dumps(rpc_result)

        except subprocess.TimeoutExpired:
            raise MCPPoolError(
                f"MCP stdio call to '{server_name}/{tool_name}' "
                f"timed out after {timeout}s"
            )
        except FileNotFoundError:
            raise MCPPoolError(
                f"MCP command '{command}' not found for server '{server_name}'"
            )
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    @staticmethod
    def _stdio_send(proc: subprocess.Popen, message: str) -> None:
        """Send a JSON-RPC message to the subprocess stdin 📤.

        Args:
            proc: Running subprocess with stdin pipe.
            message: JSON-RPC message string.
        """
        proc.stdin.write(message + "\n")
        proc.stdin.flush()

    @staticmethod
    def _stdio_read(
        proc: subprocess.Popen,
        timeout: int,
    ) -> dict[str, Any] | None:
        """Read a JSON-RPC response line from subprocess stdout 📥.

        Reads lines until a valid JSON-RPC response (with "id" or "error")
        is found, skipping notifications and non-JSON output.

        Args:
            proc: Running subprocess with stdout pipe.
            timeout: Read timeout in seconds.

        Returns:
            Parsed JSON-RPC response dict, or None on timeout/EOF.
        """
        import select
        import time as _time

        deadline = _time.monotonic() + timeout

        while _time.monotonic() < deadline:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                break

            # ⏳ Use select if available, otherwise direct read
            try:
                ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 2.0))
                if not ready:
                    if proc.poll() is not None:
                        return None
                    continue
            except (io.UnsupportedOperation, ValueError):
                # 📝 StringIO (tests) or closed fd — fall through to read
                pass

            line = proc.stdout.readline()
            if not line:
                return None

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("📡 Non-JSON MCP output: %s", line[:200])
                continue

            # 🔍 Skip JSON-RPC notifications (no "id" field)
            if "id" in data or "error" in data:
                return data

        return None

    def _call_http(
        self,
        server_name: str,
        server_config: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int,
    ) -> str:
        """Execute an MCP tool call via HTTP transport 🌐.

        Sends a JSON-RPC ``tools/call`` POST request to the
        configured endpoint.

        Args:
            server_name: MCP server name (for logging).
            server_config: Server configuration with ``endpoint`` key.
            tool_name: Tool name to invoke.
            arguments: Arguments dict forwarded to the tool.
            timeout: HTTP request timeout in seconds.

        Returns:
            Text content extracted from the MCP response.

        Raises:
            MCPPoolError: On HTTP failure or invalid response.
        """
        import urllib.request
        import urllib.error

        endpoint = server_config.get("endpoint", "")
        if not endpoint:
            raise MCPPoolError(
                f"No endpoint configured for HTTP server '{server_name}'"
            )

        rpc_request = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        ).encode("utf-8")

        logger.debug(
            "🌐 HTTP call: %s tool=%s endpoint=%s",
            server_name,
            tool_name,
            endpoint,
        )

        req = urllib.request.Request(
            endpoint,
            data=rpc_request,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_output = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise MCPPoolError(
                f"MCP HTTP call to '{server_name}/{tool_name}' failed: {exc}"
            )
        except TimeoutError:
            raise MCPPoolError(
                f"MCP HTTP call to '{server_name}/{tool_name}' "
                f"timed out after {timeout}s"
            )

        # 🔍 Parse JSON-RPC response
        try:
            rpc_response = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output

        if "error" in rpc_response:
            err = rpc_response["error"]
            raise MCPPoolError(
                f"MCP error from '{server_name}/{tool_name}': {err.get('message', err)}"
            )

        rpc_result = rpc_response.get("result", {})
        content_list = rpc_result.get("content", [])
        texts = [
            item.get("text", "") for item in content_list if item.get("type") == "text"
        ]
        return "\n".join(texts) if texts else json.dumps(rpc_result)

    def _build_cb_registry(self) -> CircuitBreakerRegistry:
        """Build a CircuitBreakerRegistry from server configs 🏗️.

        Extracts per-server circuit breaker settings from the
        server configurations and creates a registry.

        Returns:
            Configured CircuitBreakerRegistry.
        """
        per_server: dict[str, CircuitBreakerConfig] = {}

        for server_name, server_config in self.servers.items():
            cb_config = server_config.get("circuit_breaker", {})
            if cb_config:
                per_server[server_name] = CircuitBreakerConfig(
                    failure_threshold=cb_config.get("failure_threshold", 3),
                    recovery_timeout=cb_config.get(
                        "recovery_timeout",
                        cb_config.get("timeout_seconds", 60.0),
                    ),
                    half_open_max_calls=cb_config.get("half_open_max_calls", 3),
                    recovery_success_threshold=cb_config.get("success_threshold", 1),
                )

        return CircuitBreakerRegistry(per_server_configs=per_server)
