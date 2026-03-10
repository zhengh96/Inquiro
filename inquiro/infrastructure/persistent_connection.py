"""Persistent MCP connection with state machine and auto-reconnect 🔌.

Provides a long-lived connection to a single MCP server, eliminating
the subprocess-spawn-per-call overhead of the stateless model.

Architecture:
    - Each PersistentMCPConnection manages ONE MCP server.
    - A dedicated daemon thread runs an asyncio event loop for
      async MCP protocol communication.
    - Sync callers use ``run_coroutine_threadsafe()`` to bridge into
      the MCP event loop.
    - ConnectionState tracks the lifecycle: INIT → CONNECTING → READY
      → FAILED → RECONNECTING → CLOSED.
    - Auto-reconnect with exponential backoff on failures.

Thread safety:
    - State transitions protected by threading.Lock.
    - All MCP operations dispatched to the dedicated event loop.
    - ``call_tool()`` is safe to call from any thread.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# 🔧 Reconnect backoff parameters
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_BACKOFF_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# ConnectionState enum (S4-1)
# ---------------------------------------------------------------------------


class ConnectionState(str, enum.Enum):
    """MCP connection lifecycle states 🔄.

    State transitions::

        INIT ──connect()──▶ CONNECTING ──success──▶ READY
                                │                     │
                                ▼                     ▼
                             FAILED ◀──failure────── (any)
                                │
                                ▼
                          RECONNECTING ──success──▶ READY
                                │
                                ▼
                             FAILED (max retries)

        (any) ──close()──▶ CLOSED
    """

    INIT = "init"
    CONNECTING = "connecting"
    READY = "ready"
    FAILED = "failed"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# MCP Event Loop Thread (S4-2)
# ---------------------------------------------------------------------------


class MCPEventLoop:
    """Dedicated asyncio event loop running in a daemon thread 🔄.

    Provides a long-lived event loop for MCP async operations.
    The loop runs in a background daemon thread that automatically
    terminates when the main process exits.

    Usage::

        mcp_loop = MCPEventLoop()
        mcp_loop.start()
        future = mcp_loop.submit(some_coro())
        result = future.result(timeout=30)
        mcp_loop.stop()

    Attributes:
        loop: The asyncio event loop (available after start()).
    """

    def __init__(self) -> None:
        """Initialize MCPEventLoop 🔧."""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Return the underlying event loop 🔄."""
        return self._loop

    @property
    def is_running(self) -> bool:
        """Check if the event loop is running 🔍."""
        return self._loop is not None and self._loop.is_running()

    def start(self) -> None:
        """Start the event loop in a daemon thread 🚀.

        Blocks until the loop is confirmed running.
        Safe to call multiple times (no-op if already running).
        """
        if self.is_running:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="mcp-event-loop",
        )
        self._thread.start()
        # ⏳ Wait for loop to be running before returning
        self._started.wait(timeout=5.0)
        logger.info("🔄 MCP event loop started (thread=%s)", self._thread.name)

    def _run_loop(self) -> None:
        """Run the event loop (executed in daemon thread) 🔄."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def stop(self) -> None:
        """Stop the event loop and join the thread 🛑.

        Safe to call multiple times.
        """
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._loop = None
        self._thread = None
        self._started.clear()
        logger.info("🛑 MCP event loop stopped")

    def submit(self, coro: Any) -> asyncio.Future:
        """Submit an async coroutine to the MCP event loop 🚀.

        Args:
            coro: Awaitable coroutine to execute.

        Returns:
            concurrent.futures.Future wrapping the coroutine result.

        Raises:
            RuntimeError: If the event loop is not running.
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("MCP event loop is not running. Call start() first.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


# ---------------------------------------------------------------------------
# PersistentMCPConnection (S4-1 + S4-3 + S4-5)
# ---------------------------------------------------------------------------


class PersistentMCPConnection:
    """Persistent connection to a single MCP server 🔌.

    Maintains a long-lived async MCPConnection (from EvoMaster) and
    provides a synchronous ``call_tool()`` interface that bridges
    via ``run_coroutine_threadsafe()`` to the shared MCP event loop.

    Features:
        - ConnectionState lifecycle tracking 🔄
        - Auto-reconnect with exponential backoff 🔁
        - Sync-to-async bridge via shared MCPEventLoop 🌉
        - Thread-safe state management 🔒
        - Queue wait time metrics (queue_wait_ms) 📊

    Args:
        server_name: MCP server identifier.
        server_config: Server configuration dict from YAML.
        mcp_loop: Shared MCPEventLoop instance.
        max_reconnect_attempts: Max auto-reconnect attempts (0=disable).

    Attributes:
        state: Current ConnectionState.
        server_name: Server identifier.
        queue_wait_ms_total: Cumulative queue wait time in milliseconds.
        call_count: Total number of tool calls.
    """

    def __init__(
        self,
        server_name: str,
        server_config: dict[str, Any],
        mcp_loop: MCPEventLoop,
        max_reconnect_attempts: int = 5,
    ) -> None:
        """Initialize PersistentMCPConnection 🔧.

        Args:
            server_name: MCP server identifier.
            server_config: Full server configuration dict.
            mcp_loop: Shared MCP event loop.
            max_reconnect_attempts: Maximum reconnection attempts.
        """
        self.server_name = server_name
        self._config = server_config
        self._mcp_loop = mcp_loop
        self._max_reconnect = max_reconnect_attempts

        # 🔄 State machine
        self._state = ConnectionState.INIT
        self._state_lock = threading.Lock()

        # 🔌 Async connection (managed in MCP loop)
        self._connection: Any = None  # MCPConnection instance
        self._exit_stack: Any = None

        # 📊 Metrics
        self.queue_wait_ms_total: float = 0.0
        self.call_count: int = 0
        self._reconnect_count: int = 0
        self._consecutive_failures: int = 0

        # ⏱️ Backoff state
        self._current_backoff = _BASE_BACKOFF_SECONDS

        # 🔍 S8-3: Dynamic tool definitions from list_tools()
        self._tool_defs: list[dict[str, Any]] = []
        self._last_health_check: float = 0.0

    @property
    def state(self) -> ConnectionState:
        """Return current connection state 🔄."""
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: ConnectionState) -> None:
        """Transition to a new state (thread-safe) 🔄.

        Args:
            new_state: Target ConnectionState.
        """
        with self._state_lock:
            old = self._state
            self._state = new_state
        logger.info(
            "🔄 [%s] State: %s → %s",
            self.server_name,
            old.value,
            new_state.value,
        )

    # -- Connection lifecycle -----------------------------------------------

    def connect(self, timeout: float = 30.0) -> bool:
        """Establish the persistent MCP connection 🔌.

        Dispatches async connection setup to the MCP event loop
        and waits for completion.

        Args:
            timeout: Maximum time to wait for connection (seconds).

        Returns:
            True if connection established, False on failure.
        """
        if self.state == ConnectionState.READY:
            return True
        if self.state == ConnectionState.CLOSED:
            return False

        self._set_state(ConnectionState.CONNECTING)

        try:
            future = self._mcp_loop.submit(self._async_connect())
            result = future.result(timeout=timeout)
            if result:
                self._set_state(ConnectionState.READY)
                self._consecutive_failures = 0
                self._current_backoff = _BASE_BACKOFF_SECONDS
                return True
            else:
                self._set_state(ConnectionState.FAILED)
                return False
        except Exception as exc:
            logger.error(
                "❌ [%s] Connection failed: %s",
                self.server_name,
                exc,
            )
            self._set_state(ConnectionState.FAILED)
            return False

    async def _async_connect(self) -> bool:
        """Async connection setup (runs in MCP event loop) 🔌.

        Creates the EvoMaster MCPConnection and enters its async
        context manager.

        Returns:
            True if connection was established successfully.
        """
        from contextlib import AsyncExitStack
        from evomaster.agent.tools.mcp.mcp_connection import (
            create_connection,
        )

        transport = self._config.get("transport", "stdio")
        try:
            stack = AsyncExitStack()
            await stack.__aenter__()

            # 🔧 Build connection kwargs from config
            kwargs = self._build_connection_kwargs()

            conn = create_connection(transport=transport, **kwargs)
            self._connection = await stack.enter_async_context(conn)
            self._exit_stack = stack

            # ✅ Verify connection with tool list + capture defs
            tools = await self._connection.list_tools()
            self._tool_defs = [self._extract_tool_def(t) for t in tools]
            logger.info(
                "✅ [%s] Connected (%s): %d tools available",
                self.server_name,
                transport,
                len(tools),
            )
            return True

        except Exception as exc:
            logger.error(
                "❌ [%s] Async connect failed: %s",
                self.server_name,
                exc,
            )
            return False

    def _build_connection_kwargs(self) -> dict[str, Any]:
        """Build kwargs for create_connection() from server config 🔧.

        Returns:
            Keyword arguments dict for EvoMaster create_connection().
        """
        transport = self._config.get("transport", "stdio")
        kwargs: dict[str, Any] = {}

        if transport == "stdio":
            command = self._config.get("command", "")
            args = self._config.get("args", [])
            # 🔧 Resolve environment variables
            env = os.environ.copy()
            for k, v in self._config.get("env", {}).items():
                if isinstance(v, str) and not v.startswith("${"):
                    env[k] = v
            kwargs["command"] = command
            kwargs["args"] = args
            kwargs["env"] = env

        elif transport in ("http", "sse"):
            kwargs["url"] = self._config.get("endpoint", "")
            kwargs["headers"] = self._config.get("headers", {})

        return kwargs

    def close(self, timeout: float = 10.0) -> None:
        """Close the persistent connection 🧹.

        Args:
            timeout: Maximum time to wait for cleanup.
        """
        if self.state == ConnectionState.CLOSED:
            return

        self._set_state(ConnectionState.CLOSED)

        if self._exit_stack is not None:
            try:
                future = self._mcp_loop.submit(
                    self._exit_stack.__aexit__(None, None, None)
                )
                future.result(timeout=timeout)
            except Exception as exc:
                logger.warning(
                    "⚠️ [%s] Close error: %s",
                    self.server_name,
                    exc,
                )
            finally:
                self._connection = None
                self._exit_stack = None

        logger.info("🧹 [%s] Connection closed", self.server_name)

    # -- Tool execution (S4-3: sync-async bridge) ---------------------------

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float = 30.0,
    ) -> str:
        """Execute an MCP tool call via the persistent connection 🚀.

        Synchronous interface that bridges to the async MCP protocol
        via ``run_coroutine_threadsafe()``.

        Records queue_wait_ms metric for each call.

        Args:
            tool_name: Tool name on the MCP server.
            arguments: Arguments dict to pass to the tool.
            timeout: Call timeout in seconds.

        Returns:
            Text observation from the MCP server.

        Raises:
            MCPConnectionError: If connection is not READY or call fails.
        """
        # 📊 Track queue wait time
        submit_time = time.monotonic()

        # 🔄 Auto-reconnect if FAILED
        current_state = self.state
        if current_state == ConnectionState.FAILED:
            if not self._try_reconnect():
                raise MCPConnectionError(
                    f"[{self.server_name}] Connection FAILED and reconnect unsuccessful"
                )

        if self.state != ConnectionState.READY:
            raise MCPConnectionError(
                f"[{self.server_name}] Connection not ready (state={self.state.value})"
            )

        try:
            future = self._mcp_loop.submit(self._async_call_tool(tool_name, arguments))
            # 📊 Record queue wait (time between submit and execution start)
            queue_wait = (time.monotonic() - submit_time) * 1000
            self.queue_wait_ms_total += queue_wait
            self.call_count += 1

            result = future.result(timeout=timeout)
            self._consecutive_failures = 0
            return result

        except Exception as exc:
            self._consecutive_failures += 1
            logger.error(
                "❌ [%s] call_tool(%s) failed: %s",
                self.server_name,
                tool_name,
                exc,
            )
            # 🔄 Mark as FAILED for auto-reconnect on next call
            if self._consecutive_failures >= 3:
                self._set_state(ConnectionState.FAILED)
            raise MCPConnectionError(f"[{self.server_name}/{tool_name}] {exc}") from exc

    async def _async_call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Async tool call (runs in MCP event loop) 🚀.

        Args:
            tool_name: Tool name on the MCP server.
            arguments: Arguments dict.

        Returns:
            Text content from MCP response.
        """
        import json as _json

        result = await self._connection.call_tool(tool_name, arguments)

        # ✨ Extract text from MCP content list
        if isinstance(result, list):
            texts = []
            for item in result:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) if texts else _json.dumps(result)

        return str(result)

    # -- Auto-reconnect (S4-5) ----------------------------------------------

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff 🔁.

        Returns:
            True if reconnection succeeded.
        """
        if self._reconnect_count >= self._max_reconnect:
            logger.warning(
                "⚠️ [%s] Max reconnect attempts (%d) reached",
                self.server_name,
                self._max_reconnect,
            )
            return False

        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_count += 1

        # ⏳ Exponential backoff
        backoff = min(self._current_backoff, _MAX_BACKOFF_SECONDS)
        logger.info(
            "🔁 [%s] Reconnecting (attempt %d/%d, backoff=%.1fs)",
            self.server_name,
            self._reconnect_count,
            self._max_reconnect,
            backoff,
        )
        time.sleep(backoff)
        self._current_backoff *= _BACKOFF_MULTIPLIER

        # 🔌 Clean up old connection first
        if self._exit_stack is not None:
            try:
                future = self._mcp_loop.submit(
                    self._exit_stack.__aexit__(None, None, None)
                )
                future.result(timeout=5.0)
            except Exception:
                pass
            self._connection = None
            self._exit_stack = None

        # 🔌 Establish new connection
        success = self.connect(timeout=30.0)
        if success:
            self._reconnect_count = 0
            logger.info(
                "✅ [%s] Reconnected successfully",
                self.server_name,
            )
        return success

    # -- Metrics (S4-6) -----------------------------------------------------

    @property
    def avg_queue_wait_ms(self) -> float:
        """Average queue wait time in milliseconds 📊."""
        if self.call_count == 0:
            return 0.0
        return self.queue_wait_ms_total / self.call_count

    def get_metrics(self) -> dict[str, Any]:
        """Return connection metrics snapshot 📊.

        Returns:
            Dict with connection stats including queue wait times,
            call counts, reconnect counts, and current state.
        """
        return {
            "server_name": self.server_name,
            "state": self.state.value,
            "call_count": self.call_count,
            "queue_wait_ms_total": round(self.queue_wait_ms_total, 2),
            "avg_queue_wait_ms": round(self.avg_queue_wait_ms, 2),
            "reconnect_count": self._reconnect_count,
            "consecutive_failures": self._consecutive_failures,
            "tool_count": len(self._tool_defs),
        }

    # -- S8-3: Dynamic tool discovery -----------------------------------------

    def get_tool_defs(self) -> list[dict[str, Any]]:
        """Return tool definitions discovered at connect time 🔍.

        Each tool def contains name, description, and optional
        input_schema from the MCP server's list_tools() response.

        Returns:
            List of tool definition dicts.
        """
        return list(self._tool_defs)

    @staticmethod
    def _extract_tool_def(tool: Any) -> dict[str, Any]:
        """Extract a normalized tool definition from MCP tool object 📋.

        Args:
            tool: Tool object from MCPConnection.list_tools().

        Returns:
            Dict with name, description, and input_schema.
        """
        td: dict[str, Any] = {
            "name": getattr(tool, "name", str(tool)),
            "description": getattr(tool, "description", ""),
        }
        # 🔍 Extract input_schema if available
        schema = getattr(tool, "inputSchema", None)
        if schema is None:
            schema = getattr(tool, "input_schema", None)
        if schema is not None:
            if hasattr(schema, "model_dump"):
                td["input_schema"] = schema.model_dump()
            elif isinstance(schema, dict):
                td["input_schema"] = schema
        return td

    # -- S8-4: Health check ---------------------------------------------------

    def health_check(self, timeout: float = 10.0) -> dict[str, Any]:
        """Perform active health probe via list_tools() 🩺.

        Runs a lightweight list_tools() call to verify the connection
        is still responsive. Updates tool_defs on success.

        Args:
            timeout: Probe timeout in seconds.

        Returns:
            Dict with healthy status, latency, and tool count.
        """
        start = time.monotonic()
        try:
            if self.state != ConnectionState.READY:
                return {
                    "healthy": False,
                    "state": self.state.value,
                    "latency_ms": 0.0,
                }
            future = self._mcp_loop.submit(self._connection.list_tools())
            tools = future.result(timeout=timeout)
            latency = (time.monotonic() - start) * 1000
            self._tool_defs = [self._extract_tool_def(t) for t in tools]
            self._last_health_check = time.monotonic()
            return {
                "healthy": True,
                "state": self.state.value,
                "latency_ms": round(latency, 2),
                "tool_count": len(self._tool_defs),
            }
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                "⚠️ [%s] Health check failed: %s",
                self.server_name,
                exc,
            )
            return {
                "healthy": False,
                "state": self.state.value,
                "latency_ms": round(latency, 2),
                "error": str(exc),
            }


class MCPConnectionError(Exception):
    """Raised when a persistent MCP connection operation fails ❌."""
