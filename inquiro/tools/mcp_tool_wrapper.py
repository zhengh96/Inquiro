"""Inquiro MCPToolWrapper — MCP-to-BaseTool bridge 🔧.

Wraps a single MCP server tool definition as an EvoMaster BaseTool,
allowing MCP tools to be registered in ToolRegistry and invoked by
agents through the standard ``execute()`` interface.

The actual MCP protocol communication (stdio/HTTP) is delegated to
the MCPConnectionPool at call time. This wrapper only handles
parameter translation, circuit breaker gating, and evidence metadata
extraction.

Example::

    wrapper = MCPToolWrapper(
        server_name="perplexity",
        tool_name="perplexity_search",
        tool_description="Search the web using Perplexity AI",
        server_config={"transport": "stdio", "command": "uvx", ...},
        pool=mcp_pool,
        circuit_breaker=cb_registry.get_breaker("perplexity"),
    )
    registry.register(wrapper)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

    from inquiro.infrastructure.circuit_breaker import CircuitBreaker
    from inquiro.infrastructure.mcp_pool import MCPConnectionPool
    from inquiro.infrastructure.mcp_response_cache import MCPResponseCache

logger = logging.getLogger(__name__)


class MCPToolParams(BaseToolParams):
    """Generic parameter model for MCP tools 🔧.

    Accepts arbitrary JSON arguments that are forwarded to the MCP server.
    The ``name`` class var is set dynamically per wrapper instance.
    """

    name: ClassVar[str] = "mcp_tool"
    model_config = ConfigDict(extra="allow")

    arguments: str = ""
    """JSON-encoded arguments to pass to the MCP tool."""


class MCPToolWrapper(BaseTool):
    """Wraps a single MCP server tool as an EvoMaster BaseTool 🔧.

    Each instance represents one tool on one MCP server. The tool
    is registered in ToolRegistry under the name
    ``mcp__<server>__<tool_name>``.

    When ``pool`` and ``circuit_breaker`` are provided, execute()
    delegates to ``MCPConnectionPool.call_tool()`` through the
    circuit breaker for fault isolation. Without these references,
    the wrapper falls back to a standalone stub (useful for unit
    tests).

    Attributes:
        server_name: MCP server name (e.g., "perplexity").
        tool_name: Tool name on the server (e.g., "perplexity_search").
        server_config: Full server config dict from mcp_servers.yaml.
        pool: Optional back-reference to the owning MCPConnectionPool.
        circuit_breaker: Optional CircuitBreaker for fault isolation.
    """

    # ✨ Class vars required by BaseTool
    name: ClassVar[str] = "mcp_tool"
    params_class: ClassVar[type[BaseToolParams]] = MCPToolParams

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        tool_description: str,
        server_config: dict[str, Any],
        pool: MCPConnectionPool | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        input_schema: dict[str, Any] | None = None,
        tool_effectiveness_tracker: Any | None = None,
        response_cache: MCPResponseCache | None = None,
    ) -> None:
        """Initialize MCPToolWrapper 🔧.

        Args:
            server_name: MCP server name.
            tool_name: Tool name on the MCP server.
            tool_description: Human-readable tool description.
            server_config: Server configuration dict from YAML.
            pool: Optional MCPConnectionPool for delegated execution.
            circuit_breaker: Optional CircuitBreaker for fault isolation.
            input_schema: Optional JSON Schema for tool parameters
                (from dynamic discovery via list_tools()). When provided,
                get_tool_spec() uses this instead of the generic wrapper.
            tool_effectiveness_tracker: Optional ToolEffectivenessTracker
                for recording tool call success/failure metrics 📊.
            response_cache: Optional MCPResponseCache for deduplicating
                identical MCP calls across rounds. When None (default),
                all behavior is unchanged and no caching occurs.
        """
        super().__init__()
        self.server_name = server_name
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.server_config = server_config
        self.pool = pool
        self.circuit_breaker = circuit_breaker
        self.input_schema = input_schema
        self._effectiveness_tracker = tool_effectiveness_tracker
        self._response_cache = response_cache

        # 🏷️ Override class-level name for this instance
        self.name = f"mcp__{server_name}__{tool_name}"

        # 🏷️ MCP tool markers for ToolRegistry filtering
        self._is_mcp_tool = True
        self._mcp_server = server_name

    def execute(
        self, session: BaseSession, args_json: str
    ) -> tuple[str, dict[str, Any]]:
        """Execute the MCP tool call via pool + circuit breaker 🚀.

        Execution flow:
        1. Parse JSON arguments from the agent.
        2. Check the circuit breaker (if available); reject immediately
           if the circuit is OPEN.
        3. Delegate to ``MCPConnectionPool.call_tool()`` for the actual
           MCP transport (stdio / HTTP).
        4. Record success or failure on the circuit breaker.
        5. Return ``(observation, info)`` with evidence metadata.

        Falls back to a standalone stub when ``pool`` is ``None``
        (backward-compatible for unit tests).

        Args:
            session: EvoMaster session (provides context).
            args_json: JSON string of tool arguments.

        Returns:
            Tuple of (observation, info):
            - observation: Text result from the MCP server.
            - info: Metadata dict with source, duration, server.
        """
        from inquiro.infrastructure.circuit_breaker import CircuitOpenError

        start_time = time.monotonic()

        try:
            # 🔍 Step 1: Parse arguments
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                args = {"query": args_json}

            # 📦 Step 1b: Unwrap nested wrappers.
            # LLMs frequently wrap actual args in an outer key:
            #   {"arguments": "<json string>"}  — generic fallback spec
            #   {"params": {"query": "..."}}    — JSON-RPC habit
            # The MCP server expects the inner arguments directly.
            if (
                "arguments" in args
                and isinstance(args["arguments"], str)
                and len(args) == 1
            ):
                try:
                    args = json.loads(args["arguments"])
                except json.JSONDecodeError:
                    # ⚠️ Treat raw string as a query
                    args = {"query": args["arguments"]}
            elif (
                "params" in args and isinstance(args["params"], dict) and len(args) == 1
            ):
                # 🔄 Unwrap {"params": {actual args}}
                args = args["params"]

            # 🛡️ Step 1c: Strip fields unknown to the input schema.
            # MCP servers with extra="forbid" silently reject
            # unexpected fields and return empty results. Remove
            # unrecognised keys here and map common aliases.
            if self.input_schema and isinstance(self.input_schema, dict):
                known = set(self.input_schema.get("properties", {}).keys())
                if known:
                    # 🔄 Common LLM alias → canonical name
                    _ALIASES: dict[str, str] = {
                        "limit": "page_size",
                        "count": "page_size",
                        "num_results": "page_size",
                        "search_query": "query",
                        "search_term": "query",
                        "q": "query",
                        "question": "query",
                        "output_format": "response_format",
                        "format": "response_format",
                    }
                    remapped: dict[str, Any] = {}
                    for k, v in args.items():
                        canonical = _ALIASES.get(k, k)
                        if canonical in known:
                            remapped[canonical] = v
                        elif k in known:
                            remapped[k] = v
                        else:
                            logger.warning(
                                "⚠️ Stripped unknown param '%s' for %s/%s",
                                k,
                                self.server_name,
                                self.tool_name,
                            )

                    # 🛡️ Check that required params survive remapping
                    required = set(
                        self.input_schema.get("required", []),
                    )
                    missing = required - set(remapped.keys())
                    if missing:
                        logger.error(
                            "🚨 Required params missing after "
                            "remapping for %s/%s: %s "
                            "(original keys: %s)",
                            self.server_name,
                            self.tool_name,
                            missing,
                            set(args.keys())
                            if args != remapped
                            else set(remapped.keys()),
                        )

                    args = remapped

            logger.info(
                "🔧 MCP tool call: %s/%s args=%s",
                self.server_name,
                self.tool_name,
                str(args)[:200],
            )

            # 🗄️ Step 2: Check response cache before hitting the MCP server
            if self._response_cache is not None:
                cached = self._response_cache.get(
                    self.server_name, self.tool_name, args
                )
                if cached is not None:
                    logger.debug(
                        "🗄️ MCP cache hit: %s/%s",
                        self.server_name,
                        self.tool_name,
                    )
                    duration_ms = int((time.monotonic() - start_time) * 1000)
                    return cached, {
                        "server": self.server_name,
                        "tool": self.tool_name,
                        "duration_ms": duration_ms,
                        "success": True,
                        "cached": True,
                        "source": (f"mcp://{self.server_name}/{self.tool_name}"),
                    }

            # 🔌 Step 3: Circuit breaker gate
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                raise CircuitOpenError(self.server_name)

            # 🚀 Step 4: Delegate to pool or use standalone stub
            if self.pool is not None:
                observation = self.pool.call_tool(
                    self.server_name, self.tool_name, args
                )
            else:
                # ⚠️ Standalone fallback (no pool wired)
                observation = (
                    f"[MCP:{self.server_name}/{self.tool_name}] "
                    f"Tool call dispatched with args: "
                    f"{json.dumps(args)}"
                )

            # 📦 Step 5: Store successful response in cache
            if self._response_cache is not None:
                self._response_cache.put(
                    self.server_name, self.tool_name, args, observation
                )

            # ✅ Step 6: Record success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()

            duration_ms = int((time.monotonic() - start_time) * 1000)

            info = {
                "server": self.server_name,
                "tool": self.tool_name,
                "duration_ms": duration_ms,
                "success": True,
                "cached": False,
                "source": f"mcp://{self.server_name}/{self.tool_name}",
            }

            logger.info(
                "✅ MCP tool completed: %s/%s in %dms",
                self.server_name,
                self.tool_name,
                duration_ms,
            )

            # 📊 Record tool effectiveness
            if self._effectiveness_tracker:
                self._effectiveness_tracker.record(
                    tool_name=self.name,
                    yielded_evidence=True,
                    latency_ms=duration_ms,
                )

            return observation, info

        except CircuitOpenError:
            # 🔴 Circuit is open — fail fast without recording
            duration_ms = int((time.monotonic() - start_time) * 1000)
            error_msg = (
                f"[MCP:{self.server_name}/{self.tool_name}] "
                f"Circuit breaker OPEN — server temporarily "
                f"unavailable. Try a different search tool or "
                f"retry later."
            )
            logger.warning(
                "🔴 Circuit open for %s/%s, rejecting call",
                self.server_name,
                self.tool_name,
            )

            # 📊 Record circuit-open as failure
            if self._effectiveness_tracker:
                self._effectiveness_tracker.record(
                    tool_name=self.name,
                    yielded_evidence=False,
                    latency_ms=duration_ms,
                )

            return error_msg, {
                "server": self.server_name,
                "tool": self.tool_name,
                "duration_ms": duration_ms,
                "success": False,
                "error": "circuit_open",
            }

        except Exception as exc:
            # 📊 Classify error for structured logging
            from inquiro.infrastructure.errors import classify_error

            error_class = classify_error(exc)

            # ❌ Execution failed — record on circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()

            duration_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "❌ MCP tool failed: %s/%s error=%s (classified=%s)",
                self.server_name,
                self.tool_name,
                exc,
                error_class,
            )

            # 📊 Record tool failure
            if self._effectiveness_tracker:
                self._effectiveness_tracker.record(
                    tool_name=self.name,
                    yielded_evidence=False,
                    latency_ms=duration_ms,
                )

            error_msg = f"[MCP:{self.server_name}/{self.tool_name}] Error: {exc}"
            info = {
                "server": self.server_name,
                "tool": self.tool_name,
                "duration_ms": duration_ms,
                "success": False,
                "error": str(exc),
                "error_class": error_class,
            }
            return error_msg, info

    def get_tool_spec(self) -> Any:
        """Generate ToolSpec for LLM function calling 📋.

        When ``input_schema`` is available (from S8-3 dynamic discovery),
        uses the real parameter schema so the LLM sees actual tool
        parameters. Falls back to the generic ``{"arguments": "<json>"}``
        wrapper when no schema is available.

        Returns:
            ToolSpec with the MCP tool's name and description.
        """
        from evomaster.utils.types import FunctionSpec, ToolSpec

        # 🔍 S8-3: Use real input_schema when available
        if self.input_schema and isinstance(self.input_schema, dict):
            parameters = self.input_schema
        else:
            # 📦 Fallback: generic JSON wrapper
            parameters = {
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": "string",
                        "description": ("JSON-encoded arguments for the MCP tool"),
                    }
                },
                "required": ["arguments"],
            }

        return ToolSpec(
            type="function",
            function=FunctionSpec(
                name=self.name,
                description=self.tool_description,
                parameters=parameters,
                strict=None,
            ),
        )

    def __repr__(self) -> str:
        return f"<MCPToolWrapper {self.name} server={self.server_name!r}>"
