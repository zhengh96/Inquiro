"""Inquiro preflight connectivity checker 🔍.

Validates that external dependencies (MCP servers and LLM providers) are
reachable before accepting evaluation requests. Designed to run at startup
within the FastAPI lifespan and on-demand via ``GET /api/v1/preflight``.

Strategy:
- MCP servers: attempt a lightweight ``tools/list`` JSON-RPC call via the
  configured transport (stdio or HTTP). This validates the handshake
  without performing a real search.
- LLM providers: send a minimal single-message ``_call()`` to verify
  that credentials and network connectivity are valid.

Results are purely informational — the service starts regardless of
check outcomes (graceful degradation), but clear log messages report
which services are UP and which are DOWN.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from typing import Any

from inquiro.api.schemas import PreflightResponse, ServiceCheckResult

logger = logging.getLogger(__name__)
_UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}")


async def preflight_check(
    mcp_pool: Any | None = None,
    llm_pool: Any | None = None,
) -> PreflightResponse:
    """Run connectivity probes against all configured services 🔍.

    Tests each enabled MCP server and the default LLM provider,
    collecting per-service UP/DOWN status. Failures are logged as
    warnings but do NOT prevent the service from starting.

    Args:
        mcp_pool: MCPConnectionPool instance (or None in stub mode).
        llm_pool: LLMProviderPool instance (or None in stub mode).

    Returns:
        PreflightResponse with per-service results and overall status.
    """
    logger.info("🔍 Starting preflight connectivity checks...")

    mcp_results = _check_mcp_servers(mcp_pool)
    llm_results = _check_llm_providers(llm_pool)

    all_checks = mcp_results + llm_results
    overall = _compute_overall_status(all_checks)

    response = PreflightResponse(
        status=overall,
        mcp_checks=mcp_results,
        llm_checks=llm_results,
    )

    # 📊 Summary log
    up_count = sum(1 for c in all_checks if c.status == "up")
    down_count = sum(1 for c in all_checks if c.status == "down")
    logger.info(
        "🔍 Preflight complete: status=%s, up=%d, down=%d",
        overall,
        up_count,
        down_count,
    )

    return response


def _compute_overall_status(
    checks: list[ServiceCheckResult],
) -> str:
    """Derive aggregate status from individual check results 📊.

    Args:
        checks: Combined list of MCP and LLM check results.

    Returns:
        ``"all_healthy"`` if every service is up,
        ``"degraded"`` if some are down,
        ``"all_down"`` if none are reachable (or no services configured).
    """
    if not checks:
        return "all_healthy"

    up_count = sum(1 for c in checks if c.status == "up")
    if up_count == len(checks):
        return "all_healthy"
    if up_count == 0:
        return "all_down"
    return "degraded"


# =====================================================================
# 🔌 MCP Server Probes
# =====================================================================


def _check_mcp_servers(
    mcp_pool: Any | None,
) -> list[ServiceCheckResult]:
    """Probe each enabled MCP server with a tools/list call 🔌.

    Args:
        mcp_pool: MCPConnectionPool instance, or None.

    Returns:
        List of ServiceCheckResult, one per enabled server.
    """
    if mcp_pool is None:
        logger.info("🔌 No MCP pool configured — skipping MCP checks")
        return []

    enabled_servers: list[str] = mcp_pool.get_enabled_servers()
    if not enabled_servers:
        logger.info("🔌 No enabled MCP servers — skipping MCP checks")
        return []

    results: list[ServiceCheckResult] = []
    for server_name in enabled_servers:
        result = _probe_mcp_server(mcp_pool, server_name)
        results.append(result)
        if result.status == "up":
            logger.info(
                "  ✅ MCP [%s]: UP (%.0fms)",
                server_name,
                result.latency_ms or 0,
            )
        else:
            logger.warning(
                "  ⚠️ MCP [%s]: DOWN — %s",
                server_name,
                result.error,
            )

    return results


def _probe_mcp_server(
    mcp_pool: Any,
    server_name: str,
) -> ServiceCheckResult:
    """Send a tools/list probe to a single MCP server 🔌.

    Uses the server's configured transport (stdio or HTTP) to perform
    the MCP initialize handshake followed by a ``tools/list`` request.
    This validates connectivity without executing a real search.

    Args:
        mcp_pool: MCPConnectionPool instance.
        server_name: Name of the MCP server to probe.

    Returns:
        ServiceCheckResult with status and optional latency/error.
    """
    try:
        server_config = mcp_pool.get_server_config(server_name)
    except Exception as exc:
        return ServiceCheckResult(
            name=server_name,
            status="down",
            error=str(exc),
        )

    transport = server_config.get("transport", "stdio")
    timeout = min(server_config.get("timeout_seconds", 10), 15)

    start = time.monotonic()
    try:
        if transport == "stdio":
            _probe_stdio(server_name, server_config, timeout)
        elif transport == "http":
            _probe_http(server_name, server_config, timeout)
        else:
            return ServiceCheckResult(
                name=server_name,
                status="down",
                error=f"Unsupported transport: {transport}",
            )
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceCheckResult(
            name=server_name,
            status="up",
            latency_ms=round(elapsed_ms, 1),
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceCheckResult(
            name=server_name,
            status="down",
            latency_ms=round(elapsed_ms, 1),
            error=str(exc),
        )


def _probe_stdio(
    server_name: str,
    server_config: dict[str, Any],
    timeout: int,
) -> None:
    """Probe an MCP server via stdio with tools/list 📡.

    Spawns the configured command, sends initialize + initialized +
    tools/list, and validates we get a successful JSON-RPC response.

    Args:
        server_name: Server name for logging.
        server_config: Server configuration dict.
        timeout: Subprocess timeout in seconds.

    Raises:
        RuntimeError: If handshake or tools/list fails.
    """
    import os as _os

    command = server_config.get("command", "")
    cmd_args = server_config.get("args", [])
    if not command:
        raise RuntimeError(f"No command configured for stdio server '{server_name}'")

    full_cmd = [command] + list(cmd_args)

    # 🔧 Build subprocess environment
    env = _os.environ.copy()
    for k, v in server_config.get("env", {}).items():
        if isinstance(v, str) and not v.startswith("${"):
            env[k] = v

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

        # 🤝 Step 1: initialize
        init_req = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "inquiro-preflight",
                        "version": "0.1.0",
                    },
                },
            }
        )
        proc.stdin.write(init_req + "\n")
        proc.stdin.flush()

        init_resp = _read_jsonrpc_response(proc, timeout)
        if init_resp and "error" in init_resp:
            raise RuntimeError(f"MCP initialize error: {init_resp['error']}")

        # 📣 Step 2: initialized notification
        notification = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        )
        proc.stdin.write(notification + "\n")
        proc.stdin.flush()

        # 📋 Step 3: tools/list (lightweight probe)
        list_req = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
        )
        proc.stdin.write(list_req + "\n")
        proc.stdin.flush()

        list_resp = _read_jsonrpc_response(proc, timeout)
        if list_resp and "error" in list_resp:
            raise RuntimeError(f"MCP tools/list error: {list_resp['error']}")
        if list_resp is None:
            raise RuntimeError("No response to tools/list")

    except FileNotFoundError:
        raise RuntimeError(f"MCP command '{command}' not found")
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


def _read_jsonrpc_response(
    proc: subprocess.Popen,
    timeout: int,
) -> dict[str, Any] | None:
    """Read a JSON-RPC response from subprocess stdout 📥.

    Reads lines until a valid JSON-RPC response with an ``id`` or
    ``error`` field is found, skipping notifications and non-JSON.

    Args:
        proc: Running subprocess.
        timeout: Read timeout in seconds.

    Returns:
        Parsed JSON dict, or None on timeout/EOF.
    """
    import io
    import select
    import time as _time

    deadline = _time.monotonic() + timeout

    while _time.monotonic() < deadline:
        remaining = deadline - _time.monotonic()
        if remaining <= 0:
            break

        try:
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 2.0))
            if not ready:
                if proc.poll() is not None:
                    return None
                continue
        except (io.UnsupportedOperation, ValueError):
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
            continue

        if "id" in data or "error" in data:
            return data

    return None


def _probe_http(
    server_name: str,
    server_config: dict[str, Any],
    timeout: int,
) -> None:
    """Probe an MCP server via HTTP with tools/list 🌐.

    Sends a JSON-RPC ``tools/list`` POST request to the configured
    endpoint and validates the response.

    Args:
        server_name: Server name for logging.
        server_config: Server configuration dict with ``endpoint`` key.
        timeout: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If the HTTP request fails or returns an error.
    """
    import urllib.error
    import urllib.request

    endpoint = server_config.get("endpoint", "")
    if not endpoint:
        raise RuntimeError(f"No endpoint configured for HTTP server '{server_name}'")

    rpc_request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=rpc_request,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP probe failed: {exc}")
    except TimeoutError:
        raise RuntimeError(f"HTTP probe timed out after {timeout}s")

    try:
        rpc_response = json.loads(raw)
    except json.JSONDecodeError:
        # ✨ Got a response but not JSON — server is at least reachable
        return

    if "error" in rpc_response:
        err = rpc_response["error"]
        raise RuntimeError(f"tools/list error: {err.get('message', err)}")


# =====================================================================
# 🤖 LLM Provider Probes
# =====================================================================


def _check_llm_providers(
    llm_pool: Any | None,
) -> list[ServiceCheckResult]:
    """Probe configured LLM providers with a minimal ping 🤖.

    Args:
        llm_pool: LLMProviderPool instance, or None.

    Returns:
        List of ServiceCheckResult items for configured models,
        or empty list if no LLM pool is configured.
    """
    if llm_pool is None:
        logger.info("🤖 No LLM pool configured — skipping LLM checks")
        return []

    # Prefer checking all configured providers; fallback to default only.
    model_names: list[str] = []
    if hasattr(llm_pool, "get_available_models"):
        try:
            model_names = list(llm_pool.get_available_models())
        except Exception:
            model_names = []

    if not model_names:
        default_model = getattr(llm_pool, "default_model", "")
        if default_model:
            model_names = [default_model]

    if not model_names:
        logger.info("🤖 No LLM models configured — skipping LLM checks")
        return []

    # 🧹 Deduplicate while preserving order.
    model_names = list(dict.fromkeys(model_names))

    providers_map = getattr(llm_pool, "_providers", {})
    if not isinstance(providers_map, dict):
        providers_map = {}

    probe_models: list[str] = []
    for model_name in model_names:
        provider_config = providers_map.get(model_name, {})
        if isinstance(provider_config, dict) and not _provider_has_required_credentials(
            model_name, provider_config
        ):
            continue
        probe_models.append(model_name)

    if not probe_models:
        logger.info(
            "🤖 No probeable LLM models (missing credentials) — skipping LLM checks"
        )
        return []

    results: list[ServiceCheckResult] = []
    for model_name in probe_models:
        result = _probe_llm_provider(llm_pool, model_name)
        results.append(result)
        if result.status == "up":
            logger.info(
                "  ✅ LLM [%s]: UP (%.0fms)",
                model_name,
                result.latency_ms or 0,
            )
        else:
            logger.warning(
                "  ⚠️ LLM [%s]: DOWN — %s",
                model_name,
                result.error,
            )

    return results


def _provider_has_required_credentials(
    model_name: str,
    provider_config: dict[str, Any],
) -> bool:
    """Check whether provider has minimally required credentials 🔐.

    Returns False for unresolved placeholders (e.g. ``${ANTHROPIC_API_KEY}``)
    or empty credentials, so preflight can skip optional providers
    instead of producing repetitive false alarms.
    """
    provider = str(provider_config.get("provider", "")).strip().lower()

    if provider == "bedrock":
        access_key = provider_config.get("aws_access_key_id")
        secret_key = provider_config.get("aws_secret_access_key")
        missing = _is_missing_secret(access_key) or _is_missing_secret(secret_key)
        if missing:
            logger.info(
                "  ⏭️ LLM [%s]: skipped (Bedrock credentials missing)",
                model_name,
            )
            return False
        return True

    if provider in {"anthropic", "openai", "deepseek", "openrouter"}:
        api_key = provider_config.get("api_key")
        if _is_missing_secret(api_key):
            logger.info(
                "  ⏭️ LLM [%s]: skipped (%s API key missing)",
                model_name,
                provider,
            )
            return False
        return True

    # Unknown provider type: do not block probe.
    return True


def _is_missing_secret(value: Any) -> bool:
    """Return True if a credential value is empty or unresolved 🚫."""
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    return bool(_UNRESOLVED_ENV_PATTERN.search(stripped))


def _probe_llm_provider(
    llm_pool: Any,
    model_name: str,
    timeout_seconds: float = 30.0,
) -> ServiceCheckResult:
    """Send a minimal ping message to an LLM provider 🤖.

    Constructs the smallest possible request to validate
    credentials and connectivity without incurring significant cost.
    Enforces a timeout to prevent blocking startup.

    Args:
        llm_pool: LLMProviderPool instance.
        model_name: Model name to probe.
        timeout_seconds: Max seconds to wait for a response.

    Returns:
        ServiceCheckResult with status and optional latency/error.
    """
    import concurrent.futures

    start = time.monotonic()
    try:
        llm = llm_pool.get_llm(model_name)

        # 🏓 Minimal ping: short message with safe token budget.
        messages = [
            {"role": "user", "content": "ping"},
        ]

        # ⏱️ Run with timeout to prevent blocking startup
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
        ) as executor:
            future = executor.submit(
                llm._call, messages, None, max_tokens=32,
            )
            future.result(timeout=timeout_seconds)

        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceCheckResult(
            name=model_name,
            status="up",
            latency_ms=round(elapsed_ms, 1),
        )
    except concurrent.futures.TimeoutError:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceCheckResult(
            name=model_name,
            status="down",
            latency_ms=round(elapsed_ms, 1),
            error=f"Timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return ServiceCheckResult(
            name=model_name,
            status="down",
            latency_ms=round(elapsed_ms, 1),
            error=str(exc),
        )
