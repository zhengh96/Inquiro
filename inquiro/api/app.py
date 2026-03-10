"""Inquiro FastAPI application factory 🚀.

Creates and configures the FastAPI application with:
- Async lifespan (startup: init MCP pool, LLM pool, EvalTaskRunner;
  shutdown: cleanup connections)
- CORS middleware (configurable origins from service.yaml)
- Router inclusion
- Exception handlers

Usage:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8100)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inquiro.api.router import router
from inquiro.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)

# ✅ Mandatory startup connectivity gate (hard requirement)
_REQUIRED_STARTUP_LLM: tuple[str, ...] = (
    "claude-bedrock",
    "gpt-5",
    "gemini-3-pro",
)
_REQUIRED_STARTUP_MCP: tuple[str, ...] = (
    "perplexity",
    "brave",
)


# ---------------------------------------------------------------------------
# 🔧 Early .env loading (before any config resolution)
# ---------------------------------------------------------------------------
# Following PharmMaster reference: load .env files at module init time
# so that all ${ENV_VAR} placeholders in YAML configs can be resolved,
# and MCP subprocesses inherit the correct environment.
# ---------------------------------------------------------------------------


def _load_env_file(env_path: Path) -> bool:
    """Load KEY=VALUE pairs from an env file into os.environ 📋.

    Existing process environment variables take precedence —
    the .env file only fills in missing values.

    Args:
        env_path: Path to the .env file.

    Returns:
        True if the file was found and loaded.
    """
    if not env_path.exists():
        return False

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1

    logger.info("📋 Loaded %d env vars from %s", loaded, env_path)
    return True


# 🔍 Search for .env files: project root > inquiro dir > cwd
_DIMSENSE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _DIMSENSE_DIR.parent
_load_env_file(_PROJECT_ROOT / ".env")
_load_env_file(_DIMSENSE_DIR / ".env")

# 🌐 Default CORS origins for development
_DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]


def _resolve_additional_skill_roots(
    project_root: Path,
) -> list[Path]:
    """Resolve extra skill roots for SkillService 🔍.

    Priority:
    1. ``INQUIRO_ADDITIONAL_SKILLS_ROOTS`` env var (comma-separated).
       Relative paths are resolved against *project_root*.
    2. Auto-discovery of ``targetmaster/skills`` in monorepo layout.

    Missing paths are skipped with warnings.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _append(path: Path, warn_missing: bool = True) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            return
        if not resolved.is_dir():
            if warn_missing:
                logger.warning(
                    "⚠️ Additional skill root not found: %s",
                    resolved,
                )
            return
        seen.add(resolved)
        roots.append(resolved)

    env_val = os.environ.get("INQUIRO_ADDITIONAL_SKILLS_ROOTS", "").strip()
    if env_val:
        for item in env_val.split(","):
            item = item.strip()
            if not item:
                continue
            candidate = Path(item)
            if not candidate.is_absolute():
                candidate = project_root / candidate
            _append(candidate, warn_missing=True)

    # 🎯 Monorepo auto-discovery: targetmaster/skills
    _append(
        project_root / "targetmaster" / "skills",
        warn_missing=False,
    )
    return roots


def _load_service_config() -> dict[str, Any]:
    """Eagerly load service.yaml for app construction 📋.

    Called synchronously during ``create_app()`` so that CORS origins
    and other service-level settings are available before middleware
    is constructed (middleware captures config at init time).

    Returns:
        Parsed service.yaml dict, or empty dict on failure.
    """
    from pathlib import Path

    try:
        from inquiro.infrastructure.config_loader import (
            ConfigLoader,
        )

        config_dir = Path(__file__).parent.parent / "configs"
        loader = ConfigLoader(config_dir)
        return loader.get_service_config()
    except Exception as exc:
        logger.warning(
            "⚠️ Could not load service.yaml for CORS config (%s), using defaults",
            exc,
        )
        return {}


def _load_cors_origins(service_config: dict[str, Any]) -> list[str]:
    """Resolve CORS allowed origins from config and environment 🌐.

    Priority:
    1. ``CORS_ALLOWED_ORIGINS`` env var (comma-separated).
    2. ``cors.allowed_origins`` list in service.yaml.
    3. Fall back to ``_DEFAULT_CORS_ORIGINS``.

    Args:
        service_config: Parsed service.yaml dict.

    Returns:
        List of allowed origin strings.
    """
    # ✨ Env var takes highest priority (comma-separated)
    env_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if env_origins:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        logger.info(
            "🌐 CORS origins from CORS_ALLOWED_ORIGINS env: %s",
            origins,
        )
        return origins

    # 📋 Config file
    cors_config = service_config.get("cors", {})
    config_origins = cors_config.get("allowed_origins", [])
    if config_origins:
        logger.info("🌐 CORS origins from service.yaml: %s", config_origins)
        return config_origins

    # ⚠️ Defaults
    logger.info("🌐 CORS origins: using defaults (localhost only)")
    return list(_DEFAULT_CORS_ORIGINS)


def _assert_required_startup_connectivity(
    preflight_result: Any,
) -> None:
    """Enforce mandatory startup connectivity gate ✅.

    Inquiro startup is considered successful only when all required
    LLM and MCP services are present in preflight checks and report
    ``status="up"``.

    Raises:
        RuntimeError: If any required service is missing or down.
    """
    llm_checks = getattr(preflight_result, "llm_checks", []) or []
    mcp_checks = getattr(preflight_result, "mcp_checks", []) or []

    llm_status = {
        str(getattr(item, "name", "")): (
            str(getattr(item, "status", "")),
            str(getattr(item, "error", "") or ""),
        )
        for item in llm_checks
    }
    mcp_status = {
        str(getattr(item, "name", "")): (
            str(getattr(item, "status", "")),
            str(getattr(item, "error", "") or ""),
        )
        for item in mcp_checks
    }

    failures: list[str] = []
    for name in _REQUIRED_STARTUP_LLM:
        status_tuple = llm_status.get(name)
        if status_tuple is None:
            failures.append(f"LLM[{name}]=missing")
            continue
        status, error = status_tuple
        if status != "up":
            detail = f", error={error}" if error else ""
            failures.append(f"LLM[{name}]={status}{detail}")

    for name in _REQUIRED_STARTUP_MCP:
        status_tuple = mcp_status.get(name)
        if status_tuple is None:
            failures.append(f"MCP[{name}]=missing")
            continue
        status, error = status_tuple
        if status != "up":
            detail = f", error={error}" if error else ""
            failures.append(f"MCP[{name}]={status}{detail}")

    if failures:
        raise RuntimeError(
            "Startup connectivity gate failed; required services "
            "must all be UP. " + "; ".join(failures)
        )

    logger.info(
        "✅ Startup connectivity gate passed (LLM=%s, MCP=%s)",
        ",".join(_REQUIRED_STARTUP_LLM),
        ",".join(_REQUIRED_STARTUP_MCP),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async lifespan manager for Inquiro service 🔄.

    Startup:
    1. Initialize MCP connection pool (persistent connections)
    2. Initialize LLM provider pool
    3. Create EvalTaskRunner with shared resources
    4. Store runner in app.state for dependency injection

    Shutdown:
    1. Cancel any active tasks
    2. Close MCP connections
    3. Clean up LLM provider resources
    """
    # 📝 Initialize structured logging early
    from inquiro.infrastructure.logging_config import setup_logging

    setup_logging()

    logger.info("🚀 Inquiro service starting up...")

    # ✨ Initialize infrastructure pools with ConfigLoader
    try:
        from pathlib import Path

        from inquiro.infrastructure.config_loader import (
            ConfigLoadError,
            ConfigLoader,
        )
        from inquiro.infrastructure.llm_pool import LLMProviderPool
        from inquiro.infrastructure.mcp_pool import MCPConnectionPool
        from inquiro.core.runner import EvalTaskRunner

        # 📋 Load configuration from inquiro/configs/
        config_dir = Path(__file__).parent.parent / "configs"
        try:
            config_loader = ConfigLoader(config_dir)
            mcp_config = config_loader.get_mcp_config()
            llm_config = config_loader.get_llm_config()
            logger.info("✅ Configuration loaded from %s", config_dir)
        except ConfigLoadError as exc:
            logger.warning(
                "⚠️ Config loading failed (%s), using empty defaults",
                exc,
            )
            mcp_config = {}
            llm_config = {}

        # 1. Initialize MCP connection pool 🔌
        mcp_pool = MCPConnectionPool(config=mcp_config)
        await mcp_pool.initialize()
        logger.info("✅ MCP connection pool initialized")

        # 2. Initialize LLM provider pool 🤖
        llm_pool = LLMProviderPool(config=llm_config)
        logger.info("✅ LLM provider pool initialized")

        # 3. Initialize SkillService 🎯
        from inquiro.infrastructure.skill_service import SkillService

        additional_skill_roots = _resolve_additional_skill_roots(_PROJECT_ROOT)
        skill_service = SkillService(additional_roots=additional_skill_roots)
        skill_service.setup()
        logger.info(
            "✅ SkillService initialized (additional_roots=%d)",
            len(additional_skill_roots),
        )

        # 3b. Load ensemble configuration 🎭
        ensemble_config = config_loader.get_ensemble_config()
        if ensemble_config.get("enabled"):
            logger.info(
                "🎭 Ensemble enabled: mode=%s, models=%d",
                ensemble_config.get("mode", "search_once_reason_many"),
                len(ensemble_config.get("models", [])),
            )

        # 4. Create task runner 🎯
        task_runner = EvalTaskRunner(
            mcp_pool=mcp_pool,
            llm_pool=llm_pool,
            skill_service=skill_service,
            ensemble_defaults=ensemble_config,
        )
        app.state.task_runner = task_runner
        logger.info("✅ EvalTaskRunner initialized")

        # 5. Initialize evolution scheduler (if configured) 🧬
        scheduler = None
        evolution_db_url = os.getenv("EVOLUTION_DB_URL")
        if evolution_db_url:
            try:
                from inquiro.evolution.scheduler import (
                    EvolutionScheduler,
                    SchedulerConfig,
                )
                from inquiro.evolution.store_factory import get_store

                store = await get_store()
                scheduler_config = SchedulerConfig(
                    namespace=os.getenv("EVOLUTION_NAMESPACE", "targetmaster"),
                    decay_factor=float(os.getenv("EVOLUTION_DECAY_FACTOR", "0.95")),
                    prune_min_fitness=float(
                        os.getenv("EVOLUTION_PRUNE_MIN_FITNESS", "0.2")
                    ),
                )
                scheduler = EvolutionScheduler(
                    config=scheduler_config,
                    store=store,
                )
                # 🔧 Run maintenance immediately on startup
                await scheduler.run_maintenance()
                scheduler.start()
                app.state.evolution_scheduler = scheduler
                logger.info("✅ EvolutionScheduler started")
            except Exception as exc:
                logger.warning(
                    "⚠️ Evolution scheduler init failed: %s",
                    exc,
                )
                scheduler = None

        # 6. Run preflight connectivity checks 🔍
        from inquiro.api.preflight import preflight_check

        preflight_result = await preflight_check(
            mcp_pool=mcp_pool,
            llm_pool=llm_pool,
        )
        app.state.last_preflight = preflight_result
        _assert_required_startup_connectivity(preflight_result)
        if preflight_result.status == "all_down":
            logger.warning(
                "⚠️ All external services are DOWN — "
                "service will start but cannot execute tasks"
            )
        elif preflight_result.status == "degraded":
            logger.warning(
                "⚠️ Some external services are DOWN — "
                "service will start with reduced capabilities"
            )

    except ImportError as exc:
        # 📋 Graceful fallback for development/testing
        logger.warning(
            "⚠️ Could not initialize full engine (missing: %s). "
            "Running in stub mode — API accepts requests but cannot "
            "execute tasks.",
            exc,
        )
        app.state.task_runner = None

    logger.info("🚀 Inquiro service ready")

    yield  # 🔄 Application runs here

    # 🛑 Shutdown cleanup
    logger.info("🛑 Inquiro service shutting down...")

    # 🧬 Stop evolution scheduler
    evolution_scheduler = getattr(app.state, "evolution_scheduler", None)
    if evolution_scheduler is not None:
        try:
            await evolution_scheduler.stop()
            logger.info("✅ EvolutionScheduler stopped")
        except Exception as exc:
            logger.warning("⚠️ EvolutionScheduler stop failed: %s", exc)

    task_runner = getattr(app.state, "task_runner", None)
    if task_runner is not None:
        # 🛑 Cancel active tasks via public API (not private dict)
        for task_id in task_runner.get_active_task_ids():
            task_runner.cancel_task(task_id)
            logger.info("Cancelled active task: %s", task_id)

        # Close MCP connections 🔌
        if hasattr(task_runner, "mcp_pool") and task_runner.mcp_pool:
            await task_runner.mcp_pool.cleanup()
            logger.info("✅ MCP connections closed")

    logger.info("🛑 Inquiro service stopped")


def create_app() -> FastAPI:
    """Create and configure the Inquiro FastAPI application 🏗️.

    Returns:
        Configured FastAPI application instance with:
        - Async lifespan (startup/shutdown hooks)
        - CORS middleware (origins resolved eagerly)
        - API router (all /api/v1/* endpoints)
        - Global exception handlers
    """
    app = FastAPI(
        title="Inquiro — Generic Evidence Research & Synthesis Engine",
        description=(
            "Domain-agnostic evidence research and synthesis service. "
            "Provides atomic research (Search-Reason-Doubt-Reframe) "
            "and multi-report synthesis capabilities via REST API with "
            "SSE real-time progress streaming."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 🌐 CORS middleware — origins resolved eagerly at construction time
    # CORSMiddleware captures allow_origins at init, so we must resolve
    # before add_middleware(). Loads from env var → service.yaml → defaults.
    cors_origins = _load_cors_origins(_load_service_config())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 📡 Include API routers
    app.include_router(router)

    # 🧬 Include evolution management router
    from inquiro.api.evolution_router import router as evolution_router

    app.include_router(evolution_router)

    # ❌ Global exception handlers
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors as 400 Bad Request ❌."""
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                code="validation_error",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(
        request: Request, exc: RuntimeError
    ) -> JSONResponse:
        """Handle runtime errors as 503 Service Unavailable ❌."""
        logger.exception("Runtime error: %s", exc)
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                code="service_unavailable",
                message=str(exc),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors as 500 Internal Server Error ❌."""
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="internal_error",
                message="An unexpected error occurred",
            ).model_dump(),
        )

    return app
