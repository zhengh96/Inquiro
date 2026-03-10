"""Inquiro FastAPI router — all API endpoints 🌐.

Endpoints:
- POST   /api/v1/research           Submit atomic research task
- POST   /api/v1/synthesize         Submit synthesis task
- GET    /api/v1/task/{task_id}     Query task status & result
- GET    /api/v1/task/{task_id}/stream  SSE real-time progress
- DELETE /api/v1/task/{task_id}     Cancel a running task
- GET    /api/v1/health             Health check + capabilities

All endpoints follow the PRD API contract from
System_Decomposition.md Section 2.5.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from inquiro.api.dependencies import (
    get_active_task_count,
    get_event_emitter,
    get_task_runner,
    get_task_store,
)
from inquiro.api.task_store import TaskStore
from inquiro.api.schemas import (
    ErrorResponse,
    HealthResponse,
    PreflightResponse,
    ResearchRequest,
    SynthesizeRequest,
    TaskCancelResponse,
    TaskCost,
    TaskCostBreakdown,
    TaskResponse,
    TaskStatus,
    TaskSubmitResponse,
    TaskType,
)
from inquiro.core.types import (
    AdditionalResearchConfig as CoreAdditionalResearchConfig,
    AgentConfig as CoreAgentConfig,
    CostGuardConfig as CoreCostGuardConfig,
    EvaluationTask,
    QualityGateConfig as CoreQualityGateConfig,
    SynthesisTask,
    ToolsConfig as CoreToolsConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inquiro"])


# ============================================================
# 📦 Internal TaskState (in-memory state for Phase 0)
# ============================================================


class TaskState:
    """In-memory mutable state for a running task 📦.

    Tracks lifecycle, result, cost, and event emitter for each task.
    Phase 0 uses in-memory storage; production migrates to PostgreSQL.
    """

    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        event_emitter: Any,
    ):
        """Initialize TaskState 🔧.

        Args:
            task_id: Unique task identifier
            task_type: Research or synthesis
            event_emitter: SSE event emitter for this task
        """
        self.task_id = task_id
        self.task_type = task_type
        self.status: TaskStatus = TaskStatus.PENDING
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.cost: float = 0.0
        self.event_emitter = event_emitter


def _resolve_event_emitter_instance(raw_emitter: Any) -> Any:
    """Normalize event emitter to an instance (never class object) 🔧."""
    # Defensive fallback: if a class object is accidentally returned,
    # instantiate it so background cleanup calls stay safe.
    if isinstance(raw_emitter, type):
        logger.warning(
            "⚠️ get_event_emitter() returned a class object (%s); "
            "instantiating automatically",
            raw_emitter.__name__,
        )
        return raw_emitter()
    return raw_emitter


# ============================================================
# 🔬 POST /api/v1/research — Submit research task
# ============================================================


@router.post(
    "/research",
    response_model=TaskSubmitResponse,
    status_code=202,
    responses={
        202: {"description": "Task accepted for async processing"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Submit atomic research task",
)
async def submit_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    task_runner: Any = Depends(get_task_runner),
    task_store: TaskStore = Depends(get_task_store),
) -> TaskSubmitResponse:
    """Submit an atomic research task for asynchronous execution 🔬.

    The discovery pipeline executes multi-round search and analysis
    on the given topic. Use the returned stream_url for SSE
    real-time progress, or poll_url for status polling.

    Args:
        request: Research task specification (topic, rules, checklist, etc.)
        background_tasks: FastAPI background task manager
        task_runner: Injected EvalTaskRunner
        task_store: Injected task store

    Returns:
        TaskSubmitResponse with task_id, stream_url, and poll_url
    """
    logger.info(
        "Research task submitted: task_id=%s, topic=%s",
        request.task_id,
        request.task.topic[:80],
    )

    # ✨ Create event emitter and task state
    event_emitter = _resolve_event_emitter_instance(get_event_emitter())
    task_state = TaskState(
        task_id=request.task_id,
        task_type=TaskType.RESEARCH,
        event_emitter=event_emitter,
    )
    task_store[request.task_id] = task_state

    # 🔄 Run research in background
    background_tasks.add_task(
        _run_research_background, request, task_state, task_runner
    )

    return TaskSubmitResponse(
        task_id=request.task_id,
        status="accepted",
        stream_url=f"/api/v1/task/{request.task_id}/stream",
        poll_url=f"/api/v1/task/{request.task_id}",
    )


# ============================================================
# 📊 POST /api/v1/synthesize — Submit synthesis task
# ============================================================


@router.post(
    "/synthesize",
    response_model=TaskSubmitResponse,
    status_code=202,
    responses={
        202: {"description": "Task accepted for async processing"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Submit synthesis task",
)
async def submit_synthesis(
    request: SynthesizeRequest,
    background_tasks: BackgroundTasks,
    task_runner: Any = Depends(get_task_runner),
    task_store: TaskStore = Depends(get_task_store),
) -> TaskSubmitResponse:
    """Submit a multi-report synthesis task for asynchronous execution 📊.

    The SynthesisAgent reads input reports, cross-references findings,
    identifies gaps and contradictions, optionally triggers additional
    research, and produces a unified synthesized result.

    Args:
        request: Synthesis task specification (objective, input_reports, etc.)
        background_tasks: FastAPI background task manager
        task_runner: Injected EvalTaskRunner
        task_store: Injected task store

    Returns:
        TaskSubmitResponse with task_id, stream_url, and poll_url
    """
    logger.info(
        "Synthesis task submitted: task_id=%s, objective=%s, reports=%d",
        request.task_id,
        request.task.objective[:80],
        len(request.task.input_reports),
    )

    # ✨ Create event emitter and task state
    event_emitter = _resolve_event_emitter_instance(get_event_emitter())
    task_state = TaskState(
        task_id=request.task_id,
        task_type=TaskType.SYNTHESIS,
        event_emitter=event_emitter,
    )
    task_store[request.task_id] = task_state

    # 🔄 Run synthesis in background
    background_tasks.add_task(
        _run_synthesis_background, request, task_state, task_runner
    )

    return TaskSubmitResponse(
        task_id=request.task_id,
        status="accepted",
        stream_url=f"/api/v1/task/{request.task_id}/stream",
        poll_url=f"/api/v1/task/{request.task_id}",
    )


# ============================================================
# 📊 GET /api/v1/task/{task_id} — Query task status
# ============================================================


@router.get(
    "/task/{task_id}",
    response_model=TaskResponse,
    responses={
        200: {"description": "Task status and optional result"},
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
    summary="Get task status and result",
)
async def get_task(
    task_id: str,
    task_store: TaskStore = Depends(get_task_store),
) -> TaskResponse:
    """Query task status and result 📊.

    Works for both research and synthesis tasks. When the task is
    completed, the result field contains the structured report.

    Args:
        task_id: The task identifier

    Returns:
        TaskResponse with status, progress, result, and cost

    Raises:
        HTTPException: 404 if task not found
    """
    task_state: TaskState | None = task_store.get(task_id)
    if task_state is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code="task_not_found",
                message=f"Task '{task_id}' not found",
            ).model_dump(),
        )

    return TaskResponse(
        task_id=task_state.task_id,
        task_type=task_state.task_type,
        status=task_state.status,
        result=task_state.result,
        cost=TaskCost(
            total_cost_usd=task_state.cost,
            breakdown=TaskCostBreakdown(main_task=task_state.cost),
        )
        if task_state.cost > 0
        else None,
        trajectory_url=f"/api/v1/task/{task_id}/trajectory",
    )


# ============================================================
# 📡 GET /api/v1/task/{task_id}/stream — SSE real-time progress
# ============================================================


@router.get(
    "/task/{task_id}/stream",
    responses={
        200: {
            "description": "SSE event stream",
            "content": {"text/event-stream": {}},
        },
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
    summary="Stream task progress via SSE",
)
async def stream_task(
    task_id: str,
    task_store: TaskStore = Depends(get_task_store),
) -> StreamingResponse:
    """SSE endpoint for real-time task progress 📡.

    Streams structured events as the task executes:
    - task_started: Task execution begins
    - round_completed: After each search/synthesis round
    - additional_research_requested: SynthesisAgent triggers research
    - quality_gate_result: After QualityGate validation
    - task_completed: Task finished successfully
    - task_failed: Task execution failed

    Args:
        task_id: The task identifier

    Returns:
        StreamingResponse with text/event-stream content type

    Raises:
        HTTPException: 404 if task not found
    """
    task_state: TaskState | None = task_store.get(task_id)
    if task_state is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code="task_not_found",
                message=f"Task '{task_id}' not found",
            ).model_dump(),
        )

    async def event_generator():
        """Generate SSE events from task's event emitter 📡."""
        _SSE_STREAM_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
        start_time = time.monotonic()

        try:
            async for event in task_state.event_emitter.get_event_stream(task_id):
                # ⏱️ Timeout protection: 30 minutes max
                if time.monotonic() - start_time > _SSE_STREAM_TIMEOUT_SECONDS:
                    logger.warning("SSE stream timeout (30min) for task %s", task_id)
                    break

                yield (
                    f"event: {event.event_type}\ndata: {event.model_dump_json()}\n\n"
                )
                # 🛑 Terminal events end the stream
                if event.event_type in (
                    "task_completed",
                    "task_failed",
                    "task_cancelled",
                ):
                    break
        except Exception:
            logger.exception("SSE stream error for task %s", task_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# ⏹️ DELETE /api/v1/task/{task_id} — Cancel task
# ============================================================


@router.delete(
    "/task/{task_id}",
    response_model=TaskCancelResponse,
    responses={
        200: {"description": "Task cancelled successfully"},
        404: {"model": ErrorResponse, "description": "Task not found"},
        409: {
            "model": ErrorResponse,
            "description": "Task already completed/cancelled",
        },
    },
    summary="Cancel a running task",
)
async def cancel_task(
    task_id: str,
    task_runner: Any = Depends(get_task_runner),
    task_store: TaskStore = Depends(get_task_store),
) -> TaskCancelResponse:
    """Cancel a running task ⏹️.

    Sends a cancellation signal to the running agent via
    CancellationToken. The agent cooperatively stops at the
    next step boundary.

    Args:
        task_id: The task to cancel
        task_runner: Injected EvalTaskRunner
        task_store: Injected task store

    Returns:
        TaskCancelResponse with cancellation status

    Raises:
        HTTPException: 404 if not found, 409 if not cancellable
    """
    task_state: TaskState | None = task_store.get(task_id)
    if task_state is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code="task_not_found",
                message=f"Task '{task_id}' not found",
            ).model_dump(),
        )

    if task_state.status in (
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    ):
        raise HTTPException(
            status_code=409,
            detail=ErrorResponse(
                code="task_not_cancellable",
                message=(f"Task '{task_id}' is already {task_state.status.value}"),
            ).model_dump(),
        )

    # 🛑 Signal cancellation via CancellationToken
    cancelled = task_runner.cancel_task(task_id)
    if cancelled:
        task_state.status = TaskStatus.CANCELLED
        logger.info("Task cancelled: task_id=%s", task_id)

    return TaskCancelResponse(
        task_id=task_id,
        status="cancelled" if cancelled else "not_running",
        reason="user_requested",
    )


# ============================================================
# ❤️ GET /api/v1/health — Health check
# ============================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and capabilities",
)
async def health_check(
    request: Request,
    task_runner: Any = Depends(get_task_runner),
) -> HealthResponse:
    """Health check endpoint ❤️.

    Returns service health status, version, available capabilities,
    active task count, and per-MCP-server connectivity status.

    Args:
        request: FastAPI request
        task_runner: Injected EvalTaskRunner

    Returns:
        HealthResponse with service status details
    """
    # 📊 Get MCP server health if available
    mcp_health: dict[str, str] = {}
    if hasattr(task_runner, "mcp_pool") and task_runner.mcp_pool:
        mcp_health = task_runner.mcp_pool.get_health()

    active_count = get_active_task_count(request)

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        capabilities=["research", "synthesis"],
        active_tasks=active_count,
        mcp_servers=mcp_health,
    )


# ============================================================
# 🔍 GET /api/v1/preflight — On-demand connectivity check
# ============================================================


@router.get(
    "/preflight",
    response_model=PreflightResponse,
    summary="Run preflight connectivity checks",
)
async def preflight(
    request: Request,
    task_runner: Any = Depends(get_task_runner),
) -> PreflightResponse:
    """Run on-demand connectivity checks against MCP and LLM services 🔍.

    Re-executes the same probes that run during startup:
    - Each enabled MCP server is tested via a ``tools/list`` call.
    - The default LLM provider is tested via a minimal ping message.

    The response reports per-service UP/DOWN status and overall
    system readiness. Does NOT block or affect running tasks.

    Args:
        request: FastAPI request.
        task_runner: Injected EvalTaskRunner.

    Returns:
        PreflightResponse with per-service results and overall status.
    """
    from inquiro.api.preflight import preflight_check

    mcp_pool = getattr(task_runner, "mcp_pool", None)
    llm_pool = getattr(task_runner, "llm_pool", None)

    result = await preflight_check(
        mcp_pool=mcp_pool,
        llm_pool=llm_pool,
    )

    # 📋 Cache the latest result in app state
    request.app.state.last_preflight = result

    return result


# ============================================================
# 🔄 API → Core model conversion helpers
# ============================================================


def _to_evaluation_task(request: ResearchRequest) -> EvaluationTask:
    """Convert API ResearchRequest to internal EvaluationTask 🔄.

    Reconstructs the flat EvaluationTask expected by EvalTaskRunner
    from the nested API request payload. Maps API-specific config
    models (with stricter validation bounds) to their core counterparts.

    Args:
        request: The validated API research request.

    Returns:
        A fully populated EvaluationTask ready for EvalTaskRunner.
    """
    return EvaluationTask(
        task_id=request.task_id,
        topic=request.task.topic,
        rules=request.task.rules,
        checklist=request.task.checklist,
        decision_guidance=request.task.decision_guidance,
        output_schema=request.task.output_schema,
        agent_config=CoreAgentConfig(
            model=request.agent_config.model,
            max_turns=request.agent_config.max_turns,
            temperature=request.agent_config.temperature,
            system_prompt_template=(request.agent_config.system_prompt_template),
        ),
        tools_config=CoreToolsConfig(
            mcp_servers=request.tools_config.mcp_servers,
            mcp_config_override=(request.tools_config.mcp_config_override),
        ),
        ensemble_config=request.ensemble_config,
        quality_gate=CoreQualityGateConfig(
            enabled=request.quality_gate.enabled,
            max_retries=request.quality_gate.max_retries,
            checks=request.quality_gate.checks,
        ),
        cost_guard=CoreCostGuardConfig(
            max_cost_per_task=request.cost_guard.max_cost_per_task,
            overspend_strategy=request.cost_guard.overspend_strategy,
        ),
        callback_url=request.callback_url,
        # 🧬 Evolution fields (opaque pass-through)
        evolution_profile=request.evolution_profile,
        context_tags=request.context_tags,
        sub_item_id=request.sub_item_id or None,
        # 🔬 Discovery pipeline fields
        discovery_config=request.discovery_config,
        # 🔄 Evidence sharing fields
        evidence_pool_id=request.evidence_pool_id,
        # 📋 Query strategy (opaque pass-through from TargetMaster)
        query_strategy=request.query_strategy,
        # 📦 Knowledge Base injection (pass-through from TargetMaster)
        seeded_evidence=request.seeded_evidence,
        seeded_gap_hints=request.seeded_gap_hints,
    )


def _to_synthesis_task(
    request: SynthesizeRequest,
) -> SynthesisTask:
    """Convert API SynthesizeRequest to internal SynthesisTask 🔄.

    Reconstructs the flat SynthesisTask expected by EvalTaskRunner
    from the nested API request payload. Conditionally constructs
    the additional_research_config only when enabled.

    Args:
        request: The validated API synthesis request.

    Returns:
        A fully populated SynthesisTask ready for EvalTaskRunner.
    """
    # 🔬 Build additional research config only when enabled
    additional_research = None
    if (
        request.task.allow_additional_research
        and request.task.additional_research_config
    ):
        ar = request.task.additional_research_config
        additional_research = CoreAdditionalResearchConfig(
            max_tasks=ar.max_tasks,
            cost_budget=ar.cost_budget,
            tools_config=ar.tools_config,
        )

    return SynthesisTask(
        task_id=request.task_id,
        topic=request.task.objective,
        input_reports=request.task.input_reports,
        synthesis_rules=request.task.synthesis_rules,
        output_schema=request.task.output_schema,
        allow_additional_research=(request.task.allow_additional_research),
        additional_research_config=additional_research,
        agent_config=CoreAgentConfig(
            model=request.agent_config.model,
            max_turns=request.agent_config.max_turns,
            temperature=request.agent_config.temperature,
            system_prompt_template=(request.agent_config.system_prompt_template),
        ),
        quality_gate=CoreQualityGateConfig(
            enabled=request.quality_gate.enabled,
            max_retries=request.quality_gate.max_retries,
            checks=request.quality_gate.checks,
        ),
        cost_guard=CoreCostGuardConfig(
            max_cost_per_task=request.cost_guard.max_cost_per_task,
            overspend_strategy=request.cost_guard.overspend_strategy,
        ),
        callback_url=request.callback_url,
        # 🧬 Evolution fields (opaque pass-through)
        evolution_profile=request.evolution_profile,
        context_tags=request.context_tags,
    )


# ============================================================
# 🔄 Background task runners
# ============================================================


async def _run_research_background(
    request: ResearchRequest,
    task_state: TaskState,
    task_runner: Any,
) -> None:
    """Execute research task in background 🔄.

    Updates TaskState in-place as the task progresses through its
    lifecycle: pending → running → completed/failed.

    Args:
        request: Original ResearchRequest
        task_state: Mutable task state to update
        task_runner: EvalTaskRunner instance
    """
    task_state.status = TaskStatus.RUNNING
    task_state.event_emitter.emit(
        "task_started",
        request.task_id,
        {"task_type": "research"},
    )
    try:
        # 🔄 Convert API schema to internal core model
        eval_task = _to_evaluation_task(request)
        result = await task_runner.submit_research(eval_task, task_state.event_emitter)
        task_state.result = (
            result.model_dump() if hasattr(result, "model_dump") else result
        )
        task_state.status = TaskStatus.COMPLETED
        task_state.cost = result.cost if hasattr(result, "cost") else 0.0
        task_state.event_emitter.emit(
            "task_completed",
            request.task_id,
            {"task_type": "research"},
        )
        logger.info("Research task completed: task_id=%s", request.task_id)
    except Exception as exc:
        task_state.status = TaskStatus.FAILED
        task_state.error = str(exc)
        task_state.event_emitter.emit(
            "task_failed",
            request.task_id,
            {"error": str(exc)},
        )
        logger.exception("Research task failed: task_id=%s", request.task_id)
    finally:
        # 🔌 Close SSE stream for cleanup
        try:
            task_state.event_emitter.close_stream(request.task_id)
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to close event stream for task_id=%s: %s",
                request.task_id,
                exc,
            )


async def _run_synthesis_background(
    request: SynthesizeRequest,
    task_state: TaskState,
    task_runner: Any,
) -> None:
    """Execute synthesis task in background 🔄.

    Updates TaskState in-place as the task progresses through its
    lifecycle: pending → running → completed/failed.

    Args:
        request: Original SynthesizeRequest
        task_state: Mutable task state to update
        task_runner: EvalTaskRunner instance
    """
    task_state.status = TaskStatus.RUNNING
    task_state.event_emitter.emit(
        "task_started",
        request.task_id,
        {"task_type": "synthesis"},
    )
    try:
        # 🔄 Convert API schema to internal core model
        synth_task = _to_synthesis_task(request)
        result = await task_runner.submit_synthesis(
            synth_task, task_state.event_emitter
        )
        task_state.result = (
            result.model_dump() if hasattr(result, "model_dump") else result
        )
        task_state.status = TaskStatus.COMPLETED
        task_state.cost = result.cost if hasattr(result, "cost") else 0.0
        task_state.event_emitter.emit(
            "task_completed",
            request.task_id,
            {"task_type": "synthesis"},
        )
        logger.info("Synthesis task completed: task_id=%s", request.task_id)
    except Exception as exc:
        task_state.status = TaskStatus.FAILED
        task_state.error = str(exc)
        task_state.event_emitter.emit(
            "task_failed",
            request.task_id,
            {"error": str(exc)},
        )
        logger.exception("Synthesis task failed: task_id=%s", request.task_id)
    finally:
        # 🔌 Close SSE stream for cleanup
        try:
            task_state.event_emitter.close_stream(request.task_id)
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to close event stream for task_id=%s: %s",
                request.task_id,
                exc,
            )
