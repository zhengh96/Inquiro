"""Inquiro API Pydantic request/response models 📦.

All schemas match the PRD API contract defined in System_Decomposition.md
Section 2.5. Models are organized as:
- Request models: ResearchRequest, SynthesizeRequest
- Response models: TaskSubmitResponse, TaskResponse, TaskCancelResponse,
  HealthResponse, ErrorResponse
- Nested sub-models: Checklist, AgentConfig, ToolsConfig, etc.
- SSE event models: SSEEvent and typed variants

Shared domain models (enums, config types, data models) are imported from
``inquiro.core.types`` to eliminate duplication (TD-1). Only API-specific
models (request/response wrappers, SSE events, error types) and models
with API-boundary-specific validation constraints remain here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ============================================================
# 🔗 Re-exported from inquiro.core.types (dedup TD-1)
#
# These models are the single source of truth in types.py.
# Re-exported here so existing imports from schemas.py continue to work.
# ============================================================

from inquiro.core.types import (  # noqa: F401
    # 🏷️ Enums
    TaskType,
    TaskStatus,
    OverspendStrategy,
    TaskPhase,
    # 📝 Shared config/data models
    ChecklistItem,
    Checklist,
    ToolsConfig,
    EnsembleModelConfig,
    EnsembleConfig,
    DecisionGuidance,
    InputReport,
    QualityChecks,
    # 🔄 Core type imports for inheritance
    ContextConfig as CoreContextConfig,
    AgentConfig as CoreAgentConfig,
    QualityGateConfig as CoreQualityGateConfig,
    CostGuardConfig as CoreCostGuardConfig,
    AdditionalResearchConfig as CoreAdditionalResearchConfig,
)


# ============================================================
# 📡 SSE Event Type (API-only enum)
# ============================================================


class SSEEventType(str, Enum):
    """SSE event type identifiers 📡."""

    TASK_STARTED = "task_started"
    ROUND_COMPLETED = "round_completed"
    ADDITIONAL_RESEARCH_REQUESTED = "additional_research_requested"
    QUALITY_GATE_RESULT = "quality_gate_result"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    # 📡 Discovery pipeline SSE events
    DISCOVERY_STARTED = "discovery_started"
    DISCOVERY_ROUND_STARTED = "discovery_round_started"
    DISCOVERY_ROUND_COMPLETED = "discovery_round_completed"
    DISCOVERY_COVERAGE_UPDATED = "discovery_coverage_updated"
    DISCOVERY_CONVERGED = "discovery_converged"
    DISCOVERY_COMPLETED = "discovery_completed"


# ============================================================
# 🤖 API-specific Configuration Sub-models
#
# These models have API-boundary-specific validation constraints
# (e.g., stricter bounds on max_turns, temperature) that differ
# from the core types.py versions.
# ============================================================


class ContextConfig(CoreContextConfig):
    """LLM context window configuration for API requests 🧠."""

    pass  # ✅ Inherits all from core


class AgentConfig(CoreAgentConfig):
    """Agent configuration for API requests 🤖."""

    @field_validator("max_turns")
    @classmethod
    def validate_max_turns_api_bound(cls, v: int) -> int:
        """Enforce API boundary constraint on max_turns 🔒."""
        if v > 100:
            raise ValueError("max_turns must not exceed 100 at API boundary")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature_api_bound(cls, v: float) -> float:
        """Enforce API boundary constraint on temperature 🔒."""
        if v > 1.0:
            raise ValueError("temperature must not exceed 1.0 at API boundary")
        return v


# ============================================================
# ✅ Quality Gate & Cost Guard (API-specific bounds)
# ============================================================


class QualityGateConfig(CoreQualityGateConfig):
    """Quality gate config for API requests 🔍."""

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries_api_bound(cls, v: int) -> int:
        """Enforce API boundary constraint on max_retries 🔒."""
        if v > 5:
            raise ValueError("max_retries must not exceed 5 at API boundary")
        return v


class CostGuardConfig(CoreCostGuardConfig):
    """Cost guard config for API requests 💰."""

    pass  # ✅ Inherits from core


class AdditionalResearchConfig(CoreAdditionalResearchConfig):
    """Additional research config for API requests 🔬."""

    @field_validator("max_tasks")
    @classmethod
    def validate_max_tasks_api_bound(cls, v: int) -> int:
        """Enforce API boundary constraint on max_tasks 🔒."""
        if v > 10:
            raise ValueError("max_tasks must not exceed 10 at API boundary")
        return v


# ============================================================
# 📬 Top-level Request Models
# ============================================================


class ResearchTaskPayload(BaseModel):
    """Research task payload — the core research specification 🔬.

    Contains the topic, evaluation rules, search checklist, decision
    guidance, and output schema. Completely flat and domain-agnostic.
    """

    topic: str = Field(..., description="Research topic to investigate")
    rules: str = Field(
        default="",
        description=(
            "Evaluation rules in Markdown (domain-specific, injected by caller)"
        ),
    )
    checklist: Checklist = Field(
        default_factory=Checklist,
        description="Search checklist with required/optional items",
    )
    decision_guidance: DecisionGuidance | None = Field(
        default=None,
        description="Hints for classifying the research outcome",
    )
    output_schema: dict[str, Any] = Field(
        ..., description="JSON Schema for the structured output"
    )


class ResearchRequest(BaseModel):
    """POST /api/v1/research — Submit atomic research task 🔬.

    The complete request payload for submitting a research task.
    Maps to the PRD API contract in System_Decomposition.md Section 2.5.

    The ``task`` object is completely flat. There is no dimension or
    sub_item wrapper. Inquiro sees only a topic, rules, checklist,
    and output schema.
    """

    task_id: str = Field(
        ..., description="Caller-assigned unique task identifier (UUID)"
    )
    task: ResearchTaskPayload = Field(..., description="Research task specification")
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent-level configuration",
    )
    tools_config: ToolsConfig = Field(
        default_factory=ToolsConfig,
        description="MCP tools configuration",
    )
    ensemble_config: EnsembleConfig = Field(
        default_factory=EnsembleConfig,
        description="Multi-LLM ensemble configuration",
    )
    quality_gate: QualityGateConfig = Field(
        default_factory=QualityGateConfig,
        description="Quality gate configuration",
    )
    cost_guard: CostGuardConfig = Field(
        default_factory=CostGuardConfig,
        description="Cost guard configuration",
    )
    callback_url: str | None = Field(
        default=None,
        description="Webhook URL for task completion notification",
    )

    # 🧬 Evolution — injected by upper layer, opaque to Inquiro
    evolution_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Evolution profile config from upper layer. "
            "When present, enables experience-based prompt enrichment."
        ),
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Context tags for experience matching (opaque to Inquiro)",
    )
    sub_item_id: str = Field(
        default="",
        description="Sub-item identifier for experience matching (opaque to Inquiro)",
    )
    # 🔬 Discovery pipeline fields
    pipeline_mode: str = Field(
        default="discovery",
        description=(
            "Pipeline execution mode (deprecated — always 'discovery'). "
            "Retained for backward compatibility with existing API clients."
        ),
    )
    discovery_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "DiscoveryConfig parameters for the discovery pipeline."
        ),
    )
    # 🔄 Evidence sharing fields
    evidence_pool_id: str | None = Field(
        default=None,
        description=(
            "Shared evidence pool identifier for cross-task evidence reuse. "
            "Tasks with the same pool_id share evidence in DISCOVERY mode."
        ),
    )

    # 📦 Knowledge Base injection fields (opaque to Inquiro)
    seeded_evidence: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Pre-filled evidence from knowledge base. "
            "Injected into DiscoveryLoop as starting evidence set."
        ),
    )
    seeded_gap_hints: list[str] | None = Field(
        default=None,
        description=(
            "Known gap descriptions from prior KB evaluations. "
            "Injected into round-1 focus prompt."
        ),
    )

    # 📋 Query strategy — parsed template from TargetMaster, opaque to Inquiro
    query_strategy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parsed query strategy dict produced by TargetMaster's "
            "QueryTemplateParser. Contains alias_expansion, query_sections, "
            "and tool_allocations. Passed through to SearchExp for structured "
            "search prompt injection. None when no template exists."
        ),
    )


class SynthesisTaskPayload(BaseModel):
    """Synthesis task payload — the core synthesis specification 📊.

    Contains the objective, input reports, synthesis rules, optional
    additional research config, and output schema.
    """

    objective: str = Field(..., description="Synthesis objective description")
    input_reports: list[InputReport] = Field(
        ...,
        min_length=1,
        description="Research reports to synthesize",
    )
    synthesis_rules: str = Field(
        default="",
        description="Synthesis rules in Markdown (domain-specific, injected by caller)",
    )
    allow_additional_research: bool = Field(
        default=True,
        description="Whether SynthesisAgent can trigger additional research",
    )
    additional_research_config: AdditionalResearchConfig = Field(
        default_factory=AdditionalResearchConfig,
        description="Constraints for additional research",
    )
    output_schema: dict[str, Any] = Field(
        ..., description="JSON Schema for the structured synthesis output"
    )


class SynthesizeRequest(BaseModel):
    """POST /api/v1/synthesize — Submit synthesis task 📊.

    The complete request payload for submitting a synthesis task.
    Maps to the PRD API contract in System_Decomposition.md Section 2.5.

    ``input_reports`` carries the actual content of previously completed
    research reports. The SynthesisAgent reads, reasons, and synthesizes.
    """

    task_id: str = Field(
        ..., description="Caller-assigned unique task identifier (UUID)"
    )
    task: SynthesisTaskPayload = Field(..., description="Synthesis task specification")
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent-level configuration",
    )
    quality_gate: QualityGateConfig = Field(
        default_factory=QualityGateConfig,
        description="Quality gate configuration",
    )
    cost_guard: CostGuardConfig = Field(
        default_factory=CostGuardConfig,
        description="Cost guard configuration",
    )
    callback_url: str | None = Field(
        default=None,
        description="Webhook URL for task completion notification",
    )

    # 🧬 Evolution — injected by upper layer, opaque to Inquiro
    evolution_profile: dict[str, Any] | None = Field(
        default=None,
        description="Evolution profile config from upper layer",
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Context tags for experience matching (opaque to Inquiro)",
    )


# ============================================================
# 📊 Response Models
# ============================================================


class TaskSubmitResponse(BaseModel):
    """Response after submitting a research or synthesis task 📬.

    Returned immediately upon task acceptance. The task runs
    asynchronously — use poll_url or stream_url to monitor progress.
    """

    task_id: str = Field(..., description="The accepted task identifier")
    status: str = Field(default="accepted", description="Submission status")
    stream_url: str = Field(..., description="SSE endpoint URL for real-time progress")
    poll_url: str = Field(..., description="Polling endpoint URL for task status")


class TaskProgress(BaseModel):
    """Task execution progress details 🔄."""

    current_phase: TaskPhase = Field(..., description="Current execution phase")
    current_round: int = Field(
        default=0, ge=0, description="Current search/synthesis round"
    )
    max_rounds: int = Field(default=0, ge=0, description="Maximum rounds configured")
    additional_research_triggered: int = Field(
        default=0,
        ge=0,
        description="Count of additional research tasks triggered (synthesis)",
    )


class TaskCostBreakdown(BaseModel):
    """Cost breakdown for a completed task 💰."""

    main_task: float = Field(
        default=0.0, ge=0.0, description="Cost of the main task in USD"
    )
    additional_research: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost of additional research in USD",
    )


class TaskCost(BaseModel):
    """Total cost information for a task 💰."""

    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total cost in USD")
    breakdown: TaskCostBreakdown = Field(
        default_factory=TaskCostBreakdown,
        description="Cost breakdown by category",
    )


class TaskResponse(BaseModel):
    """GET /api/v1/task/{task_id} — Task status and result 📊.

    Returns current task state. When status is 'completed', the
    result field contains the structured report conforming to the
    caller's output_schema.
    """

    task_id: str = Field(..., description="Task identifier")
    task_type: TaskType = Field(..., description="Task type (research/synthesis)")
    status: TaskStatus = Field(..., description="Current task status")
    progress: TaskProgress | None = Field(
        default=None, description="Execution progress (while running)"
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Structured report (when completed)",
    )
    cost: TaskCost | None = Field(default=None, description="Cost information")
    trajectory_url: str | None = Field(
        default=None,
        description="URL to retrieve full execution trajectory",
    )


class TaskCancelResponse(BaseModel):
    """DELETE /api/v1/task/{task_id} — Task cancellation response ⏹️."""

    task_id: str = Field(..., description="Cancelled task identifier")
    status: str = Field(default="cancelled", description="Cancellation status")
    reason: str = Field(default="", description="Reason for cancellation")


class MCPServerStatus(BaseModel):
    """Health status of a single MCP server 🔌."""

    name: str = Field(..., description="MCP server name")
    status: str = Field(
        ..., description="Connection status (connected/degraded/disconnected)"
    )


class HealthResponse(BaseModel):
    """GET /api/v1/health — Service health check response ❤️.

    Returns service status, version, capabilities, active task count,
    and per-MCP-server connectivity status.
    """

    status: str = Field(default="healthy", description="Overall service health")
    version: str = Field(default="2.0.0", description="Inquiro engine version")
    capabilities: list[str] = Field(
        default_factory=lambda: ["research", "synthesis"],
        description="Available engine capabilities",
    )
    active_tasks: int = Field(
        default=0, ge=0, description="Number of currently running tasks"
    )
    mcp_servers: dict[str, str] = Field(
        default_factory=dict,
        description="MCP server name → status mapping",
    )


# ============================================================
# 🔍 Preflight Check Models
# ============================================================


class ServiceCheckResult(BaseModel):
    """Result of a single service connectivity check 🔍.

    Represents the UP/DOWN status of an individual external
    service (MCP server or LLM provider) after a probe attempt.
    """

    name: str = Field(..., description="Service identifier (e.g., MCP server name)")
    status: str = Field(
        ...,
        description="Service status: 'up' or 'down'",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Probe round-trip latency in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the service is down",
    )


class PreflightResponse(BaseModel):
    """GET /api/v1/preflight — Preflight connectivity check response 🔍.

    Aggregated result of probing all configured MCP servers and
    the default LLM provider. Used at startup and on-demand to
    verify external dependency availability.
    """

    status: str = Field(
        ...,
        description=(
            "Overall preflight status: 'all_healthy' if every service "
            "is up, 'degraded' if some are down, 'all_down' if none "
            "are reachable"
        ),
    )
    mcp_checks: list[ServiceCheckResult] = Field(
        default_factory=list,
        description="Per-MCP-server connectivity results",
    )
    llm_checks: list[ServiceCheckResult] = Field(
        default_factory=list,
        description="Per-LLM-provider connectivity results",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of the preflight check (UTC)",
    )


class ErrorDetail(BaseModel):
    """Structured error detail for API error responses ❌."""

    field: str | None = Field(default=None, description="Field that caused the error")
    message: str = Field(..., description="Error description")


class ErrorResponse(BaseModel):
    """Standard error response for all API endpoints ❌.

    Consistent error format across all endpoints.
    HTTP status codes: 400 (validation), 404 (not found),
    500 (internal), 503 (service unavailable).
    """

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Additional error details",
    )


# ============================================================
# 📡 SSE Event Models
# ============================================================


class SSEEventData(BaseModel):
    """Base SSE event data 📡."""

    task_id: str = Field(..., description="Associated task identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)",
    )


class TaskStartedEvent(SSEEventData):
    """SSE: task_started — Emitted when task execution begins 🚀."""

    task_type: TaskType = Field(..., description="Task type")


class RoundCompletedEvent(SSEEventData):
    """SSE: round_completed — Emitted after each search/synthesis round 🔄."""

    round: int = Field(..., ge=1, description="Completed round number")
    findings_count: int = Field(
        default=0, ge=0, description="Number of findings in this round"
    )
    gaps_count: int = Field(default=0, ge=0, description="Number of remaining gaps")


class AdditionalResearchRequestedEvent(SSEEventData):
    """SSE: additional_research_requested — SynthesisAgent triggers research 🔬."""

    parent_task_id: str = Field(
        ..., description="The synthesis task that triggered this research"
    )
    topic: str = Field(..., description="Topic of the additional research")


class QualityGateResultEvent(SSEEventData):
    """SSE: quality_gate_result — Emitted after QualityGate validation ✅."""

    passed: bool = Field(..., description="Whether the quality gate passed")
    soft_failures: list[str] = Field(
        default_factory=list,
        description="Non-blocking quality issues",
    )


class TaskCompletedEvent(SSEEventData):
    """SSE: task_completed — Emitted when task finishes successfully 🎉."""

    task_type: TaskType = Field(..., description="Task type")
    status: str = Field(default="completed", description="Final status")
    total_cost: float = Field(default=0.0, ge=0.0, description="Total task cost in USD")


class TaskFailedEvent(SSEEventData):
    """SSE: task_failed — Emitted when task execution fails ❌."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(
        default="internal_error", description="Machine-readable error code"
    )


# ============================================================
# 📡 Discovery Pipeline SSE Event Models
# ============================================================


class DiscoveryStartedEvent(SSEEventData):
    """SSE: discovery_started — Emitted when discovery loop begins 🔄."""

    max_rounds: int = Field(..., ge=1, description="Maximum rounds configured")
    coverage_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Target coverage threshold",
    )


class DiscoveryRoundStartedEvent(SSEEventData):
    """SSE: discovery_round_started — Emitted when a round starts 🔄."""

    round_number: int = Field(..., ge=1, description="Round number (1-based)")


class DiscoveryRoundCompletedEvent(SSEEventData):
    """SSE: discovery_round_completed — Emitted when a round finishes 📊."""

    round_number: int = Field(..., ge=1, description="Completed round number")
    coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Coverage ratio after this round",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative cost in USD after this round",
    )
    evidence_count: int = Field(
        default=0,
        ge=0,
        description="Total accumulated evidence items",
    )


class DiscoveryCoverageUpdatedEvent(SSEEventData):
    """SSE: discovery_coverage_updated — Coverage curve progression 📈."""

    coverage_curve: list[float] = Field(
        default_factory=list,
        description=("Coverage ratio at each completed round (ordered, 1-indexed)"),
    )


class DiscoveryConvergedEvent(SSEEventData):
    """SSE: discovery_converged — Convergence reached 🎯."""

    reason: str = Field(..., description="Convergence reason description")
    total_rounds: int = Field(..., ge=1, description="Total rounds executed")


class DiscoveryCompletedEvent(SSEEventData):
    """SSE: discovery_completed — Discovery pipeline finished ✅."""

    total_rounds: int = Field(..., ge=0, description="Total rounds executed")
    final_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final coverage ratio achieved",
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total pipeline cost in USD",
    )
    total_evidence: int = Field(
        default=0,
        ge=0,
        description="Total evidence items collected",
    )
    termination_reason: str = Field(default="", description="Why the pipeline stopped")
