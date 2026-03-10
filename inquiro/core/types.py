"""Inquiro core data models 🎯.

Defines all Pydantic v2 data models for the Inquiro evidence research
and synthesis engine. Models are organized into:

1. Enums — Fixed-choice value types
2. Config Models — Agent, quality, cost, and tool configurations
3. Input Models — Task definitions (EvaluationTask, SynthesisTask)
4. Output Models — Results (EvaluationResult, SynthesisResult)
5. Infrastructure Models — Internal operational types

API request/response models live in ``inquiro.api.schemas``.

All models follow the Inquiro API contract defined in the PRD.
The engine is domain-agnostic: no pharma/dimension/sub-item concepts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from inquiro.core.trajectory.models import (
        DiscoveryRoundRecord,
        SynthesisRecord,
    )


# ============================================================================
# ✨ Public API — All exported model names
# ============================================================================

__all__ = [
    # 🏷️ Enums
    "TaskStatus",
    "TaskType",
    "Decision",
    "EvidenceStrength",
    "OverspendStrategy",
    "TaskPhase",
    "TruncationStrategy",
    "LogicSignal",
    "EvidenceTier",
    "CostStatus",
    "CircuitState",
    "EnsembleMode",
    "ExpPhase",
    "SubItemStatus",
    "_VALID_TRANSITIONS",
    # ⚙️ Config Models
    "ContextConfig",
    "AgentConfig",
    "ToolsConfig",
    "EnsembleModelConfig",
    "EnsembleConfig",
    "QualityChecks",
    "QualityGateConfig",
    "CostGuardConfig",
    "AdditionalResearchConfig",
    # 📥 Input Models
    "ChecklistItem",
    "Checklist",
    "DecisionGuidance",
    "InputReport",
    "EvaluationTask",
    "SynthesisTask",
    # 📤 Output Models
    "Evidence",
    "ReasoningClaim",
    "SearchExecution",
    "ChecklistProgress",
    "RoundLog",
    "ChecklistCoverage",
    "ConfidenceBreakdown",
    "ResearchContext",
    "EvaluationResult",
    "CrossReference",
    "ContradictionSide",
    "Contradiction",
    "DeepDiveRecord",
    "SynthesisResult",
    # 🔧 Infrastructure Models
    "QualityGateResult",
    "CostSummary",
    # 🔬 Discovery Pipeline Models
    "IntensityLevel",
    "DiscoveryConfig",
    "INTENSITY_PRESETS",
    "EvidenceTag",
    "RawEvidence",
    "CleanedEvidence",
    "GapReport",
    "FocusPrompt",
    "DiscoveryRoundSummary",
    "DiscoveryResult",
    # 🧬 Evolution Models
    "RoundMetrics",
    "EvolutionProvider",
    # 🎭 Multi-Perspective Models
    "PerspectiveConfig",
    "PerspectiveOutput",
    "ConsensusOutput",
]


# ============================================================================
# 🏷️ Section 1: Enums
# ============================================================================


class TaskStatus(str, Enum):
    """Task lifecycle status 📋.

    Represents the current state of a Inquiro task (research or synthesis).
    Follows a linear progression: pending → running → completed/failed/cancelled.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of Inquiro task 🔬.

    Inquiro supports two task types matching its dual-capability architecture.
    """

    RESEARCH = "research"
    SYNTHESIS = "synthesis"


class Decision(str, Enum):
    """Assessment decision from agent output 🎯.

    Domain-agnostic decision values. Upper-layer systems (e.g., TargetMaster)
    map these to domain-specific terms (e.g., go/conditional_go/no_go).
    """

    POSITIVE = "positive"
    CAUTIOUS = "cautious"
    NEGATIVE = "negative"


class EvidenceStrength(str, Enum):
    """Strength of a reasoning claim's evidence support 💪.

    Used in ReasoningClaim to indicate how well-supported a claim is.
    """

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class OverspendStrategy(str, Enum):
    """Strategy when cost budget is exceeded 💰.

    Controls agent behavior when the CostGuard threshold is reached.
    """

    SOFT_STOP = "SoftStop"  # ⚠️ Finish current round, then stop
    HARD_STOP = "HardStop"  # 🛑 Stop immediately


class TaskPhase(str, Enum):
    """Current execution phase of a running task 🔄.

    Used in TaskProgress to indicate what the agent is currently doing.
    """

    SEARCHING = "searching"
    REASONING = "reasoning"
    SYNTHESIZING = "synthesizing"
    QUALITY_CHECK = "quality_check"


class TruncationStrategy(str, Enum):
    """Context window truncation strategy 📐.

    Controls how the agent manages its context window when it grows too large.
    """

    LATEST_HALF = "latest_half"  # Keep the most recent half of context
    SLIDING_WINDOW = "sliding_window"  # 🪟 Keep the most recent N turns


class LogicSignal(str, Enum):
    """Logic signal for domain-specific decision display 🚦.

    Mapped from Decision + Confidence by upper-layer systems.
    Included here for type compatibility across the stack.
    """

    GO = "go"
    OK = "ok"
    CAUTION = "caution"
    NO_GO = "no_go"


class EvidenceTier(str, Enum):
    """Evidence quality tier classification 📊.

    Configurable evidence quality labels. The specific meaning of each tier
    is defined by the calling system (e.g., TargetMaster defines
    Tier1=clinical data, Tier2=peer-reviewed, etc.).
    """

    TIER_1 = "tier_1"  # 🥇 Highest quality
    TIER_2 = "tier_2"  # 🥈 High quality
    TIER_3 = "tier_3"  # 🥉 Moderate quality
    TIER_4 = "tier_4"  # ⚠️ Low quality / unverified


class CostStatus(str, Enum):
    """Cost tracking status from CostTracker 💸.

    Returned after each cost recording to indicate budget health.
    Ordered by severity: OK < WARNING < MODEL_DOWNGRADE <
    BUDGET_CRITICAL < TASK_EXCEEDED / TOTAL_EXCEEDED.
    """

    OK = "ok"
    WARNING = "warning"  # 🟡 Approaching budget limit (default >50%)
    MODEL_DOWNGRADE = "model_downgrade"  # 🟠 Suggest cheaper model (>80%)
    BUDGET_CRITICAL = "budget_critical"  # 🔴 Near exhaustion (>90%)
    TASK_EXCEEDED = "task_exceeded"  # 🔴 Per-task budget exceeded
    TOTAL_EXCEEDED = "total_exceeded"  # 🔴 Total budget exceeded


class EnsembleMode(str, Enum):
    """Ensemble execution mode 🎭.

    Controls how multi-model ensemble evaluation is performed.
    """

    PARALLEL_FULL = "parallel_full"  # 🔄 Each model runs full search+reason
    SEARCH_ONCE_REASON_MANY = (
        "search_once_reason_many"  # 🚀 Search once, fan out reasoning
    )


class CircuitState(str, Enum):
    """Circuit breaker state for MCP server fault isolation 🔌.

    Standard circuit breaker pattern: CLOSED → OPEN → HALF_OPEN → CLOSED.
    """

    CLOSED = "closed"  # ✅ Normal operation
    OPEN = "open"  # 🛑 Requests blocked, waiting for recovery
    HALF_OPEN = "half_open"  # 🟡 Testing if service recovered


class ExpPhase(str, Enum):
    """Experiment lifecycle phases 🔄.

    Represents the state machine governing the Exp layer's execution
    lifecycle. Each phase has a defined set of valid successor phases
    enforced by ``_VALID_TRANSITIONS``.

    Terminal states (COMPLETED, FAILED, CANCELLED) accept no transitions.
    """

    INIT = "init"
    PROMPT_BUILDING = "prompt_building"
    AGENT_RUNNING = "agent_running"
    QG_VALIDATING = "qg_validating"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubItemStatus(str, Enum):
    """Status of a research sub-item result 📊.

    Used by upper-layer systems to classify the outcome of an
    individual research sub-item after QualityGate validation.
    Enables detection of silent failures where zero evidence
    is collected without an explicit error.
    """

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    SKIPPED = "skipped"
    ERROR = "error"


# 🔄 Valid phase transitions for the Exp lifecycle state machine
_VALID_TRANSITIONS: dict[ExpPhase, set[ExpPhase]] = {
    ExpPhase.INIT: {
        ExpPhase.PROMPT_BUILDING,
        ExpPhase.FAILED,
        ExpPhase.CANCELLED,
    },
    ExpPhase.PROMPT_BUILDING: {
        ExpPhase.AGENT_RUNNING,
        ExpPhase.FAILED,
    },
    ExpPhase.AGENT_RUNNING: {
        ExpPhase.QG_VALIDATING,
        ExpPhase.FAILED,
        ExpPhase.CANCELLED,
    },
    ExpPhase.QG_VALIDATING: {
        ExpPhase.COMPLETED,
        ExpPhase.RETRYING,
        ExpPhase.FAILED,
    },
    ExpPhase.RETRYING: {
        ExpPhase.PROMPT_BUILDING,
        ExpPhase.FAILED,
        ExpPhase.CANCELLED,
    },
    ExpPhase.COMPLETED: set(),  # 🏁 Terminal
    ExpPhase.FAILED: set(),  # 🏁 Terminal
    ExpPhase.CANCELLED: set(),  # 🏁 Terminal
}


# ============================================================================
# ⚙️ Section 2: Config Models
# ============================================================================


class ContextConfig(BaseModel):
    """Agent context window configuration 📐.

    Controls how the agent manages its LLM context window, including
    maximum token limits and truncation behavior.
    """

    max_tokens: int = Field(
        default=128000,
        description="Maximum context window size in tokens",
    )
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.LATEST_HALF,
        description="Strategy for truncating context when window is full",
    )
    preserve_system_messages: bool = Field(
        default=True,
        description="Whether to always preserve system messages during truncation",
    )
    preserve_recent_turns: int = Field(
        default=8,
        description="Number of recent conversation turns to always preserve",
    )


class AgentConfig(BaseModel):
    """Agent configuration from API request 🤖.

    Configures the LLM model, behavior, and context management for
    a research or synthesis agent instance.
    """

    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="LLM model identifier",
    )
    max_turns: int = Field(
        default=30,
        description="Maximum number of agent turns (LLM round-trips)",
        gt=0,
    )
    temperature: float = Field(
        default=0.3,
        description="LLM sampling temperature",
        ge=0.0,
        le=2.0,
    )
    system_prompt_template: str | None = Field(
        default=None,
        description="Custom system prompt template; uses default if None",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context window management configuration",
    )


class ToolsConfig(BaseModel):
    """MCP tool server selection and overrides 🔌.

    Specifies which MCP servers to use for a task and any
    per-request configuration overrides.
    """

    mcp_servers: list[str] = Field(
        default_factory=list,
        description="List of MCP server names to enable for this task",
    )
    mcp_config_override: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-request MCP configuration overrides",
    )


class EnsembleModelConfig(BaseModel):
    """Single model in a multi-LLM ensemble ⚖️.

    Defines one model participant, its voting weight, and runtime
    constraints for the ensemble evaluation.
    """

    name: str = Field(description="Logical model name for identification")
    provider_key: str = Field(
        default="",
        description="Key in LLMProviderPool (defaults to name if empty)",
    )
    model: str = Field(
        default="",
        description="LLM model identifier (legacy; prefer provider_key)",
    )
    weight: float = Field(
        default=1.0,
        description="Voting weight for this model (0-1)",
        ge=0.0,
        le=1.0,
    )
    max_turns: int = Field(
        default=30,
        description="Maximum agent turns for this model",
        gt=0,
    )
    enabled: bool = Field(
        default=True,
        description="Whether this model participates in the ensemble",
    )

    @property
    def effective_provider_key(self) -> str:
        """Resolve the provider key to use for LLM pool lookup 🔑.

        Returns:
            provider_key if set, otherwise falls back to name.
        """
        return self.provider_key or self.name


class EnsembleConfig(BaseModel):
    """Multi-LLM ensemble configuration 🎭.

    Enables running multiple models in parallel and combining their
    results via weighted voting for higher reliability.
    """

    enabled: bool = Field(
        default=False,
        description="Whether to enable multi-model ensemble",
    )
    mode: EnsembleMode = Field(
        default=EnsembleMode.SEARCH_ONCE_REASON_MANY,
        description=(
            "Ensemble execution mode: 'parallel_full' runs each model "
            "with full search+reason; 'search_once_reason_many' searches "
            "once with the primary model and fans out reasoning to all models"
        ),
    )
    strategy: str = Field(
        default="weighted_voting",
        description="Aggregation strategy: weighted_voting or majority_voting",
    )
    models: list[EnsembleModelConfig] = Field(
        default_factory=list,
        description="List of models with voting weights",
    )
    consensus_threshold: float = Field(
        default=0.7,
        description="Minimum agreement ratio required for consensus",
        ge=0.0,
        le=1.0,
    )
    min_successful_models: int = Field(
        default=2,
        description="Minimum models that must succeed for a valid result",
        ge=1,
    )

    def get_enabled_models(self) -> list[EnsembleModelConfig]:
        """Return only enabled model configurations 🎯.

        Returns:
            List of EnsembleModelConfig where enabled is True.
        """
        return [m for m in self.models if m.enabled]


class QualityChecks(BaseModel):
    """Individual quality check toggles ✅.

    Controls which quality gate checks are enabled for a task.
    Each check can be independently toggled on/off.
    """

    schema_validation: bool = Field(
        default=True,
        description="Validate output against JSON Schema (hard fail)",
    )
    coverage_check: bool = Field(
        default=True,
        description="Check search checklist coverage ratio (soft fail)",
    )
    evidence_reference_check: bool = Field(
        default=True,
        description="Verify evidence reference integrity (soft fail)",
    )
    cross_reference_check: bool = Field(
        default=False,
        description="Check cross-report reference consistency (synthesis only)",
    )
    source_diversity_check: bool = Field(
        default=False,
        description="Check evidence source diversity (soft fail, opt-in)",
    )
    evidence_url_check: bool = Field(
        default=True,
        description="Check evidence URL presence (soft fail)",
    )


class QualityGateConfig(BaseModel):
    """Quality gate configuration 🔍.

    Configures the deterministic output quality validation that runs
    after each agent execution attempt.
    """

    enabled: bool = Field(
        default=True,
        description="Whether quality gate validation is active",
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts on hard quality failures",
        ge=0,
    )
    checks: QualityChecks = Field(
        default_factory=QualityChecks,
        description="Individual check toggles",
    )


class CostGuardConfig(BaseModel):
    """Cost guard configuration 💰.

    Controls budget limits, overspend behavior, and wall-clock
    timeout for a task. Costs are tracked in USD.
    """

    max_cost_per_task: float = Field(
        default=1.5,
        description="Maximum cost in USD for a single task",
        gt=0.0,
    )
    overspend_strategy: OverspendStrategy = Field(
        default=OverspendStrategy.SOFT_STOP,
        description="Behavior when budget is exceeded",
    )
    timeout_seconds: float = Field(
        default=1200.0,
        description=(
            "Wall-clock timeout in seconds for task execution. "
            "When exceeded, the task is cancelled and a partial "
            "result is returned. Default is 1200s (20 minutes)."
        ),
        gt=0.0,
    )


class AdditionalResearchConfig(BaseModel):
    """Configuration for SynthesisAgent's supplementary research 🔬.

    Controls scope and budget when SynthesisAgent triggers
    additional SearchAgent instances to fill evidence gaps.
    """

    max_tasks: int = Field(
        default=3,
        description="Maximum number of additional research tasks",
        gt=0,
    )
    cost_budget: float = Field(
        default=2.0,
        description="Total cost budget in USD for additional research",
        gt=0.0,
    )
    tools_config: ToolsConfig = Field(
        default_factory=ToolsConfig,
        description="MCP server configuration for supplementary research",
    )


# ============================================================================
# 📥 Section 3: Input Models
# ============================================================================


class ChecklistItem(BaseModel):
    """Single item in a search checklist 📌.

    Represents a required or optional research topic that the
    SearchAgent should investigate.
    """

    id: str = Field(description="Unique identifier for this checklist item")
    description: str = Field(
        description="Human-readable description of what to search for"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Suggested search keywords",
    )
    suggested_sources: list[str] = Field(
        default_factory=list,
        description="Suggested MCP server names for this item",
    )


class Checklist(BaseModel):
    """Search checklist with required and optional items 📋.

    Defines the structured research agenda for a SearchAgent.
    Coverage is measured against required items only.
    """

    required: list[ChecklistItem] = Field(
        default_factory=list,
        description="Required search items (coverage measured against these)",
    )
    optional: list[ChecklistItem] = Field(
        default_factory=list,
        description="Optional search items (not counted for coverage)",
    )
    coverage_threshold: float = Field(
        default=0.8,
        description="Minimum fraction of required items that must be covered",
        ge=0.0,
        le=1.0,
    )


class DecisionGuidance(BaseModel):
    """Decision guidance for agent reasoning 🧭.

    Provides examples of what constitutes positive, cautious, or negative
    assessments. Injected into the agent prompt to guide decision-making.
    """

    positive: list[str] = Field(
        default_factory=list,
        description="Examples of conditions supporting a positive decision",
    )
    cautious: list[str] = Field(
        default_factory=list,
        description="Examples of conditions warranting a cautious decision",
    )
    negative: list[str] = Field(
        default_factory=list,
        description="Examples of conditions supporting a negative decision",
    )


class InputReport(BaseModel):
    """Input report for synthesis 📖.

    Carries the full content of a previously completed research report,
    to be consumed by the SynthesisAgent.
    """

    report_id: str = Field(description="Unique identifier for this report")
    label: str = Field(description="Human-readable label for this report")
    content: dict[str, Any] = Field(
        description="Full structured report data (e.g., EvaluationResult dict)",
    )


class EvaluationTask(BaseModel):
    """Atomic research task definition 🔬.

    Complete input for a SearchAgent. Contains all domain-specific
    knowledge injected by the calling system (topic, rules, checklist,
    guidance, output schema) plus engine configuration.

    NOTE: This is the *internal* domain model, NOT the API request body.
    The API uses ``ResearchRequest`` (in schemas.py) which has a different
    shape (nested ``task`` field). See ``router.py`` for the conversion.
    """

    task_id: str = Field(description="Unique task identifier (UUID)")
    topic: str = Field(description="Research topic / question to investigate")
    rules: str = Field(
        default="",
        description="Evaluation rules in Markdown (injected into agent prompt)",
    )
    checklist: Checklist = Field(
        default_factory=Checklist,
        description="Structured search checklist with required/optional items",
    )
    decision_guidance: DecisionGuidance | None = Field(
        default=None,
        description="Decision guidance examples for the agent",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema defining the required output structure",
    )
    prior_context: str | None = Field(
        default=None,
        description="Previous research context for incremental evaluation",
    )
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent behavior configuration",
    )
    tools_config: ToolsConfig = Field(
        default_factory=ToolsConfig,
        description="MCP tool server selection",
    )
    ensemble_config: EnsembleConfig = Field(
        default_factory=EnsembleConfig,
        description="Multi-LLM ensemble configuration",
    )
    quality_gate: QualityGateConfig = Field(
        default_factory=QualityGateConfig,
        description="Output quality validation configuration",
    )
    cost_guard: CostGuardConfig = Field(
        default_factory=CostGuardConfig,
        description="Cost budget and overspend configuration",
    )
    callback_url: str | None = Field(
        default=None,
        description="Webhook URL for task completion notification",
    )

    # 💾 Trajectory persistence — optional, for post-hoc analysis
    trajectory_dir: str | None = Field(
        default=None,
        description=(
            "Directory for trajectory persistence. When set, the Exp "
            "layer writes a JSONL file per task after execution completes. "
            "None disables persistence. Concurrent-safe (per-task file)."
        ),
    )
    trajectory_streaming: bool = Field(
        default=False,
        description=(
            "When True and trajectory_dir is set, append trajectory steps "
            "after each step (Synthesis only for now). Research still "
            "writes once at end. Only effective when trajectory_dir is set."
        ),
    )

    # 🧬 Evolution — injected by upper layer, opaque to Inquiro
    evolution_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Evolution profile config from upper layer. "
            "Contains extraction/enrichment/fitness configuration. "
            "Opaque to Inquiro — passed through to evolution components."
        ),
    )
    query_strategy: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parsed query strategy for guided search. "
            "Opaque to Inquiro — injected by the upstream orchestration layer. "
            "When present, SearchExp uses it for alias expansion and query routing."
        ),
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Context tags for experience matching (opaque to Inquiro). "
            "Injected by upper layer, e.g., ['category:TypeA']."
        ),
    )
    # 🔬 Discovery Pipeline — optional config
    discovery_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "DiscoveryConfig parameters for the discovery pipeline. "
            "Opaque to upper layers — parsed internally by DiscoveryLoop."
        ),
    )

    sub_item_id: str | None = Field(
        default=None,
        description=(
            "Sub-item identifier for Progressive Disclosure and experience "
            "matching. When set, rules/checklist are rendered as summary + "
            "get_reference; when None or empty, full rules/checklist are used."
        ),
    )

    # 🔄 Evidence Sharing — optional pool identifier for cross-task reuse
    evidence_pool_id: str | None = Field(
        default=None,
        description=(
            "Shared evidence pool identifier for cross-task evidence reuse. "
            "Tasks with the same pool_id share evidence via a "
            "SharedEvidencePool instance managed by the Runner. "
            "When None, no evidence sharing occurs (default behavior)."
        ),
    )

    # 📦 Knowledge Base injection — set by upper layer (TargetMaster)
    seeded_evidence: list[Any] | None = Field(
        default=None,
        description=(
            "Pre-filled evidence from knowledge base. "
            "Injected into DiscoveryLoop as starting evidence set. "
            "When None, no KB pre-fill occurs."
        ),
    )
    seeded_gap_hints: list[str] | None = Field(
        default=None,
        description=(
            "Known gap descriptions from prior KB evaluations. "
            "Injected into round-1 focus prompt to guide search."
        ),
    )

    # 💡 Evolution hints — injected by TargetMaster, opaque to Inquiro
    evolution_hints_preamble: str | None = Field(
        default=None,
        description=(
            "Structured search hints derived from cross-evaluation learning. "
            "Injected into Round 1 SearchExp prompt after query_global_preamble. "
            "Generated by EvolutionHintLoader from evolution_hints/{sub_item}.yaml."
        ),
    )


class SynthesisTask(BaseModel):
    """Multi-report synthesis task definition 📊.

    Complete input for a SynthesisAgent. Contains input reports to
    synthesize, rules for synthesis, and optional deep-dive configuration.

    NOTE: This is the *internal* domain model, NOT the API request body.
    The API uses ``SynthesizeRequest`` (in schemas.py) which has a different
    shape (nested ``task`` field). See ``router.py`` for the conversion.
    """

    task_id: str = Field(description="Unique task identifier (UUID)")
    topic: str = Field(description="Synthesis objective / topic")
    input_reports: list[InputReport] = Field(
        description="Research reports to synthesize",
        min_length=1,
    )
    synthesis_rules: str = Field(
        default="",
        description="Synthesis rules in Markdown (injected into agent prompt)",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema defining the required output structure",
    )
    allow_additional_research: bool = Field(
        default=True,
        description="Whether SynthesisAgent can trigger supplementary research",
    )
    additional_research_config: AdditionalResearchConfig | None = Field(
        default=None,
        description="Configuration for supplementary research (if allowed)",
    )
    agent_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Agent behavior configuration",
    )
    quality_gate: QualityGateConfig = Field(
        default_factory=QualityGateConfig,
        description="Output quality validation configuration",
    )
    cost_guard: CostGuardConfig = Field(
        default_factory=CostGuardConfig,
        description="Cost budget and overspend configuration",
    )
    callback_url: str | None = Field(
        default=None,
        description="Webhook URL for task completion notification",
    )

    # 💾 Trajectory persistence — optional, for post-hoc analysis
    trajectory_dir: str | None = Field(
        default=None,
        description=(
            "Directory for trajectory persistence. When set, the Exp "
            "layer writes a JSONL file per task after execution completes. "
            "None disables persistence. Concurrent-safe (per-task file)."
        ),
    )
    trajectory_streaming: bool = Field(
        default=False,
        description=(
            "When True and trajectory_dir is set, Synthesis appends each "
            "step to the JSONL file after agent._step(). Only effective "
            "when trajectory_dir is set."
        ),
    )

    # 🧬 Evolution — injected by upper layer, opaque to Inquiro
    evolution_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Evolution profile config from upper layer. "
            "Opaque to Inquiro — passed through to evolution components."
        ),
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Context tags for experience matching (opaque to Inquiro)",
    )


# ============================================================================
# 📤 Section 4: Output Models
# ============================================================================


class EvidenceMetadata(BaseModel):
    """Rich structured metadata extracted by LLM from evidence text 📋.

    Flat model with optional fields — only fields extractable from
    the evidence text will be populated.  All fields default to None
    so partial extraction is safe.
    """

    # Common fields (all evidence types)
    title: str | None = Field(default=None, description="Title of the source material")
    authors: list[str] = Field(default_factory=list, description="Author names")
    publication_year: int | None = Field(default=None, description="Publication year")
    publication_date: str | None = Field(
        default=None,
        description="Publication date (ISO 8601 or partial, e.g. '2024-03')",
    )

    # Academic literature
    journal: str | None = Field(default=None, description="Journal or conference name")
    pmid: str | None = Field(default=None, description="PubMed ID")

    # Clinical trials
    trial_phase: str | None = Field(
        default=None,
        description="Clinical trial phase (e.g. 'Phase 1', 'Phase 2/3')",
    )
    trial_status: str | None = Field(
        default=None,
        description="Trial status (e.g. 'Recruiting', 'Completed')",
    )
    sponsor: str | None = Field(default=None, description="Trial sponsor organization")
    enrollment: int | None = Field(default=None, description="Enrollment count")

    # Patents
    patent_number: str | None = Field(
        default=None,
        description="Patent number (e.g. 'US11234567B2')",
    )
    patent_assignee: str | None = Field(default=None, description="Patent assignee")

    # News
    news_agency: str | None = Field(
        default=None,
        description="News agency or publisher name",
    )


class Evidence(BaseModel):
    """Single piece of evidence collected during research 🔗.

    Each evidence item is auto-tagged with a unique ID when collected
    via MCP tool calls. Evidence provenance enables full audit trails.
    """

    id: str = Field(description="Unique evidence identifier (e.g., 'E1', 'E2')")
    source: str = Field(description="Source MCP server or data origin")
    url: str | None = Field(
        default=None,
        description="URL of the source material (if available)",
    )
    query: str = Field(description="Search query that produced this evidence")
    summary: str = Field(description="Brief summary of the evidence content")
    quality_label: str | None = Field(
        default=None,
        description="Evidence quality tier label (configurable by caller)",
    )
    round_number: int | None = Field(
        default=None,
        description="Search round in which this evidence was collected",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="Timestamp when this evidence was collected",
    )
    source_report_id: str | None = Field(
        default=None,
        description="Report ID if evidence originates from a synthesis input report",
    )

    # 🏷️ KB extension fields — populated by EvidencePipeline
    evidence_tag: str | None = Field(
        default=None,
        description=(
            "Source type tag populated by EvidencePipeline "
            "(academic, patent, clinical_trial, regulatory, other)"
        ),
    )
    clinical_trial_id: str | None = Field(
        default=None,
        description="NCT ID extracted from URL or summary (e.g. NCT12345678)",
    )
    doi: str | None = Field(
        default=None,
        description="DOI extracted from URL or summary content",
    )
    metadata: EvidenceMetadata | None = Field(
        default=None,
        description="Rich structured metadata extracted by MetadataEnricher",
    )


class ReasoningClaim(BaseModel):
    """Single reasoning entry with evidence support 🧠.

    Each claim in the agent's reasoning must reference at least one
    evidence item. Claims without evidence are flagged as gaps.
    """

    claim: str = Field(description="The reasoning claim / assertion")
    evidence_ids: list[str] = Field(
        description="Evidence IDs supporting this claim",
    )
    strength: EvidenceStrength = Field(
        description="Strength of evidence support for this claim",
    )
    direction: str | None = Field(
        default=None,
        description=(
            "Claim direction: positive (supports hypothesis), "
            "negative (opposes), or neutral"
        ),
    )


class SearchExecution(BaseModel):
    """Record of a single search execution within a round 🔍.

    Captures what was searched and from which source during a
    research round.
    """

    tool: str | None = Field(
        default=None,
        description="MCP tool / server used for this search",
    )
    query: str | None = Field(
        default=None,
        description="Search query executed",
    )
    results_count: int | None = Field(
        default=None,
        description="Number of results returned",
    )


class ChecklistProgress(BaseModel):
    """Checklist coverage progress within a round ✅.

    Tracks which checklist items have been covered and which remain.
    """

    covered: list[str] = Field(
        default_factory=list,
        description="Checklist item IDs that have been covered",
    )
    remaining: list[str] = Field(
        default_factory=list,
        description="Checklist item IDs still remaining",
    )


class RoundLog(BaseModel):
    """Per-round structured log from research execution 📝.

    Each search-reason-doubt-reframe cycle produces one round log,
    enabling detailed observability of the research process.
    """

    round_number: int = Field(description="1-based round number")
    searches_executed: list[SearchExecution] = Field(
        default_factory=list,
        description="Searches performed in this round",
    )
    findings_summary: str = Field(
        default="",
        description="Summary of findings from this round",
    )
    checklist_progress: ChecklistProgress | None = Field(
        default=None,
        description="Checklist coverage status after this round",
    )
    gaps_identified: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps identified in this round",
    )
    doubts_identified: list[str] = Field(
        default_factory=list,
        description="Doubts / weaknesses identified in this round",
    )
    reframed_prompt_summary: str | None = Field(
        default=None,
        description="Summary of the reframed search strategy for next round",
    )


class ChecklistCoverage(BaseModel):
    """Final checklist coverage report 📊.

    Summarizes which required checklist items were covered vs. missed
    at the end of the research task.
    """

    required_covered: list[str] = Field(
        default_factory=list,
        description="IDs of required checklist items that were covered",
    )
    required_missing: list[str] = Field(
        default_factory=list,
        description="IDs of required checklist items that were NOT covered",
    )


class ConfidenceBreakdown(BaseModel):
    """Multi-dimensional confidence breakdown 📊.

    Decomposes a single confidence score into quality dimensions
    for transparent evaluation of result reliability.

    Attributes:
        evidence_strength: Score based on source quality tier.
        evidence_coverage: Score based on checklist item coverage.
        evidence_consistency: Score based on evidence agreement
            (1.0 = no conflicts).
        information_recency: Score based on evidence freshness.
        overall: Weighted composite of all dimensions.
    """

    evidence_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Score based on source quality tier "
            "(primary=1.0, secondary=0.7, tertiary=0.4)"
        ),
    )
    evidence_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score based on checklist item coverage ratio",
    )
    evidence_consistency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=("1.0 if no conflicts, reduced per conflict found"),
    )
    information_recency: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Score based on evidence freshness",
    )
    overall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted composite of all dimensions",
    )


class ResearchContext(BaseModel):
    """Structured metadata from research phase for synthesis consumption 🔬.

    Captures the research process metadata that helps SynthesisAgent
    make informed decisions about additional research requests.

    Attributes:
        coverage_map: Mapping of checklist items to their coverage status.
        information_gaps: List of identified information gaps.
        conflicting_evidence: Pairs of conflicting evidence items.
        search_strategies_used: List of search strategies that were employed.
        tool_effectiveness: Mapping of tool names to their success rates.
    """

    coverage_map: dict[str, bool] = Field(
        default_factory=dict,
        description="Checklist item ID -> whether it was covered",
    )
    information_gaps: list[str] = Field(
        default_factory=list,
        description="Information that could not be found",
    )
    conflicting_evidence: list[list[str]] = Field(
        default_factory=list,
        description=("Pairs of conflicting evidence IDs [[E1, E3], [E2, E5]]"),
    )
    search_strategies_used: list[str] = Field(
        default_factory=list,
        description="Search strategies employed during research",
    )
    tool_effectiveness: dict[str, float] = Field(
        default_factory=dict,
        description="Tool name -> success rate (0.0 to 1.0)",
    )


class EvaluationResult(BaseModel):
    """Structured output from SearchAgent 📋.

    The complete result of an atomic research task, conforming to
    the default EvaluationResult schema. Contains the agent's decision,
    evidence trail, reasoning chain, and research process logs.
    """

    task_id: str = Field(description="ID of the research task")
    decision: Decision = Field(description="Overall assessment decision")
    confidence: float = Field(
        description="Confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: list[ReasoningClaim] = Field(
        default_factory=list,
        description="Evidence-backed reasoning claims",
    )
    evidence_index: list[Evidence] = Field(
        default_factory=list,
        description="Complete index of all evidence collected",
    )
    search_rounds: int = Field(
        default=0,
        description="Total number of search rounds executed",
        ge=0,
    )
    round_logs: list[RoundLog] = Field(
        default_factory=list,
        description="Structured logs for each search-reason round",
    )
    checklist_coverage: ChecklistCoverage = Field(
        default_factory=ChecklistCoverage,
        description="Final checklist coverage report",
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps remaining at task completion",
    )
    doubts_remaining: list[str] = Field(
        default_factory=list,
        description="Unresolved doubts at task completion",
    )
    cost: float = Field(
        default=0.0,
        description="Total cost in USD for this research task",
        ge=0.0,
    )
    confidence_breakdown: ConfidenceBreakdown | None = Field(
        default=None,
        description="Multi-dimensional confidence breakdown (optional)",
    )
    research_context: ResearchContext | None = Field(
        default=None,
        description=(
            "Structured research metadata for synthesis consumption. "
            "Built from QG results and evidence after research completes."
        ),
    )

    # 🔬 Discovery Pipeline fields — populated for DISCOVERY mode
    pipeline_mode: str = Field(
        default="discovery",
        description="Pipeline execution mode (always 'discovery')",
    )
    confidence_source: str = Field(
        default="agent_self_assessment",
        description=(
            "Source of the confidence value. "
            "'agent_self_assessment': Agent's own confidence. "
            "'coverage_ratio': Checklist coverage proportion. "
            "'consensus_confidence': Multi-model synthesis consensus."
        ),
    )
    discovery_rounds: int = Field(
        default=0,
        description="Number of DiscoveryLoop rounds (0 if no loop ran)",
        ge=0,
    )
    discovery_coverage: float = Field(
        default=0.0,
        description="Final coverage ratio from DiscoveryLoop (0.0 if not applicable)",
        ge=0.0,
        le=1.0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible metadata bag. Used for ensemble info, "
            "discovery details, and custom upper-layer data."
        ),
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range 🔒.

        Args:
            v: The confidence value to validate.

        Returns:
            The validated confidence value.

        Raises:
            ValueError: If confidence is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    def get_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        """Look up a single evidence item by its ID 🔍.

        Args:
            evidence_id: The unique evidence identifier (e.g., "E1").

        Returns:
            The matching Evidence object, or None if not found.
        """
        for ev in self.evidence_index:
            if ev.id == evidence_id:
                return ev
        return None

    def get_covered_ratio(self) -> float:
        """Calculate the ratio of covered required checklist items 📊.

        Returns:
            Float between 0.0 and 1.0. Returns 1.0 if there are no
            required items at all.
        """
        covered = len(self.checklist_coverage.required_covered)
        missing = len(self.checklist_coverage.required_missing)
        total = covered + missing
        if total == 0:
            return 1.0
        return covered / total


class CrossReference(BaseModel):
    """Cross-report reference in synthesis 🔗.

    Records a claim that appears across multiple input reports,
    noting which reports support or contradict it.
    """

    claim: str = Field(description="The cross-referenced claim")
    supporting_reports: list[str] = Field(
        default_factory=list,
        description="Report IDs that support this claim",
    )
    contradicting_reports: list[str] = Field(
        default_factory=list,
        description="Report IDs that contradict this claim",
    )
    resolution: str | None = Field(
        default=None,
        description="How the cross-reference was resolved",
    )


class ContradictionSide(BaseModel):
    """One side of a contradiction between reports ⚔️.

    Captures a specific report's claim and evidence in a contradiction.
    """

    report_id: str = Field(description="ID of the report making this claim")
    claim: str = Field(description="The specific claim from this report")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting this side",
    )


class Contradiction(BaseModel):
    """Explicit contradiction found between reports ⚠️.

    Documents a disagreement between two input reports on a
    specific topic, including how it was resolved (if at all).
    """

    topic: str = Field(description="Topic of the contradiction")
    report_a: ContradictionSide = Field(description="First report's position")
    report_b: ContradictionSide = Field(description="Second report's position")
    resolution: str = Field(
        default="",
        description="How the contradiction was resolved or noted",
    )


class DeepDiveRecord(BaseModel):
    """Record of a deep-dive research task triggered during synthesis 🔬.

    When the SynthesisAgent identifies a critical gap, it can trigger
    additional research via RequestResearchTool. Each trigger is recorded.
    """

    topic: str = Field(description="Deep-dive research topic")
    task_id: str = Field(description="Task ID of the spawned research task")
    result_summary: str = Field(
        default="",
        description="Summary of the deep-dive research result",
    )
    evidence_count: int = Field(
        default=0,
        description="Number of evidence items collected by this deep-dive",
        ge=0,
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps still unresolved after this deep-dive",
    )


class SynthesisResult(BaseModel):
    """Structured output from SynthesisAgent 📊.

    The complete result of a synthesis task, conforming to the default
    SynthesisResult schema. Contains the synthesized decision,
    cross-references, contradictions, and any deep-dive results.
    """

    task_id: str = Field(description="ID of the synthesis task")
    decision: Decision = Field(description="Overall synthesized assessment decision")
    confidence: float = Field(
        description="Confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: list[ReasoningClaim] = Field(
        default_factory=list,
        description="Evidence-backed reasoning claims",
    )
    evidence_index: list[Evidence] = Field(
        default_factory=list,
        description="Complete index of all evidence (from input reports + deep dives)",
    )
    source_reports: list[str] = Field(
        default_factory=list,
        description="IDs of all input reports consumed",
    )
    cross_references: list[CrossReference] = Field(
        default_factory=list,
        description="Claims referenced across multiple reports",
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Explicit contradictions found between reports",
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps remaining after synthesis",
    )
    deep_dives_triggered: list[DeepDiveRecord] = Field(
        default_factory=list,
        description="Additional research tasks triggered during synthesis",
    )
    cost: float = Field(
        default=0.0,
        description="Total cost in USD for this synthesis task",
        ge=0.0,
    )
    raw_output: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Complete raw output from the synthesis agent "
            "(preserves schema-specific fields)"
        ),
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is within valid range 🔒.

        Args:
            v: The confidence value to validate.

        Returns:
            The validated confidence value.

        Raises:
            ValueError: If confidence is not between 0.0 and 1.0.
        """
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v

    def get_evidence_by_id(self, evidence_id: str) -> Evidence | None:
        """Look up a single evidence item by its ID 🔍.

        Args:
            evidence_id: The unique evidence identifier (e.g., "E1").

        Returns:
            The matching Evidence object, or None if not found.
        """
        for ev in self.evidence_index:
            if ev.id == evidence_id:
                return ev
        return None


# ============================================================================
# 🔧 Section 5: Infrastructure Models
# ============================================================================


class QualityGateResult(BaseModel):
    """Result from quality gate validation ✅.

    Contains the pass/fail verdict and details of any failures.
    Hard failures trigger retries; soft failures are warnings.
    """

    passed: bool = Field(description="Whether all hard checks passed")
    hard_failures: list[str] = Field(
        default_factory=list,
        description="Hard failures that must be fixed (e.g., schema violations)",
    )
    soft_failures: list[str] = Field(
        default_factory=list,
        description="Soft failures (warnings, e.g., low coverage)",
    )
    confidence_cap: float | None = Field(
        default=None,
        description="Confidence ceiling imposed by soft failures",
    )


class CostSummary(BaseModel):
    """Aggregate cost summary from CostTracker 💰.

    Provides a snapshot of costs across all tracked tasks.
    """

    task_costs: dict[str, float] = Field(
        default_factory=dict,
        description="Per-task cost breakdown {task_id: cost_usd}",
    )
    total_cost: float = Field(
        default=0.0,
        description="Total cost across all tasks in USD",
    )
    budget_remaining: float = Field(
        default=0.0,
        description="Remaining budget in USD",
    )


# ============================================================================
# 🔬 Section 7: Discovery Pipeline Models
# ============================================================================


class IntensityLevel(str, Enum):
    """Research intensity level ⚡.

    Controls the depth, cost, and thoroughness of the research
    pipeline.  Each level maps to a DiscoveryConfig preset —
    no separate code paths.

    Two levels:
    - STANDARD: 2 rounds, 3 analysis models, $20 budget
    - DISCOVERY: 5 rounds, 3 analysis models, $200 budget
    """

    STANDARD = "standard"
    DISCOVERY = "discovery"


class DiscoveryConfig(BaseModel):
    """Configuration for DiscoveryLoop behavior 🔧.

    Controls all configurable parameters for the multi-round
    search-analyze-gap discovery pipeline.  Supports three-level
    override: global defaults < mode-level < sub_item-level.
    """

    max_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum search-analysis iterations",
    )
    max_cost_per_subitem: float = Field(
        default=8.0,
        gt=0,
        description="Budget cap per sub-item in USD",
    )
    coverage_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Checklist coverage target to stop iteration",
    )
    convergence_delta: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Minimum coverage improvement to continue",
    )
    convergence_patience: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Number of consecutive rounds below convergence_delta "
            "before declaring diminishing returns"
        ),
    )
    min_evidence_per_round: int = Field(
        default=3,
        ge=0,
        description="Stop if fewer new evidence items found",
    )
    timeout_per_round: int = Field(
        default=300,
        gt=0,
        description="Per-round timeout in seconds",
    )
    timeout_total: int = Field(
        default=1200,
        gt=0,
        description="Total DiscoveryLoop timeout in seconds",
    )
    analysis_models: list[str] = Field(
        default_factory=list,
        description=(
            "LLM models for parallel analysis. "
            "When empty, the Runner injects models from its LLM pool configuration."
        ),
    )
    gap_focus_max_items: int = Field(
        default=3,
        ge=1,
        description="Max uncovered items to focus on per round",
    )
    input_cost_per_token: float = Field(
        default=3.0 / 1_000_000,
        description=(
            "Cost per input token in USD (default: $3/M — Claude Sonnet 3.5 pricing)."
        ),
    )
    output_cost_per_token: float = Field(
        default=15.0 / 1_000_000,
        description=(
            "Cost per output token in USD (default: $15/M — Claude Sonnet 3.5 pricing)."
        ),
    )
    enable_parallel_search: bool = Field(
        default=True,
        description="Enable parallel multi-agent search with query-section splitting.",
    )
    max_parallel_agents: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Maximum number of parallel search agents when enabled.",
    )
    shared_pool_prefill_limit: int = Field(
        default=50,
        ge=1,
        description="Max items to pre-fill from shared evidence pool.",
    )
    condenser_tier1_threshold: int = Field(
        default=150,
        ge=1,
        description=(
            "Evidence count at or below which Tier 0 passthrough is used. "
            "Above this EvidenceCondenser activates Tier 1 (keyword scoring)."
        ),
    )
    condenser_tier2_threshold: int = Field(
        default=400,
        ge=1,
        description=(
            "Evidence count at or below which Tier 1 filter is used. "
            "Above this Tier 2 activates (Tier 1 + tag-group summaries)."
        ),
    )
    condenser_doi_prefix_map: dict[str, float] = Field(
        default_factory=dict,
        description="DOI prefix quality map injected by caller for journal scoring.",
    )
    condenser_doi_prefix_default: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Default score for DOIs not in prefix map.",
    )
    condenser_tag_quality_map: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Custom evidence_tag quality score overrides injected by caller. "
            "Merged on top of CondenserConfig.tag_quality_map defaults. "
            "Empty dict means use CondenserConfig defaults unchanged."
        ),
    )
    condenser_summarizer_model: str = Field(
        default="",
        description=(
            "LLM model name for Tier 2 group summarisation. "
            "When non-empty, excluded evidence groups are summarised by LLM "
            "instead of template text. Recommend a fast/cheap model (e.g. haiku). "
            "Empty string disables LLM summarisation (template-only fallback)."
        ),
    )

    analysis_model_count: int = Field(
        default=3,
        ge=0,
        le=5,
        description=(
            "Number of models for parallel analysis. "
            "1 = single model (no voting), 3 = triple voting."
        ),
    )
    enable_synthesis: bool = Field(
        default=True,
        description="Whether to run multi-model synthesis after analysis rounds.",
    )
    coverage_judge_model: str = Field(
        default="claude-bedrock",
        description=(
            "LLM model name for coverage judge. Must match a key in "
            "llm_providers.yaml. Empty string disables LLM judge (mock fallback)."
        ),
    )
    coverage_judge_mode: str = Field(
        default="always",
        description=(
            "Coverage judge mode: 'always' runs LLM judge every round (ignoring "
            "pre_computed_coverage), 'fallback' runs LLM judge only when no "
            "pre_computed_coverage is available."
        ),
    )
    intensity: str = Field(
        default="standard",
        description="Intensity level preset name. Overridden by explicit field values.",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> DiscoveryConfig:
        """Validate that config fields are internally consistent 🔍."""
        if self.timeout_total < self.timeout_per_round:
            raise ValueError(
                f"timeout_total ({self.timeout_total}) must be >= "
                f"timeout_per_round ({self.timeout_per_round})"
            )
        return self


INTENSITY_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {
        "max_rounds": 2,
        "coverage_threshold": 0.75,
        "convergence_delta": 0.08,
        "convergence_patience": 1,
        "min_evidence_per_round": 3,
        "timeout_per_round": 600,
        "timeout_total": 1800,
        "max_cost_per_subitem": 20.0,
        "enable_parallel_search": True,
        "max_parallel_agents": 3,
        "gap_focus_max_items": 3,
        "enable_synthesis": True,
        "analysis_model_count": 3,
        "coverage_judge_model": "haiku",
        "coverage_judge_mode": "always",
        "intensity": "standard",
    },
    "discovery": {
        "max_rounds": 5,
        "coverage_threshold": 0.85,
        "convergence_delta": 0.05,
        "convergence_patience": 2,
        "min_evidence_per_round": 5,
        "timeout_per_round": 300,
        "timeout_total": 1500,
        "max_cost_per_subitem": 200.0,
        "enable_parallel_search": True,
        "max_parallel_agents": 3,
        "gap_focus_max_items": 3,
        "enable_synthesis": True,
        "analysis_model_count": 3,
        "condenser_summarizer_model": "gemini-3-flash",
        "coverage_judge_model": "haiku",
        "coverage_judge_mode": "always",
        "intensity": "discovery",
    },
}


class EvidenceTag(str, Enum):
    """Evidence source type classification 🏷️.

    Assigned by EvidencePipeline during deterministic cleaning.
    Based on URL pattern matching — domain-agnostic.
    """

    ACADEMIC = "academic"
    PATENT = "patent"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY = "regulatory"
    OTHER = "other"


class RawEvidence(BaseModel):
    """Unprocessed evidence from MCP search 📄.

    Captures the raw observation from a single MCP tool call
    before any cleaning or classification.
    """

    id: str = Field(description="Evidence identifier (e.g., 'E1')")
    source_query: str = Field(
        default="",
        description="Query text that produced this evidence",
    )
    mcp_server: str = Field(
        default="",
        description="MCP server that produced this result",
    )
    observation: str = Field(
        default="",
        description="Raw MCP response text",
    )
    url: str | None = Field(
        default=None,
        description="Source URL if available",
    )


class CleanedEvidence(BaseModel):
    """Evidence after pipeline cleaning 🧹.

    Produced by EvidencePipeline after dedup, noise filter,
    and source classification.
    """

    id: str = Field(description="Evidence identifier")
    summary: str = Field(
        default="",
        description="Cleaned and truncated summary text",
    )
    url: str | None = Field(
        default=None,
        description="Source URL if available",
    )
    tag: EvidenceTag = Field(
        default=EvidenceTag.OTHER,
        description="Source type classification",
    )
    source_query: str = Field(
        default="",
        description="Query that produced this evidence",
    )
    mcp_server: str = Field(
        default="",
        description="MCP server source",
    )

    # 🏷️ Extended fields propagated from Evidence (H5 fix)
    doi: str | None = Field(
        default=None,
        description="DOI extracted from URL or summary content",
    )
    clinical_trial_id: str | None = Field(
        default=None,
        description="NCT ID extracted from URL or summary (e.g. NCT12345678)",
    )
    quality_label: str | None = Field(
        default=None,
        description="Quality label from EvidencePipeline classification",
    )


class GapReport(BaseModel):
    """Gap analysis result for one round 🎯.

    Contains coverage assessment and convergence status
    from a single round of gap analysis.
    """

    round_number: int = Field(
        description="Which round this report is for",
    )
    coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of checklist items covered",
    )
    covered_items: list[str] = Field(
        default_factory=list,
        description="Checklist items with sufficient evidence",
    )
    uncovered_items: list[str] = Field(
        default_factory=list,
        description="Checklist items lacking evidence",
    )
    conflict_signals: list[str] = Field(
        default_factory=list,
        description="Items with contradictory evidence",
    )
    converged: bool = Field(
        default=False,
        description="Whether the loop should stop",
    )
    convergence_reason: str | None = Field(
        default=None,
        description="Reason for convergence (if converged)",
    )
    judge_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost of LLM coverage judge call in USD.",
    )


class FocusPrompt(BaseModel):
    """LLM-generated search focus for next round 🔍.

    Produced by DiscoveryLoop to guide the next round's search
    toward uncovered checklist items.
    """

    prompt_text: str = Field(
        default="",
        description="Generated search guidance text",
    )
    target_gaps: list[str] = Field(
        default_factory=list,
        description="Uncovered items this prompt targets",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Recommended search queries",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Recommended MCP tools for next round",
    )
    generation_model: str = Field(
        default="",
        description="LLM model used to generate this prompt",
    )
    generation_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost to generate this focus prompt",
    )


class DiscoveryRoundSummary(BaseModel):
    """Summary of a single discovery round 📊.

    Lightweight record for tracking round-level metrics
    without full trajectory detail.
    """

    round_number: int = Field(description="Round index (1-based)")
    queries_executed: int = Field(default=0, description="Search queries run")
    raw_evidence_count: int = Field(
        default=0,
        description="Evidence before cleaning",
    )
    cleaned_evidence_count: int = Field(
        default=0,
        description="Evidence after cleaning",
    )
    coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Checklist coverage after this round",
    )
    coverage_delta: float = Field(
        default=0.0,
        description="Coverage improvement from previous round",
    )
    round_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost for this round in USD",
    )
    converged: bool = Field(
        default=False,
        description="Whether loop stopped after this round",
    )
    convergence_reason: str | None = Field(
        default=None,
        description="Reason for convergence (if converged)",
    )


class DiscoveryResult(BaseModel):
    """Complete result from a DiscoveryLoop run 📊.

    Wraps the final output of a multi-round discovery pipeline,
    including all accumulated evidence, claims, and gap reports.
    """

    task_id: str = Field(description="Task identifier")
    pipeline_mode: str = Field(
        default="discovery",
        description="Pipeline execution mode (always 'discovery')",
    )
    total_rounds: int = Field(
        default=0,
        description="Number of rounds executed",
    )
    final_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final checklist coverage ratio",
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total cost across all rounds",
    )
    termination_reason: str = Field(
        default="",
        description="Why the loop terminated",
    )
    evidence: list[CleanedEvidence] = Field(
        default_factory=list,
        description="All cleaned evidence accumulated",
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Consensus claims from analysis voting",
    )
    gap_reports: list[GapReport] = Field(
        default_factory=list,
        description="Gap report from each round",
    )
    round_summaries: list[DiscoveryRoundSummary] = Field(
        default_factory=list,
        description="Per-round summary metrics",
    )
    trajectory_id: str | None = Field(
        default=None,
        description="Trajectory ID for process tracing",
    )


# ============================================================================
# 🧬 Section 8: Evolution Models
# ============================================================================


class RoundMetrics(BaseModel):
    """Per-round metrics for fitness evaluation — before/after comparison 📊.

    Used by EvolutionProvider to evaluate whether injected experiences
    improved research quality between rounds.

    Attributes:
        evidence_count: Total evidence items accumulated.
        new_evidence_count: New evidence found in this round.
        coverage: Checklist coverage ratio (0.0-1.0).
        cost_usd: Cumulative cost in USD for this round.
        round_index: Zero-based round index for array access.
    """

    evidence_count: int = Field(
        default=0,
        description="Total evidence items accumulated",
    )
    new_evidence_count: int = Field(
        default=0,
        description="New evidence found in this round",
    )
    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Checklist coverage ratio (0.0-1.0)",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative cost in USD for this round",
    )
    round_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based round index for array access",
    )


class EvolutionProvider(Protocol):
    """Domain-agnostic evolution interface injected into DiscoveryLoop 🧬.

    Follows the same injection pattern as FeedbackProvider:
    - Defined in Inquiro (domain-agnostic)
    - Implemented in TargetMaster (domain-specific)
    - Injected via Runner → DiscoveryLoop constructor

    This protocol enables the DISCOVERY pipeline to:
    1. Inject learned experiences into search/analysis/synthesis prompts
    2. Collect trajectory data after each round
    3. Extract and store reusable experiences
    4. Evaluate fitness of previously injected experiences
    """

    def get_search_enrichment(
        self,
        round_num: int,
        gap_items: list[str],
    ) -> str | None:
        """Return prompt section to inject into SearchExp system prompt 🔍.

        Args:
            round_num: Current Discovery round (1-based, matching
                DiscoveryLoop).
            gap_items: Uncovered checklist item IDs from gap analysis.

        Returns:
            Markdown text to append to system prompt, or None.
        """
        ...

    def get_analysis_enrichment(self) -> str | None:
        """Return prompt section to inject into AnalysisExp system prompt 🔬."""
        ...

    def get_synthesis_enrichment(self) -> str | None:
        """Return prompt section to inject into SynthesisExp system prompt 📝."""
        ...

    async def on_round_complete(
        self,
        round_num: int,
        round_record: DiscoveryRoundRecord,
        round_metrics: RoundMetrics,
    ) -> None:
        """Called after each Discovery round completes 🔄.

        Implementations should:
        1. Collect trajectory data from round_record
        2. Extract reusable experiences (may call LLM)
        3. Store experiences
        4. Evaluate fitness of previously injected experiences

        Args:
            round_num: Completed round number (1-based).
            round_record: Full record of the completed round.
            round_metrics: Computed metrics for fitness comparison.
        """
        ...

    async def on_synthesis_complete(
        self,
        synthesis_record: SynthesisRecord,
        final_metrics: RoundMetrics,
    ) -> None:
        """Called after final synthesis completes 🏁.

        Args:
            synthesis_record: Full synthesis execution record.
            final_metrics: Final accumulated metrics.
        """
        ...


# ============================================================================
# 🎭 Section 9: Multi-Perspective Models
# ============================================================================


class PerspectiveConfig(BaseModel):
    """Configuration for a single analysis perspective 🎭.

    Defines a named perspective with its system prompt context and relative
    weight for consensus aggregation.  Domain-specific role descriptions
    are injected by the orchestrating layer (e.g., TargetMaster).

    Attributes:
        perspective_id: Unique identifier for this perspective.
        system_prompt_context: System prompt content defining the
            perspective's focus.
        weight: Relative weight in consensus aggregation (default 1.0).
    """

    perspective_id: str = Field(
        description="Unique identifier for this perspective.",
    )
    system_prompt_context: str = Field(
        description="System prompt content for this perspective.",
    )
    weight: float = Field(
        default=1.0,
        description="Relative weight in consensus (0.0-1.0).",
    )


class PerspectiveOutput(BaseModel):
    """Structured output from a single perspective analysis 📋.

    Attributes:
        perspective_id: Which perspective produced this output.
        decision: Assessment decision (GO / CONDITIONAL_GO / NO_GO).
        key_insight: Core insight from this perspective (1-2 sentences).
        concern: Primary concern raised (1-2 sentences).
        confidence: Confidence level (0.0-1.0).
        recommendation: Actionable recommendation (1 sentence).
    """

    perspective_id: str = Field(
        description="Which perspective produced this output.",
    )
    decision: str = Field(
        description="GO / CONDITIONAL_GO / NO_GO.",
    )
    key_insight: str = Field(
        description="Core insight (1-2 sentences).",
    )
    concern: str = Field(
        description="Primary concern (1-2 sentences).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence 0.0-1.0.",
    )
    recommendation: str = Field(
        default="",
        description="Actionable recommendation (1 sentence).",
    )


class ConsensusOutput(BaseModel):
    """Aggregated consensus from multiple perspectives 🤝.

    Attributes:
        unified_recommendation: Overall committee recommendation.
        consensus_narrative: Brief narrative summarizing the consensus
            (~200 words).
        dissent_notes: Notable dissenting opinions from individual
            perspectives.
        key_action_items: Recommended next actions based on the panel
            discussion.
    """

    unified_recommendation: str = Field(
        description="GO / CONDITIONAL_GO / NO_GO.",
    )
    consensus_narrative: str = Field(
        description="Consensus summary (~200 words).",
    )
    dissent_notes: list[str] = Field(
        default_factory=list,
        description="Dissenting opinions.",
    )
    key_action_items: list[str] = Field(
        default_factory=list,
        description="Recommended actions.",
    )
