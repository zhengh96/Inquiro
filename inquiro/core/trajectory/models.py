"""Discovery Trajectory data models 📊.

Defines the full hierarchy of Trajectory recording models
for the Discovery pipeline.  Uses composition over inheritance
with EvoMaster's base Trajectory types.

Hierarchy:
    DiscoveryTrajectory (one complete Discovery run)
    ├── config_snapshot
    ├── task_snapshot
    ├── rounds: list[DiscoveryRoundRecord]
    │   ├── search_phase: SearchPhaseRecord
    │   ├── cleaning_phase: CleaningPhaseRecord
    │   ├── analysis_phase: AnalysisPhaseRecord
    │   └── gap_phase: GapPhaseRecord
    ├── synthesis_record: SynthesisRecord | None
    ├── summary: DiscoverySummary
    └── events: list[TrajectoryEvent]
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

_UTC = ZoneInfo("UTC")


# ============================================================================
# 🏷️ Enums
# ============================================================================


class TrajectoryEventType(str, Enum):
    """Types of trajectory events ⏱️."""

    DISCOVERY_STARTED = "discovery_started"
    ROUND_STARTED = "round_started"
    SEARCH_COMPLETED = "search_completed"
    CLEANING_COMPLETED = "cleaning_completed"
    ANALYSIS_COMPLETED = "analysis_completed"
    GAP_COMPLETED = "gap_completed"
    FOCUS_PROMPT_GENERATED = "focus_prompt_generated"
    ROUND_COMPLETED = "round_completed"
    CONVERGENCE_REACHED = "convergence_reached"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    DISCOVERY_COMPLETED = "discovery_completed"
    ERROR = "error"


# ============================================================================
# 📊 Phase-level records
# ============================================================================


class QueryRecord(BaseModel):
    """Record of a single search query execution 🔍.

    Attributes:
        query_text: The search query string.
        mcp_tool: MCP tool that executed this query.
        result_count: Number of evidence items returned.
        cost_usd: Cost for this query execution.
    """

    query_text: str = Field(description="Search query text")
    mcp_tool: str = Field(default="", description="MCP tool used")
    result_count: int = Field(default=0, description="Results returned")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Query cost")


class ServerStats(BaseModel):
    """Per-MCP-server search effectiveness statistics 📡.

    Attributes:
        queries_sent: Number of queries dispatched to this server.
        results_returned: Total evidence items returned.
        hit_rate: Fraction of queries that returned at least one result.
    """

    queries_sent: int = Field(default=0, description="Queries dispatched")
    results_returned: int = Field(
        default=0, description="Total evidence items returned"
    )
    hit_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of queries with at least one result",
    )


class SearchPhaseRecord(BaseModel):
    """Record of the search phase in one round 🔍.

    Attributes:
        queries: Individual query execution records.
        total_raw_evidence: Total raw evidence collected.
        agent_trajectory_ref: Path to the agent's JSONL trajectory.
        agent_trajectory_refs: All agent JSONL trajectory file paths
            (one per parallel section).
        duration_seconds: Time taken for search phase.
        server_effectiveness: Per-MCP-server hit rates and result counts.
        query_diversity_score: Measure of query variety (0-1).
    """

    queries: list[QueryRecord] = Field(
        default_factory=list,
        description="Individual query execution records",
    )
    total_raw_evidence: int = Field(
        default=0,
        description="Total raw evidence items collected",
    )
    agent_trajectory_ref: str | None = Field(
        default=None,
        description="Path to agent JSONL trajectory file",
    )
    agent_trajectory_refs: list[str] = Field(
        default_factory=list,
        description=(
            "All agent JSONL trajectory file paths (one per parallel section)."
        ),
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Search phase duration",
    )
    server_effectiveness: dict[str, ServerStats] = Field(
        default_factory=dict,
        description="Per-MCP-server hit rates and result counts",
    )
    query_diversity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Measure of query variety across round (0=identical, 1=unique)",
    )


class CleaningPhaseRecord(BaseModel):
    """Record of the evidence cleaning phase 🧹.

    Attributes:
        input_count: Evidence before cleaning.
        output_count: Evidence after cleaning.
        dedup_removed: Items removed by deduplication.
        noise_removed: Items removed by noise filter.
        tag_distribution: Count of each EvidenceTag.
        duration_seconds: Time taken for cleaning.
    """

    input_count: int = Field(default=0, description="Evidence before cleaning")
    output_count: int = Field(default=0, description="Evidence after cleaning")
    dedup_removed: int = Field(default=0, description="Dedup removed count")
    noise_removed: int = Field(default=0, description="Noise removed count")
    tag_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each evidence source tag",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Cleaning phase duration",
    )


class ModelAnalysisRecord(BaseModel):
    """Analysis output from a single LLM model 🧠.

    Attributes:
        model_name: LLM model identifier.
        claims_count: Number of claims produced.
        decision: Model's decision (positive/cautious/negative).
        confidence: Model's confidence score.
        cost_usd: Cost for this model's analysis.
    """

    model_name: str = Field(description="LLM model identifier")
    claims_count: int = Field(default=0, description="Claims produced")
    decision: str = Field(default="", description="Model decision")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence",
    )
    cost_usd: float = Field(default=0.0, ge=0.0, description="Model cost")


class ConsensusRecord(BaseModel):
    """Consensus from multi-model voting 🤝.

    Attributes:
        consensus_decision: Final aggregated decision.
        consensus_ratio: Fraction of models agreeing.
        total_claims: Total merged claims after dedup.
    """

    consensus_decision: str = Field(
        default="", description="Aggregated consensus decision"
    )
    consensus_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model agreement ratio",
    )
    total_claims: int = Field(default=0, description="Merged claims after dedup")


class AnalysisPhaseRecord(BaseModel):
    """Record of the analysis phase in one round 🧠.

    Attributes:
        model_results: Per-model analysis records.
        consensus: Consensus voting result.
        evidence_quality_distribution: Count of evidence items per quality tier.
        duration_seconds: Time taken for analysis.
    """

    model_results: list[ModelAnalysisRecord] = Field(
        default_factory=list,
        description="Per-model analysis records",
    )
    consensus: ConsensusRecord = Field(
        default_factory=ConsensusRecord,
        description="Consensus voting result",
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Information gaps identified in this round",
    )
    doubts_remaining: list[str] = Field(
        default_factory=list,
        description="Evidence contradictions found in this round",
    )
    evidence_quality_distribution: dict[str, int] = Field(
        default_factory=dict,
        description='Count per quality tier, e.g. {"high": 5, "medium": 10, "low": 3}',
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Analysis phase duration",
    )


class FocusPromptRecord(BaseModel):
    """Record of focus prompt generation 🔍.

    Attributes:
        prompt_text: Generated focus prompt text.
        target_gaps: Gaps this prompt addresses.
        generation_model: LLM model used.
        cost_usd: Generation cost.
    """

    prompt_text: str = Field(default="", description="Generated focus prompt")
    target_gaps: list[str] = Field(
        default_factory=list,
        description="Gaps this prompt targets",
    )
    generation_model: str = Field(default="", description="LLM model used")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Cost")


class GapPhaseRecord(BaseModel):
    """Record of the gap analysis phase 🎯.

    Attributes:
        coverage_ratio: Checklist coverage after analysis.
        covered_items: Items with sufficient evidence.
        uncovered_items: Items lacking evidence.
        conflict_signals: Items with contradictory evidence.
        newly_covered_items: Checklist items first covered in this round.
        convergence_reason: Why the loop stopped (if converged).
        focus_prompt: Generated focus prompt for next round.
        duration_seconds: Time taken for gap analysis.
    """

    coverage_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Checklist coverage ratio",
    )
    covered_items: list[str] = Field(
        default_factory=list, description="Covered checklist items"
    )
    uncovered_items: list[str] = Field(
        default_factory=list, description="Uncovered checklist items"
    )
    conflict_signals: list[str] = Field(
        default_factory=list, description="Contradictory evidence items"
    )
    newly_covered_items: list[str] = Field(
        default_factory=list,
        description="Checklist items first covered in this round",
    )
    convergence_reason: str | None = Field(
        default=None, description="Convergence reason"
    )
    focus_prompt: FocusPromptRecord | None = Field(
        default=None, description="Generated focus prompt"
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Gap analysis phase duration",
    )


# ============================================================================
# 📊 Round-level record
# ============================================================================


class DiscoveryRoundRecord(BaseModel):
    """Complete record of one discovery round 📊.

    Captures all four phases: search, cleaning, analysis, gap.

    Attributes:
        round_number: Round index (1-based).
        search_phase: Search phase record.
        cleaning_phase: Cleaning phase record.
        analysis_phase: Analysis phase record.
        gap_phase: Gap analysis phase record.
        round_cost_usd: Total cost for this round.
        round_duration_seconds: Total time for this round.
    """

    round_number: int = Field(description="Round index (1-based)")
    search_phase: SearchPhaseRecord = Field(
        default_factory=SearchPhaseRecord,
        description="Search phase record",
    )
    cleaning_phase: CleaningPhaseRecord = Field(
        default_factory=CleaningPhaseRecord,
        description="Cleaning phase record",
    )
    analysis_phase: AnalysisPhaseRecord = Field(
        default_factory=AnalysisPhaseRecord,
        description="Analysis phase record",
    )
    gap_phase: GapPhaseRecord = Field(
        default_factory=GapPhaseRecord,
        description="Gap analysis phase record",
    )
    round_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total cost for this round",
    )
    round_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total round duration in seconds",
    )


# ============================================================================
# 📊 Synthesis record
# ============================================================================


class SynthesisRecord(BaseModel):
    """Record of the synthesis phase 📝.

    Attributes:
        model_results: Per-model synthesis outputs.
        consensus_decision: Final synthesis decision.
        consensus_ratio: Model agreement ratio.
        cost_usd: Synthesis total cost.
        duration_seconds: Synthesis duration.
    """

    model_results: list[ModelAnalysisRecord] = Field(
        default_factory=list,
        description="Per-model synthesis outputs",
    )
    consensus_decision: str = Field(default="", description="Final synthesis decision")
    consensus_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model agreement ratio",
    )
    cost_usd: float = Field(default=0.0, ge=0.0, description="Synthesis cost")
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Synthesis duration",
    )


# ============================================================================
# 📊 Summary and events
# ============================================================================


class TrajectoryEvent(BaseModel):
    """Timestamped event in the trajectory timeline ⏱️.

    Attributes:
        event_type: Type of event.
        timestamp: When the event occurred.
        data: Optional additional data.
    """

    event_type: TrajectoryEventType = Field(description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=_UTC),
        description="Event timestamp",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional event data",
    )


class DiscoverySummary(BaseModel):
    """Aggregate summary of a complete discovery run 📊.

    Lightweight record (< 2KB) for permanent retention.

    Attributes:
        total_rounds: Number of rounds executed.
        final_coverage: Final checklist coverage ratio.
        total_cost_usd: Total cost across all phases.
        total_evidence: Total cleaned evidence items.
        total_claims: Total consensus claims.
        total_duration_seconds: Total wall-clock time.
        termination_reason: Why the loop stopped.
        evidence_yield_rate: cleaned / queries ratio.
        cost_normalized_quality: coverage / cost ratio.
    """

    total_rounds: int = Field(default=0, description="Rounds executed")
    final_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final coverage ratio",
    )
    total_cost_usd: float = Field(default=0.0, ge=0.0, description="Total cost")
    total_evidence: int = Field(default=0, description="Total cleaned evidence")
    total_claims: int = Field(default=0, description="Total consensus claims")
    total_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Total duration"
    )
    termination_reason: str = Field(default="", description="Loop termination reason")
    evidence_yield_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Evidence yield: cleaned / queries executed",
    )
    cost_normalized_quality: float = Field(
        default=0.0,
        ge=0.0,
        description="Quality metric: coverage / cost_usd",
    )


# ============================================================================
# 📊 Top-level Discovery Trajectory
# ============================================================================


class DiscoveryTrajectory(BaseModel):
    """Complete trajectory record for one Discovery run 📊.

    Top-level container composing all phase records, summary,
    and events.  Uses composition (not inheritance) with EvoMaster
    Trajectory types — agent-level trajectories are referenced by
    file path, not embedded.

    Attributes:
        trajectory_id: Unique identifier for this trajectory.
        task_id: Associated task identifier.
        config_snapshot: Runtime DiscoveryConfig parameters.
        task_snapshot: Task rules and checklist snapshot.
        rounds: Per-round detailed records.
        synthesis_record: Synthesis phase record (if performed).
        summary: Aggregate summary for lightweight querying.
        events: Timeline of key events.
        created_at: When the trajectory was created.
        completed_at: When the trajectory was finalized.
    """

    trajectory_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trajectory identifier",
    )
    task_id: str = Field(description="Associated task identifier")
    config_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime DiscoveryConfig parameters",
    )
    task_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Task rules and checklist snapshot",
    )
    rounds: list[DiscoveryRoundRecord] = Field(
        default_factory=list,
        description="Per-round detailed records",
    )
    synthesis_record: SynthesisRecord | None = Field(
        default=None,
        description="Synthesis phase record",
    )
    summary: DiscoverySummary = Field(
        default_factory=DiscoverySummary,
        description="Aggregate summary",
    )
    events: list[TrajectoryEvent] = Field(
        default_factory=list,
        description="Timeline of key events",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=_UTC),
        description="Trajectory creation time",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Trajectory completion time",
    )
