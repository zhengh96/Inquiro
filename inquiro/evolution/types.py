"""Inquiro Evolution data models — domain-agnostic containers 🧬.

All models are generic infrastructure types. String fields (category,
source, context_tags, insight) are treated as **opaque** by Inquiro.
Their vocabularies and semantics are defined by the upper-layer platform
(e.g., TargetMaster) via ``EvolutionProfile``.

Models:
- ``Experience`` — A single learned insight (opaque container)
- ``ExperienceQuery`` — Query to find relevant experiences
- ``TrajectorySnapshot`` — Structured execution data from a task run
- ``ToolCallRecord`` — Single tool call within a trajectory
- ``ResultMetrics`` — Outcome metrics from a task run
- ``FitnessUpdate`` — Delta update for experience fitness score
- ``PruneConfig`` — Configuration for experience pruning
- ``EnrichmentResult`` — Result of prompt enrichment
- ``MechanismType`` — Enum of supported learning mechanisms
- ``ToolStats`` — Bandit statistics for a single (context, tool) pair
- ``ReflectionRecord`` — Structured output from a round reflection
- ``ActionPrinciple`` — A learned operating principle with A/B status
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = [
    "Experience",
    "ExperienceQuery",
    "TrajectorySnapshot",
    "ToolCallRecord",
    "ResultMetrics",
    "FitnessUpdate",
    "PruneConfig",
    "EnrichmentResult",
    "MechanismType",
    "ToolStats",
    "ReflectionRecord",
    "ActionPrinciple",
]


# ============================================================================
# 🧬 Mechanism Type Enum
# ============================================================================


class MechanismType(str, Enum):
    """Identifies which learning mechanism produced an experience 🧬.

    Used for filtering, fitness attribution, and analytics.
    """

    EXPERIENCE_EXTRACTION = "experience_extraction"
    TOOL_SELECTION = "tool_selection"
    ROUND_REFLECTION = "round_reflection"
    ACTION_PRINCIPLES = "action_principles"


# ============================================================================
# 🎰 Bandit Statistics
# ============================================================================


class ToolStats(BaseModel):
    """Beta distribution statistics for a single (context, tool) pair 🎰.

    Used by ToolSelectionBandit to track success/failure counts
    for Thompson Sampling.
    """

    alpha: float = Field(
        default=1.0,
        description="Beta distribution alpha (successes + prior)",
        gt=0.0,
    )
    beta: float = Field(
        default=1.0,
        description="Beta distribution beta (failures + prior)",
        gt=0.0,
    )
    total_observations: int = Field(
        default=0,
        description="Total number of reward observations",
        ge=0,
    )


# ============================================================================
# 🪞 Reflection Record
# ============================================================================


class ReflectionRecord(BaseModel):
    """Structured output from a round-level reflection 🪞.

    Captures what worked, what failed, and strategic adjustments
    for the next round.
    """

    round_number: int = Field(
        description="Round number this reflection is about",
        ge=1,
    )
    what_worked: str = Field(
        default="",
        description="Effective strategies observed in this round",
    )
    what_failed: str = Field(
        default="",
        description="Failed attempts or ineffective strategies",
    )
    strategy: str = Field(
        default="",
        description="Strategic adjustment for the next round",
    )
    tool_recommendations: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Tool-level recommendations: {tool_name: 'increase'|'decrease'|'maintain'}"
        ),
    )
    priority_gaps: list[str] = Field(
        default_factory=list,
        description="High-priority gap items to cover next",
    )


# ============================================================================
# 📜 Action Principle
# ============================================================================


class ActionPrinciple(BaseModel):
    """A learned operating principle with A/B evaluation status 📜.

    Distilled from accumulated insights across multiple evaluations.
    Principles go through an A/B test before being promoted to active.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique principle identifier",
    )
    text: str = Field(
        description="The principle text (imperative sentence)",
    )
    status: str = Field(
        default="candidate",
        description="Lifecycle status: 'candidate' | 'active' | 'retired'",
    )
    treatment_coverage: float = Field(
        default=0.0,
        description="Average coverage when principle was injected",
        ge=0.0,
        le=1.0,
    )
    control_coverage: float = Field(
        default=0.0,
        description="Average coverage when principle was NOT injected",
        ge=0.0,
        le=1.0,
    )
    evaluation_count: int = Field(
        default=0,
        description="Number of evaluations in A/B test so far",
        ge=0,
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this principle was first distilled",
    )
    source_insight_ids: list[str] = Field(
        default_factory=list,
        description="IDs of experiences that were distilled into this principle",
    )


# ============================================================================
# 📝 Trajectory Data Models
# ============================================================================


class ToolCallRecord(BaseModel):
    """Record of a single tool call within a trajectory 🔧.

    Captures the essential information about an MCP tool invocation
    without interpreting the domain meaning of the tool or its results.
    """

    tool_name: str = Field(description="Name of the MCP tool invoked")
    arguments_summary: str = Field(
        default="",
        description="Abbreviated summary of tool arguments (not full payload)",
    )
    result_size: int = Field(
        default=0,
        description="Approximate size of tool result in characters",
        ge=0,
    )
    success: bool = Field(
        default=True,
        description="Whether the tool call completed successfully",
    )
    round_number: int = Field(
        default=0,
        description="Search round in which this call occurred",
        ge=0,
    )


class ResultMetrics(BaseModel):
    """Outcome metrics from a completed task run 📊.

    Generic metric container — all fields are domain-agnostic numbers.
    The upper layer decides which metrics matter via ``FitnessDimension``.
    """

    evidence_count: int = Field(
        default=0,
        description="Number of evidence items collected",
        ge=0,
    )
    confidence: float = Field(
        default=0.0,
        description="Agent-reported confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    search_rounds: int = Field(
        default=0,
        description="Number of search rounds executed",
        ge=0,
    )
    cost_usd: float = Field(
        default=0.0,
        description="Total cost in USD",
        ge=0.0,
    )
    decision: str = Field(
        default="",
        description="Decision string (opaque to evolution layer)",
    )
    checklist_coverage: float = Field(
        default=0.0,
        description="Fraction of required checklist items covered (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class TrajectorySnapshot(BaseModel):
    """Structured execution data from a completed task run 📸.

    Contains the tool calls, result metrics, and task context needed
    for experience extraction. Inquiro collects this data generically;
    the upper layer's extraction prompt decides what to make of it.
    """

    # 🆔 Identity
    evaluation_id: str = Field(
        description="ID of the completed evaluation task",
    )
    task_id: str = Field(
        description="Inquiro task ID within the evaluation",
    )

    # 📋 Task context (opaque strings from upper layer)
    topic: str = Field(
        default="",
        description="Research topic that was evaluated",
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Context tags from the task (opaque to Inquiro)",
    )
    sub_item_id: str = Field(
        default="",
        description="Sub-item identifier (opaque to Inquiro)",
    )

    # 🔧 Tool call records
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="All MCP tool calls made during the task",
    )

    # 📊 Outcome metrics
    metrics: ResultMetrics = Field(
        default_factory=ResultMetrics,
        description="Outcome metrics from the completed task",
    )

    # ⏱️ Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task start time (UTC)",
    )
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task completion time (UTC)",
    )
    wall_time_seconds: float = Field(
        default=0.0,
        description="Wall-clock time in seconds",
        ge=0.0,
    )


# ============================================================================
# 🧬 Experience Model
# ============================================================================


class Experience(BaseModel):
    """A single learned insight — domain-agnostic container 🧬.

    Inquiro treats all string fields as **opaque**. It does NOT
    interpret category, insight, or context_tags semantically.
    Vocabulary and semantics are defined by the upper layer's
    ``EvolutionProfile``.
    """

    # 🆔 Identity
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique experience identifier (UUID)",
    )
    namespace: str = Field(
        description=(
            "Namespace for data isolation. Each upper-layer platform "
            "(e.g., 'targetmaster') gets its own namespace."
        ),
    )

    # 🏷️ Classification (opaque to Inquiro)
    category: str = Field(
        description=(
            "Experience category — opaque string. "
            "Vocabulary defined by upper layer's EvolutionProfile."
        ),
    )

    # 📝 Content (opaque to Inquiro)
    insight: str = Field(
        description="The learned insight text, injected into prompts",
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Context tags for matching (e.g., 'category:TypeA'). "
            "Opaque to Inquiro — vocabulary from upper layer."
        ),
    )
    applicable_sub_items: list[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "Sub-item IDs this experience applies to. ['*'] means all sub-items."
        ),
    )

    # 📊 Fitness (generic math, thresholds from EvolutionProfile)
    fitness_score: float = Field(
        default=0.5,
        description="Fitness score (0.0-1.0), updated by FitnessEvaluator",
        ge=0.0,
        le=1.0,
    )
    times_used: int = Field(
        default=0,
        description="Number of times this experience was injected",
        ge=0,
    )
    times_helpful: int = Field(
        default=0,
        description="Number of times injection led to metric improvement",
        ge=0,
    )

    # 🧬 Mechanism provenance
    mechanism_type: str = Field(
        default=MechanismType.EXPERIENCE_EXTRACTION,
        description=(
            "Which learning mechanism produced this experience. "
            "Defaults to 'experience_extraction' for backward compatibility."
        ),
    )

    # 📎 Provenance
    source: str = Field(
        description=(
            "Experience source — opaque string. "
            "Vocabulary defined by upper layer "
            "(e.g., 'trajectory_extraction', 'human_feedback')."
        ),
    )
    source_evaluation_id: str | None = Field(
        default=None,
        description="ID of the evaluation that produced this experience",
    )
    source_trajectory_step: int | None = Field(
        default=None,
        description="Step number in the source trajectory",
    )

    # ⏱️ Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC)",
    )

    @field_validator("fitness_score")
    @classmethod
    def validate_fitness_score(cls, v: float) -> float:
        """Clamp fitness score to [0.0, 1.0] 🔒.

        Args:
            v: Raw fitness score.

        Returns:
            Clamped fitness score.
        """
        return max(0.0, min(1.0, v))

    @property
    def helpfulness_ratio(self) -> float:
        """Calculate helpfulness ratio (times_helpful / times_used) 📊.

        Returns:
            Float between 0.0 and 1.0. Returns 0.0 if never used.
        """
        if self.times_used == 0:
            return 0.0
        return self.times_helpful / self.times_used


# ============================================================================
# 🔍 Query Model
# ============================================================================


class ExperienceQuery(BaseModel):
    """Query to find relevant experiences for a task 🔍.

    All queries MUST include namespace — cross-namespace queries
    are not allowed to enforce data isolation.
    """

    namespace: str = Field(
        description="Required — only query experiences within this namespace",
    )
    context_tags: list[str] = Field(
        default_factory=list,
        description="Tags to match against experience context_tags",
    )
    sub_item: str | None = Field(
        default=None,
        description="Filter by applicable_sub_items (None = any)",
    )
    category: str | None = Field(
        default=None,
        description="Filter by category (None = any)",
    )
    min_fitness: float = Field(
        default=0.3,
        description="Minimum fitness score threshold",
        ge=0.0,
        le=1.0,
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of experiences to return",
        gt=0,
    )


# ============================================================================
# 📊 Fitness and Ranking Models
# ============================================================================


class FitnessUpdate(BaseModel):
    """Delta update for experience fitness score 📈.

    Produced by FitnessEvaluator after comparing task outcomes
    with and without experience injection.
    """

    experience_id: str = Field(
        description="ID of the experience to update",
    )
    signal: float = Field(
        description=(
            "Current signal strength (0.0-1.0). "
            "Computed from fitness_dimensions in EvolutionProfile."
        ),
        ge=0.0,
        le=1.0,
    )
    was_helpful: bool = Field(
        description="Whether the experience contributed to improvement",
    )
    metric_deltas: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-metric deltas: {metric_name: delta_value}. "
            "Positive = improvement, negative = regression."
        ),
    )


class PruneConfig(BaseModel):
    """Configuration for experience pruning — values from EvolutionProfile 🗑️.

    Inquiro does not define default values for these fields;
    they MUST be provided by the upper layer via EvolutionProfile.
    """

    min_fitness: float = Field(
        description="Minimum fitness score to keep (below = prune)",
        ge=0.0,
        le=1.0,
    )
    min_uses: int = Field(
        description=(
            "Minimum times_used before eligible for pruning "
            "(give new experiences a chance)"
        ),
        ge=0,
    )
    decay_factor: float = Field(
        description="Multiplicative decay factor (e.g., 0.95)",
        gt=0.0,
        le=1.0,
    )
    decay_interval_days: int = Field(
        description="Days between decay applications",
        gt=0,
    )


# ============================================================================
# 💉 Enrichment Result
# ============================================================================


class EnrichmentResult(BaseModel):
    """Result of prompt enrichment — tracks what was injected 💉.

    Used by FitnessEvaluator to know which experiences were active
    during a particular task run, enabling credit assignment.
    """

    injected_experience_ids: list[str] = Field(
        default_factory=list,
        description="IDs of experiences that were injected into the prompt",
    )
    enrichment_text: str = Field(
        default="",
        description="The formatted text that was injected into the prompt",
    )
    token_count: int = Field(
        default=0,
        description="Approximate token count of the enrichment text",
        ge=0,
    )
    truncated: bool = Field(
        default=False,
        description="Whether the enrichment was truncated due to token budget",
    )
