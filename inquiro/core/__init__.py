"""Inquiro Core module 🎯.

Provides the service-level orchestrator, shared infrastructure,
and all core data models.

Components:
    EvalTaskRunner: Service-level task orchestrator 🚀
    AggregationEngine: Result aggregation across models 📊
    types: All Pydantic v2 data models for Inquiro 📦
"""

from inquiro.core.aggregation import (
    AggregatedResult,
    AggregationEngine,
    ConflictInfo,
    ConflictResolution,
    EnsembleResult,
)
from inquiro.core.evidence_pool import SharedEvidencePool
from inquiro.core.runner import EvalTaskRunner
from inquiro.core.types import (
    # 🏷️ Enums
    CircuitState,
    CostStatus,
    Decision,
    EnsembleMode,
    EvidenceStrength,
    EvidenceTier,
    ExpPhase,
    LogicSignal,
    OverspendStrategy,
    TaskPhase,
    TaskStatus,
    TaskType,
    TruncationStrategy,
    _VALID_TRANSITIONS,
    # ⚙️ Config Models
    AdditionalResearchConfig,
    AgentConfig,
    ContextConfig,
    CostGuardConfig,
    EnsembleConfig,
    EnsembleModelConfig,
    QualityChecks,
    QualityGateConfig,
    ToolsConfig,
    # 📥 Input Models
    Checklist,
    ChecklistItem,
    DecisionGuidance,
    EvaluationTask,
    InputReport,
    SynthesisTask,
    # 📤 Output Models
    ChecklistCoverage,
    ChecklistProgress,
    Contradiction,
    ContradictionSide,
    CrossReference,
    DeepDiveRecord,
    Evidence,
    EvaluationResult,
    ReasoningClaim,
    RoundLog,
    SearchExecution,
    SynthesisResult,
    # 🔧 Infrastructure Models
    CostSummary,
    QualityGateResult,
)

__all__ = [
    # 🚀 Orchestrator
    "EvalTaskRunner",
    # 🔄 Evidence Pool
    "SharedEvidencePool",
    # 📊 Aggregation
    "AggregatedResult",
    "AggregationEngine",
    "ConflictInfo",
    "ConflictResolution",
    "EnsembleResult",
    # 🏷️ Enums
    "CircuitState",
    "CostStatus",
    "Decision",
    "EnsembleMode",
    "EvidenceStrength",
    "EvidenceTier",
    "ExpPhase",
    "LogicSignal",
    "_VALID_TRANSITIONS",
    "OverspendStrategy",
    "TaskPhase",
    "TaskStatus",
    "TaskType",
    "TruncationStrategy",
    # ⚙️ Config Models
    "AdditionalResearchConfig",
    "AgentConfig",
    "ContextConfig",
    "CostGuardConfig",
    "EnsembleConfig",
    "EnsembleModelConfig",
    "QualityChecks",
    "QualityGateConfig",
    "ToolsConfig",
    # 📥 Input Models
    "Checklist",
    "ChecklistItem",
    "DecisionGuidance",
    "EvaluationTask",
    "InputReport",
    "SynthesisTask",
    # 📤 Output Models
    "ChecklistCoverage",
    "ChecklistProgress",
    "Contradiction",
    "ContradictionSide",
    "CrossReference",
    "DeepDiveRecord",
    "Evidence",
    "EvaluationResult",
    "ReasoningClaim",
    "RoundLog",
    "SearchExecution",
    "SynthesisResult",
    # 🔧 Infrastructure Models
    "CostSummary",
    "QualityGateResult",
]
