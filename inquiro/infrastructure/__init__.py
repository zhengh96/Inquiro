"""Inquiro Infrastructure -- shared cross-cutting components 🏗️.

Provides quality validation, cost tracking, circuit breaking,
event streaming, cancellation support, configuration loading,
logging configuration, mode resolution, error classification,
metrics collection, graceful degradation, resource pool management,
and cross-task evidence memory used across the engine.

Components:
    QualityGate: Deterministic output validation 🔍
    CostTracker: Real-time token cost tracking 💰
    CircuitBreaker / CircuitBreakerRegistry: MCP fault isolation 🔌
    EventEmitter: Async SSE event system 📡
    CancellationToken: Cooperative task cancellation 🛑
    ConfigLoader: YAML config loading with env var interpolation 📋
    LoggingConfig: Structured logging setup 📝
    ModeResolver: Evaluation mode preset resolution 🎚️
    ErrorClassification: Transient / permanent error hierarchy 🔍
    MetricsCollector: Structured execution metrics 📊
    DegradationManager: Graceful 3-level fallback strategy 🛡️
    LLMProviderPool: Service-level LLM provider management 🤖
    MCPConnectionPool: Service-level MCP server management 🔌
    EvidenceMemory: Cross-task evidence reuse within a session 🧠
"""

from inquiro.infrastructure.cancellation import (
    CancellationToken,
    CancelledError,
)
from inquiro.infrastructure.errors import (
    AuthenticationError,
    ConfigurationError,
    InquiroError,
    LLMTimeoutError,
    MCPTransientError,
    PermanentError,
    RateLimitError,
    SchemaValidationError,
    TransientError,
    classify_error,
)
from inquiro.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from inquiro.infrastructure.config_loader import (
    ConfigLoadError,
    ConfigLoader,
)
from inquiro.infrastructure.logging_config import (
    ConsoleFormatter,
    JSONFormatter,
    setup_logging,
)
from inquiro.infrastructure.logging_context import (
    ContextConsoleFormatter,
    ContextFilter,
    ContextJSONFormatter,
    clear_logging_context,
    get_logging_context,
    install_context_logging,
    set_logging_context,
)
from inquiro.infrastructure.cost_tracker import (
    CostRecord,
    CostStatus,
    CostSummary,
    CostTracker,
    OverspendStrategy,
)
from inquiro.infrastructure.event_emitter import (
    InquiroEvent,
    EventCallback,
    EventData,
    EventEmitter,
)
from inquiro.infrastructure.mode_resolver import (
    ModeConfig,
    ModeResolver,
    ModeResolverError,
)

# ⚠️ llm_pool and mcp_pool are NOT eagerly imported here to avoid
# circular imports (mcp_pool -> tools -> core -> runner -> mcp_pool).
# Import them directly: from inquiro.infrastructure.llm_pool import ...
from inquiro.infrastructure.quality_gate import (
    QualityGate,
    QualityGateCheck,
    QualityGateChecksConfig,
    QualityGateConfig,
    QualityGateResult,
)
from inquiro.infrastructure.metrics import (
    MetricPoint,
    MetricsCollector,
)
from inquiro.infrastructure.degradation import (
    DegradationEvent,
    DegradationLevel,
    DegradationManager,
)
from inquiro.infrastructure.tool_effectiveness import (
    ToolEffectivenessTracker,
)
from inquiro.infrastructure.tool_routing import (
    ToolRoutingStrategy,
)
from inquiro.infrastructure.tracing import (
    TraceFilter,
    get_evaluation_id,
    get_trace_id,
    install_trace_filter,
    set_trace_context,
)
from inquiro.infrastructure.evidence_memory import (
    EvidenceMemory,
    StoredEvidence,
)

__all__ = [
    # 🛑 Cancellation
    "CancellationToken",
    "CancelledError",
    # 🔍 Error Classification
    "AuthenticationError",
    "ConfigurationError",
    "InquiroError",
    "LLMTimeoutError",
    "MCPTransientError",
    "PermanentError",
    "RateLimitError",
    "SchemaValidationError",
    "TransientError",
    "classify_error",
    # 🔌 Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
    # 📋 Config Loader
    "ConfigLoadError",
    "ConfigLoader",
    # 📝 Logging
    "ConsoleFormatter",
    "JSONFormatter",
    "setup_logging",
    # 📋 Logging Context
    "ContextConsoleFormatter",
    "ContextFilter",
    "ContextJSONFormatter",
    "clear_logging_context",
    "get_logging_context",
    "install_context_logging",
    "set_logging_context",
    # 💰 Cost Tracker
    "CostRecord",
    "CostStatus",
    "CostSummary",
    "CostTracker",
    "OverspendStrategy",
    # 📡 Event Emitter
    "InquiroEvent",
    "EventCallback",
    "EventData",
    "EventEmitter",
    # 🎚️ Mode Resolver
    "ModeConfig",
    "ModeResolver",
    "ModeResolverError",
    # 🤖 LLM Provider Pool (import from inquiro.infrastructure.llm_pool)
    "LLMProviderError",
    "LLMProviderPool",
    # 🔌 MCP Connection Pool (import from inquiro.infrastructure.mcp_pool)
    "MCPConnectionPool",
    "MCPPoolError",
    # 🔍 Quality Gate
    "QualityGate",
    "QualityGateCheck",
    "QualityGateChecksConfig",
    "QualityGateConfig",
    "QualityGateResult",
    # 📊 Metrics
    "MetricPoint",
    "MetricsCollector",
    # 🛡️ Degradation
    "DegradationEvent",
    "DegradationLevel",
    "DegradationManager",
    # 🎯 Tool Routing & Effectiveness
    "ToolEffectivenessTracker",
    "ToolRoutingStrategy",
    # 🔍 Tracing
    "TraceFilter",
    "get_evaluation_id",
    "get_trace_id",
    "install_trace_filter",
    "set_trace_context",
    # 🧠 Evidence Memory
    "EvidenceMemory",
    "StoredEvidence",
]


# 🔄 Lazy imports for modules with circular dependency potential
_LAZY_IMPORTS = {
    "LLMProviderError": "inquiro.infrastructure.llm_pool",
    "LLMProviderPool": "inquiro.infrastructure.llm_pool",
    "MCPConnectionPool": "inquiro.infrastructure.mcp_pool",
    "MCPPoolError": "inquiro.infrastructure.mcp_pool",
}


def __getattr__(name: str):  # noqa: ANN001
    """Lazy import for pool classes to avoid circular imports 🔄."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
