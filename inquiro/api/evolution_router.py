"""Evolution management API endpoints 🧬.

Provides management endpoints for the self-evolution system, including:
- Experience querying and statistics
- Fitness decay and pruning operations
- Conflict detection and resolution
- Evolution metrics and A/B test analysis

All operations require a namespace parameter to enforce data isolation
between different upper-layer platforms (e.g., TargetMaster).

Endpoints:
- GET    /api/v1/evolution/experiences        Query experiences
- GET    /api/v1/evolution/stats              Get evolution statistics
- GET    /api/v1/evolution/metrics            Get evolution metrics + A/B breakdown
- POST   /api/v1/evolution/decay              Trigger fitness decay
- POST   /api/v1/evolution/prune              Trigger pruning
- POST   /api/v1/evolution/conflicts/detect   Detect conflicts
- POST   /api/v1/evolution/conflicts/resolve  Resolve conflicts
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from inquiro.api.schemas import ErrorResponse
from inquiro.evolution.metrics import (
    EvolutionMetricsRecorder,
    EvolutionMetricsSummary,
)
from inquiro.evolution.ranker import ExperienceRanker
from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.store_factory import get_store
from inquiro.evolution.types import (
    Experience,
    ExperienceQuery,
    PruneConfig,
)

logger = logging.getLogger(__name__)

# ============================================================
# 🔧 Router Instance
# ============================================================

router = APIRouter(prefix="/api/v1/evolution", tags=["evolution"])


# ============================================================
# 💾 Store Dependency (via shared factory)
# ============================================================

_ranker_instance: ExperienceRanker | None = None

# 📊 Module-level metrics recorder singleton
_metrics_recorder: EvolutionMetricsRecorder | None = None


def get_metrics_recorder() -> EvolutionMetricsRecorder:
    """Get or create the module-level EvolutionMetricsRecorder singleton 📊.

    Returns:
        Shared EvolutionMetricsRecorder instance.
    """
    global _metrics_recorder
    if _metrics_recorder is None:
        _metrics_recorder = EvolutionMetricsRecorder()
    return _metrics_recorder


async def _get_store() -> ExperienceStore:
    """Get ExperienceStore via shared singleton factory 💾.

    Delegates to ``inquiro.evolution.store_factory.get_store()``
    for centralized, lazy-initialized store management.

    Returns:
        Initialized ExperienceStore instance.
    """
    return await get_store()


async def _get_ranker(
    store: ExperienceStore = Depends(_get_store),
) -> ExperienceRanker:
    """Get or initialize ExperienceRanker singleton 🗑️.

    Creates a ranker instance that wraps the store for lifecycle
    management operations (pruning, decay, conflict resolution).

    Args:
        store: Injected ExperienceStore dependency.

    Returns:
        Initialized ExperienceRanker instance.
    """
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = ExperienceRanker(store)
        logger.info("ExperienceRanker initialized ✅")
    return _ranker_instance


# ============================================================
# 📋 Request/Response Models
# ============================================================


class ExperienceQueryResponse(BaseModel):
    """Response for experience query endpoint 🔍.

    Returns a list of matching Experience objects.
    """

    experiences: list[Experience] = Field(
        description="List of experiences matching the query",
    )
    total_count: int = Field(
        description="Number of experiences returned",
    )


class StatsResponse(BaseModel):
    """Evolution statistics for a namespace 📊.

    Provides aggregate metrics about the experience store.
    """

    namespace: str = Field(
        description="Namespace these stats apply to",
    )
    total_count: int = Field(
        description="Total number of experiences in namespace",
    )
    avg_fitness: float = Field(
        description="Average fitness score across all experiences",
    )
    category_breakdown: dict[str, int] = Field(
        description="Count of experiences per category",
    )


class DecayRequest(BaseModel):
    """Request to trigger fitness decay 📉.

    Applies multiplicative decay to all experiences in a namespace.
    """

    namespace: str = Field(
        description="Namespace to apply decay to",
    )
    decay_factor: float = Field(
        default=0.95,
        description="Multiplicative decay factor (0.0-1.0)",
        gt=0.0,
        le=1.0,
    )


class DecayResponse(BaseModel):
    """Response from decay operation 📉."""

    affected_count: int = Field(
        description="Number of experiences affected by decay",
    )


class PruneRequest(BaseModel):
    """Request to trigger experience pruning 🗑️.

    Removes low-fitness experiences that have been tested enough times.
    """

    namespace: str = Field(
        description="Namespace to prune",
    )
    min_fitness: float = Field(
        default=0.2,
        description="Minimum fitness score to keep",
        ge=0.0,
        le=1.0,
    )
    min_uses: int = Field(
        default=5,
        description="Minimum times_used before eligible for pruning",
        ge=0,
    )


class PruneResponse(BaseModel):
    """Response from prune operation 🗑️."""

    pruned_count: int = Field(
        description="Number of experiences pruned",
    )


class ConflictDetectRequest(BaseModel):
    """Request to detect conflicting experiences 🔍."""

    namespace: str = Field(
        description="Namespace to scan for conflicts",
    )


class ConflictDetectResponse(BaseModel):
    """Response from conflict detection 🔍."""

    conflicts: list[tuple[str, str]] = Field(
        description="List of [exp_id_1, exp_id_2] conflict pairs",
    )
    count: int = Field(
        description="Total number of conflict pairs found",
    )


class ConflictResolveRequest(BaseModel):
    """Request to resolve detected conflicts 🔧."""

    namespace: str = Field(
        description="Namespace containing the conflicts",
    )
    conflicts: list[tuple[str, str]] = Field(
        description="List of [exp_id_1, exp_id_2] pairs to resolve",
    )


class ConflictResolveResponse(BaseModel):
    """Response from conflict resolution 🔧."""

    resolved_count: int = Field(
        description="Number of conflict pairs resolved",
    )


# ============================================================
# 🔍 GET /api/v1/evolution/experiences — Query experiences
# ============================================================


@router.get(
    "/experiences",
    response_model=ExperienceQueryResponse,
    responses={
        200: {"description": "List of matching experiences"},
        400: {"model": ErrorResponse, "description": "Invalid query parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Query experiences by namespace and filters",
)
async def query_experiences(
    namespace: str = Query(
        ...,
        description="Namespace to query (required for data isolation)",
    ),
    category: str | None = Query(
        None,
        description="Filter by category (optional)",
    ),
    min_fitness: float = Query(
        0.3,
        description="Minimum fitness score threshold",
        ge=0.0,
        le=1.0,
    ),
    max_results: int = Query(
        20,
        description="Maximum number of results to return",
        gt=0,
        le=100,
    ),
    context_tags: str | None = Query(
        None,
        description="Comma-separated context tags (optional)",
    ),
    store: ExperienceStore = Depends(_get_store),
) -> ExperienceQueryResponse:
    """Query experiences by namespace and optional filters 🔍.

    Returns a list of experiences matching the specified criteria.
    All queries must include a namespace for data isolation.

    Args:
        namespace: Required namespace for data isolation.
        category: Optional category filter.
        min_fitness: Minimum fitness score (default 0.3).
        max_results: Maximum results to return (default 20, max 100).
        context_tags: Comma-separated context tags.
        store: Injected ExperienceStore.

    Returns:
        ExperienceQueryResponse with matching experiences.

    Raises:
        HTTPException: 400 if query parameters are invalid, 500 on error.
    """
    try:
        # 🔄 Parse context_tags from comma-separated string
        parsed_tags: list[str] = []
        if context_tags:
            parsed_tags = [
                tag.strip() for tag in context_tags.split(",") if tag.strip()
            ]

        # 🔍 Build query object
        query = ExperienceQuery(
            namespace=namespace,
            context_tags=parsed_tags,
            category=category,
            min_fitness=min_fitness,
            max_results=max_results,
        )

        logger.info(
            "Querying experiences: namespace=%s, category=%s, min_fitness=%.2f 🔍",
            namespace,
            category,
            min_fitness,
        )

        # 📊 Execute query
        experiences = await store.query(query)

        logger.info(
            "Query returned %d experiences for namespace=%s ✅",
            len(experiences),
            namespace,
        )

        return ExperienceQueryResponse(
            experiences=experiences,
            total_count=len(experiences),
        )

    except ValueError as exc:
        logger.warning("Invalid query parameters: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="invalid_query",
                message=str(exc),
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Error querying experiences: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="query_failed",
                message=f"Failed to query experiences: {exc}",
            ).model_dump(),
        )


# ============================================================
# 📊 GET /api/v1/evolution/stats — Get evolution statistics
# ============================================================


@router.get(
    "/stats",
    response_model=StatsResponse,
    responses={
        200: {"description": "Evolution statistics for namespace"},
        400: {"model": ErrorResponse, "description": "Invalid namespace"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get evolution statistics for a namespace",
)
async def get_stats(
    namespace: str = Query(
        ...,
        description="Namespace to get stats for (required)",
    ),
    store: ExperienceStore = Depends(_get_store),
) -> StatsResponse:
    """Get evolution statistics for a namespace 📊.

    Returns aggregate metrics including total count, average fitness,
    and breakdown by category.

    Args:
        namespace: Namespace to get stats for.
        store: Injected ExperienceStore.

    Returns:
        StatsResponse with aggregate metrics.

    Raises:
        HTTPException: 400 if namespace is invalid, 500 on error.
    """
    try:
        logger.info("Getting stats for namespace=%s 📊", namespace)

        # 📊 Get stats from store
        stats = await store.get_stats(namespace)

        return StatsResponse(
            namespace=namespace,
            total_count=stats.get("total_count", 0),
            avg_fitness=stats.get("avg_fitness", 0.0),
            category_breakdown=stats.get("category_breakdown", {}),
        )

    except Exception as exc:
        logger.exception("Error getting stats: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="stats_failed",
                message=f"Failed to get statistics: {exc}",
            ).model_dump(),
        )


# ============================================================
# 📉 POST /api/v1/evolution/decay — Trigger fitness decay
# ============================================================


@router.post(
    "/decay",
    response_model=DecayResponse,
    responses={
        200: {"description": "Decay applied successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Apply fitness decay to all experiences in namespace",
)
async def apply_decay(
    request: DecayRequest,
    ranker: ExperienceRanker = Depends(_get_ranker),
) -> DecayResponse:
    """Apply multiplicative fitness decay to all experiences 📉.

    Reduces fitness scores over time to prevent stale experiences
    from dominating. For each experience: new_fitness = old_fitness * decay_factor

    Args:
        request: Decay configuration (namespace, decay_factor).
        ranker: Injected ExperienceRanker.

    Returns:
        DecayResponse with number of experiences affected.

    Raises:
        HTTPException: 400 if parameters invalid, 500 on error.
    """
    try:
        logger.info(
            "Applying decay to namespace=%s with factor=%.3f 📉",
            request.namespace,
            request.decay_factor,
        )

        # 📉 Apply decay via ranker
        affected_count = await ranker.apply_decay(
            request.namespace,
            request.decay_factor,
        )

        logger.info(
            "Decay applied to %d experiences in namespace=%s ✅",
            affected_count,
            request.namespace,
        )

        return DecayResponse(affected_count=affected_count)

    except ValueError as exc:
        logger.warning("Invalid decay request: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="invalid_decay_factor",
                message=str(exc),
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Error applying decay: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="decay_failed",
                message=f"Failed to apply decay: {exc}",
            ).model_dump(),
        )


# ============================================================
# 🗑️ POST /api/v1/evolution/prune — Trigger pruning
# ============================================================


@router.post(
    "/prune",
    response_model=PruneResponse,
    responses={
        200: {"description": "Pruning completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Prune low-fitness experiences from namespace",
)
async def prune_experiences(
    request: PruneRequest,
    ranker: ExperienceRanker = Depends(_get_ranker),
) -> PruneResponse:
    """Prune low-fitness experiences from namespace 🗑️.

    Removes experiences that have low fitness scores and have been
    tested enough times. This prevents the store from accumulating
    unhelpful experiences.

    Args:
        request: Prune configuration (namespace, min_fitness, min_uses).
        ranker: Injected ExperienceRanker.

    Returns:
        PruneResponse with number of experiences pruned.

    Raises:
        HTTPException: 400 if parameters invalid, 500 on error.
    """
    try:
        logger.info(
            "Pruning namespace=%s with min_fitness=%.2f, min_uses=%d 🗑️",
            request.namespace,
            request.min_fitness,
            request.min_uses,
        )

        # 🗑️ Build prune config
        config = PruneConfig(
            min_fitness=request.min_fitness,
            min_uses=request.min_uses,
            decay_factor=0.95,  # Default, not used in prune operation
            decay_interval_days=7,  # Default, not used in prune operation
        )

        # 🗑️ Prune via ranker
        pruned_count = await ranker.prune(request.namespace, config)

        logger.info(
            "Pruned %d experiences from namespace=%s ✅",
            pruned_count,
            request.namespace,
        )

        return PruneResponse(pruned_count=pruned_count)

    except ValueError as exc:
        logger.warning("Invalid prune request: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="invalid_prune_config",
                message=str(exc),
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception("Error pruning experiences: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="prune_failed",
                message=f"Failed to prune experiences: {exc}",
            ).model_dump(),
        )


# ============================================================
# 🔍 POST /api/v1/evolution/conflicts/detect — Detect conflicts
# ============================================================


@router.post(
    "/conflicts/detect",
    response_model=ConflictDetectResponse,
    responses={
        200: {"description": "Conflict detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Detect conflicting experiences in namespace",
)
async def detect_conflicts(
    request: ConflictDetectRequest,
    ranker: ExperienceRanker = Depends(_get_ranker),
) -> ConflictDetectResponse:
    """Detect conflicting experiences within namespace 🔍.

    Identifies pairs of experiences within the same category that
    contradict each other (e.g., "use X" vs "avoid X").

    Args:
        request: Namespace to scan for conflicts.
        ranker: Injected ExperienceRanker.

    Returns:
        ConflictDetectResponse with list of conflict pairs.

    Raises:
        HTTPException: 400 if request invalid, 500 on error.
    """
    try:
        logger.info(
            "Detecting conflicts in namespace=%s 🔍",
            request.namespace,
        )

        # 🔍 Detect conflicts via ranker
        conflicts = await ranker.detect_conflicts(request.namespace)

        logger.info(
            "Found %d conflict pairs in namespace=%s ✅",
            len(conflicts),
            request.namespace,
        )

        return ConflictDetectResponse(
            conflicts=conflicts,
            count=len(conflicts),
        )

    except Exception as exc:
        logger.exception("Error detecting conflicts: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="conflict_detection_failed",
                message=f"Failed to detect conflicts: {exc}",
            ).model_dump(),
        )


# ============================================================
# 🔧 POST /api/v1/evolution/conflicts/resolve — Resolve conflicts
# ============================================================


@router.post(
    "/conflicts/resolve",
    response_model=ConflictResolveResponse,
    responses={
        200: {"description": "Conflict resolution completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Resolve detected conflicts between experiences",
)
async def resolve_conflicts(
    request: ConflictResolveRequest,
    ranker: ExperienceRanker = Depends(_get_ranker),
) -> ConflictResolveResponse:
    """Resolve detected conflicts between experiences 🔧.

    Resolves conflict pairs by keeping the higher-fitness experience
    and removing the lower-fitness one.

    Args:
        request: Namespace and list of conflict pairs to resolve.
        ranker: Injected ExperienceRanker.

    Returns:
        ConflictResolveResponse with number of conflicts resolved.

    Raises:
        HTTPException: 400 if request invalid, 500 on error.
    """
    try:
        logger.info(
            "Resolving %d conflicts in namespace=%s 🔧",
            len(request.conflicts),
            request.namespace,
        )

        # 🔧 Resolve conflicts via ranker
        resolved_count = await ranker.resolve_conflicts(request.conflicts)

        logger.info(
            "Resolved %d conflict pairs in namespace=%s ✅",
            resolved_count,
            request.namespace,
        )

        return ConflictResolveResponse(resolved_count=resolved_count)

    except Exception as exc:
        logger.exception("Error resolving conflicts: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="conflict_resolution_failed",
                message=f"Failed to resolve conflicts: {exc}",
            ).model_dump(),
        )


# ============================================================
# 📊 Evolution Metrics
# ============================================================


@router.get(
    "/metrics",
    response_model=EvolutionMetricsSummary,
    summary="Get evolution metrics and A/B breakdown",
)
async def get_evolution_metrics(
    namespace: str = Query(
        ...,
        description="Namespace to retrieve metrics for",
    ),
    ab_group: str | None = Query(
        None,
        description=(
            "Filter by A/B group ('control' or 'treatment'). Omit for full breakdown."
        ),
    ),
) -> EvolutionMetricsSummary:
    """Get evolution metrics summary with A/B test breakdown 📊.

    Returns aggregated metrics including enrichment injection rate,
    average coverage, rounds to convergence, and per-group comparison
    for A/B test analysis.

    Args:
        namespace: Required namespace for data isolation.
        ab_group: Optional filter to retrieve only one A/B group.

    Returns:
        EvolutionMetricsSummary with aggregated metrics.
    """
    try:
        recorder = get_metrics_recorder()
        summary = recorder.summarize(
            namespace=namespace,
            ab_group=ab_group,
        )

        logger.info(
            "📊 Metrics retrieved: namespace=%s, evaluations=%d, rounds=%d",
            namespace,
            summary.total_evaluations,
            summary.total_rounds,
        )

        return summary

    except Exception as exc:
        logger.exception("Error retrieving evolution metrics: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="metrics_retrieval_failed",
                message=f"Failed to retrieve metrics: {exc}",
            ).model_dump(),
        )
