"""Shared pytest fixtures for Inquiro test suite 🧪.

Provides realistic sample data matching PRD examples, mock objects
for external dependencies, and async test client for API testing.

Fixtures:
- sample_evaluation_task: Realistic EvaluationTask for research tests
- sample_synthesis_task: Realistic SynthesisTask for synthesis tests
- sample_evaluation_result: Expected research output structure
- sample_synthesis_result: Expected synthesis output structure
- sample_research_request: Full API request payload for POST /research
- sample_synthesize_request: Full API request payload for POST /synthesize
- mock_llm: Mocked LLM provider
- mock_mcp_pool: Mocked MCP connection pool
- mock_task_runner: Mocked EvalTaskRunner
- mock_event_emitter: Mocked EventEmitter
- async_client: httpx AsyncClient for FastAPI testing
- test_app: FastAPI test application with dependency overrides
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# ✨ Import schemas for test data construction
from inquiro.api.schemas import (
    AdditionalResearchConfig,
    AgentConfig,
    Checklist,
    ChecklistItem,
    ContextConfig,
    CostGuardConfig,
    DecisionGuidance,
    EnsembleConfig,
    InputReport,
    OverspendStrategy,
    QualityChecks,
    QualityGateConfig,
    ResearchRequest,
    ResearchTaskPayload,
    SynthesisTaskPayload,
    SynthesizeRequest,
    ToolsConfig,
)


# ============================================================
# 📋 Sample Output Schemas (matching PRD Section 4)
# ============================================================


@pytest.fixture
def sample_output_schema() -> dict[str, Any]:
    """Default EvaluationResult JSON Schema from PRD Section 4.1 📋."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["decision", "confidence", "reasoning", "evidence_index"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["positive", "cautious", "negative"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["claim", "evidence_ids", "strength"],
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "strength": {
                            "type": "string",
                            "enum": ["weak", "moderate", "strong"],
                        },
                    },
                },
            },
            "evidence_index": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "source", "query", "summary"],
                    "properties": {
                        "id": {"type": "string"},
                        "source": {"type": "string"},
                        "query": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                },
            },
        },
    }


@pytest.fixture
def sample_synthesis_output_schema() -> dict[str, Any]:
    """Default SynthesisResult JSON Schema from PRD Section 4.2 📊."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "decision",
            "confidence",
            "reasoning",
            "evidence_index",
            "source_reports",
            "cross_references",
            "gaps_remaining",
        ],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["positive", "cautious", "negative"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {"type": "array"},
            "evidence_index": {"type": "array"},
            "source_reports": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cross_references": {"type": "array"},
            "gaps_remaining": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


# ============================================================
# 🔬 Sample Research Data (matching PRD Section 2.5 examples)
# ============================================================


@pytest.fixture
def sample_research_request(
    sample_output_schema: dict[str, Any],
) -> ResearchRequest:
    """Realistic POST /research request payload 🔬.

    Based on the EGFR market size example from PRD Section 2.5.
    """
    return ResearchRequest(
        task_id="test-research-001",
        task=ResearchTaskPayload(
            topic=(
                "Market size and growth potential for EGFR-targeted therapies in NSCLC"
            ),
            rules=(
                "# Evaluation Rules\n\n"
                "Focus on market data from 2020-2025.\n"
                "## Evidence Requirements\n"
                "Prefer peer-reviewed market analysis reports."
            ),
            checklist=Checklist(
                required=[
                    ChecklistItem(
                        id="item_1",
                        description=("Current global market size for EGFR therapies"),
                        keywords=["EGFR", "market size", "oncology"],
                        suggested_sources=["perplexity"],
                    ),
                ],
                optional=[
                    ChecklistItem(
                        id="item_2",
                        description="Projected CAGR for next 5 years",
                        keywords=["EGFR", "market growth", "forecast"],
                    ),
                ],
                coverage_threshold=0.8,
            ),
            decision_guidance=DecisionGuidance(
                positive=["Large addressable market with growing demand"],
                cautious=["Mature market with slow growth"],
                negative=["Shrinking market with declining demand"],
            ),
            output_schema=sample_output_schema,
        ),
        agent_config=AgentConfig(
            model="claude-sonnet-4-20250514",
            max_turns=30,
            temperature=0.3,
            context=ContextConfig(
                max_tokens=128000,
                truncation_strategy="latest_half",
            ),
        ),
        tools_config=ToolsConfig(
            mcp_servers=["perplexity", "biomcp"],
        ),
        ensemble_config=EnsembleConfig(enabled=False),
        quality_gate=QualityGateConfig(
            enabled=True,
            max_retries=2,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=True,
                evidence_reference_check=True,
            ),
        ),
        cost_guard=CostGuardConfig(
            max_cost_per_task=1.5,
            overspend_strategy=OverspendStrategy.SOFT_STOP,
        ),
        callback_url="https://targetmaster.example.com/api/webhooks/inquiro",
    )


@pytest.fixture
def sample_synthesize_request(
    sample_synthesis_output_schema: dict[str, Any],
) -> SynthesizeRequest:
    """Realistic POST /synthesize request payload 📊.

    Based on the clinical feasibility example from PRD Section 2.5.
    """
    return SynthesizeRequest(
        task_id="test-synthesis-001",
        task=SynthesisTaskPayload(
            objective=(
                "Synthesize clinical feasibility assessment "
                "from sub-item research reports"
            ),
            input_reports=[
                InputReport(
                    report_id="uuid-report-1",
                    label="Target Biology",
                    content={
                        "decision": "positive",
                        "confidence": 0.85,
                        "reasoning": [
                            {
                                "claim": "EGFR is well-validated target",
                                "evidence_ids": ["E1", "E2"],
                                "strength": "strong",
                            }
                        ],
                        "evidence_index": [
                            {
                                "id": "E1",
                                "source": "perplexity",
                                "query": "EGFR target validation",
                                "summary": "Multiple approved drugs",
                            },
                            {
                                "id": "E2",
                                "source": "biomcp",
                                "query": "EGFR biology",
                                "summary": "Well-characterized pathway",
                            },
                        ],
                    },
                ),
                InputReport(
                    report_id="uuid-report-2",
                    label="Biomarker Availability",
                    content={
                        "decision": "cautious",
                        "confidence": 0.7,
                        "reasoning": [
                            {
                                "claim": "Biomarker testing available but adoption varies",
                                "evidence_ids": ["E3"],
                                "strength": "moderate",
                            }
                        ],
                        "evidence_index": [
                            {
                                "id": "E3",
                                "source": "perplexity",
                                "query": "EGFR biomarker testing",
                                "summary": "IHC and FISH widely available",
                            }
                        ],
                    },
                ),
                InputReport(
                    report_id="uuid-report-3",
                    label="Patient Stratification",
                    content={
                        "decision": "positive",
                        "confidence": 0.75,
                        "reasoning": [
                            {
                                "claim": "Clear mutation-based stratification",
                                "evidence_ids": ["E4"],
                                "strength": "strong",
                            }
                        ],
                        "evidence_index": [
                            {
                                "id": "E4",
                                "source": "perplexity",
                                "query": "EGFR patient stratification",
                                "summary": "L858R and exon 19 deletions",
                            }
                        ],
                    },
                ),
            ],
            synthesis_rules=(
                "# Synthesis Rules\n\n"
                "Cross-reference findings across all input reports.\n"
                "## Cross-cutting Analysis\n"
                "Identify correlations between target biology and "
                "biomarker availability."
            ),
            allow_additional_research=True,
            additional_research_config=AdditionalResearchConfig(
                max_tasks=3,
                cost_budget=2.0,
                tools_config=ToolsConfig(
                    mcp_servers=["perplexity", "biomcp"],
                ),
            ),
            output_schema=sample_synthesis_output_schema,
        ),
        agent_config=AgentConfig(
            model="claude-sonnet-4-20250514",
            max_turns=20,
            temperature=0.3,
        ),
        quality_gate=QualityGateConfig(
            enabled=True,
            max_retries=2,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=True,
                cross_reference_check=True,
            ),
        ),
        cost_guard=CostGuardConfig(
            max_cost_per_task=3.0,
            overspend_strategy=OverspendStrategy.SOFT_STOP,
        ),
        callback_url="https://targetmaster.example.com/api/webhooks/inquiro",
    )


# ============================================================
# 📊 Sample Result Objects
# ============================================================


@pytest.fixture
def sample_evaluation_result() -> dict[str, Any]:
    """Expected EvaluationResult structure matching PRD output schema 📊."""
    return {
        "task_id": "test-research-001",
        "decision": "positive",
        "confidence": 0.85,
        "reasoning": [
            {
                "claim": "EGFR therapy market exceeds $20B globally",
                "evidence_ids": ["E1", "E2"],
                "strength": "strong",
            },
            {
                "claim": "Market growing at 8% CAGR",
                "evidence_ids": ["E3"],
                "strength": "moderate",
            },
        ],
        "evidence_index": [
            {
                "id": "E1",
                "source": "perplexity",
                "query": "EGFR market size 2024",
                "summary": "Global EGFR therapy market valued at $22.5B",
            },
            {
                "id": "E2",
                "source": "biomcp",
                "query": "EGFR oncology market",
                "summary": "Consistent growth in targeted therapy segment",
            },
            {
                "id": "E3",
                "source": "perplexity",
                "query": "EGFR market growth forecast",
                "summary": "Projected 8.2% CAGR through 2029",
            },
        ],
        "search_rounds": 3,
        "round_logs": [
            {
                "round_number": 1,
                "searches_executed": [{"query": "EGFR market size"}],
                "findings_summary": "Found initial market data",
                "gaps_identified": ["CAGR projection missing"],
                "doubts_identified": ["Single source for market size"],
            }
        ],
        "checklist_coverage": {
            "required_covered": ["item_1"],
            "required_missing": [],
        },
        "gaps_remaining": [],
        "doubts_remaining": [],
    }


@pytest.fixture
def sample_synthesis_result() -> dict[str, Any]:
    """Expected SynthesisResult structure matching PRD output schema 📊."""
    return {
        "task_id": "test-synthesis-001",
        "decision": "positive",
        "confidence": 0.80,
        "reasoning": [
            {
                "claim": "Strong target biology combined with available biomarkers",
                "evidence_ids": ["E1", "E2", "E3"],
                "strength": "strong",
            }
        ],
        "evidence_index": [
            {
                "id": "E1",
                "source": "perplexity",
                "query": "EGFR target validation",
                "summary": "Multiple approved drugs",
                "source_report_id": "uuid-report-1",
            }
        ],
        "source_reports": [
            "uuid-report-1",
            "uuid-report-2",
            "uuid-report-3",
        ],
        "cross_references": [
            {
                "claim": "EGFR well-validated as therapeutic target",
                "supporting_reports": ["uuid-report-1", "uuid-report-3"],
                "contradicting_reports": [],
            }
        ],
        "contradictions": [],
        "gaps_remaining": [],
        "deep_dives_triggered": [],
    }


# ============================================================
# 🤖 Mock Objects
# ============================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mocked LLM provider for unit testing 🤖.

    Returns a mock that simulates LLM responses without
    making actual API calls.
    """
    llm = MagicMock()
    llm.generate = MagicMock(return_value="Mocked LLM response")
    llm.generate_async = AsyncMock(return_value="Mocked LLM response")
    return llm


@pytest.fixture
def mock_mcp_pool() -> MagicMock:
    """Mocked MCP connection pool 🔌.

    Provides a mock that simulates MCP server connections
    and tool registry filtering.
    """
    pool = MagicMock()
    pool.initialize = AsyncMock()
    pool.cleanup = AsyncMock()
    pool.get_filtered_registry = MagicMock(return_value=MagicMock())
    pool.get_health = MagicMock(
        return_value={
            "perplexity": "connected",
            "biomcp": "connected",
        }
    )
    return pool


@pytest.fixture
def mock_event_emitter() -> MagicMock:
    """Mocked EventEmitter for testing event emission 📡."""
    emitter = MagicMock()
    emitter.emit = MagicMock()
    emitter.subscribe = MagicMock(return_value=asyncio.Queue())
    return emitter


@pytest.fixture
def mock_task_runner(mock_mcp_pool: MagicMock) -> MagicMock:
    """Mocked EvalTaskRunner for API testing 🎯.

    Provides mock submit_research/submit_synthesis that return
    sample results without executing real agents.
    """
    runner = MagicMock()
    runner.mcp_pool = mock_mcp_pool
    runner._active_tasks = {}
    runner.get_active_task_count = MagicMock(return_value=0)
    runner.get_active_task_ids = MagicMock(return_value=[])

    # ✨ Return properly structured results to avoid Pydantic errors
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {
        "decision": "positive",
        "confidence": 0.85,
    }
    mock_result.cost = 0.5

    runner.submit_research = AsyncMock(return_value=mock_result)
    runner.submit_synthesis = AsyncMock(return_value=mock_result)
    runner.cancel_task = MagicMock(return_value=True)
    return runner


# ============================================================
# 🌐 FastAPI Test Application
# ============================================================


@pytest.fixture
def test_task_store() -> dict[str, Any]:
    """Shared in-memory task store for testing 📦.

    Provides a fresh dict for each test, shared between the test
    application and test functions that need direct store access.
    """
    return {}


@pytest.fixture
def test_app(
    mock_task_runner: MagicMock,
    test_task_store: dict[str, Any],
) -> Any:
    """FastAPI test application with dependency overrides 🌐.

    Overrides the task_runner dependency to use mock objects,
    allowing API endpoint testing without real infrastructure.
    """
    from inquiro.api.app import create_app
    from inquiro.api.dependencies import get_task_runner, get_task_store

    app = create_app()

    # 🔧 Override dependencies
    app.dependency_overrides[get_task_runner] = lambda: mock_task_runner
    app.dependency_overrides[get_task_store] = lambda: test_task_store

    # 🎯 Set app state for the health endpoint
    app.state.task_runner = mock_task_runner

    return app


@pytest_asyncio.fixture
async def async_client(test_app: Any):
    """httpx AsyncClient for async API testing 🌐.

    Provides an async HTTP client bound to the test FastAPI application.
    Use for testing all API endpoints without starting a real server.
    """
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
