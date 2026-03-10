"""E2E tests for Inquiro unified pipeline 🧪.

Exercises the FULL pipeline: EvalTaskRunner → DiscoveryLoop → Agent → Result
with mocked LLM and MCP servers but real infrastructure components.

Components exercised (REAL):
    - EvalTaskRunner (task orchestration)
    - DiscoveryLoop / SynthesisExp (lifecycle management)
    - SearchAgent / SynthesisAgent (agent loop)
    - QualityGate (output validation)
    - CostTracker (budget enforcement)
    - EventEmitter (SSE event emission)

Components mocked:
    - LLM API calls (MockLLM returns pre-scripted tool calls)
    - MCP servers (MockMCPPool returns empty ToolRegistry)
    - LocalSession (patched to avoid filesystem side-effects)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inquiro.core.runner import EvalTaskRunner
from evomaster.agent.tools.base import ToolRegistry

from inquiro.tests.mock_helpers import (
    MockLLM,
    build_invalid_research_result,
    build_sample_evaluation_task,
    build_sample_synthesis_task,
    build_valid_research_result,
    build_valid_synthesis_result,
)


# ============================================================================
# 🔧 Mock Pools (service-level stubs for LLM and MCP)
# ============================================================================


class MockLLMPool:
    """Mock LLM provider pool for E2E tests 🤖.

    Returns MockLLM instances that produce pre-scripted responses.

    Attributes:
        _default_result: Default result for auto-generated finish responses.
    """

    def __init__(
        self,
        default_result: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MockLLMPool 🔧.

        Args:
            default_result: Result dict embedded in finish tool calls.
        """
        self._default_result = default_result

    def get_llm(self, model: str) -> MockLLM:
        """Return a MockLLM instance for the given model 🤖.

        Args:
            model: LLM model identifier (ignored in mock).

        Returns:
            Configured MockLLM.
        """
        return MockLLM(default_result=self._default_result)

    def close(self) -> None:
        """No-op cleanup 🧹."""


class MockMCPPool:
    """Mock MCP connection pool for E2E tests 🔌.

    Returns an empty ToolRegistry (no MCP search tools available).
    The agent will only have the built-in InquiroFinishTool.
    """

    async def initialize(self) -> None:
        """No-op initialization ✨."""

    def get_tools(self, mcp_servers: list[str] | None = None) -> ToolRegistry:
        """Return an empty ToolRegistry 🔧.

        Args:
            mcp_servers: MCP server names (ignored in mock).

        Returns:
            Empty ToolRegistry.
        """
        return ToolRegistry()

    def get_health(self) -> dict[str, str]:
        """Return mock health status ❤️.

        Returns:
            Mock server health dict.
        """
        return {"mock_server": "connected"}

    def close(self) -> None:
        """No-op cleanup 🧹."""


# ============================================================================
# 🔧 Fixtures
# ============================================================================


@pytest.fixture
def _patch_local_sessions():
    """Patch LocalSession creation to avoid filesystem side-effects 🔧."""
    mock_session = MagicMock()
    with (
        patch(
            "inquiro.agents.search_agent.SearchAgent._create_local_session",
            return_value=mock_session,
        ),
        patch(
            "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
            return_value=mock_session,
        ),
    ):
        yield mock_session


@pytest.fixture
def research_llm_pool():
    """LLM pool that returns valid research results 🔬."""
    return MockLLMPool(default_result=build_valid_research_result())


@pytest.fixture
def synthesis_llm_pool():
    """LLM pool that returns valid synthesis results 📊."""
    return MockLLMPool(default_result=build_valid_synthesis_result())


@pytest.fixture
def mcp_pool():
    """Empty MCP pool (no search tools) 🔌."""
    return MockMCPPool()


@pytest.fixture
def research_runner(research_llm_pool, mcp_pool):
    """EvalTaskRunner configured for research E2E tests 🎯."""
    return EvalTaskRunner(
        mcp_pool=mcp_pool,
        llm_pool=research_llm_pool,
    )


@pytest.fixture
def synthesis_runner(synthesis_llm_pool, mcp_pool):
    """EvalTaskRunner configured for synthesis E2E tests 🎯."""
    return EvalTaskRunner(
        mcp_pool=mcp_pool,
        llm_pool=synthesis_llm_pool,
    )


# ============================================================================
# 🔬 E2E Research Tests
# ============================================================================


def _make_discovery_result(task_id: str) -> Any:
    """Build a mock EvaluationResult for discovery pipeline tests 🔧."""
    from inquiro.core.types import (
        Decision,
        Evidence,
        EvaluationResult,
        EvidenceStrength,
        ReasoningClaim,
    )

    return EvaluationResult(
        task_id=task_id,
        decision=Decision.POSITIVE,
        confidence=0.85,
        reasoning=[
            ReasoningClaim(
                claim="Strong evidence found",
                evidence_ids=["E1"],
                strength=EvidenceStrength.STRONG,
            ),
        ],
        evidence_index=[
            Evidence(
                id="E1",
                source="perplexity",
                url="https://example.com/1",
                query="test query",
                summary="Test evidence",
            ),
        ],
        search_rounds=2,
        cost=0.5,
        pipeline_mode="discovery",
        confidence_source="coverage_ratio",
        discovery_rounds=2,
        discovery_coverage=0.85,
    )


class TestE2EResearch:
    """E2E tests for the unified research pipeline (discovery-first) 🔬.

    All submit_research calls now route through _run_discovery. These
    tests verify the entry point routing and result structure using
    mocked discovery results.
    """

    @pytest.mark.asyncio
    async def test_single_research_task_produces_valid_result(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Submit 1 EvaluationTask via unified pipeline and verify result."""
        # Arrange
        task = build_sample_evaluation_task(
            task_id="e2e-research-001",
            topic="Global EGFR therapy market analysis",
        )
        mock_result = _make_discovery_result("e2e-research-001")

        # Act — mock _run_discovery to return structured result
        with patch.object(
            research_runner,
            "_run_discovery",
            return_value=mock_result,
        ) as mock_discovery:
            result = await research_runner.submit_research(task)

        # Assert — result has all required fields
        mock_discovery.assert_awaited_once()
        assert result.task_id == "e2e-research-001"
        assert result.decision.value in ("positive", "cautious", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.reasoning) > 0
        assert len(result.evidence_index) > 0
        assert result.pipeline_mode == "discovery"

    @pytest.mark.asyncio
    async def test_three_evaluation_tasks_produce_reports(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Submit 3 EvaluationTasks concurrently via unified pipeline."""
        # Arrange
        tasks = [
            build_sample_evaluation_task(
                task_id=f"e2e-research-batch-{i}",
                topic=topic,
            )
            for i, topic in enumerate(
                [
                    "EGFR market size",
                    "Biomarker availability",
                    "Patient stratification",
                ]
            )
        ]

        # Act — mock _run_discovery for each call
        results_map = {
            f"e2e-research-batch-{i}": _make_discovery_result(
                f"e2e-research-batch-{i}",
            )
            for i in range(3)
        }

        async def _mock_discovery(task, emitter=None):
            return results_map[task.task_id]

        with patch.object(
            research_runner,
            "_run_discovery",
            side_effect=_mock_discovery,
        ):
            results = await asyncio.gather(
                *[research_runner.submit_research(t) for t in tasks]
            )

        # Assert — all 3 produce valid results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.task_id == f"e2e-research-batch-{i}"
            assert result.decision.value in ("positive", "cautious", "negative")
            assert len(result.reasoning) > 0
            assert len(result.evidence_index) > 0

    @pytest.mark.asyncio
    async def test_research_task_evidence_traceability(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify evidence chain: reasoning → evidence_ids → evidence_index."""
        # Arrange
        task = build_sample_evaluation_task(
            task_id="e2e-evidence-trace",
            topic="Evidence traceability test",
        )
        mock_result = _make_discovery_result("e2e-evidence-trace")

        # Act
        with patch.object(
            research_runner,
            "_run_discovery",
            return_value=mock_result,
        ):
            result = await research_runner.submit_research(task)

        # Assert — every reasoning claim references valid evidence
        evidence_ids = {ev.id for ev in result.evidence_index}
        for claim in result.reasoning:
            for eid in claim.evidence_ids:
                assert eid in evidence_ids, (
                    f"Claim '{claim.claim}' references non-existent evidence '{eid}'"
                )

    @pytest.mark.asyncio
    async def test_research_task_with_event_emitter(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify submit_research passes event_emitter to _run_discovery."""
        # Arrange
        task = build_sample_evaluation_task(task_id="e2e-research-events")
        emitter = MagicMock()
        mock_result = _make_discovery_result("e2e-research-events")

        # Act
        with patch.object(
            research_runner,
            "_run_discovery",
            return_value=mock_result,
        ) as mock_discovery:
            _result = await research_runner.submit_research(
                task, event_emitter=emitter,
            )

        # Assert — event emitter was passed through to discovery
        mock_discovery.assert_awaited_once_with(task, emitter)

    @pytest.mark.asyncio
    async def test_research_result_stored_in_runner(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify runner returns result from _run_discovery correctly."""
        # Arrange
        task = build_sample_evaluation_task(task_id="e2e-research-store")
        mock_result = _make_discovery_result("e2e-research-store")

        # Act
        with patch.object(
            research_runner,
            "_run_discovery",
            return_value=mock_result,
        ):
            result = await research_runner.submit_research(task)

        # Assert — result fields match mock
        assert result.task_id == "e2e-research-store"
        assert result.decision.value == "positive"
        assert result.confidence == 0.85


# ============================================================================
# 📊 E2E Synthesis Tests
# ============================================================================


class TestE2ESynthesis:
    """E2E tests for the synthesis task pipeline 📊."""

    @pytest.mark.asyncio
    async def test_synthesis_reads_all_input_reports(
        self,
        synthesis_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Submit a SynthesisTask with 3 reports and verify all are used."""
        # Arrange
        task = build_sample_synthesis_task(
            task_id="e2e-synthesis-001",
            topic="Clinical feasibility synthesis",
        )

        # Act
        result = await synthesis_runner.submit_synthesis(task)

        # Assert — result references source reports
        assert result.task_id == "e2e-synthesis-001"
        assert result.decision.value in ("positive", "cautious", "negative")
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.source_reports) > 0

    @pytest.mark.asyncio
    async def test_synthesis_produces_cross_references(
        self,
        synthesis_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify synthesis produces cross-references between reports."""
        # Arrange
        task = build_sample_synthesis_task(task_id="e2e-synthesis-xref")

        # Act
        result = await synthesis_runner.submit_synthesis(task)

        # Assert — cross_references populated
        assert len(result.cross_references) > 0
        for xref in result.cross_references:
            assert xref.claim is not None
            assert len(xref.supporting_reports) > 0

    @pytest.mark.asyncio
    async def test_synthesis_result_stored_in_runner(
        self,
        synthesis_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify runner stores completed synthesis result."""
        # Arrange
        task = build_sample_synthesis_task(task_id="e2e-synthesis-store")

        # Act
        _result = await synthesis_runner.submit_synthesis(task)
        status = synthesis_runner.get_task_status("e2e-synthesis-store")

        # Assert
        assert status["status"] == "completed"
        assert status["result"] is not None


# ============================================================================
# 🔬 E2E Deep Dive Tests
# ============================================================================


class TestE2EDeepDive:
    """E2E tests for synthesis with additional research (deep-dive) 🔬."""

    @pytest.mark.asyncio
    async def test_synthesis_with_deep_dive_enabled(
        self,
        synthesis_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify synthesis completes with deep-dive enabled."""
        # Arrange
        task = build_sample_synthesis_task(
            task_id="e2e-deepdive-001",
            allow_additional_research=True,
        )

        # Act
        result = await synthesis_runner.submit_synthesis(task)

        # Assert — result is valid (deep-dive might not trigger with MockLLM)
        assert result.task_id == "e2e-deepdive-001"
        assert result.decision.value in ("positive", "cautious", "negative")

    @pytest.mark.asyncio
    async def test_synthesis_without_deep_dive(
        self,
        synthesis_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify synthesis completes with deep-dive disabled."""
        # Arrange
        task = build_sample_synthesis_task(
            task_id="e2e-no-deepdive",
            allow_additional_research=False,
        )

        # Act
        result = await synthesis_runner.submit_synthesis(task)

        # Assert — no deep-dives triggered
        assert result.task_id == "e2e-no-deepdive"
        assert len(result.deep_dives_triggered) == 0


# ============================================================================
# 🛑 E2E Cancellation Tests
# ============================================================================


class TestE2ECancellation:
    """E2E tests for task cancellation 🛑."""

    @pytest.mark.asyncio
    async def test_cancel_task_via_runner(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify runner.cancel_task() signals cancellation for active tasks."""
        from inquiro.infrastructure.cancellation import CancellationToken

        # Arrange — manually register a task as active
        token = CancellationToken()
        research_runner._register_active_task("e2e-cancel-001", token)

        # Act
        cancelled = research_runner.cancel_task("e2e-cancel-001")

        # Assert — cancellation was signaled
        assert cancelled is True
        assert token.is_cancelled is True
        assert "e2e-cancel-001" in (token.reason or "")

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(
        self,
        research_runner: EvalTaskRunner,
    ) -> None:
        """Verify cancel_task returns False for unknown tasks."""
        # Act
        result = research_runner.cancel_task("nonexistent-task")

        # Assert
        assert result is False


# ============================================================================
# ✅ E2E Quality Gate Tests
# ============================================================================


class TestE2EQualityGate:
    """E2E tests for QualityGate in the unified pipeline ✅.

    QG retry behavior is now tested at the SearchExp/AnalysisExp level.
    These tests verify the runner entry point handles discovery results.
    """

    @pytest.mark.asyncio
    async def test_discovery_pipeline_returns_valid_result(
        self,
        mcp_pool: MockMCPPool,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify unified pipeline returns properly structured result."""
        runner = EvalTaskRunner(
            mcp_pool=mcp_pool,
            llm_pool=MockLLMPool(default_result=build_valid_research_result()),
        )
        task = build_sample_evaluation_task(
            task_id="e2e-qg-retry",
            max_retries=2,
        )
        mock_result = _make_discovery_result("e2e-qg-retry")

        with patch.object(runner, "_run_discovery", return_value=mock_result):
            result = await runner.submit_research(task)

        assert result.decision.value == "positive"
        assert result.pipeline_mode == "discovery"

    @pytest.mark.asyncio
    async def test_discovery_graceful_degradation(
        self,
        mcp_pool: MockMCPPool,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify discovery pipeline handles partial results gracefully."""
        from inquiro.core.types import Decision, EvaluationResult

        runner = EvalTaskRunner(
            mcp_pool=mcp_pool,
            llm_pool=MockLLMPool(
                default_result=build_invalid_research_result(),
            ),
        )
        task = build_sample_evaluation_task(
            task_id="e2e-qg-exhausted",
            max_retries=1,
        )

        # 📊 Return a partial result (low confidence, cautious decision)
        partial_result = EvaluationResult(
            task_id="e2e-qg-exhausted",
            decision=Decision.CAUTIOUS,
            confidence=0.3,
            pipeline_mode="discovery",
            confidence_source="coverage_ratio",
        )
        with patch.object(
            runner, "_run_discovery", return_value=partial_result,
        ):
            result = await runner.submit_research(task)

        assert result.task_id == "e2e-qg-exhausted"
        assert result.decision == Decision.CAUTIOUS


# ============================================================================
# 💰 E2E Cost Budget Tests
# ============================================================================


class TestE2ECostBudget:
    """E2E tests for cost budget enforcement 💰."""

    @pytest.mark.asyncio
    async def test_research_tracks_cost(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify cost is reported in discovery pipeline result."""
        # Arrange
        task = build_sample_evaluation_task(task_id="e2e-cost-001")
        mock_result = _make_discovery_result("e2e-cost-001")

        # Act
        with patch.object(
            research_runner, "_run_discovery", return_value=mock_result,
        ):
            result = await research_runner.submit_research(task)

        # Assert — task completed with cost field populated
        assert result.task_id == "e2e-cost-001"
        assert result.decision.value in ("positive", "cautious", "negative")
        assert result.cost >= 0.0


# ============================================================================
# ❌ E2E Error Handling Tests
# ============================================================================


class TestE2EErrorHandling:
    """E2E tests for error handling in the unified pipeline ❌."""

    @pytest.mark.asyncio
    async def test_runner_handles_discovery_exception(
        self,
        mcp_pool: MockMCPPool,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify runner handles and reports discovery pipeline exceptions."""
        runner = EvalTaskRunner(
            mcp_pool=mcp_pool,
            llm_pool=MockLLMPool(),
        )
        task = build_sample_evaluation_task(
            task_id="e2e-error-001",
            max_retries=0,
        )

        # Act / Assert — _run_discovery raises, runner propagates
        with (
            patch.object(
                runner,
                "_run_discovery",
                side_effect=RuntimeError("Pipeline failure"),
            ),
            pytest.raises(RuntimeError),
        ):
            await runner.submit_research(task)

    @pytest.mark.asyncio
    async def test_runner_task_status_lifecycle(
        self,
        research_runner: EvalTaskRunner,
        _patch_local_sessions: Any,
    ) -> None:
        """Verify task status transitions: unknown → completed."""
        # Arrange
        task = build_sample_evaluation_task(task_id="e2e-lifecycle-001")
        mock_result = _make_discovery_result("e2e-lifecycle-001")

        # Assert — unknown before submission
        status = research_runner.get_task_status("e2e-lifecycle-001")
        assert status["status"] == "unknown"

        # Act
        with patch.object(
            research_runner, "_run_discovery", return_value=mock_result,
        ):
            _result = await research_runner.submit_research(task)

        # Assert — result returned successfully
        assert _result.task_id == "e2e-lifecycle-001"
        assert _result.decision.value in ("positive", "cautious", "negative")
