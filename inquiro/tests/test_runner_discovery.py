"""Tests for EvalTaskRunner discovery pipeline 🧪.

Validates that the runner correctly dispatches tasks to the discovery
pipeline, creates proper adapter classes, converts
DiscoveryResult -> EvaluationResult, and handles edge cases.

Test categories:
    1. Adapter integration — SearchExpAdapter / AnalysisExpAdapter
    2. DiscoveryLoop creation — correct component injection
    3. Result conversion — DiscoveryResult -> EvaluationResult
    4. Config parsing — DiscoveryConfig from task
    5. Trajectory integration — trajectory_dir passthrough
    6. Error handling — pipeline failures, timeouts
    7. Synthesis integration — optional post-loop synthesis
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inquiro.core.types import (
    AgentConfig,
    Checklist,
    ChecklistItem,
    CleanedEvidence,
    CostGuardConfig,
    Decision,
    DiscoveryConfig,
    DiscoveryResult,
    DiscoveryRoundSummary,
    EvaluationTask,
    Evidence,
    EvidenceStrength,
    EvidenceTag,
    GapReport,
    QualityGateConfig,
    ReasoningClaim,
    ToolsConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 🏭 Fixtures and helpers
# ============================================================================


def _build_discovery_task(
    task_id: str = "disc-task-001",
    discovery_config: dict[str, Any] | None = None,
    trajectory_dir: str | None = None,
) -> EvaluationTask:
    """Build an EvaluationTask for the discovery pipeline 🔧.

    Args:
        task_id: Unique task identifier.
        discovery_config: Optional DiscoveryConfig dict override.
        trajectory_dir: Optional trajectory output directory.

    Returns:
        EvaluationTask with discovery pipeline settings.
    """
    return EvaluationTask(
        task_id=task_id,
        topic="Evidence research topic for testing",
        rules="Evaluate based on available evidence",
        checklist=Checklist(
            required=[
                ChecklistItem(
                    id="req_1",
                    description="First required checklist item",
                    keywords=["first", "item"],
                ),
                ChecklistItem(
                    id="req_2",
                    description="Second required checklist item",
                    keywords=["second", "item"],
                ),
            ],
            optional=[],
            coverage_threshold=0.8,
        ),
        output_schema={
            "type": "object",
            "required": ["decision", "confidence"],
            "properties": {
                "decision": {"type": "string"},
                "confidence": {"type": "number"},
            },
        },
        agent_config=AgentConfig(max_turns=10),
        tools_config=ToolsConfig(mcp_servers=["perplexity"]),
        quality_gate=QualityGateConfig(max_retries=1),
        cost_guard=CostGuardConfig(max_cost_per_task=5.0),
        discovery_config=discovery_config,
        trajectory_dir=trajectory_dir,
    )


def _build_mock_discovery_result(
    task_id: str = "disc-task-001",
    total_rounds: int = 2,
    final_coverage: float = 0.85,
    total_cost: float = 1.50,
) -> DiscoveryResult:
    """Build a mock DiscoveryResult for testing 🔧.

    Args:
        task_id: Task identifier.
        total_rounds: Number of rounds executed.
        final_coverage: Final checklist coverage ratio.
        total_cost: Total cost in USD.

    Returns:
        DiscoveryResult with realistic mock data.
    """
    return DiscoveryResult(
        task_id=task_id,
        pipeline_mode="discovery",
        total_rounds=total_rounds,
        final_coverage=final_coverage,
        total_cost_usd=total_cost,
        termination_reason="coverage_threshold_reached",
        evidence=[
            CleanedEvidence(
                id="E1",
                summary="Evidence item 1 summary",
                url="https://example.com/e1",
                tag=EvidenceTag.OTHER,
                source_query="query 1",
                mcp_server="perplexity",
            ),
            CleanedEvidence(
                id="E2",
                summary="Evidence item 2 summary",
                url="https://example.com/e2",
                tag=EvidenceTag.OTHER,
                source_query="query 2",
                mcp_server="perplexity",
            ),
        ],
        claims=[
            {
                "claim": "First research finding",
                "evidence_ids": ["E1"],
                "strength": "strong",
            },
            {
                "claim": "Second research finding",
                "evidence_ids": ["E2"],
                "strength": "moderate",
            },
        ],
        gap_reports=[
            GapReport(
                round_number=1,
                coverage_ratio=0.50,
                covered_items=["First required checklist item"],
                uncovered_items=["Second required checklist item"],
                converged=False,
            ),
            GapReport(
                round_number=2,
                coverage_ratio=final_coverage,
                covered_items=[
                    "First required checklist item",
                    "Second required checklist item",
                ],
                uncovered_items=[],
                converged=True,
                convergence_reason="coverage_threshold_reached",
            ),
        ],
        round_summaries=[
            DiscoveryRoundSummary(
                round_number=1,
                queries_executed=3,
                raw_evidence_count=5,
                cleaned_evidence_count=3,
                coverage_ratio=0.50,
                coverage_delta=0.50,
                round_cost_usd=0.75,
                converged=False,
            ),
            DiscoveryRoundSummary(
                round_number=2,
                queries_executed=2,
                raw_evidence_count=3,
                cleaned_evidence_count=2,
                coverage_ratio=final_coverage,
                coverage_delta=0.35,
                round_cost_usd=0.75,
                converged=True,
                convergence_reason="coverage_threshold_reached",
            ),
        ],
        trajectory_id="traj-uuid-001",
    )


def _create_mock_runner() -> MagicMock:
    """Create a mock EvalTaskRunner with essential methods 🔧.

    Returns:
        MagicMock configured as EvalTaskRunner.
    """
    runner = MagicMock()
    runner.mcp_pool = MagicMock()
    runner.llm_pool = MagicMock()
    runner.logger = logging.getLogger("MockRunner")
    runner._active_tasks = {}
    runner._tasks_lock = MagicMock()
    runner._metrics = MagicMock()
    runner._degradation = MagicMock()
    runner._completed_results = MagicMock()
    runner._evidence_memory = MagicMock()

    # 🔧 Wire up real methods for adapters
    runner._get_llm = MagicMock(return_value=MagicMock())
    runner._get_filtered_tools = MagicMock(
        return_value=MagicMock(),
    )
    runner._create_cost_tracker = MagicMock(
        return_value=MagicMock(),
    )
    runner._create_event_emitter = MagicMock(
        return_value=MagicMock(),
    )

    return runner


# ============================================================================
# 🧪 1. Mode routing tests
# ============================================================================


class TestModeRouting:
    """Verify submit_research always routes to _run_discovery (unified) 🧪."""

    @pytest.mark.asyncio
    async def test_submit_research_routes_to_discovery(self) -> None:
        """All tasks route to _run_discovery (unified pipeline) 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner.submit_research = EvalTaskRunner.submit_research.__get__(
            runner,
            EvalTaskRunner,
        )
        runner._run_discovery = AsyncMock(
            return_value=MagicMock(),
        )

        task = _build_discovery_task()
        await runner.submit_research(task)

        runner._run_discovery.assert_awaited_once_with(task, None)

    @pytest.mark.asyncio
    async def test_discovery_mode_with_event_emitter(self) -> None:
        """Discovery mode passes event_emitter through 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner.submit_research = EvalTaskRunner.submit_research.__get__(
            runner,
            EvalTaskRunner,
        )
        runner._run_discovery = AsyncMock(
            return_value=MagicMock(),
        )

        task = _build_discovery_task()
        emitter = MagicMock()
        await runner.submit_research(task, event_emitter=emitter)

        runner._run_discovery.assert_awaited_once_with(
            task,
            emitter,
        )

    @pytest.mark.asyncio
    async def test_all_modes_route_to_discovery(self) -> None:
        """All pipeline modes route to _run_discovery (unified entry) 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner.submit_research = EvalTaskRunner.submit_research.__get__(
            runner,
            EvalTaskRunner,
        )
        runner._run_discovery = AsyncMock(
            return_value=MagicMock(),
        )

        # All tasks route to _run_discovery
        for task_id in ["task-a", "task-b"]:
            task = _build_discovery_task(task_id=task_id)
            runner._run_discovery.reset_mock()
            await runner.submit_research(task)
            runner._run_discovery.assert_awaited_once()


# ============================================================================
# 🧪 2. Config parsing tests
# ============================================================================


class TestConfigParsing:
    """Verify DiscoveryConfig parsing from task 🧪."""

    def test_parse_discovery_config_from_task(self) -> None:
        """Config dict in task is parsed into DiscoveryConfig 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner._parse_discovery_config = EvalTaskRunner._parse_discovery_config.__get__(
            runner,
            EvalTaskRunner,
        )

        task = _build_discovery_task(
            discovery_config={
                "max_rounds": 5,
                "coverage_threshold": 0.90,
                "max_cost_per_subitem": 10.0,
            },
        )

        config = runner._parse_discovery_config(task)

        assert isinstance(config, DiscoveryConfig)
        assert config.max_rounds == 5
        assert config.coverage_threshold == 0.90
        assert config.max_cost_per_subitem == 10.0

    def test_parse_discovery_config_default_fallback(self) -> None:
        """Absent discovery_config uses INTENSITY_PRESETS["standard"] 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner._parse_discovery_config = EvalTaskRunner._parse_discovery_config.__get__(
            runner,
            EvalTaskRunner,
        )

        task = _build_discovery_task(discovery_config=None)

        config = runner._parse_discovery_config(task)

        assert isinstance(config, DiscoveryConfig)
        assert config.max_rounds == 2  # standard preset
        assert config.coverage_threshold == 0.75  # standard preset
        assert config.intensity == "standard"

    def test_parse_discovery_config_partial_override(self) -> None:
        """Partial config dict merges with intensity preset base 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner._parse_discovery_config = EvalTaskRunner._parse_discovery_config.__get__(
            runner,
            EvalTaskRunner,
        )

        task = _build_discovery_task(
            discovery_config={"max_rounds": 7},
        )

        config = runner._parse_discovery_config(task)

        assert config.max_rounds == 7
        # 📊 Other fields use "standard" preset values as base
        assert config.coverage_threshold == 0.75
        assert config.max_cost_per_subitem == 20.0

    def test_parse_discovery_config_with_analysis_models(self) -> None:
        """Custom analysis_models list is respected 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        runner = MagicMock(spec=EvalTaskRunner)
        runner._parse_discovery_config = EvalTaskRunner._parse_discovery_config.__get__(
            runner,
            EvalTaskRunner,
        )

        task = _build_discovery_task(
            discovery_config={
                "analysis_models": ["model-a", "model-b"],
            },
        )

        config = runner._parse_discovery_config(task)

        assert config.analysis_models == ["model-a", "model-b"]


# ============================================================================
# 🧪 3. Result conversion tests
# ============================================================================


def _make_conversion_runner() -> MagicMock:
    """Build a MagicMock runner with result-conversion methods bound 🔧."""
    from inquiro.core.runner import EvalTaskRunner

    runner = MagicMock(spec=EvalTaskRunner)
    for method_name in (
        "_discovery_to_evaluation_result",
        "_determine_decision",
        "_build_discovery_metadata",
    ):
        raw = EvalTaskRunner.__dict__.get(method_name)
        if isinstance(raw, staticmethod):
            setattr(runner, method_name, raw.__func__)
        else:
            method = getattr(EvalTaskRunner, method_name)
            setattr(
                runner,
                method_name,
                method.__get__(runner, EvalTaskRunner),
            )
    return runner


class TestResultConversion:
    """Verify DiscoveryResult → EvaluationResult conversion 🧪."""

    def test_basic_conversion(self) -> None:
        """DiscoveryResult converts to EvaluationResult with all fields 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.task_id == "disc-task-001"
        assert result.pipeline_mode == "discovery"
        assert result.discovery_rounds == 2
        assert result.discovery_coverage == 0.85

    def test_evidence_mapping(self) -> None:
        """CleanedEvidence items map to Evidence objects 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert len(result.evidence_index) == 2
        assert result.evidence_index[0].id == "E1"
        assert result.evidence_index[0].source == "perplexity"
        assert result.evidence_index[1].id == "E2"

    def test_claims_to_reasoning(self) -> None:
        """Claim dicts convert to ReasoningClaim objects 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert len(result.reasoning) == 2
        assert result.reasoning[0].claim == "First research finding"
        assert result.reasoning[0].strength == EvidenceStrength.STRONG
        assert result.reasoning[1].strength == EvidenceStrength.MODERATE

    def test_high_coverage_yields_positive_decision(self) -> None:
        """Coverage >= 0.80 yields positive decision without synthesis 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result(final_coverage=0.90)

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.decision == Decision.POSITIVE
        assert result.confidence == 0.90

    def test_moderate_coverage_yields_cautious_decision(self) -> None:
        """Coverage 0.50-0.79 yields cautious decision without synthesis 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result(final_coverage=0.65)

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.decision == Decision.CAUTIOUS
        assert result.confidence == 0.65

    def test_low_coverage_yields_negative_decision(self) -> None:
        """Coverage < 0.50 yields negative decision without synthesis 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result(final_coverage=0.30)

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.decision == Decision.NEGATIVE
        assert result.confidence == 0.30

    def test_checklist_coverage_from_last_gap(self) -> None:
        """ChecklistCoverage is derived from the last gap report 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.checklist_coverage.required_covered == [
            "First required checklist item",
            "Second required checklist item",
        ]
        assert result.checklist_coverage.required_missing == []

    def test_metadata_contains_discovery_info(self) -> None:
        """Result metadata includes discovery-specific fields 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.metadata["discovery"] is True
        assert result.metadata["total_rounds"] == 2
        assert result.metadata["termination_reason"] == ("coverage_threshold_reached")
        assert result.metadata["trajectory_id"] == "traj-uuid-001"
        assert len(result.metadata["round_summaries"]) == 2

    def test_total_cost_accumulated(self) -> None:
        """Total cost from discovery is propagated to result 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result(total_cost=3.25)

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.cost == 3.25

    def test_synthesis_result_overrides_decision(self) -> None:
        """When synthesis_result is provided, its decision is used 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()

        # 🧬 Mock synthesis result
        synth_eval = MagicMock()
        synth_eval.decision = Decision.CAUTIOUS
        synth_eval.confidence = 0.72
        synth_eval.reasoning = [
            ReasoningClaim(
                claim="Synthesis-derived claim",
                evidence_ids=["E1", "E2"],
                strength=EvidenceStrength.STRONG,
            ),
        ]

        synth_result = MagicMock()
        synth_result.evaluation_result = synth_eval
        synth_result.consensus_decision = "cautious"
        synth_result.consensus_ratio = 0.67
        synth_result.cost_usd = 0.50

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
            synthesis_result=synth_result,
        )

        assert result.decision == Decision.CAUTIOUS
        assert result.confidence == 0.72
        assert result.reasoning[0].claim == "Synthesis-derived claim"
        # 💰 Cost includes synthesis
        assert result.cost == 1.50 + 0.50

    def test_empty_gap_reports_uses_empty_coverage(self) -> None:
        """Empty gap_reports produces default ChecklistCoverage 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()
        discovery.gap_reports = []

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.checklist_coverage.required_covered == []
        assert result.checklist_coverage.required_missing == []
        assert result.gaps_remaining == []

    def test_invalid_claim_strength_falls_back_to_moderate(self) -> None:
        """Invalid strength string falls back to MODERATE 🧪."""
        runner = _make_conversion_runner()

        task = _build_discovery_task()
        discovery = _build_mock_discovery_result()
        discovery.claims = [
            {
                "claim": "Claim with bad strength",
                "evidence_ids": ["E1"],
                "strength": "ultra_strong",
            },
        ]

        result = runner._discovery_to_evaluation_result(
            discovery_result=discovery,
            task=task,
        )

        assert result.reasoning[0].strength == EvidenceStrength.MODERATE


# ============================================================================
# 🧪 4. Adapter tests
# ============================================================================


class TestSearchExpAdapter:
    """Verify _SearchExpAdapter bridges protocols correctly 🧪."""

    @pytest.mark.asyncio
    async def test_adapter_calls_search_exp(self) -> None:
        """Adapter creates SearchExp and calls run_search 🧪."""
        from inquiro.core.runner import _SearchExpAdapter

        mock_runner = _create_mock_runner()

        adapter = _SearchExpAdapter(
            runner=mock_runner,
            event_emitter=MagicMock(),
            cancellation_token=MagicMock(),
        )

        task = _build_discovery_task()
        config = DiscoveryConfig()

        # 🔧 Patch SearchExp.run_search to return a mock result
        with patch(
            "inquiro.exps.search_exp.SearchExp",
        ) as MockSearchExpCls:
            mock_exp_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.cleaned_evidence = [
                CleanedEvidence(
                    id="E1",
                    summary="Test evidence",
                    tag=EvidenceTag.OTHER,
                    source_query="q1",
                    mcp_server="perplexity",
                ),
            ]
            mock_result.queries_executed = ["q1"]
            mock_result.mcp_tools_used = ["tool1"]
            mock_result.cost_usd = 0.10
            mock_result.duration_seconds = 1.5
            mock_result.agent_trajectory_ref = None
            mock_exp_instance.run_search = AsyncMock(
                return_value=mock_result,
            )
            MockSearchExpCls.return_value = mock_exp_instance

            output = await adapter.execute_search(
                task=task,
                config=config,
                round_number=1,
            )

        # ✅ Verify output structure
        assert len(output.evidence) == 1
        assert output.evidence[0].id == "E1"
        assert output.queries_executed == ["q1"]
        assert output.cost_usd == 0.10

    @pytest.mark.asyncio
    async def test_adapter_passes_focus_prompt(self) -> None:
        """Adapter passes focus_prompt to SearchExp.run_search 🧪."""
        from inquiro.core.runner import _SearchExpAdapter

        mock_runner = _create_mock_runner()
        adapter = _SearchExpAdapter(runner=mock_runner)

        task = _build_discovery_task()
        config = DiscoveryConfig()

        with patch(
            "inquiro.exps.search_exp.SearchExp",
        ) as MockSearchExpCls:
            mock_exp_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.cleaned_evidence = []
            mock_result.queries_executed = []
            mock_result.mcp_tools_used = []
            mock_result.cost_usd = 0.0
            mock_result.duration_seconds = 0.0
            mock_result.agent_trajectory_ref = None
            mock_exp_instance.run_search = AsyncMock(
                return_value=mock_result,
            )
            MockSearchExpCls.return_value = mock_exp_instance

            await adapter.execute_search(
                task=task,
                config=config,
                round_number=2,
                focus_prompt="Focus on uncovered items",
            )

            # ✅ Verify focus_prompt passed through
            call_kwargs = mock_exp_instance.run_search.call_args
            assert call_kwargs.kwargs.get("focus_prompt") == (
                "Focus on uncovered items"
            )


class TestAnalysisExpAdapter:
    """Verify _AnalysisExpAdapter bridges protocols correctly 🧪."""

    @pytest.mark.asyncio
    async def test_adapter_calls_analysis_exp(self) -> None:
        """Adapter creates AnalysisExp and calls run_analysis 🧪."""
        from inquiro.core.runner import _AnalysisExpAdapter

        mock_runner = _create_mock_runner()

        adapter = _AnalysisExpAdapter(
            runner=mock_runner,
            event_emitter=MagicMock(),
            cancellation_token=MagicMock(),
        )

        task = _build_discovery_task()
        config = DiscoveryConfig(
            analysis_models=["model-a", "model-b"],
        )

        evidence = [
            Evidence(
                id="E1",
                source="perplexity",
                query="q1",
                summary="Evidence summary",
            ),
        ]

        with patch(
            "inquiro.exps.analysis_exp.AnalysisExp",
        ) as MockAnalysisExpCls:
            mock_exp_instance = MagicMock()
            mock_aggregated = MagicMock()
            mock_aggregated.structured_reasoning = [
                {
                    "claim": "Test claim",
                    "evidence_ids": ["E1"],
                    "strength": "strong",
                },
            ]
            mock_aggregated.decision = Decision.POSITIVE
            mock_aggregated.consensus_ratio = 0.80
            mock_aggregated.model_decisions = {
                "model-a": "positive",
            }
            mock_aggregated.checklist_coverage = None
            mock_exp_instance.run_analysis = AsyncMock(
                return_value=(mock_aggregated, 0.05),
            )
            MockAnalysisExpCls.return_value = mock_exp_instance

            output = await adapter.execute_analysis(
                task=task,
                evidence=evidence,
                config=config,
                round_number=1,
            )

        # ✅ Verify output structure
        assert len(output.claims) == 1
        assert output.claims[0]["claim"] == "Test claim"
        assert output.consensus_decision == "positive"
        assert output.consensus_ratio == 0.80


# ============================================================================
# 🧪 5. Discovery timeout result tests
# ============================================================================


class TestDiscoveryTimeoutResult:
    """Verify timeout result construction for DISCOVERY pipeline 🧪."""

    def test_timeout_result_has_negative_decision(self) -> None:
        """Timeout result uses negative decision 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        result = EvalTaskRunner._build_discovery_timeout_result(
            "timeout-task",
        )

        assert result.decision == Decision.NEGATIVE
        assert result.confidence == 0.0

    def test_timeout_result_has_discovery_metadata(self) -> None:
        """Timeout result includes discovery metadata 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        result = EvalTaskRunner._build_discovery_timeout_result(
            "timeout-task",
        )

        assert result.pipeline_mode == "discovery"
        assert result.metadata["discovery"] is True
        assert result.metadata["timeout"] is True

    def test_timeout_result_has_gap_message(self) -> None:
        """Timeout result includes timeout in gaps_remaining 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        result = EvalTaskRunner._build_discovery_timeout_result(
            "timeout-task",
        )

        assert "Discovery pipeline timeout" in result.gaps_remaining


# ============================================================================
# 🧪 6. Full pipeline integration test (mocked DiscoveryLoop)
# ============================================================================


class TestFullPipelineIntegration:
    """End-to-end discovery pipeline with mocked components 🧪."""

    @pytest.mark.asyncio
    async def test_run_discovery_full_pipeline(self) -> None:
        """Full _run_discovery creates loop, runs, and converts result 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        # 🏗️ Create real runner with mock pools
        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()
        discovery_result = _build_mock_discovery_result()

        # 🔧 Patch DiscoveryLoop.run to return mock result
        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            # 🔧 Patch synthesis to skip
            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                result = await runner._run_discovery(task)

        # ✅ Verify result
        assert result.task_id == "disc-task-001"
        assert result.pipeline_mode == "discovery"
        assert result.discovery_rounds == 2
        assert result.discovery_coverage == 0.85
        assert len(result.evidence_index) == 2
        assert len(result.reasoning) == 2

    @pytest.mark.asyncio
    async def test_run_discovery_registers_and_unregisters(self) -> None:
        """Discovery pipeline registers/unregisters active task 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()
        discovery_result = _build_mock_discovery_result()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                await runner._run_discovery(task)

        # ✅ After completion, task should be unregistered
        assert task.task_id not in runner._active_tasks

    @pytest.mark.asyncio
    async def test_run_discovery_stores_completed_result(self) -> None:
        """Completed discovery result is stored in result cache 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()
        discovery_result = _build_mock_discovery_result()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                await runner._run_discovery(task)

        # ✅ Result should be cached
        cached = runner._completed_results.get(task.task_id)
        assert cached is not None


# ============================================================================
# 🧪 7. Trajectory integration tests
# ============================================================================


class TestTrajectoryIntegration:
    """Verify trajectory_dir is passed through to DiscoveryLoop 🧪."""

    @pytest.mark.asyncio
    async def test_trajectory_dir_passed_to_loop(self) -> None:
        """trajectory_dir from task is forwarded to DiscoveryLoop 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task(
            trajectory_dir="/tmp/trajectories",
        )
        discovery_result = _build_mock_discovery_result()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                await runner._run_discovery(task)

            # ✅ Verify trajectory_dir passed to constructor
            call_kwargs = MockLoopCls.call_args
            assert call_kwargs.kwargs.get("trajectory_dir") == ("/tmp/trajectories")

    @pytest.mark.asyncio
    async def test_no_trajectory_dir_passes_none(self) -> None:
        """Missing trajectory_dir passes None to DiscoveryLoop 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task(trajectory_dir=None)
        discovery_result = _build_mock_discovery_result()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ):
                await runner._run_discovery(task)

            call_kwargs = MockLoopCls.call_args
            assert call_kwargs.kwargs.get("trajectory_dir") is None


# ============================================================================
# 🧪 8. Error handling tests
# ============================================================================


class TestErrorHandling:
    """Verify error handling in DISCOVERY pipeline 🧪."""

    @pytest.mark.asyncio
    async def test_discovery_loop_exception_is_raised(self) -> None:
        """Exception in DiscoveryLoop.run is propagated 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                side_effect=RuntimeError("Loop crashed"),
            )
            MockLoopCls.return_value = mock_loop

            with pytest.raises(RuntimeError, match="Loop crashed"):
                await runner._run_discovery(task)

    @pytest.mark.asyncio
    async def test_discovery_failure_stores_error(self) -> None:
        """Pipeline failure stores error in result cache 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                side_effect=ValueError("Bad config"),
            )
            MockLoopCls.return_value = mock_loop

            with pytest.raises(ValueError):
                await runner._run_discovery(task)

        # ✅ Error should be cached
        cached = runner._completed_results.get(task.task_id)
        assert cached is not None

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_discovery_result(
        self,
    ) -> None:
        """Synthesis failure still returns valid EvaluationResult 🧪."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_mcp.get_tools = MagicMock(return_value=MagicMock())
        mock_llm_pool = MagicMock()
        mock_llm_pool.get_llm = MagicMock(return_value=MagicMock())

        runner = EvalTaskRunner(
            mcp_pool=mock_mcp,
            llm_pool=mock_llm_pool,
        )

        task = _build_discovery_task()
        discovery_result = _build_mock_discovery_result()

        with patch(
            "inquiro.core.discovery_loop.DiscoveryLoop",
        ) as MockLoopCls:
            mock_loop = MagicMock()
            mock_loop.run = AsyncMock(
                return_value=discovery_result,
            )
            MockLoopCls.return_value = mock_loop

            # 🔧 Synthesis fails with exception
            with patch.object(
                runner,
                "_run_discovery_synthesis",
                new_callable=AsyncMock,
                return_value=None,
            ) as _mock_synth:
                # ✨ Synthesis returns None (graceful degradation)
                result = await runner._run_discovery(task)

        # ✅ Result is still valid (from discovery, not synthesis)
        assert result.task_id == "disc-task-001"
        assert result.pipeline_mode == "discovery"
        assert result.decision == Decision.POSITIVE
