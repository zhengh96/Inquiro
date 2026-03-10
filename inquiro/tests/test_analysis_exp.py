"""Integration tests for AnalysisExp lifecycle 🧪.

Tests cover:
    - AnalysisExp initialization and inheritance
    - Prompt rendering (system prompt, user prompt)
    - Evidence formatting for prompt injection
    - Round context formatting
    - JSON extraction from LLM responses
    - Response parsing into EvaluationResult
    - Parallel execution setup with asyncio.gather
    - Aggregation flow with mocked model results
    - Fallback behavior when all models fail
    - Cancellation behavior
    - Cost extraction from response metadata
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from inquiro.core.aggregation import AggregatedResult, AggregationEngine
from inquiro.core.types import (
    CleanedEvidence,
    Decision,
    EvidenceTag,
    EvaluationResult,
)
from inquiro.exps.analysis_exp import AnalysisExp
from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.infrastructure.cancellation import (
    CancellationToken,
    CancelledError,
)
from inquiro.infrastructure.cost_tracker import CostTracker

from inquiro.tests.mock_helpers import (
    build_sample_evaluation_task,
)


# ============================================================================
# 🏭 Test Fixtures and Helpers
# ============================================================================


def _build_cleaned_evidence(count: int = 3) -> list[CleanedEvidence]:
    """Build a list of sample cleaned evidence items 🧹.

    Args:
        count: Number of evidence items to generate.

    Returns:
        List of CleanedEvidence instances.
    """
    items: list[CleanedEvidence] = []
    for i in range(1, count + 1):
        items.append(
            CleanedEvidence(
                id=f"E{i}",
                summary=f"Evidence summary for item {i}",
                url=f"https://example.com/evidence/{i}",
                tag=EvidenceTag.ACADEMIC,
                source_query=f"query for item {i}",
                mcp_server="test_server",
            )
        )
    return items


def _build_analysis_exp(
    task: Any | None = None,
    llm_pool: Any | None = None,
    event_emitter: Any | None = None,
    cost_tracker: Any | None = None,
    cancellation_token: Any | None = None,
) -> AnalysisExp:
    """Create an AnalysisExp instance with sensible test defaults 🔬.

    Args:
        task: EvaluationTask (auto-built if None).
        llm_pool: LLMProviderPool mock (None = use default llm).
        event_emitter: EventEmitter (MagicMock if None).
        cost_tracker: CostTracker (real instance if None).
        cancellation_token: CancellationToken (real instance if None).

    Returns:
        Configured AnalysisExp instance.
    """
    if task is None:
        task = build_sample_evaluation_task()

    if event_emitter is None:
        event_emitter = MagicMock()
        event_emitter.emit = MagicMock()

    mock_llm = MagicMock()

    return AnalysisExp(
        task=task,
        llm=mock_llm,
        llm_pool=llm_pool,
        quality_gate_config=task.quality_gate,
        cost_tracker=(
            cost_tracker
            if cost_tracker is not None
            else CostTracker(max_per_task=10.0, max_total=100.0)
        ),
        event_emitter=event_emitter,
        cancellation_token=(
            cancellation_token
            if cancellation_token is not None
            else CancellationToken()
        ),
    )


def _make_mock_llm_response(result_dict: dict[str, Any]) -> MagicMock:
    """Build a mock AssistantMessage with JSON content 🤖.

    Args:
        result_dict: The structured result to serialize as JSON.

    Returns:
        MagicMock simulating an AssistantMessage.
    """
    mock_response = MagicMock()
    mock_response.content = json.dumps(result_dict)
    mock_response.meta = {
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
        },
    }
    return mock_response


def _build_valid_analysis_result() -> dict[str, Any]:
    """Build a valid analysis result dict ✅.

    Returns:
        Dict conforming to the expected analysis output schema.
    """
    return {
        "decision": "positive",
        "confidence": 0.85,
        "reasoning": [
            {
                "claim": "Strong evidence for item 1",
                "evidence_ids": ["E1"],
                "strength": "strong",
            },
            {
                "claim": "Moderate evidence for item 2",
                "evidence_ids": ["E2"],
                "strength": "moderate",
            },
        ],
        "evidence_index": [
            {
                "id": "E1",
                "source": "test_server",
                "query": "query for item 1",
                "summary": "Evidence summary for item 1",
            },
            {
                "id": "E2",
                "source": "test_server",
                "query": "query for item 2",
                "summary": "Evidence summary for item 2",
            },
        ],
        "checklist_coverage": {
            "required_covered": ["item_1", "item_2"],
            "required_missing": [],
        },
        "gaps_remaining": [],
    }


# ============================================================================
# 🏗️ Initialization and Inheritance Tests
# ============================================================================


class TestAnalysisExpInit:
    """Tests for AnalysisExp initialization and inheritance 🏗️."""

    def test_inherits_from_inquiro_base_exp(self) -> None:
        """AnalysisExp must inherit from InquiroBaseExp."""
        exp = _build_analysis_exp()
        assert isinstance(exp, InquiroBaseExp)

    def test_exp_name_is_analysis(self) -> None:
        """exp_name property must return 'Analysis'."""
        exp = _build_analysis_exp()
        assert exp.exp_name == "Analysis"

    def test_init_sets_task(self) -> None:
        """Initialization must store the evaluation task."""
        task = build_sample_evaluation_task(task_id="analysis-init-001")
        exp = _build_analysis_exp(task=task)
        assert exp.task.task_id == "analysis-init-001"

    def test_init_creates_aggregation_engine(self) -> None:
        """Initialization must create an AggregationEngine instance."""
        exp = _build_analysis_exp()
        assert isinstance(exp.aggregation_engine, AggregationEngine)

    def test_init_stores_llm_pool(self) -> None:
        """Initialization must store the llm_pool reference."""
        mock_pool = MagicMock()
        exp = _build_analysis_exp(llm_pool=mock_pool)
        assert exp.llm_pool is mock_pool

    def test_init_with_none_llm_pool(self) -> None:
        """Initialization with None llm_pool stores None."""
        exp = _build_analysis_exp(llm_pool=None)
        assert exp.llm_pool is None

    def test_init_with_default_cost_tracker(self) -> None:
        """When cost_tracker=None, a default is created."""
        exp = AnalysisExp(
            task=build_sample_evaluation_task(),
            llm=MagicMock(),
        )
        assert exp.cost_tracker is not None

    def test_init_with_default_event_emitter(self) -> None:
        """When event_emitter=None, a default is created."""
        exp = AnalysisExp(
            task=build_sample_evaluation_task(),
            llm=MagicMock(),
        )
        assert exp.event_emitter is not None

    def test_init_with_default_cancellation_token(self) -> None:
        """When cancellation_token=None, a fresh token is created."""
        exp = AnalysisExp(
            task=build_sample_evaluation_task(),
            llm=MagicMock(),
        )
        assert exp.cancellation_token is not None
        assert not exp.cancellation_token.is_cancelled


# ============================================================================
# 📝 Prompt Rendering Tests
# ============================================================================


class TestPromptRendering:
    """Tests for system and user prompt rendering 📝."""

    def test_system_prompt_includes_rules(self) -> None:
        """System prompt must include the task's evaluation rules."""
        task = build_sample_evaluation_task(
            rules="Focus on primary research data",
        )
        exp = _build_analysis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "Focus on primary research data" in prompt

    def test_system_prompt_includes_checklist(self) -> None:
        """System prompt must include formatted checklist items."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "item_1" in prompt
        assert "item_2" in prompt

    def test_system_prompt_includes_output_schema(self) -> None:
        """System prompt must include the output JSON schema."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "decision" in prompt
        assert "confidence" in prompt

    def test_system_prompt_includes_identity(self) -> None:
        """System prompt must include AGENT IDENTITY section."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "AGENT IDENTITY" in prompt

    def test_user_prompt_includes_evidence(self) -> None:
        """User prompt must include formatted evidence items."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        evidence = _build_cleaned_evidence(2)

        prompt = exp._render_user_prompt(task, evidence, round_number=1)

        assert "[E1]" in prompt
        assert "[E2]" in prompt
        assert "Evidence summary for item 1" in prompt

    def test_user_prompt_includes_round_context(self) -> None:
        """User prompt must include round context."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        evidence = _build_cleaned_evidence(1)

        prompt = exp._render_user_prompt(task, evidence, round_number=3)

        assert "round **3**" in prompt

    def test_user_prompt_initial_round_context(self) -> None:
        """Round 1 context should indicate initial analysis."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        evidence = _build_cleaned_evidence(1)

        prompt = exp._render_user_prompt(task, evidence, round_number=1)

        assert "initial analysis" in prompt


# ============================================================================
# 📋 Evidence Formatting Tests
# ============================================================================


class TestEvidenceFormatting:
    """Tests for evidence list formatting 📋."""

    def test_format_empty_evidence(self) -> None:
        """Empty evidence list produces fallback message."""
        result = AnalysisExp._format_evidence_list([])
        assert "No evidence items available" in result

    def test_format_evidence_includes_ids(self) -> None:
        """Formatted evidence must include evidence IDs."""
        evidence = _build_cleaned_evidence(3)
        result = AnalysisExp._format_evidence_list(evidence)
        assert "[E1]" in result
        assert "[E2]" in result
        assert "[E3]" in result

    def test_format_evidence_includes_summaries(self) -> None:
        """Formatted evidence must include summaries."""
        evidence = _build_cleaned_evidence(2)
        result = AnalysisExp._format_evidence_list(evidence)
        assert "Evidence summary for item 1" in result
        assert "Evidence summary for item 2" in result

    def test_format_evidence_includes_urls(self) -> None:
        """Formatted evidence must include URLs when present."""
        evidence = _build_cleaned_evidence(1)
        result = AnalysisExp._format_evidence_list(evidence)
        assert "https://example.com/evidence/1" in result

    def test_format_evidence_includes_tags(self) -> None:
        """Formatted evidence must include source type tags."""
        evidence = _build_cleaned_evidence(1)
        result = AnalysisExp._format_evidence_list(evidence)
        assert "academic" in result

    def test_format_evidence_includes_source_query(self) -> None:
        """Formatted evidence must include the source query."""
        evidence = _build_cleaned_evidence(1)
        result = AnalysisExp._format_evidence_list(evidence)
        assert "query for item 1" in result


# ============================================================================
# 🔄 Round Context Tests
# ============================================================================


class TestRoundContext:
    """Tests for round context formatting 🔄."""

    def test_round_1_is_initial(self) -> None:
        """Round 1 should produce initial analysis context."""
        result = AnalysisExp._format_round_context(1)
        assert "initial analysis" in result

    def test_round_2_shows_round_number(self) -> None:
        """Round > 1 should show the round number."""
        result = AnalysisExp._format_round_context(2)
        assert "round **2**" in result

    def test_round_5_mentions_gaps(self) -> None:
        """Later rounds should mention previous gaps."""
        result = AnalysisExp._format_round_context(5)
        assert "gaps" in result.lower()


# ============================================================================
# 🔍 JSON Extraction Tests
# ============================================================================


class TestJsonExtraction:
    """Tests for JSON extraction from LLM output 🔍."""

    def test_extract_pure_json(self) -> None:
        """Extracts valid JSON from pure JSON string."""
        text = '{"decision": "positive", "confidence": 0.8}'
        result = AnalysisExp._extract_json_from_text(text)
        assert result["decision"] == "positive"
        assert result["confidence"] == 0.8

    def test_extract_json_from_code_block(self) -> None:
        """Extracts JSON from markdown code block."""
        text = 'Here is my analysis:\n```json\n{"decision": "cautious"}\n```'
        result = AnalysisExp._extract_json_from_text(text)
        assert result["decision"] == "cautious"

    def test_extract_json_with_surrounding_text(self) -> None:
        """Extracts JSON surrounded by commentary."""
        text = 'Analysis: {"decision": "negative"} end.'
        result = AnalysisExp._extract_json_from_text(text)
        assert result["decision"] == "negative"

    def test_extract_returns_empty_on_invalid(self) -> None:
        """Returns empty dict for completely invalid text."""
        text = "No JSON here at all."
        result = AnalysisExp._extract_json_from_text(text)
        assert result == {}


# ============================================================================
# 📊 Response Parsing Tests
# ============================================================================


class TestResponseParsing:
    """Tests for LLM response parsing into EvaluationResult 📊."""

    def test_parse_valid_response(self) -> None:
        """Parses a valid analysis response into EvaluationResult."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        response = _make_mock_llm_response(_build_valid_analysis_result())

        result = exp._parse_analysis_response(response, "test-model", task)

        assert isinstance(result, EvaluationResult)
        assert result.decision == Decision.POSITIVE
        assert result.confidence == 0.85
        assert len(result.reasoning) == 2
        assert len(result.evidence_index) == 2

    def test_parse_response_with_unknown_decision(self) -> None:
        """Unknown decision defaults to CAUTIOUS."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        data = _build_valid_analysis_result()
        data["decision"] = "unknown_value"
        response = _make_mock_llm_response(data)

        result = exp._parse_analysis_response(response, "test-model", task)

        assert result.decision == Decision.CAUTIOUS

    def test_parse_response_includes_model_in_task_id(self) -> None:
        """Parsed result task_id must include the model name."""
        task = build_sample_evaluation_task(task_id="task-001")
        exp = _build_analysis_exp(task=task)
        response = _make_mock_llm_response(_build_valid_analysis_result())

        result = exp._parse_analysis_response(
            response,
            "gpt-5.2",
            task,
        )

        assert "task-001" in result.task_id
        assert "gpt-5.2" in result.task_id
        assert "analysis" in result.task_id

    def test_parse_response_clamps_confidence(self) -> None:
        """Confidence > 1.0 or < 0.0 must be clamped."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        data = _build_valid_analysis_result()
        data["confidence"] = 1.5
        response = _make_mock_llm_response(data)

        result = exp._parse_analysis_response(response, "test-model", task)

        assert result.confidence == 1.0

    def test_parse_empty_response(self) -> None:
        """Empty response produces cautious result with 0.5 confidence."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.meta = {}

        result = exp._parse_analysis_response(
            mock_response,
            "test-model",
            task,
        )

        assert result.decision == Decision.CAUTIOUS
        assert result.confidence == 0.5

    def test_parse_response_sets_pipeline_mode(self) -> None:
        """Parsed result must have pipeline_mode='discovery'."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        response = _make_mock_llm_response(_build_valid_analysis_result())

        result = exp._parse_analysis_response(response, "test-model", task)

        assert result.pipeline_mode == "discovery"


# ============================================================================
# 💰 Cost Extraction Tests
# ============================================================================


class TestCostExtraction:
    """Tests for cost extraction from LLM response metadata 💰."""

    def test_extract_cost_with_usage(self) -> None:
        """Extracts non-zero cost when usage metadata is present."""
        mock_response = MagicMock()
        mock_response.meta = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        cost = AnalysisExp._extract_cost_from_response(mock_response)
        assert cost > 0.0

    def test_extract_cost_without_meta(self) -> None:
        """Returns 0.0 when no metadata is available."""
        mock_response = MagicMock()
        mock_response.meta = None
        cost = AnalysisExp._extract_cost_from_response(mock_response)
        assert cost == 0.0

    def test_extract_cost_empty_usage(self) -> None:
        """Returns 0.0 when usage dict is empty."""
        mock_response = MagicMock()
        mock_response.meta = {"usage": {}}
        cost = AnalysisExp._extract_cost_from_response(mock_response)
        assert cost == 0.0


# ============================================================================
# 🚀 Parallel Execution Tests (Async)
# ============================================================================


class TestParallelExecution:
    """Tests for parallel model execution and aggregation 🚀."""

    @pytest.mark.asyncio
    async def test_run_analysis_with_three_models(self) -> None:
        """run_analysis with 3 models should produce aggregated result."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        evidence = _build_cleaned_evidence(3)
        models = ["model-a", "model-b", "model-c"]

        # 🤖 Mock LLM responses
        valid_result = _build_valid_analysis_result()
        mock_response = _make_mock_llm_response(valid_result)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        result, cost = await exp.run_analysis(
            task=task,
            cleaned_evidence=evidence,
            analysis_models=models,
            round_number=1,
        )

        assert isinstance(result, AggregatedResult)
        assert cost >= 0.0
        assert result.decision in (
            Decision.POSITIVE,
            Decision.CAUTIOUS,
            Decision.NEGATIVE,
        )
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.consensus_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_run_analysis_empty_models_raises(self) -> None:
        """run_analysis with empty models list should raise ValueError."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task)
        evidence = _build_cleaned_evidence(1)

        with pytest.raises(ValueError, match="at least one model"):
            await exp.run_analysis(
                task=task,
                cleaned_evidence=evidence,
                analysis_models=[],
                round_number=1,
            )

    @pytest.mark.asyncio
    async def test_run_analysis_emits_events(self) -> None:
        """run_analysis should emit start and completion events."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task, event_emitter=emitter)
        evidence = _build_cleaned_evidence(2)

        # 🤖 Mock LLM
        valid_result = _build_valid_analysis_result()
        mock_response = _make_mock_llm_response(valid_result)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        await exp.run_analysis(
            task=task,
            cleaned_evidence=evidence,
            analysis_models=["model-a"],
            round_number=1,
        )

        # ✅ Check that events were emitted
        call_args_list = emitter.emit.call_args_list
        event_types = [call.args[0] for call in call_args_list]
        from inquiro.infrastructure.event_emitter import InquiroEvent

        assert InquiroEvent.TASK_STARTED in event_types
        assert InquiroEvent.TASK_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_run_analysis_with_failing_model(self) -> None:
        """One failing model should not prevent aggregation."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()

        # 🤖 Model A succeeds, Model B fails
        good_response = _make_mock_llm_response(
            _build_valid_analysis_result(),
        )
        good_llm = MagicMock()
        good_llm.query = MagicMock(return_value=good_response)

        bad_llm = MagicMock()
        bad_llm.query = MagicMock(side_effect=RuntimeError("Model error"))

        def get_llm_side_effect(key: str) -> Any:
            if key == "model-bad":
                return bad_llm
            return good_llm

        mock_pool.get_llm = MagicMock(side_effect=get_llm_side_effect)
        exp = _build_analysis_exp(task=task, llm_pool=mock_pool)
        evidence = _build_cleaned_evidence(2)

        result, cost = await exp.run_analysis(
            task=task,
            cleaned_evidence=evidence,
            analysis_models=["model-good", "model-bad"],
            round_number=1,
        )

        # ✅ Should still produce a valid result
        assert isinstance(result, AggregatedResult)

    @pytest.mark.asyncio
    async def test_run_analysis_all_models_fail_returns_fallback(
        self,
    ) -> None:
        """When all models fail, return cautious fallback."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()
        bad_llm = MagicMock()
        bad_llm.query = MagicMock(
            side_effect=RuntimeError("All models down"),
        )
        mock_pool.get_llm = MagicMock(return_value=bad_llm)

        exp = _build_analysis_exp(task=task, llm_pool=mock_pool)
        evidence = _build_cleaned_evidence(2)

        result, cost = await exp.run_analysis(
            task=task,
            cleaned_evidence=evidence,
            analysis_models=["model-a", "model-b"],
            round_number=1,
        )

        assert isinstance(result, AggregatedResult)
        assert result.decision == Decision.CAUTIOUS
        assert result.confidence == 0.0
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_run_analysis_cancellation(self) -> None:
        """run_analysis should raise CancelledError when cancelled."""
        task = build_sample_evaluation_task()
        token = CancellationToken()
        token.cancel("Test cancellation")
        exp = _build_analysis_exp(task=task, cancellation_token=token)
        evidence = _build_cleaned_evidence(1)

        with pytest.raises(CancelledError):
            await exp.run_analysis(
                task=task,
                cleaned_evidence=evidence,
                analysis_models=["model-a"],
                round_number=1,
            )


# ============================================================================
# 🔧 LLM Pool Fallback Tests
# ============================================================================


class TestLLMPoolFallback:
    """Tests for LLM pool resolution and fallback 🔧."""

    def test_get_llm_uses_pool_when_available(self) -> None:
        """Should use llm_pool.get_llm() when pool is available."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()
        mock_llm_instance = MagicMock()
        mock_pool.get_llm = MagicMock(return_value=mock_llm_instance)

        exp = _build_analysis_exp(task=task, llm_pool=mock_pool)
        result = exp._get_llm_for_model("test-model")

        assert result is mock_llm_instance
        mock_pool.get_llm.assert_called_once_with("test-model")

    def test_get_llm_falls_back_to_default(self) -> None:
        """Should fall back to self.llm when pool raises."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()
        mock_pool.get_llm = MagicMock(side_effect=KeyError("not found"))

        exp = _build_analysis_exp(task=task, llm_pool=mock_pool)
        default_llm = exp.llm
        result = exp._get_llm_for_model("nonexistent-model")

        assert result is default_llm

    def test_get_llm_without_pool(self) -> None:
        """Should return self.llm when llm_pool is None."""
        task = build_sample_evaluation_task()
        exp = _build_analysis_exp(task=task, llm_pool=None)
        default_llm = exp.llm
        result = exp._get_llm_for_model("any-model")

        assert result is default_llm


# ============================================================================
# 🔄 run_sync Interface Tests
# ============================================================================


class TestRunSyncInterface:
    """Tests for the run_sync() BaseExp interface 🔄."""

    def test_run_sync_returns_empty_dict(self) -> None:
        """run_sync() should return empty dict (async-only exp)."""
        exp = _build_analysis_exp()
        result = exp.run_sync()
        assert result == {}
