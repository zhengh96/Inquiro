"""Integration tests for DiscoverySynthesisExp lifecycle 🧪.

Tests cover:
    - Initialization and inheritance
    - Prompt rendering (system prompt, user prompt)
    - Evidence formatting for prompt injection
    - Claims summary formatting
    - Coverage info formatting
    - Round context formatting
    - JSON extraction from LLM responses
    - Claim extraction and parsing
    - Multi-model parallel execution (mocked)
    - Consensus voting (majority, tie-breaking)
    - Consensus confidence computation
    - Claim merging and deduplication
    - Partial failure (1 model fails, 2 succeed)
    - Total failure (all models fail)
    - EvaluationResult construction
    - SynthesisRecord generation for trajectory
    - Cost tracking across models
    - Cancellation behavior
    - Empty evidence handling
    - Large evidence set handling
    - ModelSynthesisOutput model validation
    - SynthesisResult model validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from inquiro.core.aggregation import AggregationEngine
from inquiro.core.trajectory.models import (
    ModelAnalysisRecord,
    SynthesisRecord,
)
from inquiro.core.types import (
    Decision,
    DiscoveryConfig,
    Evidence,
    EvidenceStrength,
    EvaluationResult,
    ReasoningClaim,
)
from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.exps.discovery_synthesis_exp import (
    DiscoverySynthesisExp,
    ModelSynthesisOutput,
    SynthesisResult,
)
from inquiro.infrastructure.cancellation import (
    CancellationToken,
    CancelledError,
)
from inquiro.infrastructure.cost_tracker import CostTracker

from inquiro.tests.mock_helpers import build_sample_evaluation_task


# ============================================================================
# 🏭 Test Fixtures and Helpers
# ============================================================================


def _build_evidence(count: int = 3) -> list[Evidence]:
    """Build a list of sample Evidence items 🔗.

    Args:
        count: Number of evidence items to generate.

    Returns:
        List of Evidence instances with summaries > 50 chars.
    """
    items: list[Evidence] = []
    for i in range(1, count + 1):
        items.append(
            Evidence(
                id=f"E{i}",
                source="test_server",
                query=f"search query for evidence item number {i}",
                summary=(
                    f"This is a detailed evidence summary for item {i} "
                    f"containing enough content to exceed the minimum "
                    f"character requirement for valid evidence entries"
                ),
                url=f"https://example.com/evidence/{i}",
                quality_label="medium",
                round_number=1,
            )
        )
    return items


def _build_claims(count: int = 2) -> list[ReasoningClaim]:
    """Build a list of sample ReasoningClaim items 🧠.

    Args:
        count: Number of claims to generate.

    Returns:
        List of ReasoningClaim instances.
    """
    claims: list[ReasoningClaim] = []
    for i in range(1, count + 1):
        claims.append(
            ReasoningClaim(
                claim=f"Analysis claim number {i} based on collected evidence",
                evidence_ids=[f"E{i}"],
                strength=EvidenceStrength.MODERATE,
            )
        )
    return claims


def _build_discovery_config(
    models: list[str] | None = None,
) -> DiscoveryConfig:
    """Build a DiscoveryConfig for testing 🔧.

    Args:
        models: Analysis model names. Defaults to 3 test models.
            Pass an explicit empty list to test empty-models behavior.

    Returns:
        DiscoveryConfig instance.
    """
    if models is not None:
        return DiscoveryConfig(analysis_models=models)
    return DiscoveryConfig(
        analysis_models=[
            "model-alpha",
            "model-beta",
            "model-gamma",
        ],
    )


def _build_round_summaries(count: int = 2) -> list[dict[str, Any]]:
    """Build round summary dicts for testing 🔄.

    Args:
        count: Number of round summaries.

    Returns:
        List of round summary dicts.
    """
    summaries: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        summaries.append(
            {
                "round_number": i,
                "evidence_count": 5 * i,
                "coverage_ratio": 0.3 * i,
            }
        )
    return summaries


def _build_discovery_synthesis_exp(
    task: Any | None = None,
    llm_pool: Any | None = None,
    event_emitter: Any | None = None,
    cost_tracker: Any | None = None,
    cancellation_token: Any | None = None,
) -> DiscoverySynthesisExp:
    """Create a DiscoverySynthesisExp with sensible test defaults 🧬.

    Args:
        task: EvaluationTask (auto-built if None).
        llm_pool: LLMProviderPool mock (None = use default llm).
        event_emitter: EventEmitter (MagicMock if None).
        cost_tracker: CostTracker (real instance if None).
        cancellation_token: CancellationToken (real instance if None).

    Returns:
        Configured DiscoverySynthesisExp instance.
    """
    if task is None:
        task = build_sample_evaluation_task()

    if event_emitter is None:
        event_emitter = MagicMock()
        event_emitter.emit = MagicMock()

    mock_llm = MagicMock()

    return DiscoverySynthesisExp(
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


def _make_mock_llm_response(
    result_dict: dict[str, Any],
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
) -> MagicMock:
    """Build a mock AssistantMessage with JSON content 🤖.

    Args:
        result_dict: The structured result to serialize as JSON.
        prompt_tokens: Input token count for cost estimation.
        completion_tokens: Output token count for cost estimation.

    Returns:
        MagicMock simulating an AssistantMessage.
    """
    mock_response = MagicMock()
    mock_response.content = json.dumps(result_dict)
    mock_response.meta = {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return mock_response


def _build_valid_synthesis_response() -> dict[str, Any]:
    """Build a valid synthesis response dict ✅.

    Returns:
        Dict conforming to the expected synthesis output format.
    """
    return {
        "decision": "positive",
        "confidence": 0.85,
        "claims": [
            {
                "claim": (
                    "Strong evidence supports the primary hypothesis "
                    "based on multiple independent sources"
                ),
                "evidence_ids": ["E1", "E2"],
                "strength": "strong",
            },
            {
                "claim": (
                    "Moderate evidence for secondary finding based "
                    "on analysis of collected data items"
                ),
                "evidence_ids": ["E3"],
                "strength": "moderate",
            },
        ],
        "summary": (
            "Comprehensive synthesis showing strong positive indicators "
            "across multiple evidence sources and analysis rounds"
        ),
        "gaps_remaining": [],
        "checklist_coverage": {
            "required_covered": ["item_1", "item_2"],
            "required_missing": [],
        },
    }


# ============================================================================
# 🏗️ Initialization and Inheritance Tests
# ============================================================================


class TestDiscoverySynthesisExpInit:
    """Tests for DiscoverySynthesisExp initialization 🏗️."""

    def test_inherits_from_inquiro_base_exp(self) -> None:
        """DiscoverySynthesisExp must inherit from InquiroBaseExp."""
        exp = _build_discovery_synthesis_exp()
        assert isinstance(exp, InquiroBaseExp)

    def test_exp_name_is_discovery_synthesis(self) -> None:
        """exp_name property must return 'DiscoverySynthesis'."""
        exp = _build_discovery_synthesis_exp()
        assert exp.exp_name == "DiscoverySynthesis"

    def test_init_sets_task(self) -> None:
        """Initialization must store the evaluation task."""
        task = build_sample_evaluation_task(task_id="synth-init-001")
        exp = _build_discovery_synthesis_exp(task=task)
        assert exp.task.task_id == "synth-init-001"

    def test_init_creates_aggregation_engine(self) -> None:
        """Initialization must create an AggregationEngine instance."""
        exp = _build_discovery_synthesis_exp()
        assert isinstance(exp.aggregation_engine, AggregationEngine)

    def test_init_stores_llm_pool(self) -> None:
        """Initialization must store the llm_pool reference."""
        mock_pool = MagicMock()
        exp = _build_discovery_synthesis_exp(llm_pool=mock_pool)
        assert exp.llm_pool is mock_pool

    def test_init_with_none_llm_pool(self) -> None:
        """Initialization with None llm_pool stores None."""
        exp = _build_discovery_synthesis_exp(llm_pool=None)
        assert exp.llm_pool is None

    def test_init_with_default_cost_tracker(self) -> None:
        """When cost_tracker=None, a default is created."""
        exp = DiscoverySynthesisExp(
            task=build_sample_evaluation_task(),
            llm=MagicMock(),
        )
        assert exp.cost_tracker is not None

    def test_init_with_default_event_emitter(self) -> None:
        """When event_emitter=None, a default is created."""
        exp = DiscoverySynthesisExp(
            task=build_sample_evaluation_task(),
            llm=MagicMock(),
        )
        assert exp.event_emitter is not None

    def test_init_with_default_cancellation_token(self) -> None:
        """When cancellation_token=None, a fresh token is created."""
        exp = DiscoverySynthesisExp(
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
            rules="Focus on primary research data and peer-reviewed sources",
        )
        exp = _build_discovery_synthesis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "Focus on primary research data" in prompt

    def test_system_prompt_includes_checklist(self) -> None:
        """System prompt must include formatted checklist items."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "item_1" in prompt
        assert "item_2" in prompt

    def test_system_prompt_includes_output_schema(self) -> None:
        """System prompt must include the output JSON schema."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "decision" in prompt
        assert "confidence" in prompt

    def test_system_prompt_includes_identity(self) -> None:
        """System prompt must include AGENT IDENTITY section."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "AGENT IDENTITY" in prompt

    def test_system_prompt_includes_synthesis_protocol(self) -> None:
        """System prompt must include SYNTHESIS PROTOCOL section."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        prompt = exp._render_system_prompt(task)
        assert "SYNTHESIS PROTOCOL" in prompt

    def test_user_prompt_includes_evidence(self) -> None:
        """User prompt must include formatted evidence items."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(2)
        claims = _build_claims(1)

        prompt = exp._render_user_prompt(
            task,
            evidence,
            claims,
            0.75,
            _build_round_summaries(),
        )

        assert "[E1]" in prompt
        assert "[E2]" in prompt

    def test_user_prompt_includes_claims(self) -> None:
        """User prompt must include pre-existing claims."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(1)
        claims = _build_claims(2)

        prompt = exp._render_user_prompt(
            task,
            evidence,
            claims,
            0.6,
            _build_round_summaries(),
        )

        assert "Analysis claim number 1" in prompt
        assert "Analysis claim number 2" in prompt

    def test_user_prompt_includes_coverage_info(self) -> None:
        """User prompt must include coverage information."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        prompt = exp._render_user_prompt(
            task,
            _build_evidence(1),
            _build_claims(1),
            0.85,
            _build_round_summaries(),
        )

        assert "85%" in prompt

    def test_user_prompt_includes_round_context(self) -> None:
        """User prompt must include round summaries."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        summaries = _build_round_summaries(3)

        prompt = exp._render_user_prompt(
            task,
            _build_evidence(1),
            _build_claims(1),
            0.7,
            summaries,
        )

        assert "Round 1" in prompt
        assert "Round 2" in prompt
        assert "Round 3" in prompt


# ============================================================================
# 📋 Evidence Formatting Tests
# ============================================================================


class TestEvidenceFormatting:
    """Tests for evidence list formatting 📋."""

    def test_format_empty_evidence(self) -> None:
        """Empty evidence list produces fallback message."""
        result = DiscoverySynthesisExp._format_evidence_list([])
        assert "No evidence items available" in result

    def test_format_evidence_includes_ids(self) -> None:
        """Formatted evidence must include evidence IDs."""
        evidence = _build_evidence(3)
        result = DiscoverySynthesisExp._format_evidence_list(evidence)
        assert "[E1]" in result
        assert "[E2]" in result
        assert "[E3]" in result

    def test_format_evidence_includes_summaries(self) -> None:
        """Formatted evidence must include summaries."""
        evidence = _build_evidence(1)
        result = DiscoverySynthesisExp._format_evidence_list(evidence)
        assert "detailed evidence summary" in result

    def test_format_evidence_includes_source(self) -> None:
        """Formatted evidence must include source."""
        evidence = _build_evidence(1)
        result = DiscoverySynthesisExp._format_evidence_list(evidence)
        assert "test_server" in result

    def test_format_evidence_includes_urls(self) -> None:
        """Formatted evidence must include URLs."""
        evidence = _build_evidence(1)
        result = DiscoverySynthesisExp._format_evidence_list(evidence)
        assert "https://example.com/evidence/1" in result


# ============================================================================
# 🧠 Claims Summary Formatting Tests
# ============================================================================


class TestClaimsFormatting:
    """Tests for claims summary formatting 🧠."""

    def test_format_empty_claims(self) -> None:
        """Empty claims list produces fallback message."""
        result = DiscoverySynthesisExp._format_claims_summary([])
        assert "No prior claims" in result

    def test_format_claims_includes_text(self) -> None:
        """Claims summary must include claim text."""
        claims = _build_claims(2)
        result = DiscoverySynthesisExp._format_claims_summary(claims)
        assert "Analysis claim number 1" in result
        assert "Analysis claim number 2" in result

    def test_format_claims_includes_strength(self) -> None:
        """Claims summary must include strength labels."""
        claims = _build_claims(1)
        result = DiscoverySynthesisExp._format_claims_summary(claims)
        assert "moderate" in result

    def test_format_claims_includes_evidence_refs(self) -> None:
        """Claims summary must include evidence references."""
        claims = _build_claims(1)
        result = DiscoverySynthesisExp._format_claims_summary(claims)
        assert "E1" in result


# ============================================================================
# 📊 Coverage Info Formatting Tests
# ============================================================================


class TestCoverageFormatting:
    """Tests for coverage info formatting 📊."""

    def test_coverage_high(self) -> None:
        """High coverage shows good status."""
        result = DiscoverySynthesisExp._format_coverage_info(0.85)
        assert "85%" in result
        assert "Good coverage" in result

    def test_coverage_moderate(self) -> None:
        """Moderate coverage shows moderate status."""
        result = DiscoverySynthesisExp._format_coverage_info(0.65)
        assert "65%" in result
        assert "Moderate coverage" in result

    def test_coverage_low(self) -> None:
        """Low coverage shows limited status."""
        result = DiscoverySynthesisExp._format_coverage_info(0.30)
        assert "30%" in result
        assert "Limited coverage" in result


# ============================================================================
# 🔄 Round Context Formatting Tests
# ============================================================================


class TestRoundContextFormatting:
    """Tests for round context formatting 🔄."""

    def test_empty_round_summaries(self) -> None:
        """Empty round summaries produces fallback message."""
        result = DiscoverySynthesisExp._format_round_context([])
        assert "No round summaries" in result

    def test_round_summaries_include_round_number(self) -> None:
        """Round context must include round numbers."""
        summaries = _build_round_summaries(2)
        result = DiscoverySynthesisExp._format_round_context(summaries)
        assert "Round 1" in result
        assert "Round 2" in result

    def test_round_summaries_include_evidence_count(self) -> None:
        """Round context must include evidence counts."""
        summaries = [{"round_number": 1, "evidence_count": 7, "coverage_ratio": 0.5}]
        result = DiscoverySynthesisExp._format_round_context(summaries)
        assert "7 evidence items" in result


# ============================================================================
# 🔍 JSON Extraction Tests
# ============================================================================


class TestJsonExtraction:
    """Tests for JSON extraction from LLM output 🔍."""

    def test_extract_pure_json(self) -> None:
        """Extracts valid JSON from pure JSON string."""
        text = '{"decision": "positive", "confidence": 0.8}'
        result = DiscoverySynthesisExp._extract_json_from_text(text)
        assert result["decision"] == "positive"
        assert result["confidence"] == 0.8

    def test_extract_json_from_code_block(self) -> None:
        """Extracts JSON from markdown code block."""
        text = 'Analysis:\n```json\n{"decision": "cautious"}\n```'
        result = DiscoverySynthesisExp._extract_json_from_text(text)
        assert result["decision"] == "cautious"

    def test_extract_json_with_surrounding_text(self) -> None:
        """Extracts JSON surrounded by commentary."""
        text = 'Result: {"decision": "negative"} end.'
        result = DiscoverySynthesisExp._extract_json_from_text(text)
        assert result["decision"] == "negative"

    def test_extract_returns_empty_on_invalid(self) -> None:
        """Returns empty dict for completely invalid text."""
        text = "No JSON here at all."
        result = DiscoverySynthesisExp._extract_json_from_text(text)
        assert result == {}

    def test_extract_nested_json(self) -> None:
        """Extracts nested JSON objects correctly."""
        text = json.dumps(
            {
                "decision": "positive",
                "claims": [{"claim": "test", "evidence_ids": ["E1"]}],
            }
        )
        result = DiscoverySynthesisExp._extract_json_from_text(text)
        assert result["decision"] == "positive"
        assert len(result["claims"]) == 1


# ============================================================================
# 🧠 Claim Extraction Tests
# ============================================================================


class TestClaimExtraction:
    """Tests for claim extraction from parsed JSON 🧠."""

    def test_extract_claims_from_claims_field(self) -> None:
        """Extracts claims from 'claims' field."""
        parsed = {
            "claims": [
                {
                    "claim": "Finding one from synthesis analysis",
                    "evidence_ids": ["E1"],
                    "strength": "strong",
                },
                {
                    "claim": "Finding two from evidence review process",
                    "evidence_ids": ["E2"],
                    "strength": "moderate",
                },
            ],
        }
        result = DiscoverySynthesisExp._extract_claims(parsed)
        assert len(result) == 2
        assert result[0]["claim"] == "Finding one from synthesis analysis"

    def test_extract_claims_from_reasoning_field(self) -> None:
        """Extracts claims from 'reasoning' field as fallback."""
        parsed = {
            "reasoning": [
                {
                    "claim": "Fallback reasoning claim text",
                    "evidence_ids": ["E1"],
                    "strength": "weak",
                },
            ],
        }
        result = DiscoverySynthesisExp._extract_claims(parsed)
        assert len(result) == 1
        assert result[0]["strength"] == "weak"

    def test_extract_claims_empty(self) -> None:
        """Returns empty list when no claims or reasoning."""
        result = DiscoverySynthesisExp._extract_claims({})
        assert result == []

    def test_extract_claims_skips_empty_claim_text(self) -> None:
        """Skips claim entries with empty claim text."""
        parsed = {
            "claims": [
                {"claim": "", "evidence_ids": ["E1"]},
                {"claim": "Valid claim with sufficient text", "evidence_ids": ["E2"]},
            ],
        }
        result = DiscoverySynthesisExp._extract_claims(parsed)
        assert len(result) == 1
        assert result[0]["claim"] == "Valid claim with sufficient text"

    def test_extract_claims_defaults_strength(self) -> None:
        """Defaults strength to 'moderate' when missing."""
        parsed = {
            "claims": [
                {"claim": "Claim without strength annotation", "evidence_ids": ["E1"]},
            ],
        }
        result = DiscoverySynthesisExp._extract_claims(parsed)
        assert result[0]["strength"] == "moderate"


# ============================================================================
# ⚖️ Consensus Voting Tests
# ============================================================================


class TestConsensusVoting:
    """Tests for consensus decision computation ⚖️."""

    def test_unanimous_positive(self) -> None:
        """All positive -> positive with 1.0 ratio."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.9),
            ModelSynthesisOutput(model_name="b", decision="positive", confidence=0.8),
            ModelSynthesisOutput(model_name="c", decision="positive", confidence=0.85),
        ]
        decision, ratio = DiscoverySynthesisExp._compute_consensus(models)
        assert decision == "positive"
        assert ratio == 1.0

    def test_majority_positive(self) -> None:
        """2 positive, 1 cautious -> positive with 2/3 ratio."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.9),
            ModelSynthesisOutput(model_name="b", decision="positive", confidence=0.8),
            ModelSynthesisOutput(model_name="c", decision="cautious", confidence=0.5),
        ]
        decision, ratio = DiscoverySynthesisExp._compute_consensus(models)
        assert decision == "positive"
        assert abs(ratio - 2.0 / 3.0) < 0.01

    def test_tie_defaults_to_cautious(self) -> None:
        """3 different decisions defaults to cautious."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.9),
            ModelSynthesisOutput(model_name="b", decision="cautious", confidence=0.5),
            ModelSynthesisOutput(model_name="c", decision="negative", confidence=0.3),
        ]
        decision, ratio = DiscoverySynthesisExp._compute_consensus(models)
        assert decision == "cautious"

    def test_empty_models_returns_cautious(self) -> None:
        """No models -> cautious with 0.0 ratio."""
        decision, ratio = DiscoverySynthesisExp._compute_consensus([])
        assert decision == "cautious"
        assert ratio == 0.0

    def test_single_model(self) -> None:
        """Single model -> that model's decision with 1.0 ratio."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="negative", confidence=0.3),
        ]
        decision, ratio = DiscoverySynthesisExp._compute_consensus(models)
        assert decision == "negative"
        assert ratio == 1.0

    def test_two_way_tie_with_cautious(self) -> None:
        """2-way tie including cautious -> cautious wins."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.8),
            ModelSynthesisOutput(model_name="b", decision="cautious", confidence=0.6),
        ]
        decision, ratio = DiscoverySynthesisExp._compute_consensus(models)
        assert decision == "cautious"


# ============================================================================
# 📊 Consensus Confidence Tests
# ============================================================================


class TestConsensusConfidence:
    """Tests for consensus confidence computation 📊."""

    def test_unanimous_confidence(self) -> None:
        """Unanimous decision averages all confidences."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.9),
            ModelSynthesisOutput(model_name="b", decision="positive", confidence=0.8),
        ]
        conf = DiscoverySynthesisExp._compute_consensus_confidence(
            models,
            "positive",
        )
        assert abs(conf - 0.85) < 0.01

    def test_mixed_confidence_uses_agreeing(self) -> None:
        """Mixed decisions uses only agreeing models' confidence."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=0.9),
            ModelSynthesisOutput(model_name="b", decision="positive", confidence=0.8),
            ModelSynthesisOutput(model_name="c", decision="negative", confidence=0.3),
        ]
        conf = DiscoverySynthesisExp._compute_consensus_confidence(
            models,
            "positive",
        )
        assert abs(conf - 0.85) < 0.01

    def test_empty_models_returns_zero(self) -> None:
        """No models -> 0.0 confidence."""
        conf = DiscoverySynthesisExp._compute_consensus_confidence(
            [],
            "cautious",
        )
        assert conf == 0.0

    def test_confidence_clamped(self) -> None:
        """Confidence is clamped to [0.0, 1.0]."""
        models = [
            ModelSynthesisOutput(model_name="a", decision="positive", confidence=1.0),
        ]
        conf = DiscoverySynthesisExp._compute_consensus_confidence(
            models,
            "positive",
        )
        assert 0.0 <= conf <= 1.0


# ============================================================================
# 🧠 Claim Merging Tests
# ============================================================================


class TestClaimMerging:
    """Tests for claim merging and deduplication 🧠."""

    def test_merge_unique_claims(self) -> None:
        """Unique claims are all preserved after merging."""
        models = [
            ModelSynthesisOutput(
                model_name="a",
                claims=[
                    {
                        "claim": "Unique finding from model alpha analysis",
                        "evidence_ids": ["E1"],
                    }
                ],
            ),
            ModelSynthesisOutput(
                model_name="b",
                claims=[
                    {
                        "claim": "Different finding from model beta analysis",
                        "evidence_ids": ["E2"],
                    }
                ],
            ),
        ]
        result = DiscoverySynthesisExp._merge_claims(models)
        assert len(result) == 2

    def test_merge_dedup_substrings(self) -> None:
        """Substring claims are removed during deduplication."""
        models = [
            ModelSynthesisOutput(
                model_name="a",
                claims=[{"claim": "Strong evidence found", "evidence_ids": ["E1"]}],
            ),
            ModelSynthesisOutput(
                model_name="b",
                claims=[
                    {
                        "claim": "Strong evidence found in multiple sources",
                        "evidence_ids": ["E1", "E2"],
                    },
                ],
            ),
        ]
        result = DiscoverySynthesisExp._merge_claims(models)
        # ✅ "Strong evidence found" is substring of longer claim
        assert len(result) == 1
        assert "multiple sources" in result[0]["claim"]

    def test_merge_empty_claims(self) -> None:
        """No claims from any model -> empty result."""
        models = [
            ModelSynthesisOutput(model_name="a", claims=[]),
            ModelSynthesisOutput(model_name="b", claims=[]),
        ]
        result = DiscoverySynthesisExp._merge_claims(models)
        assert result == []

    def test_merge_identical_claims_deduplicated(self) -> None:
        """Exact duplicate claims from different models are deduplicated 🧹."""
        models = [
            ModelSynthesisOutput(
                model_name="a",
                claims=[
                    {"claim": "Exact same claim text here", "evidence_ids": ["E1"]}
                ],
            ),
            ModelSynthesisOutput(
                model_name="b",
                claims=[
                    {"claim": "Exact same claim text here", "evidence_ids": ["E2"]}
                ],
            ),
        ]
        result = DiscoverySynthesisExp._merge_claims(models)
        # ✅ Exact duplicates are removed — only first occurrence kept
        assert len(result) == 1
        assert result[0]["claim"] == "Exact same claim text here"


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
        cost = DiscoverySynthesisExp._extract_cost_from_response(
            mock_response,
        )
        assert cost > 0.0

    def test_extract_cost_without_meta(self) -> None:
        """Returns 0.0 when no metadata is available."""
        mock_response = MagicMock()
        mock_response.meta = None
        cost = DiscoverySynthesisExp._extract_cost_from_response(
            mock_response,
        )
        assert cost == 0.0

    def test_extract_cost_empty_usage(self) -> None:
        """Returns 0.0 when usage dict is empty."""
        mock_response = MagicMock()
        mock_response.meta = {"usage": {}}
        cost = DiscoverySynthesisExp._extract_cost_from_response(
            mock_response,
        )
        assert cost == 0.0

    def test_extract_cost_calculation(self) -> None:
        """Verifies cost calculation formula."""
        mock_response = MagicMock()
        mock_response.meta = {
            "usage": {
                "prompt_tokens": 1_000_000,
                "completion_tokens": 100_000,
            },
        }
        cost = DiscoverySynthesisExp._extract_cost_from_response(
            mock_response,
        )
        # 💰 input: 1M * 3.0 / 1M = 3.0, output: 100K * 15.0 / 1M = 1.5
        expected = 3.0 + 1.5
        assert abs(cost - expected) < 0.01


# ============================================================================
# 🔧 LLM Pool Fallback Tests
# ============================================================================


class TestLLMPoolFallback:
    """Tests for LLM pool resolution and fallback 🔧."""

    def test_get_llm_uses_pool_when_available(self) -> None:
        """Should use llm_pool.get_llm() when pool is available."""
        mock_pool = MagicMock()
        mock_llm_instance = MagicMock()
        mock_pool.get_llm = MagicMock(return_value=mock_llm_instance)

        exp = _build_discovery_synthesis_exp(llm_pool=mock_pool)
        result = exp._get_llm_for_model("test-model")

        assert result is mock_llm_instance

    def test_get_llm_falls_back_to_default(self) -> None:
        """Should fall back to self.llm when pool raises."""
        mock_pool = MagicMock()
        mock_pool.get_llm = MagicMock(side_effect=KeyError("not found"))

        exp = _build_discovery_synthesis_exp(llm_pool=mock_pool)
        default_llm = exp.llm
        result = exp._get_llm_for_model("nonexistent-model")

        assert result is default_llm

    def test_get_llm_without_pool(self) -> None:
        """Should return self.llm when llm_pool is None."""
        exp = _build_discovery_synthesis_exp(llm_pool=None)
        default_llm = exp.llm
        result = exp._get_llm_for_model("any-model")

        assert result is default_llm


# ============================================================================
# 🚀 Parallel Execution Tests (Async)
# ============================================================================


class TestParallelExecution:
    """Tests for parallel model execution and synthesis 🚀."""

    @pytest.mark.asyncio
    async def test_run_synthesis_with_three_models(self) -> None:
        """run_synthesis with 3 models should produce valid result."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(3)
        claims = _build_claims(2)
        config = _build_discovery_config()

        # 🤖 Mock LLM responses
        valid_response = _build_valid_synthesis_response()
        mock_response = _make_mock_llm_response(valid_response)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        result = await exp.run_synthesis(
            task=task,
            config=config,
            all_evidence=evidence,
            all_claims=claims,
            coverage_ratio=0.8,
            round_summaries=_build_round_summaries(),
        )

        assert isinstance(result, SynthesisResult)
        assert result.consensus_decision in (
            "positive",
            "cautious",
            "negative",
        )
        assert 0.0 <= result.consensus_ratio <= 1.0
        assert result.total_claims > 0
        assert result.cost_usd > 0.0
        assert result.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_run_synthesis_empty_models_raises(self) -> None:
        """run_synthesis with empty models list raises ValueError."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        config = _build_discovery_config(models=[])

        with pytest.raises(ValueError, match="at least one model"):
            await exp.run_synthesis(
                task=task,
                config=config,
                all_evidence=_build_evidence(1),
                all_claims=_build_claims(1),
                coverage_ratio=0.5,
                round_summaries=[],
            )

    @pytest.mark.asyncio
    async def test_run_synthesis_emits_events(self) -> None:
        """run_synthesis should emit start and completion events."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(
            task=task,
            event_emitter=emitter,
        )

        # 🤖 Mock LLM
        valid_response = _build_valid_synthesis_response()
        mock_response = _make_mock_llm_response(valid_response)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(models=["model-a"]),
            all_evidence=_build_evidence(1),
            all_claims=_build_claims(1),
            coverage_ratio=0.7,
            round_summaries=[],
        )

        # ✅ Check that events were emitted
        call_args_list = emitter.emit.call_args_list
        event_types = [call.args[0] for call in call_args_list]
        from inquiro.infrastructure.event_emitter import InquiroEvent

        assert InquiroEvent.TASK_STARTED in event_types
        assert InquiroEvent.TASK_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_run_synthesis_with_one_failing_model(self) -> None:
        """One failing model should not prevent synthesis."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()

        # 🤖 Model A+B succeed, Model C fails
        good_response = _make_mock_llm_response(
            _build_valid_synthesis_response(),
        )
        good_llm = MagicMock()
        good_llm.query = MagicMock(return_value=good_response)

        bad_llm = MagicMock()
        bad_llm.query = MagicMock(
            side_effect=RuntimeError("Model error"),
        )

        def get_llm_side_effect(key: str) -> Any:
            if key == "model-bad":
                return bad_llm
            return good_llm

        mock_pool.get_llm = MagicMock(side_effect=get_llm_side_effect)
        exp = _build_discovery_synthesis_exp(
            task=task,
            llm_pool=mock_pool,
        )

        result = await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(
                models=["model-good-1", "model-good-2", "model-bad"],
            ),
            all_evidence=_build_evidence(2),
            all_claims=_build_claims(1),
            coverage_ratio=0.6,
            round_summaries=[],
        )

        # ✅ Should still produce a valid result
        assert isinstance(result, SynthesisResult)
        assert len(result.model_results) == 2
        assert result.consensus_decision != "error"

    @pytest.mark.asyncio
    async def test_run_synthesis_all_models_fail_returns_error(
        self,
    ) -> None:
        """When all models fail, return error result."""
        task = build_sample_evaluation_task()
        mock_pool = MagicMock()
        bad_llm = MagicMock()
        bad_llm.query = MagicMock(
            side_effect=RuntimeError("All models down"),
        )
        mock_pool.get_llm = MagicMock(return_value=bad_llm)

        exp = _build_discovery_synthesis_exp(
            task=task,
            llm_pool=mock_pool,
        )

        result = await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(
                models=["model-a", "model-b"],
            ),
            all_evidence=_build_evidence(2),
            all_claims=_build_claims(1),
            coverage_ratio=0.5,
            round_summaries=[],
        )

        assert isinstance(result, SynthesisResult)
        assert result.consensus_decision == "error"
        assert result.consensus_ratio == 0.0
        assert result.total_claims == 0
        assert len(result.model_results) == 0

    @pytest.mark.asyncio
    async def test_run_synthesis_cancellation(self) -> None:
        """run_synthesis should raise CancelledError when cancelled."""
        task = build_sample_evaluation_task()
        token = CancellationToken()
        token.cancel("Test cancellation")
        exp = _build_discovery_synthesis_exp(
            task=task,
            cancellation_token=token,
        )

        with pytest.raises(CancelledError):
            await exp.run_synthesis(
                task=task,
                config=_build_discovery_config(),
                all_evidence=_build_evidence(1),
                all_claims=_build_claims(1),
                coverage_ratio=0.5,
                round_summaries=[],
            )


# ============================================================================
# 🏗️ EvaluationResult Construction Tests
# ============================================================================


class TestEvaluationResultConstruction:
    """Tests for EvaluationResult building 🏗️."""

    def test_builds_valid_evaluation_result(self) -> None:
        """Should build a valid EvaluationResult."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(2)
        claims = [
            {
                "claim": "Test claim for evaluation result construction",
                "evidence_ids": ["E1"],
                "strength": "strong",
            },
        ]

        result = exp._build_evaluation_result(
            task=task,
            all_evidence=evidence,
            merged_claims=claims,
            decision="positive",
            confidence=0.85,
            coverage_ratio=0.8,
            total_cost=1.5,
        )

        assert isinstance(result, EvaluationResult)
        assert result.decision == Decision.POSITIVE
        assert result.confidence == 0.85
        assert len(result.evidence_index) == 2
        assert len(result.reasoning) == 1
        assert result.pipeline_mode == "discovery"
        assert result.discovery_coverage == 0.8
        assert result.cost == 1.5

    def test_task_id_includes_synthesis_suffix(self) -> None:
        """EvaluationResult task_id must include '::synthesis'."""
        task = build_sample_evaluation_task(task_id="task-123")
        exp = _build_discovery_synthesis_exp(task=task)

        result = exp._build_evaluation_result(
            task=task,
            all_evidence=[],
            merged_claims=[],
            decision="cautious",
            confidence=0.5,
            coverage_ratio=0.6,
            total_cost=0.0,
        )

        assert "task-123" in result.task_id
        assert "synthesis" in result.task_id

    def test_unknown_decision_defaults_to_cautious(self) -> None:
        """Unknown decision string defaults to CAUTIOUS."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        result = exp._build_evaluation_result(
            task=task,
            all_evidence=[],
            merged_claims=[],
            decision="unknown_value",
            confidence=0.5,
            coverage_ratio=0.5,
            total_cost=0.0,
        )

        assert result.decision == Decision.CAUTIOUS

    def test_unknown_strength_defaults_to_moderate(self) -> None:
        """Unknown strength defaults to MODERATE."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        result = exp._build_evaluation_result(
            task=task,
            all_evidence=[],
            merged_claims=[
                {
                    "claim": "Test claim text here",
                    "evidence_ids": [],
                    "strength": "unknown",
                },
            ],
            decision="cautious",
            confidence=0.5,
            coverage_ratio=0.5,
            total_cost=0.0,
        )

        assert result.reasoning[0].strength == EvidenceStrength.MODERATE


# ============================================================================
# 📊 SynthesisRecord Construction Tests
# ============================================================================


class TestSynthesisRecordConstruction:
    """Tests for SynthesisRecord building 📊."""

    def test_builds_valid_synthesis_record(self) -> None:
        """Should build a valid SynthesisRecord."""
        exp = _build_discovery_synthesis_exp()
        successful = [
            ModelSynthesisOutput(
                model_name="model-a",
                decision="positive",
                confidence=0.9,
                claims=[
                    {"claim": "Claim from model A analysis", "evidence_ids": ["E1"]}
                ],
                cost_usd=0.5,
            ),
            ModelSynthesisOutput(
                model_name="model-b",
                decision="positive",
                confidence=0.8,
                claims=[
                    {"claim": "Claim from model B analysis", "evidence_ids": ["E2"]}
                ],
                cost_usd=0.6,
            ),
        ]

        record = exp._build_synthesis_record(
            successful=successful,
            consensus_decision="positive",
            consensus_ratio=1.0,
            total_cost=1.1,
            duration=5.5,
        )

        assert isinstance(record, SynthesisRecord)
        assert record.consensus_decision == "positive"
        assert record.consensus_ratio == 1.0
        assert record.cost_usd == 1.1
        assert record.duration_seconds == 5.5
        assert len(record.model_results) == 2

    def test_model_records_have_correct_data(self) -> None:
        """Model records should capture per-model data."""
        exp = _build_discovery_synthesis_exp()
        successful = [
            ModelSynthesisOutput(
                model_name="alpha",
                decision="cautious",
                confidence=0.6,
                claims=[
                    {"claim": "First claim text here", "evidence_ids": ["E1"]},
                    {"claim": "Second claim text here", "evidence_ids": ["E2"]},
                ],
                cost_usd=0.3,
            ),
        ]

        record = exp._build_synthesis_record(
            successful=successful,
            consensus_decision="cautious",
            consensus_ratio=1.0,
            total_cost=0.3,
            duration=2.0,
        )

        model_rec = record.model_results[0]
        assert isinstance(model_rec, ModelAnalysisRecord)
        assert model_rec.model_name == "alpha"
        assert model_rec.decision == "cautious"
        assert model_rec.confidence == 0.6
        assert model_rec.claims_count == 2
        assert model_rec.cost_usd == 0.3


# ============================================================================
# 📦 Model Validation Tests
# ============================================================================


class TestModelValidation:
    """Tests for Pydantic model validation 📦."""

    def test_model_synthesis_output_defaults(self) -> None:
        """ModelSynthesisOutput should have sensible defaults."""
        output = ModelSynthesisOutput(model_name="test")
        assert output.decision == "cautious"
        assert output.confidence == 0.5
        assert output.claims == []
        assert output.summary_text == ""
        assert output.cost_usd == 0.0
        assert output.raw_response == ""

    def test_synthesis_result_defaults(self) -> None:
        """SynthesisResult should have sensible defaults."""
        result = SynthesisResult(evaluation_result={})
        assert result.consensus_decision == "cautious"
        assert result.consensus_ratio == 0.0
        assert result.total_claims == 0
        assert result.cost_usd == 0.0
        assert result.duration_seconds == 0.0

    def test_model_synthesis_output_confidence_clamped(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(Exception):
            ModelSynthesisOutput(
                model_name="test",
                confidence=1.5,
            )


# ============================================================================
# 🔄 run_sync Interface Tests
# ============================================================================


class TestRunSyncInterface:
    """Tests for the run_sync() BaseExp interface 🔄."""

    def test_run_sync_returns_empty_dict(self) -> None:
        """run_sync() should return empty dict (async-only exp)."""
        exp = _build_discovery_synthesis_exp()
        result = exp.run_sync()
        assert result == {}


# ============================================================================
# 📋 Empty Evidence Handling Tests
# ============================================================================


class TestEmptyEvidenceHandling:
    """Tests for synthesis with no evidence 📋."""

    @pytest.mark.asyncio
    async def test_synthesis_with_no_evidence(self) -> None:
        """Synthesis with empty evidence returns cautious result."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        # 🤖 Mock LLM response for no-evidence case
        cautious_response = {
            "decision": "cautious",
            "confidence": 0.2,
            "claims": [],
            "summary": "Insufficient evidence for assessment",
            "gaps_remaining": ["No evidence collected"],
        }
        mock_response = _make_mock_llm_response(cautious_response)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        result = await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(models=["model-a"]),
            all_evidence=[],
            all_claims=[],
            coverage_ratio=0.0,
            round_summaries=[],
        )

        assert isinstance(result, SynthesisResult)
        assert result.consensus_decision == "cautious"


# ============================================================================
# 📊 Large Evidence Set Tests
# ============================================================================


class TestLargeEvidenceSet:
    """Tests for synthesis with many evidence items 📊."""

    @pytest.mark.asyncio
    async def test_synthesis_with_large_evidence_set(self) -> None:
        """Synthesis with 50 evidence items completes successfully."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(50)
        claims = _build_claims(20)

        # 🤖 Mock LLM response
        valid_response = _build_valid_synthesis_response()
        mock_response = _make_mock_llm_response(valid_response)
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        result = await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(models=["model-a"]),
            all_evidence=evidence,
            all_claims=claims,
            coverage_ratio=0.9,
            round_summaries=_build_round_summaries(5),
        )

        assert isinstance(result, SynthesisResult)
        # ✅ All evidence should be in the final result
        assert len(result.evaluation_result.evidence_index) == 50


# ============================================================================
# 🏗️ Error Result Tests
# ============================================================================


class TestErrorResult:
    """Tests for error result construction 🏗️."""

    def test_error_result_has_cautious_decision(self) -> None:
        """Error result must have cautious decision."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(2)

        result = exp._build_error_result(task, evidence, 5.0)

        assert isinstance(result, SynthesisResult)
        assert result.consensus_decision == "error"
        assert result.evaluation_result.decision == Decision.CAUTIOUS
        assert result.evaluation_result.confidence == 0.0
        assert result.duration_seconds == 5.0

    def test_error_result_preserves_evidence(self) -> None:
        """Error result must preserve all evidence."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)
        evidence = _build_evidence(3)

        result = exp._build_error_result(task, evidence, 1.0)

        assert len(result.evaluation_result.evidence_index) == 3

    def test_error_result_has_empty_model_results(self) -> None:
        """Error result must have empty model results."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        result = exp._build_error_result(task, [], 0.0)

        assert result.model_results == []
        assert result.total_claims == 0


# ============================================================================
# 💰 Cost Accumulation Tests
# ============================================================================


class TestCostAccumulation:
    """Tests for cost accumulation across models 💰."""

    @pytest.mark.asyncio
    async def test_total_cost_accumulated(self) -> None:
        """Total cost should sum all successful model costs."""
        task = build_sample_evaluation_task()
        exp = _build_discovery_synthesis_exp(task=task)

        # 🤖 Mock LLM with known token counts
        valid_response = _build_valid_synthesis_response()
        mock_response = _make_mock_llm_response(
            valid_response,
            prompt_tokens=1000,
            completion_tokens=500,
        )
        exp.llm = MagicMock()
        exp.llm.query = MagicMock(return_value=mock_response)

        result = await exp.run_synthesis(
            task=task,
            config=_build_discovery_config(
                models=["model-a", "model-b", "model-c"],
            ),
            all_evidence=_build_evidence(2),
            all_claims=_build_claims(1),
            coverage_ratio=0.7,
            round_summaries=[],
        )

        # ✅ Cost should be 3x the single model cost
        single_cost = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        expected_total = single_cost * 3
        assert abs(result.cost_usd - expected_total) < 0.001
