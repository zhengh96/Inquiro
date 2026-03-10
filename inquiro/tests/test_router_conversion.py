"""Tests for API → Core model conversion functions 🧪.

Validates that ``_to_evaluation_task`` and ``_to_synthesis_task``
correctly map API request schemas (with stricter bounds) to internal
core models expected by EvalTaskRunner.

Test coverage:
- Default config round-trip for research
- Custom agent/tools config for research
- Default config round-trip for synthesis
- Synthesis with additional_research disabled
- Synthesis with additional_research enabled
"""

from __future__ import annotations

from typing import Any

import pytest

from inquiro.api.router import _to_evaluation_task, _to_synthesis_task
from inquiro.api.schemas import (
    AdditionalResearchConfig,
    AgentConfig,
    Checklist,
    ChecklistItem,
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
from inquiro.core.types import (
    EvaluationTask,
    SynthesisTask,
)


# ============================================================
# 📋 Shared fixtures
# ============================================================


@pytest.fixture
def minimal_output_schema() -> dict[str, Any]:
    """Minimal JSON Schema for output validation 📋."""
    return {
        "type": "object",
        "required": ["decision", "confidence"],
        "properties": {
            "decision": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }


@pytest.fixture
def default_research_request(
    minimal_output_schema: dict[str, Any],
) -> ResearchRequest:
    """ResearchRequest with all defaults 🔬.

    Uses default AgentConfig, ToolsConfig, QualityGateConfig,
    CostGuardConfig, and EnsembleConfig values.
    """
    return ResearchRequest(
        task_id="res-default-001",
        task=ResearchTaskPayload(
            topic="Test research topic for conversion",
            output_schema=minimal_output_schema,
        ),
    )


@pytest.fixture
def custom_research_request(
    minimal_output_schema: dict[str, Any],
) -> ResearchRequest:
    """ResearchRequest with fully customized configs 🔧.

    Exercises all config fields to ensure complete mapping.
    """
    return ResearchRequest(
        task_id="res-custom-001",
        task=ResearchTaskPayload(
            topic="Custom topic with full config",
            rules="# Custom Rules\nEvaluate thoroughly.",
            checklist=Checklist(
                required=[
                    ChecklistItem(
                        id="chk_1",
                        description="First required item",
                        keywords=["alpha", "beta"],
                        suggested_sources=["perplexity"],
                    ),
                    ChecklistItem(
                        id="chk_2",
                        description="Second required item",
                        keywords=["gamma"],
                    ),
                ],
                optional=[
                    ChecklistItem(
                        id="chk_opt",
                        description="Optional deep-dive",
                    ),
                ],
                coverage_threshold=0.9,
            ),
            decision_guidance=DecisionGuidance(
                positive=["Strong evidence"],
                cautious=["Mixed signals"],
                negative=["Weak support"],
            ),
            output_schema=minimal_output_schema,
        ),
        agent_config=AgentConfig(
            model="claude-opus-4-20250514",
            max_turns=50,
            temperature=0.7,
            system_prompt_template="You are a custom agent.",
        ),
        tools_config=ToolsConfig(
            mcp_servers=["perplexity", "biomcp", "custom-mcp"],
            mcp_config_override={"perplexity": {"api_key": "test"}},
        ),
        ensemble_config=EnsembleConfig(enabled=False),
        quality_gate=QualityGateConfig(
            enabled=True,
            max_retries=3,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=False,
                evidence_reference_check=True,
            ),
        ),
        cost_guard=CostGuardConfig(
            max_cost_per_task=5.0,
            overspend_strategy=OverspendStrategy.HARD_STOP,
        ),
        callback_url="https://example.com/webhook",
    )


@pytest.fixture
def default_synthesis_request(
    minimal_output_schema: dict[str, Any],
) -> SynthesizeRequest:
    """SynthesizeRequest with all defaults 📊.

    Uses two minimal input reports and default configs.
    """
    return SynthesizeRequest(
        task_id="syn-default-001",
        task=SynthesisTaskPayload(
            objective="Synthesize findings from reports",
            input_reports=[
                InputReport(
                    report_id="rpt-1",
                    label="Report A",
                    content={"decision": "positive", "confidence": 0.8},
                ),
                InputReport(
                    report_id="rpt-2",
                    label="Report B",
                    content={"decision": "cautious", "confidence": 0.6},
                ),
            ],
            output_schema=minimal_output_schema,
        ),
    )


@pytest.fixture
def synthesis_no_additional_research(
    minimal_output_schema: dict[str, Any],
) -> SynthesizeRequest:
    """SynthesizeRequest with additional research disabled 🚫.

    Ensures the conversion correctly sets additional_research_config
    to None when allow_additional_research is False.
    """
    return SynthesizeRequest(
        task_id="syn-no-ar-001",
        task=SynthesisTaskPayload(
            objective="Synthesize without additional research",
            input_reports=[
                InputReport(
                    report_id="rpt-1",
                    label="Only report",
                    content={"decision": "positive", "confidence": 0.9},
                ),
            ],
            allow_additional_research=False,
            output_schema=minimal_output_schema,
        ),
    )


@pytest.fixture
def synthesis_with_additional_research(
    minimal_output_schema: dict[str, Any],
) -> SynthesizeRequest:
    """SynthesizeRequest with additional research explicitly configured 🔬.

    Provides custom AdditionalResearchConfig to verify full mapping.
    """
    return SynthesizeRequest(
        task_id="syn-ar-001",
        task=SynthesisTaskPayload(
            objective="Synthesize with deep-dive capability",
            input_reports=[
                InputReport(
                    report_id="rpt-1",
                    label="Report Alpha",
                    content={"decision": "positive", "confidence": 0.8},
                ),
                InputReport(
                    report_id="rpt-2",
                    label="Report Beta",
                    content={"decision": "negative", "confidence": 0.5},
                ),
            ],
            synthesis_rules="# Rules\nCross-reference all findings.",
            allow_additional_research=True,
            additional_research_config=AdditionalResearchConfig(
                max_tasks=5,
                cost_budget=4.0,
                tools_config=ToolsConfig(
                    mcp_servers=["perplexity", "biomcp"],
                ),
            ),
            output_schema=minimal_output_schema,
        ),
        agent_config=AgentConfig(
            model="claude-opus-4-20250514",
            max_turns=40,
            temperature=0.5,
            system_prompt_template="Custom synthesis prompt.",
        ),
        quality_gate=QualityGateConfig(
            enabled=True,
            max_retries=4,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=True,
                cross_reference_check=True,
            ),
        ),
        cost_guard=CostGuardConfig(
            max_cost_per_task=8.0,
            overspend_strategy=OverspendStrategy.HARD_STOP,
        ),
        callback_url="https://example.com/synthesis-done",
    )


# ============================================================
# 🔬 TestToEvaluationTask
# ============================================================


class TestToEvaluationTask:
    """Tests for _to_evaluation_task conversion 🔬."""

    def test_default_config_produces_valid_task(
        self,
        default_research_request: ResearchRequest,
    ) -> None:
        """Default ResearchRequest converts to a valid EvaluationTask 🎯."""
        result = _to_evaluation_task(default_research_request)

        assert isinstance(result, EvaluationTask)
        assert result.task_id == "res-default-001"
        assert result.topic == "Test research topic for conversion"
        assert result.rules == ""
        assert result.decision_guidance is None
        assert result.callback_url is None

        # ✅ Verify default agent config transferred
        assert result.agent_config.model == "claude-sonnet-4-20250514"
        assert result.agent_config.max_turns == 30
        assert result.agent_config.temperature == 0.3
        assert result.agent_config.system_prompt_template is None

        # ✅ Verify default tools config transferred
        assert result.tools_config.mcp_servers == []
        assert result.tools_config.mcp_config_override == {}

        # ✅ Verify default quality gate transferred
        assert result.quality_gate.enabled is True
        assert result.quality_gate.max_retries == 2

        # ✅ Verify default cost guard transferred
        assert result.cost_guard.max_cost_per_task == 1.5

    def test_custom_config_maps_all_fields(
        self,
        custom_research_request: ResearchRequest,
    ) -> None:
        """Custom ResearchRequest maps all config fields correctly 🔧."""
        result = _to_evaluation_task(custom_research_request)

        assert isinstance(result, EvaluationTask)
        assert result.task_id == "res-custom-001"
        assert result.topic == "Custom topic with full config"
        assert "Custom Rules" in result.rules

        # ✅ Verify custom agent config
        assert result.agent_config.model == "claude-opus-4-20250514"
        assert result.agent_config.max_turns == 50
        assert result.agent_config.temperature == 0.7
        assert result.agent_config.system_prompt_template == ("You are a custom agent.")

        # ✅ Verify custom tools config
        assert result.tools_config.mcp_servers == [
            "perplexity",
            "biomcp",
            "custom-mcp",
        ]
        assert "perplexity" in result.tools_config.mcp_config_override

        # ✅ Verify custom quality gate
        assert result.quality_gate.enabled is True
        assert result.quality_gate.max_retries == 3
        assert result.quality_gate.checks.schema_validation is True
        assert result.quality_gate.checks.coverage_check is False

        # ✅ Verify custom cost guard
        assert result.cost_guard.max_cost_per_task == 5.0
        assert result.cost_guard.overspend_strategy.value == "HardStop"

        # ✅ Verify callback URL
        assert result.callback_url == "https://example.com/webhook"

    def test_checklist_and_guidance_transferred(
        self,
        custom_research_request: ResearchRequest,
    ) -> None:
        """Checklist and decision guidance carry through conversion 📋."""
        result = _to_evaluation_task(custom_research_request)

        # ✅ Checklist items preserved
        assert len(result.checklist.required) == 2
        assert result.checklist.required[0].id == "chk_1"
        assert result.checklist.required[0].keywords == [
            "alpha",
            "beta",
        ]
        assert len(result.checklist.optional) == 1
        assert result.checklist.coverage_threshold == 0.9

        # ✅ Decision guidance preserved
        assert result.decision_guidance is not None
        assert result.decision_guidance.positive == ["Strong evidence"]
        assert result.decision_guidance.negative == ["Weak support"]

    def test_output_schema_preserved(
        self,
        default_research_request: ResearchRequest,
        minimal_output_schema: dict[str, Any],
    ) -> None:
        """Output schema passes through unchanged 📦."""
        result = _to_evaluation_task(default_research_request)
        assert result.output_schema == minimal_output_schema

    def test_ensemble_config_passed_through(
        self,
        custom_research_request: ResearchRequest,
    ) -> None:
        """Ensemble config is passed through from request 🎭."""
        result = _to_evaluation_task(custom_research_request)
        assert result.ensemble_config.enabled is False


# ============================================================
# 📊 TestToSynthesisTask
# ============================================================


class TestToSynthesisTask:
    """Tests for _to_synthesis_task conversion 📊."""

    def test_default_config_produces_valid_task(
        self,
        default_synthesis_request: SynthesizeRequest,
    ) -> None:
        """Default SynthesizeRequest converts to valid SynthesisTask 🎯."""
        result = _to_synthesis_task(default_synthesis_request)

        assert isinstance(result, SynthesisTask)
        assert result.task_id == "syn-default-001"
        assert result.topic == "Synthesize findings from reports"
        assert result.synthesis_rules == ""
        assert result.callback_url is None

        # ✅ Verify input reports transferred
        assert len(result.input_reports) == 2
        assert result.input_reports[0].report_id == "rpt-1"
        assert result.input_reports[1].report_id == "rpt-2"

        # ✅ Verify default agent config
        assert result.agent_config.model == "claude-sonnet-4-20250514"
        assert result.agent_config.max_turns == 30

        # ✅ Verify default quality gate
        assert result.quality_gate.enabled is True

        # ✅ Verify default cost guard
        assert result.cost_guard.max_cost_per_task == 1.5

    def test_additional_research_disabled(
        self,
        synthesis_no_additional_research: SynthesizeRequest,
    ) -> None:
        """Disabled additional research results in None config 🚫."""
        result = _to_synthesis_task(synthesis_no_additional_research)

        assert isinstance(result, SynthesisTask)
        assert result.task_id == "syn-no-ar-001"
        assert result.allow_additional_research is False
        assert result.additional_research_config is None

    def test_additional_research_enabled_maps_config(
        self,
        synthesis_with_additional_research: SynthesizeRequest,
    ) -> None:
        """Enabled additional research maps full config correctly 🔬."""
        result = _to_synthesis_task(synthesis_with_additional_research)

        assert isinstance(result, SynthesisTask)
        assert result.task_id == "syn-ar-001"
        assert result.allow_additional_research is True

        # ✅ Additional research config fully mapped
        ar_config = result.additional_research_config
        assert ar_config is not None
        assert ar_config.max_tasks == 5
        assert ar_config.cost_budget == 4.0
        assert ar_config.tools_config.mcp_servers == [
            "perplexity",
            "biomcp",
        ]

    def test_custom_agent_and_quality_gate(
        self,
        synthesis_with_additional_research: SynthesizeRequest,
    ) -> None:
        """Custom agent config and quality gate map correctly 🔧."""
        result = _to_synthesis_task(synthesis_with_additional_research)

        # ✅ Custom agent config
        assert result.agent_config.model == "claude-opus-4-20250514"
        assert result.agent_config.max_turns == 40
        assert result.agent_config.temperature == 0.5
        assert result.agent_config.system_prompt_template == (
            "Custom synthesis prompt."
        )

        # ✅ Custom quality gate
        assert result.quality_gate.enabled is True
        assert result.quality_gate.max_retries == 4
        assert result.quality_gate.checks.cross_reference_check is True

        # ✅ Custom cost guard
        assert result.cost_guard.max_cost_per_task == 8.0
        assert result.cost_guard.overspend_strategy.value == "HardStop"

        # ✅ Callback URL
        assert result.callback_url == ("https://example.com/synthesis-done")

    def test_objective_maps_to_topic(
        self,
        default_synthesis_request: SynthesizeRequest,
    ) -> None:
        """SynthesisTaskPayload.objective maps to SynthesisTask.topic 🏷️."""
        result = _to_synthesis_task(default_synthesis_request)

        # ✅ The API uses "objective" but the core model uses "topic"
        assert result.topic == (default_synthesis_request.task.objective)

    def test_synthesis_rules_transferred(
        self,
        synthesis_with_additional_research: SynthesizeRequest,
    ) -> None:
        """Synthesis rules carry through conversion 📝."""
        result = _to_synthesis_task(synthesis_with_additional_research)
        assert "Cross-reference all findings" in result.synthesis_rules

    def test_output_schema_preserved(
        self,
        default_synthesis_request: SynthesizeRequest,
        minimal_output_schema: dict[str, Any],
    ) -> None:
        """Output schema passes through unchanged 📦."""
        result = _to_synthesis_task(default_synthesis_request)
        assert result.output_schema == minimal_output_schema
