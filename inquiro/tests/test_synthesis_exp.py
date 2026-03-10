"""Integration tests for SynthesisExp lifecycle 🧪.

Tests cover:
    - Prompt rendering (system prompt with report list, synthesis rules)
    - Agent creation with ReadReport + RequestResearch + Finish tools
    - Additional research (deep-dive) triggering
    - Quality Gate retry logic
    - Event emission during lifecycle
    - Result extraction from trajectory
    - Full lifecycle integration tests
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


from inquiro.infrastructure.cost_tracker import CostTracker
from inquiro.infrastructure.event_emitter import InquiroEvent
from inquiro.infrastructure.quality_gate import QualityGateResult

from inquiro.tests.mock_helpers import (
    build_invalid_research_result,
    build_mock_agent_for_synthesis,
    build_sample_synthesis_task,
    build_valid_synthesis_result,
    create_finish_step,
    create_synthesis_exp,
    create_trajectory,
)


# ============================================================================
# 📝 Prompt Rendering Tests
# ============================================================================


class TestSynthesisExpPromptRendering:
    """Tests for SynthesisExp._render_system_prompt() and _render_user_prompt() 📝."""

    def test_system_prompt_includes_topic(self) -> None:
        """System prompt should reference the synthesis topic in synthesis_rules."""
        # Arrange
        task = build_sample_synthesis_task(
            synthesis_rules="Evaluate clinical feasibility of the target.",
        )
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert
        assert "clinical feasibility" in prompt.lower()

    def test_system_prompt_includes_report_list(self) -> None:
        """System prompt should list all input report labels and IDs."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert — all 3 report IDs listed
        assert "report-001" in prompt
        assert "report-002" in prompt
        assert "report-003" in prompt
        # Labels should appear too
        assert "Target Biology" in prompt
        assert "Biomarker Availability" in prompt

    def test_system_prompt_includes_synthesis_rules(self) -> None:
        """System prompt should inject caller-provided synthesis rules."""
        # Arrange
        rules = "Cross-reference findings across all dimensions."
        task = build_sample_synthesis_task(synthesis_rules=rules)
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert
        assert "SYNTHESIS RULES" in prompt
        assert rules in prompt

    def test_system_prompt_includes_output_schema(self) -> None:
        """System prompt should include the output JSON Schema."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert
        assert "OUTPUT FORMAT" in prompt
        assert '"decision"' in prompt
        assert '"source_reports"' in prompt
        assert '"cross_references"' in prompt

    def test_system_prompt_deep_dive_enabled(self) -> None:
        """System prompt should mention request_research when deep-dive enabled."""
        # Arrange
        task = build_sample_synthesis_task(allow_additional_research=True)
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert — deep-dive section mentions request_research tool
        assert "request_research" in prompt

    def test_system_prompt_available_skills_summary(self) -> None:
        """When skill_registry is set, available_skills is summary + get_reference."""
        from unittest.mock import MagicMock

        task = build_sample_synthesis_task()
        registry = MagicMock()
        skill = MagicMock()
        skill.meta_info = MagicMock()
        skill.meta_info.name = "cross-reference-rules"
        skill.meta_info.description = "Rules for cross-report synthesis."
        registry.get_all_skills.return_value = [skill]
        exp = create_synthesis_exp(task=task, skill_registry=registry)
        prompt = exp._render_system_prompt()
        assert "get_reference" in prompt
        assert "cross-reference-rules" in prompt
        assert "Available Skills" in prompt

    def test_system_prompt_deep_dive_disabled(self) -> None:
        """System prompt should say deep-dive is disabled when not allowed."""
        # Arrange
        task = build_sample_synthesis_task(allow_additional_research=False)
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_system_prompt()

        # Assert
        assert "disabled" in prompt.lower()

    def test_user_prompt_includes_report_ids(self) -> None:
        """User prompt should list all input report IDs."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Act
        prompt = exp._render_user_prompt()

        # Assert
        assert "report-001" in prompt
        assert "report-002" in prompt
        assert "report-003" in prompt
        assert "Topic" in prompt


# ============================================================================
# 🤖 Agent Creation Tests
# ============================================================================


class TestSynthesisExpAgentCreation:
    """Tests for SynthesisExp._create_synthesis_agent() 🤖."""

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_creates_synthesis_agent(self, _mock_session: Any) -> None:
        """_create_synthesis_agent should return a SynthesisAgent."""
        # Arrange
        from inquiro.agents.synthesis_agent import SynthesisAgent

        exp = create_synthesis_exp()

        # Act
        agent, _rr_tool = exp._create_synthesis_agent()

        # Assert
        assert isinstance(agent, SynthesisAgent)

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_agent_has_read_report_tool(self, _mock_session: Any) -> None:
        """Agent should have ReadReportTool registered."""
        # Arrange
        exp = create_synthesis_exp()

        # Act
        agent, _ = exp._create_synthesis_agent()

        # Assert
        read_tool = agent.tools.get_tool("read_report")
        assert read_tool is not None
        assert read_tool.name == "read_report"

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_agent_has_finish_tool(self, _mock_session: Any) -> None:
        """Agent should have InquiroFinishTool registered."""
        # Arrange
        exp = create_synthesis_exp()

        # Act
        agent, _ = exp._create_synthesis_agent()

        # Assert
        finish_tool = agent.tools.get_tool("finish")
        assert finish_tool is not None
        assert finish_tool.name == "finish"

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_agent_has_request_research_when_deep_dive_enabled(
        self,
        _mock_session: Any,
    ) -> None:
        """Agent should have RequestResearchTool when deep_dive=True."""
        # Arrange
        task = build_sample_synthesis_task(allow_additional_research=True)
        exp = create_synthesis_exp(task=task)

        # Act
        agent, rr_tool = exp._create_synthesis_agent()

        # Assert — request_research tool is returned and registered
        assert rr_tool is not None
        rr_in_agent = agent.tools.get_tool("request_research")
        assert rr_in_agent is not None

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_agent_no_request_research_when_deep_dive_disabled(
        self,
        _mock_session: Any,
    ) -> None:
        """Agent should NOT have RequestResearchTool when deep_dive=False."""
        # Arrange
        task = build_sample_synthesis_task(allow_additional_research=False)
        exp = create_synthesis_exp(task=task)

        # Act
        agent, rr_tool = exp._create_synthesis_agent()

        # Assert
        assert rr_tool is None
        rr_in_agent = agent.tools.get_tool("request_research")
        assert rr_in_agent is None

    @patch(
        "inquiro.agents.synthesis_agent.SynthesisAgent._create_local_session",
        return_value=MagicMock(),
    )
    def test_reports_preloaded_into_read_report_tool(
        self,
        _mock_session: Any,
    ) -> None:
        """Input reports should be pre-loaded into ReadReportTool."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Act
        agent, _ = exp._create_synthesis_agent()

        # Assert — ReadReportTool has all 3 reports loaded
        read_tool = agent.tools.get_tool("read_report")
        assert read_tool is not None
        # Access internal _reports dict
        assert "report-001" in read_tool._reports
        assert "report-002" in read_tool._reports
        assert "report-003" in read_tool._reports


# ============================================================================
# 🔬 Additional Research (Deep Dive) Tests
# ============================================================================


class TestSynthesisExpAdditionalResearch:
    """Tests for SynthesisExp additional research (deep-dive) capability 🔬."""

    def test_request_research_calls_task_runner(self) -> None:
        """_handle_research_request should call task_runner.run_research_sync."""
        # Arrange
        mock_task_runner = MagicMock()
        mock_research_result = MagicMock()
        mock_research_result.model_dump.return_value = {
            "decision": "positive",
            "confidence": 0.9,
        }
        mock_research_result.decision.value = "positive"
        mock_task_runner.run_research_sync.return_value = mock_research_result

        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task, task_runner=mock_task_runner)

        # Build a mock EvaluationTask with proper agent_config 🔧
        mock_eval_task = MagicMock()
        mock_eval_task.task_id = "deepdive-001"
        mock_eval_task.topic = "Detailed mechanism of action"
        mock_eval_task.agent_config.max_turns = 10
        mock_eval_task.output_schema = {}
        mock_eval_task.tools_config.mcp_servers = []

        mock_agent = MagicMock()

        # Act
        exp._handle_research_request(mock_eval_task, mock_agent)

        # Assert
        mock_task_runner.run_research_sync.assert_called_once_with(
            mock_eval_task,
        )

    def test_additional_research_result_available_to_agent(self) -> None:
        """Research result should be injected into agent via add_research_result."""
        # Arrange
        mock_task_runner = MagicMock()
        result_dict = {"decision": "cautious", "confidence": 0.65}
        mock_research_result = MagicMock()
        mock_research_result.model_dump.return_value = result_dict
        mock_research_result.decision.value = "cautious"
        mock_task_runner.run_research_sync.return_value = mock_research_result

        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task, task_runner=mock_task_runner)

        mock_eval_task = MagicMock()
        mock_eval_task.task_id = "deepdive-002"
        mock_eval_task.topic = "Biomarker sensitivity analysis"
        mock_eval_task.agent_config.max_turns = 10
        mock_eval_task.output_schema = {}
        mock_eval_task.tools_config.mcp_servers = []

        mock_agent = MagicMock()

        # Act
        exp._handle_research_request(mock_eval_task, mock_agent)

        # Assert — result injected into agent
        mock_agent.add_research_result.assert_called_once_with(result_dict)

        # Assert — recorded in _additional_research
        assert len(exp._additional_research) == 1
        assert exp._additional_research[0]["task_id"] == "deepdive-002"
        assert "result" in exp._additional_research[0]

    def test_additional_research_failure_handled(self) -> None:
        """Failed research should be recorded but not crash synthesis."""
        # Arrange
        mock_task_runner = MagicMock()
        mock_task_runner.run_research_sync.side_effect = RuntimeError(
            "LLM API timeout",
        )

        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task, task_runner=mock_task_runner)

        mock_eval_task = MagicMock()
        mock_eval_task.task_id = "deepdive-fail"
        mock_eval_task.topic = "Failing topic"
        mock_eval_task.agent_config.max_turns = 10
        mock_eval_task.output_schema = {}
        mock_eval_task.tools_config.mcp_servers = []

        mock_agent = MagicMock()

        # Act — should NOT raise
        exp._handle_research_request(mock_eval_task, mock_agent)

        # Assert — failure recorded
        assert len(exp._additional_research) == 1
        assert "error" in exp._additional_research[0]
        assert "LLM API timeout" in exp._additional_research[0]["error"]

        # Assert — agent NOT called with result
        mock_agent.add_research_result.assert_not_called()


# ============================================================================
# ✅ Quality Gate Retry Tests
# ============================================================================


class TestSynthesisExpQualityGateRetry:
    """Tests for SynthesisExp run_sync() quality gate retry logic ✅."""

    def test_passes_on_first_attempt(self) -> None:
        """Valid result should pass QG on first attempt."""
        # Arrange
        task = build_sample_synthesis_task(max_retries=2)
        valid_result = build_valid_synthesis_result()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        # Build mock agent
        mock_agent = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )
        mock_rr_tool = None

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, mock_rr_tool),
        ):
            result = exp.run_sync()

        # Assert
        assert result.decision.value == "positive"
        assert result.confidence == 0.80
        # _step called once (returned True=finish)
        assert mock_agent._step.call_count == 1

    def test_retries_on_hard_failure(self) -> None:
        """Hard QG failure should trigger retry on the synthesis loop."""
        # Arrange
        task = build_sample_synthesis_task(max_retries=1)
        invalid_result = build_invalid_research_result()  # Missing "decision"
        valid_result = build_valid_synthesis_result()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        # First attempt: invalid, second: valid
        mock_agent_1 = build_mock_agent_for_synthesis(
            result_dict=invalid_result,
            task_id=task.task_id,
        )
        mock_agent_2 = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        call_count = {"n": 0}

        def create_agent_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_agent_1, None
            return mock_agent_2, None

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            side_effect=create_agent_side_effect,
        ):
            result = exp.run_sync()

        # Assert — retried once, then passed
        assert call_count["n"] == 2
        assert result.decision.value == "positive"

    def test_max_retries_exhausted(self) -> None:
        """Exhausted retries should return best-effort result."""
        # Arrange
        task = build_sample_synthesis_task(max_retries=1)
        invalid_result = build_invalid_research_result()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        mock_agent = build_mock_agent_for_synthesis(
            result_dict=invalid_result,
            task_id=task.task_id,
        )

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, None),
        ):
            result = exp.run_sync()

        # Assert — returns best-effort result, defaults to cautious
        assert result.task_id == task.task_id
        assert result.decision.value == "cautious"


# ============================================================================
# 📡 Event Tests
# ============================================================================


class TestSynthesisExpEvents:
    """Tests for SynthesisExp event emission 📡."""

    def test_emits_synthesis_started_event(self) -> None:
        """run_sync should emit SYNTHESIS_STARTED at the beginning."""
        # Arrange
        task = build_sample_synthesis_task()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        valid_result = build_valid_synthesis_result()
        mock_agent = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, None),
        ):
            exp.run_sync()

        # Assert — SYNTHESIS_STARTED emitted
        started_calls = [
            c
            for c in event_emitter.emit.call_args_list
            if c[0][0] == InquiroEvent.SYNTHESIS_STARTED
        ]
        assert len(started_calls) == 1
        payload = started_calls[0][0][2]
        assert payload["topic"] == task.topic
        assert payload["report_count"] == 3

    def test_emits_quality_gate_result(self) -> None:
        """run_sync should emit QUALITY_GATE_RESULT after validation."""
        # Arrange
        task = build_sample_synthesis_task()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        valid_result = build_valid_synthesis_result()
        mock_agent = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, None),
        ):
            exp.run_sync()

        # Assert — QUALITY_GATE_RESULT emitted
        qg_calls = [
            c
            for c in event_emitter.emit.call_args_list
            if c[0][0] == InquiroEvent.QUALITY_GATE_RESULT
        ]
        assert len(qg_calls) == 1
        payload = qg_calls[0][0][2]
        assert "passed" in payload


# ============================================================================
# 📝 Result Extraction Tests
# ============================================================================


class TestSynthesisExpResultExtraction:
    """Tests for SynthesisExp result extraction and building 📝."""

    def test_extracts_result_from_finish_tool(self) -> None:
        """Should extract synthesis result from finish tool call."""
        # Arrange
        exp = create_synthesis_exp()
        expected = build_valid_synthesis_result()
        finish_step = create_finish_step(expected)
        traj = create_trajectory(steps=[finish_step])

        # Act
        result = exp._extract_result(traj)

        # Assert
        assert result["decision"] == "positive"
        assert result["confidence"] == 0.80
        assert "source_reports" in result
        assert "cross_references" in result

    def test_empty_trajectory_returns_empty_dict(self) -> None:
        """Empty trajectory should return empty dict."""
        # Arrange
        exp = create_synthesis_exp()
        traj = create_trajectory(steps=[])

        # Act
        result = exp._extract_result(traj)

        # Assert
        assert result == {}

    def test_build_result_includes_warnings(self) -> None:
        """Soft QG failures should cap confidence in the built result."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        raw_result = build_valid_synthesis_result()
        raw_result["confidence"] = 0.95  # High confidence

        # Create a QG result with soft failures and confidence_cap
        qg_result = QualityGateResult(
            passed=True,
            hard_failures=[],
            soft_failures=["evidence_reference: orphan evidence E99"],
            confidence_cap=0.69,
        )
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, qg_result, traj)

        # Assert — confidence capped from 0.95 to 0.69
        assert result.confidence == 0.69
        assert result.decision.value == "positive"

    def test_build_result_includes_cost(self) -> None:
        """_build_result should work correctly when cost_tracker has data."""
        # Arrange
        task = build_sample_synthesis_task()
        cost_tracker = CostTracker(max_per_task=10.0, max_total=100.0)
        # Record some cost
        cost_tracker.record(
            task_id=task.task_id,
            model="claude-sonnet-4-20250514",
            input_tokens=5000,
            output_tokens=2000,
        )
        exp = create_synthesis_exp(task=task, cost_tracker=cost_tracker)

        raw_result = build_valid_synthesis_result()
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act — should not crash even with cost data
        result = exp._build_result(raw_result, None, traj)

        # Assert — result is valid regardless of cost tracking
        assert result.decision.value == "positive"
        assert result.confidence == 0.80
        # Cost tracker should have recorded data
        assert cost_tracker.get_task_cost(task.task_id) > 0


# ============================================================================
# 🔗 Integration Tests
# ============================================================================


class TestSynthesisExpIntegration:
    """Integration-like tests for SynthesisExp lifecycle 🔗."""

    def test_full_lifecycle_with_mocked_agent(self) -> None:
        """Test complete lifecycle: create -> run -> QG -> result."""
        # Arrange
        task = build_sample_synthesis_task()
        valid_result = build_valid_synthesis_result()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        mock_agent = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, None),
        ):
            result = exp.run_sync()

        # Assert — complete SynthesisResult returned
        assert result.task_id == task.task_id
        assert result.decision.value == "positive"
        assert result.confidence == 0.80
        assert "report-001" in result.source_reports
        assert len(result.cross_references) == 1

        # Assert — events emitted
        event_types = [c[0][0] for c in event_emitter.emit.call_args_list]
        assert InquiroEvent.SYNTHESIS_STARTED in event_types
        assert InquiroEvent.QUALITY_GATE_RESULT in event_types
        assert InquiroEvent.TASK_COMPLETED in event_types

    def test_lifecycle_with_additional_research(self) -> None:
        """Lifecycle with deep-dive should include deep_dives_triggered."""
        # Arrange
        task = build_sample_synthesis_task(allow_additional_research=True)
        event_emitter = MagicMock()
        mock_task_runner = MagicMock()

        # Mock research result for deep-dive
        mock_research_result = MagicMock()
        mock_research_result.model_dump.return_value = {
            "decision": "positive",
            "confidence": 0.88,
        }
        mock_research_result.decision.value = "positive"
        mock_task_runner.run_research_sync.return_value = mock_research_result

        exp = create_synthesis_exp(
            task=task,
            event_emitter=event_emitter,
            task_runner=mock_task_runner,
        )

        # Simulate: agent produces valid result AND deep-dive was triggered
        valid_result = build_valid_synthesis_result()

        # Pre-record a deep-dive in _additional_research
        exp._additional_research.append(
            {
                "task_id": "deepdive-int-001",
                "topic": "Mechanism of EGFR resistance",
                "result": {"decision": "cautious", "confidence": 0.7},
            }
        )

        mock_agent = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            return_value=(mock_agent, None),
        ):
            result = exp.run_sync()

        # Assert — deep_dives_triggered populated
        assert len(result.deep_dives_triggered) == 1
        assert result.deep_dives_triggered[0].topic == ("Mechanism of EGFR resistance")

    def test_lifecycle_with_quality_gate_retry(self) -> None:
        """Lifecycle with QG retry on first attempt should succeed on second."""
        # Arrange
        task = build_sample_synthesis_task(max_retries=1)
        invalid_result = build_invalid_research_result()
        valid_result = build_valid_synthesis_result()
        event_emitter = MagicMock()
        exp = create_synthesis_exp(task=task, event_emitter=event_emitter)

        mock_agent_1 = build_mock_agent_for_synthesis(
            result_dict=invalid_result,
            task_id=task.task_id,
        )
        mock_agent_2 = build_mock_agent_for_synthesis(
            result_dict=valid_result,
            task_id=task.task_id,
        )

        call_count = {"n": 0}

        def create_agent_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                return mock_agent_1, None
            return mock_agent_2, None

        # Act
        with patch.object(
            exp,
            "_create_synthesis_agent",
            side_effect=create_agent_side_effect,
        ):
            result = exp.run_sync()

        # Assert — final result from second attempt
        assert result.decision.value == "positive"
        assert call_count["n"] == 2

        # Assert — retry event emitted
        retry_calls = [
            c
            for c in event_emitter.emit.call_args_list
            if c[0][0] == InquiroEvent.QUALITY_GATE_RETRY
        ]
        assert len(retry_calls) >= 1


# ============================================================================
# 🔍 Evidence Backfill Tests
# ============================================================================


class TestCollectSourceEvidence:
    """Tests for SynthesisExp._collect_source_evidence() 🔍."""

    def test_collects_evidence_from_input_reports(self) -> None:
        """Should collect evidence from all input reports with source_report_id."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Act
        evidence = exp._collect_source_evidence()

        # Assert — 3 reports × 1 evidence each = 3 evidence items
        assert len(evidence) == 3
        ids = {ev.id for ev in evidence}
        assert ids == {"E1", "E2", "E3"}

        # Assert — source_report_id is set
        source_map = {ev.id: ev.source_report_id for ev in evidence}
        assert source_map["E1"] == "report-001"
        assert source_map["E2"] == "report-002"
        assert source_map["E3"] == "report-003"

    def test_collects_evidence_from_deep_dives(self) -> None:
        """Should collect evidence from additional research results."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        # Inject a deep-dive result with evidence
        exp._additional_research.append(
            {
                "task_id": "dd-001",
                "topic": "Mechanism deep-dive",
                "result": {
                    "decision": "positive",
                    "confidence": 0.9,
                    "evidence_index": [
                        {
                            "id": "DD_E1",
                            "source": "pubmed",
                            "query": "EGFR mechanism",
                            "summary": "Detailed mechanism analysis",
                        },
                        {
                            "id": "DD_E2",
                            "source": "pubmed",
                            "query": "resistance mechanisms",
                            "summary": "Known resistance pathways",
                        },
                    ],
                    "gaps_remaining": ["long-term data"],
                },
            }
        )

        # Act
        evidence = exp._collect_source_evidence()

        # Assert — 3 from reports + 2 from deep-dive = 5
        assert len(evidence) == 5
        dd_evidence = [ev for ev in evidence if ev.source_report_id == "dd-001"]
        assert len(dd_evidence) == 2
        dd_ids = {ev.id for ev in dd_evidence}
        assert dd_ids == {"DD_E1", "DD_E2"}

    def test_skips_failed_deep_dives(self) -> None:
        """Should not crash on deep-dives with errors (no 'result' key)."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        exp._additional_research.append(
            {
                "task_id": "dd-fail",
                "topic": "Failed deep-dive",
                "error": "LLM timeout",
            }
        )

        # Act
        evidence = exp._collect_source_evidence()

        # Assert — only evidence from input reports (no crash)
        assert len(evidence) == 3

    def test_handles_invalid_evidence_data(self) -> None:
        """Should skip invalid evidence entries gracefully."""
        # Arrange
        from inquiro.core.types import InputReport

        task = build_sample_synthesis_task()
        # Add a report with invalid evidence data
        task.input_reports.append(
            InputReport(
                report_id="report-bad",
                label="Bad Report",
                content={
                    "evidence_index": [
                        "not_a_dict",  # Invalid: string instead of dict
                        {"id": "E_OK", "source": "test", "query": "q", "summary": "s"},
                    ],
                },
            )
        )
        exp = create_synthesis_exp(task=task)

        # Act
        evidence = exp._collect_source_evidence()

        # Assert — skips invalid, keeps valid
        ids = {ev.id for ev in evidence}
        assert "E_OK" in ids


class TestBuildResultRawOutput:
    """Tests for raw_output preservation in SynthesisExp._build_result() 📦."""

    def test_build_result_preserves_raw_output(self) -> None:
        """_build_result should store full raw_result as raw_output 📦."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        raw_result = build_valid_synthesis_result()
        # ✨ Add domain-specific fields that are NOT in SynthesisResult schema
        raw_result["summary"] = "TL;DR: Target looks promising."
        raw_result["cross_dimension_insights"] = [
            {"theme": "Biology-Safety alignment", "detail": "Consistent."},
        ]
        raw_result["information_gaps"] = [
            {"gap": "No Phase 3 data", "priority": "high"},
        ]
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert — raw_output preserves ALL keys including domain-specific ones
        assert result.raw_output is not None
        assert result.raw_output["summary"] == "TL;DR: Target looks promising."
        assert len(result.raw_output["cross_dimension_insights"]) == 1
        assert len(result.raw_output["information_gaps"]) == 1
        # Standard fields are also present in raw_output
        assert result.raw_output["decision"] == "positive"
        assert result.raw_output["confidence"] == 0.80

    def test_build_result_raw_output_empty_when_no_extra_fields(self) -> None:
        """raw_output should still contain standard fields even without extras ✅."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        raw_result = build_valid_synthesis_result()
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert — raw_output has standard fields
        assert result.raw_output["decision"] == "positive"
        assert "source_reports" in result.raw_output


class TestBuildResultEvidenceBackfill:
    """Tests for evidence backfill in SynthesisExp._build_result() 📊."""

    def test_build_result_merges_source_evidence(self) -> None:
        """_build_result should merge LLM evidence with source report evidence."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        raw_result = build_valid_synthesis_result()
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert — evidence_index is non-empty (backfilled from source reports)
        assert len(result.evidence_index) > 0

        # Assert — source report evidence has source_report_id set
        source_tagged = [ev for ev in result.evidence_index if ev.source_report_id]
        assert len(source_tagged) > 0

    def test_build_result_llm_evidence_takes_priority(self) -> None:
        """LLM-provided evidence with same ID should take priority over source."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        raw_result = build_valid_synthesis_result()
        # LLM provides E1 with different summary
        raw_result["evidence_index"] = [
            {
                "id": "E1",
                "source": "llm_synthesis",
                "query": "synthesized query",
                "summary": "LLM-synthesized summary",
            },
        ]
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert — E1 from LLM (no source_report_id)
        e1_items = [ev for ev in result.evidence_index if ev.id == "E1"]
        assert len(e1_items) == 1
        assert e1_items[0].source == "llm_synthesis"

        # Assert — E2, E3 from source reports (backfilled)
        backfilled = [
            ev for ev in result.evidence_index if ev.source_report_id is not None
        ]
        assert len(backfilled) >= 2


class TestDeepDiveRecordEnhanced:
    """Tests for enhanced DeepDiveRecord fields in _build_result() 🔬."""

    def test_deep_dive_record_includes_evidence_count(self) -> None:
        """DeepDiveRecord should include evidence_count from result."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        exp._additional_research.append(
            {
                "task_id": "dd-001",
                "topic": "Safety deep-dive",
                "result": {
                    "decision": "cautious",
                    "confidence": 0.7,
                    "evidence_index": [
                        {"id": "DE1", "source": "pubmed", "query": "q", "summary": "s"},
                        {
                            "id": "DE2",
                            "source": "pubmed",
                            "query": "q2",
                            "summary": "s2",
                        },
                    ],
                    "gaps_remaining": ["cardiac safety data", "long-term follow-up"],
                },
            }
        )

        raw_result = build_valid_synthesis_result()
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert — deep-dive record has evidence_count and gaps
        assert len(result.deep_dives_triggered) == 1
        dd = result.deep_dives_triggered[0]
        assert dd.evidence_count == 2
        assert dd.gaps_remaining == ["cardiac safety data", "long-term follow-up"]

    def test_deep_dive_record_failure_has_zero_evidence(self) -> None:
        """Failed deep-dive should have evidence_count=0 and empty gaps."""
        # Arrange
        task = build_sample_synthesis_task()
        exp = create_synthesis_exp(task=task)

        exp._additional_research.append(
            {
                "task_id": "dd-fail",
                "topic": "Failed deep-dive",
                "error": "LLM API timeout",
            }
        )

        raw_result = build_valid_synthesis_result()
        traj = create_trajectory(steps=[create_finish_step(raw_result)])

        # Act
        result = exp._build_result(raw_result, None, traj)

        # Assert
        assert len(result.deep_dives_triggered) == 1
        dd = result.deep_dives_triggered[0]
        assert dd.evidence_count == 0
        assert dd.gaps_remaining == []
        assert "Failed" in dd.result_summary
