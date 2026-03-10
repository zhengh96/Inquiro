"""Mock infrastructure for Inquiro Exp integration tests 🧪.

Provides:
- EvoMaster type factory functions (ToolCall, StepRecord, Trajectory)
- MockLLM for controlled agent responses
- Sample data builders for EvaluationTask and SynthesisTask
- Convenience helpers for building test Exp instances
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock

from evomaster.utils.types import (
    AssistantMessage,
    FunctionCall,
    StepRecord,
    ToolCall,
    ToolMessage,
    Trajectory,
)


# ============================================================
# 🏭 EvoMaster Type Factories
# ============================================================


def create_tool_call(
    name: str,
    arguments: dict[str, Any] | str,
    call_id: str | None = None,
) -> ToolCall:
    """Create a ToolCall with auto-generated ID 🔧.

    Args:
        name: Tool function name (e.g., "finish", "search").
        arguments: Tool arguments as dict or JSON string.
        call_id: Optional custom call ID; auto-generated if None.

    Returns:
        Fully constructed ToolCall instance.
    """
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return ToolCall(
        id=call_id or f"call_{uuid.uuid4().hex[:12]}",
        function=FunctionCall(name=name, arguments=arguments),
    )


def create_finish_tool_call(
    result_dict: dict[str, Any],
    task_completed: bool = True,
) -> ToolCall:
    """Create a finish tool call with nested result_json 📋.

    Args:
        result_dict: The structured result to embed.
        task_completed: Whether the task is fully complete.

    Returns:
        ToolCall for the finish tool.
    """
    arguments = {
        "result_json": json.dumps(result_dict),
        "task_completed": task_completed,
    }
    return create_tool_call("finish", arguments)


def create_step_record(
    step_id: int,
    tool_calls: list[ToolCall] | None = None,
    tool_responses: list[ToolMessage] | None = None,
    content: str | None = None,
) -> StepRecord:
    """Create a StepRecord with assistant message and tool responses 📝.

    Args:
        step_id: Step number (1-based).
        tool_calls: Optional list of tool calls in the assistant message.
        tool_responses: Optional list of tool response messages.
        content: Optional assistant message text content.

    Returns:
        Fully constructed StepRecord.
    """
    assistant_msg = AssistantMessage(
        content=content or "",
        tool_calls=tool_calls,
    )
    return StepRecord(
        step_id=step_id,
        assistant_message=assistant_msg,
        tool_responses=tool_responses or [],
    )


def create_finish_step(
    result_dict: dict[str, Any],
    step_id: int = 1,
) -> StepRecord:
    """Create a StepRecord containing a finish tool call 📋.

    Args:
        result_dict: The structured result to embed in the finish call.
        step_id: Step number (default 1).

    Returns:
        StepRecord with a finish tool call.
    """
    finish_call = create_finish_tool_call(result_dict)
    return create_step_record(
        step_id=step_id,
        tool_calls=[finish_call],
        content="Research complete. Submitting result.",
    )


def create_search_step(
    step_id: int = 1,
    tool_name: str = "perplexity_search",
    query: str = "test query",
) -> StepRecord:
    """Create a StepRecord with a search tool call 🔍.

    Args:
        step_id: Step number.
        tool_name: Name of the search tool.
        query: Search query string.

    Returns:
        StepRecord with a search tool call.
    """
    search_call = create_tool_call(
        name=tool_name,
        arguments={"query": query},
    )
    tool_response = ToolMessage(
        content=f"Search results for: {query}",
        tool_call_id=search_call.id,
        name=tool_name,
    )
    return create_step_record(
        step_id=step_id,
        tool_calls=[search_call],
        tool_responses=[tool_response],
        content=f"Searching for: {query}",
    )


def create_trajectory(
    task_id: str = "test-task-001",
    steps: list[StepRecord] | None = None,
    status: str = "completed",
) -> Trajectory:
    """Create a Trajectory with optional steps 🗺️.

    Args:
        task_id: Task identifier.
        steps: List of step records. Empty list if None.
        status: Trajectory status (default "completed").

    Returns:
        Fully constructed Trajectory.
    """
    traj = Trajectory(task_id=task_id, status="running")
    for step in steps or []:
        traj.add_step(step)
    if status != "running":
        traj.finish(status)
    return traj


# ============================================================
# 🤖 Mock LLM
# ============================================================


class MockLLM:
    """Configurable mock LLM for testing agent execution 🤖.

    Supports configurable response sequences. Each call to query()
    returns the next response in the sequence.

    Example::

        llm = MockLLM(responses=[
            make_search_response(),
            make_finish_response({"decision": "positive"}),
        ])
    """

    def __init__(
        self,
        responses: list[AssistantMessage] | None = None,
        default_result: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MockLLM 🔧.

        Args:
            responses: Ordered list of responses.
            default_result: Default result for auto-generated finish responses.
        """
        self._default_result = default_result or {
            "decision": "positive",
            "confidence": 0.85,
            "reasoning": [],
            "evidence_index": [],
        }
        if responses is not None:
            self._responses = list(responses)
        else:
            self._responses = [
                make_finish_response(self._default_result),
            ]
        self._call_count = 0

    def query(self, dialog: Any) -> AssistantMessage:
        """Return the next pre-configured response 📤.

        Args:
            dialog: Dialog object (ignored in mock).

        Returns:
            Next AssistantMessage in the sequence.
        """
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = make_finish_response(self._default_result)
        self._call_count += 1
        return response

    @property
    def call_count(self) -> int:
        """Return the number of times query() was called 📊."""
        return self._call_count


def make_finish_response(
    result_dict: dict[str, Any],
) -> AssistantMessage:
    """Build an AssistantMessage with a finish tool call 📋.

    Args:
        result_dict: Result to embed.

    Returns:
        AssistantMessage with finish tool call.
    """
    finish_call = create_finish_tool_call(result_dict)
    return AssistantMessage(
        content="Research complete. Submitting structured result.",
        tool_calls=[finish_call],
    )


def make_search_response(
    tool_name: str = "perplexity_search",
    query: str = "test query",
) -> AssistantMessage:
    """Build an AssistantMessage with a search tool call 🔍.

    Args:
        tool_name: Name of the search tool.
        query: Search query string.

    Returns:
        AssistantMessage with search tool call.
    """
    search_call = create_tool_call(
        name=tool_name,
        arguments={"query": query},
    )
    return AssistantMessage(
        content=f"Searching for: {query}",
        tool_calls=[search_call],
    )


# ============================================================
# 📋 Sample Data Builders
# ============================================================


def build_sample_evaluation_task(
    task_id: str = "test-research-001",
    topic: str = "EGFR market analysis",
    rules: str = "Focus on peer-reviewed data",
    prior_context: str | None = None,
    max_turns: int = 30,
    max_retries: int = 2,
    sub_item_id: str | None = None,
) -> Any:
    """Build a sample EvaluationTask for testing 🔬.

    Args:
        task_id: Unique task identifier.
        topic: Research topic.
        rules: Evaluation rules string.
        prior_context: Optional prior context.
        max_turns: Maximum agent turns.
        max_retries: Maximum QG retry attempts.
        sub_item_id: Optional sub-item id for Progressive Disclosure.

    Returns:
        EvaluationTask instance with realistic test data.
    """
    from inquiro.core.types import (
        AgentConfig,
        Checklist,
        ChecklistItem,
        DecisionGuidance,
        EvaluationTask,
        QualityChecks,
        QualityGateConfig,
    )

    return EvaluationTask(
        task_id=task_id,
        topic=topic,
        rules=rules,
        checklist=Checklist(
            required=[
                ChecklistItem(
                    id="item_1",
                    description="Current global market size",
                    keywords=["market size", "global"],
                    suggested_sources=["perplexity"],
                ),
                ChecklistItem(
                    id="item_2",
                    description="Growth forecast",
                    keywords=["CAGR", "forecast"],
                ),
            ],
            optional=[
                ChecklistItem(
                    id="opt_1",
                    description="Regional breakdown",
                    keywords=["regional", "breakdown"],
                ),
            ],
            coverage_threshold=0.8,
        ),
        decision_guidance=DecisionGuidance(
            positive=["Large market"],
            cautious=["Mature market"],
            negative=["Shrinking market"],
        ),
        output_schema={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": [
                "decision",
                "confidence",
                "reasoning",
                "evidence_index",
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
            },
        },
        prior_context=prior_context,
        sub_item_id=sub_item_id,
        agent_config=AgentConfig(max_turns=max_turns),
        quality_gate=QualityGateConfig(
            max_retries=max_retries,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=True,
                evidence_reference_check=True,
            ),
        ),
    )


def build_sample_synthesis_task(
    task_id: str = "test-synthesis-001",
    topic: str = "Clinical feasibility synthesis",
    synthesis_rules: str = "Cross-reference findings",
    allow_additional_research: bool = True,
    max_turns: int = 20,
    max_retries: int = 2,
) -> Any:
    """Build a sample SynthesisTask for testing 📊.

    Args:
        task_id: Unique task identifier.
        topic: Synthesis topic/objective.
        synthesis_rules: Synthesis rules string.
        allow_additional_research: Whether deep-dive is enabled.
        max_turns: Maximum agent turns.
        max_retries: Maximum QG retry attempts.

    Returns:
        SynthesisTask instance with realistic test data.
    """
    from inquiro.core.types import (
        AgentConfig,
        InputReport,
        QualityChecks,
        QualityGateConfig,
        SynthesisTask,
    )

    return SynthesisTask(
        task_id=task_id,
        topic=topic,
        input_reports=[
            InputReport(
                report_id="report-001",
                label="Target Biology",
                content={
                    "decision": "positive",
                    "confidence": 0.85,
                    "reasoning": [
                        {
                            "claim": "Well-validated target",
                            "evidence_ids": ["E1"],
                            "strength": "strong",
                        },
                    ],
                    "evidence_index": [
                        {
                            "id": "E1",
                            "source": "perplexity",
                            "query": "target validation",
                            "summary": "Multiple approved drugs",
                        },
                    ],
                },
            ),
            InputReport(
                report_id="report-002",
                label="Biomarker Availability",
                content={
                    "decision": "cautious",
                    "confidence": 0.7,
                    "reasoning": [
                        {
                            "claim": "Biomarker testing available",
                            "evidence_ids": ["E2"],
                            "strength": "moderate",
                        },
                    ],
                    "evidence_index": [
                        {
                            "id": "E2",
                            "source": "perplexity",
                            "query": "biomarker testing",
                            "summary": "IHC and FISH available",
                        },
                    ],
                },
            ),
            InputReport(
                report_id="report-003",
                label="Patient Stratification",
                content={
                    "decision": "positive",
                    "confidence": 0.75,
                    "reasoning": [
                        {
                            "claim": "Clear stratification",
                            "evidence_ids": ["E3"],
                            "strength": "strong",
                        },
                    ],
                    "evidence_index": [
                        {
                            "id": "E3",
                            "source": "perplexity",
                            "query": "patient stratification",
                            "summary": "L858R and exon 19 deletions",
                        },
                    ],
                },
            ),
        ],
        synthesis_rules=synthesis_rules,
        output_schema={
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
        },
        allow_additional_research=allow_additional_research,
        agent_config=AgentConfig(max_turns=max_turns),
        quality_gate=QualityGateConfig(
            max_retries=max_retries,
            checks=QualityChecks(
                schema_validation=True,
                coverage_check=True,
                cross_reference_check=True,
            ),
        ),
    )


def build_valid_research_result() -> dict[str, Any]:
    """Build a valid research result that passes schema validation ✅.

    Includes checklist_coverage to pass the QG coverage check.

    Returns:
        Dict conforming to the default EvaluationResult output schema.
    """
    return {
        "decision": "positive",
        "confidence": 0.85,
        "reasoning": [
            {
                "claim": "Market exceeds $20B globally",
                "evidence_ids": ["E1"],
                "strength": "strong",
            },
            {
                "claim": "Market CAGR forecast at 8.5%",
                "evidence_ids": ["E2"],
                "strength": "strong",
            },
        ],
        "evidence_index": [
            {
                "id": "E1",
                "source": "perplexity",
                "query": "EGFR market size",
                "summary": "Global EGFR therapy market valued at $22.5B",
            },
            {
                "id": "E2",
                "source": "perplexity",
                "query": "EGFR market CAGR forecast",
                "summary": "Market growing at 8.5% CAGR through 2030",
            },
        ],
        # 📊 Checklist coverage data used by QG coverage check
        "checklist_coverage": {
            "covered": ["item_1", "item_2"],
            "missing": [],
        },
    }


def build_invalid_research_result() -> dict[str, Any]:
    """Build an invalid result that fails schema validation ❌.

    Missing the required "decision" field.

    Returns:
        Dict that will fail schema validation.
    """
    return {
        "confidence": 0.5,
        "reasoning": [],
        "evidence_index": [],
    }


def build_soft_failure_research_result() -> dict[str, Any]:
    """Build a result that passes schema but triggers soft failures ⚠️.

    Has a reasoning claim referencing non-existent evidence (E99),
    triggering the evidence_reference_check soft failure.
    Includes valid checklist_coverage to avoid stacking soft failures.

    Returns:
        Dict that passes schema but fails evidence reference check.
    """
    return {
        "decision": "positive",
        "confidence": 0.9,
        "reasoning": [
            {
                "claim": "Market is growing",
                "evidence_ids": ["E99"],
                "strength": "moderate",
            },
        ],
        "evidence_index": [
            {
                "id": "E1",
                "source": "perplexity",
                "query": "EGFR market size",
                "summary": "Global market valued at $22.5B",
            },
        ],
        # 📊 Coverage passes, but evidence reference will fail (E99 orphan)
        "checklist_coverage": {
            "covered": ["item_1", "item_2"],
            "missing": [],
        },
    }


def build_valid_synthesis_result() -> dict[str, Any]:
    """Build a valid synthesis result that passes schema validation ✅.

    Returns:
        Dict conforming to the default SynthesisResult output schema.
    """
    return {
        "decision": "positive",
        "confidence": 0.80,
        "reasoning": [
            {
                "claim": "Strong target biology with biomarkers",
                "evidence_ids": ["E1", "E2"],
                "strength": "strong",
            },
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
                "source": "perplexity",
                "query": "biomarker testing",
                "summary": "IHC and FISH available",
            },
        ],
        "source_reports": ["report-001", "report-002", "report-003"],
        "cross_references": [
            {
                "claim": "EGFR well-validated",
                "supporting_reports": ["report-001", "report-003"],
                "contradicting_reports": [],
            },
        ],
        "gaps_remaining": [],
    }


# ============================================================
# 🏗️ Exp Instance Builders
# ============================================================



def create_synthesis_exp(
    task: Any | None = None,
    llm: Any | None = None,
    task_runner: Any | None = None,
    event_emitter: Any | None = None,
    cost_tracker: Any | None = None,
    cancellation_token: Any | None = None,
    skill_registry: Any | None = None,
) -> Any:
    """Create a SynthesisExp instance with sensible defaults 📊.

    Args:
        task: SynthesisTask (auto-built if None).
        llm: LLM instance (MockLLM if None).
        task_runner: EvalTaskRunner mock (MagicMock if None).
        event_emitter: EventEmitter (MagicMock if None).
        cost_tracker: CostTracker (real instance if None).
        cancellation_token: CancellationToken (real instance if None).
        skill_registry: Optional SkillRegistry for skills in prompt.

    Returns:
        Configured SynthesisExp instance.
    """
    from inquiro.exps.synthesis_exp import SynthesisExp
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker

    if task is None:
        task = build_sample_synthesis_task()

    if event_emitter is None:
        event_emitter = MagicMock()
        event_emitter.emit = MagicMock()

    # ⚠️ Use explicit `is None` — CancellationToken.__bool__ returns False
    # when cancelled, which breaks `token or CancellationToken()`.
    return SynthesisExp(
        task=task,
        llm=llm if llm is not None else MockLLM(),
        task_runner=task_runner if task_runner is not None else MagicMock(),
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
        skill_registry=skill_registry,
    )


def build_mock_agent_for_synthesis(
    result_dict: dict[str, Any],
    task_id: str = "test-synthesis-001",
    max_turns: int = 20,
) -> MagicMock:
    """Build a mock SynthesisAgent with controlled behavior 🤖.

    The mock agent completes in one step, producing a trajectory
    with a finish step containing the provided result.

    Args:
        result_dict: Result to embed in the finish step.
        task_id: Task ID for the trajectory.
        max_turns: Value for agent.config.max_turns.

    Returns:
        Configured MagicMock agent.
    """
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.max_turns = max_turns

    # 🗺️ Build trajectory with finish step
    finish_step = create_finish_step(result_dict)
    mock_traj = Trajectory(task_id=task_id)
    mock_traj.add_step(finish_step)

    mock_agent.trajectory = mock_traj

    # ✨ _step returns True on first call (agent finishes)
    mock_agent._step.return_value = True

    return mock_agent
