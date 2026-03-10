"""Inquiro SynthesisExp — Single synthesis task lifecycle 📊.

Manages the complete lifecycle of one synthesis task:
    1. Pre-load input reports into ReadReportTool 📖
    2. Create SynthesisAgent with ReadReport + RequestResearch + FinishTool 🤖
    3. Run agent (multi-round Read-Reason-Synthesize loop) 🔄
    4. Handle additional research triggered by the agent 🔬
    5. Quality Gate validation ✅
    6. Retry on hard failures 🔁
    7. Return structured SynthesisResult 📊

Inherits from InquiroBaseExp for shared lifecycle logic.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.agent import AgentConfig as EvoAgentConfig
from evomaster.agent.tools.base import BaseTool, BaseToolParams, ToolRegistry
from evomaster.utils.types import TaskInstance

from inquiro.core.types import ExpPhase
from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.exps.evolution_helper import EvolutionHelper
from inquiro.infrastructure.cancellation import CancelledError
from inquiro.infrastructure.event_emitter import InquiroEvent
from inquiro.infrastructure.quality_gate import QualityGateResult
from inquiro.prompts.loader import PromptLoader
from inquiro.prompts.section_builder import PromptSectionBuilder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from evomaster.agent.session import BaseSession
    from inquiro.agents.synthesis_agent import SynthesisAgent
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter
    from inquiro.infrastructure.quality_gate import QualityGateResult
    from inquiro.core.types import (
        QualityGateConfig,
        SynthesisResult,
        SynthesisTask,
    )


# 📝 Module-level PromptLoader instance (shared across all SynthesisExp)
_prompt_loader = PromptLoader()

# ✨ Legacy alias for backward compatibility with tests
DEFAULT_SYNTHESIS_SYSTEM_PROMPT = _prompt_loader.load("synthesis_system")


# ---------------------------------------------------------------------------
# 📖 ReadReportTool — internal tool for on-demand report reading
# ---------------------------------------------------------------------------


class _ReadReportToolParams(BaseToolParams):
    """Read a specific research report by its ID.

    Use this tool to read the content of one of the available input
    reports. You must provide the report_id from the available reports
    list in your instructions.
    """

    name: ClassVar[str] = "read_report"

    report_id: str = Field(description="The unique identifier of the report to read.")


class _ReadReportTool(BaseTool):
    """Internal tool for on-demand report reading 📖.

    Pre-loaded with report content at SynthesisExp initialization.
    Returns the full report content when called by the SynthesisAgent.
    """

    name: ClassVar[str] = "read_report"
    params_class: ClassVar[type[BaseToolParams]] = _ReadReportToolParams

    def __init__(self, reports: dict[str, dict[str, Any]]) -> None:
        """Initialize with pre-loaded report content 🔧.

        Args:
            reports: Mapping of report_id to report content dict.
        """
        super().__init__()
        self._reports = reports

    def execute(
        self,
        session: "BaseSession",
        args_json: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return the content of a specific report 📖.

        Args:
            session: Session instance (unused).
            args_json: JSON string with report_id parameter.

        Returns:
            Tuple of (observation, info) with report content.
        """
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            msg = f"Parameter validation error: {e}"
            return msg, {"error": msg}

        assert isinstance(params, _ReadReportToolParams)
        report_id = params.report_id

        if report_id not in self._reports:
            available = ", ".join(self._reports.keys())
            msg = f"Report '{report_id}' not found. Available reports: {available}"
            return msg, {
                "error": "report_not_found",
                "report_id": report_id,
            }

        content = self._reports[report_id]
        content_str = json.dumps(content, indent=2, ensure_ascii=False)

        return content_str, {
            "report_id": report_id,
            "status": "read",
        }


# ---------------------------------------------------------------------------
# 📊 SynthesisExp
# ---------------------------------------------------------------------------


class SynthesisExp(InquiroBaseExp):
    """Single synthesis task lifecycle 📊.

    Manages the complete lifecycle of one synthesis task:
        1. Pre-load input reports into ReadReportTool
        2. Create SynthesisAgent with ReadReport + RequestResearch + Finish
        3. Run agent (multi-round read-reason-synthesize loop)
        4. Handle additional research tasks spawned by the agent
        5. Quality Gate validation
        6. Retry on hard failures
        7. Return structured SynthesisResult

    Attributes:
        task: The synthesis task definition.
        llm: LLM instance for the synthesis agent.
        task_runner: EvalTaskRunner for spawning research sub-tasks.
        quality_gate: Quality gate validator.
        cost_tracker: Cost tracking instance.
        event_emitter: SSE event emitter for progress.
        cancellation_token: Cancellation signal.
        _additional_research: List of additional research results.
    """

    # ⏱️ Wall-clock time budget for the entire synthesis task (seconds).
    # After this many seconds, skip remaining deep-dives and force the
    # agent to synthesize with available evidence.
    SYNTHESIS_TIME_BUDGET_SECS: ClassVar[int] = 300

    # 🔄 Maximum turns cap for deep-dive sub-searches. Deep-dives are
    # focused sub-searches — they should NOT inherit the full parent
    # max_turns (sized for synthesis). Cap at 10 to allow 4-5 searches
    # across multiple MCP sources + think + finish.
    DEEP_DIVE_MAX_TURNS_CAP: ClassVar[int] = 10

    def __init__(
        self,
        task: SynthesisTask,
        llm: BaseLLM,
        task_runner: Any,  # EvalTaskRunner — forward reference
        quality_gate_config: QualityGateConfig,
        cost_tracker: CostTracker,
        event_emitter: EventEmitter,
        cancellation_token: CancellationToken,
        skill_registry: Any = None,
    ):
        """Initialize SynthesisExp 🔧.

        Args:
            task: Synthesis task definition containing topic, input_reports,
                synthesis_rules, output_schema, and agent configuration.
            llm: LLM instance for the synthesis agent.
            task_runner: EvalTaskRunner reference for spawning research
                sub-tasks when the SynthesisAgent requests deep dives.
            quality_gate_config: Quality validation configuration.
            cost_tracker: Cost tracking instance for budget enforcement.
            event_emitter: SSE event emitter for progress updates.
            cancellation_token: Cancellation signal for cooperative stop.
            skill_registry: Optional SkillRegistry for agent skills.
        """
        self._init_base(
            task=task,
            llm=llm,
            quality_gate_config=quality_gate_config,
            cost_tracker=cost_tracker,
            event_emitter=event_emitter,
            cancellation_token=cancellation_token,
        )
        self.task_runner = task_runner
        self._additional_research: list[dict[str, Any]] = []
        self.skill_registry = skill_registry
        # 🧬 Evolution: delegate to shared EvolutionHelper (composition)
        self._evolution_helper: EvolutionHelper | None = None
        if task.evolution_profile:
            self._evolution_helper = EvolutionHelper(
                task=task,
                llm=llm,
                cost_tracker=cost_tracker,
                logger=self.logger,
            )

        # 🔬 Deep-dive budget: enforce max_tasks from config
        self._deep_dive_count: int = 0
        self._max_deep_dives: int = (
            self.task.additional_research_config.max_tasks
            if self.task.additional_research_config
            else 2  # ✨ Default cap when no config specified
        )

    @property
    def exp_name(self) -> str:
        """Return experiment name 🏷️.

        Returns:
            "Synthesis" as the experiment type name.
        """
        return "Synthesis"

    def run_sync(self) -> SynthesisResult:
        """Execute synthesis task with Quality Gate retry loop 🔄.

        Runs the synthesis agent step-by-step, checking for pending
        research requests after each step. Validates output via
        QualityGate, and retries on hard failures.

        Returns:
            SynthesisResult with synthesized decision and cross-references.
        """

        task_id = self.task.task_id
        max_retries = self.quality_gate.max_retries

        # ⏱️ Track synthesis wall-clock time against class-level budget
        _synthesis_start = time.monotonic()

        # 📡 Emit synthesis started event
        self.event_emitter.emit(
            InquiroEvent.SYNTHESIS_STARTED,
            task_id,
            {
                "topic": self.task.topic,
                "report_count": len(self.task.input_reports),
                "max_retries": max_retries,
            },
        )

        raw_result: dict[str, Any] = {}
        qg_result: QualityGateResult | None = None
        trajectory = None

        for attempt in range(max_retries + 1):
            self.logger.info(
                "📊 Synthesis attempt %s/%s for task %s",
                attempt + 1,
                max_retries + 1,
                task_id,
            )

            # 🛑 Check cancellation before each attempt
            if self.cancellation_token.is_cancelled:
                self._safe_transition(ExpPhase.CANCELLED)
                raise CancelledError(self.cancellation_token.reason or "Task cancelled")

            # 💰 Check cost budget
            if not self._check_cost():
                self.logger.warning(
                    "⚠️ Cost budget exceeded for task %s, returning best result so far",
                    task_id,
                )
                break

            try:
                # 🔄 Phase: building prompts
                self._safe_transition(ExpPhase.PROMPT_BUILDING)

                # 🔄 Reset deep-dive counter on each attempt
                self._deep_dive_count = 0

                # 🤖 Step 1-2: Create agent and tools
                agent, request_research_tool = self._create_synthesis_agent()

                # 🔄 Step 3: Initialize agent (creates dialog)
                task_instance = TaskInstance(
                    task_id=task_id,
                    task_type="synthesis",
                    description=self.task.topic,
                )
                # 🔄 Phase: agent running
                self._safe_transition(ExpPhase.AGENT_RUNNING)

                agent._initialize(task_instance)

                # 📖 Step 4: Inject input reports into agent context
                # ⚠️ MUST come after _initialize — dialog is
                # created during initialization.
                agent._inject_reports_to_context(self.task.input_reports)
                trajectory = agent.trajectory

                if (
                    getattr(self.task, "trajectory_streaming", False)
                    and self.task.trajectory_dir
                ):
                    self._init_trajectory_stream(trajectory)

                for turn in range(agent.config.max_turns):
                    # 🛑 Cancellation check
                    if self.cancellation_token.is_cancelled:
                        raise CancelledError(
                            self.cancellation_token.reason or "Task cancelled"
                        )

                    should_finish = agent._step()

                    if (
                        getattr(self.task, "trajectory_streaming", False)
                        and self.task.trajectory_dir
                    ):
                        self._persist_trajectory_step(
                            agent.trajectory,
                            len(agent.trajectory.steps) - 1,
                        )

                    # 🔬 Step 5: Check for pending research requests
                    if request_research_tool is not None:
                        pending = request_research_tool.drain_pending_requests()
                        for eval_task in pending:
                            # ⏱️ Enforce elapsed-time budget
                            _elapsed = time.monotonic() - _synthesis_start
                            if _elapsed > self.SYNTHESIS_TIME_BUDGET_SECS:
                                self.logger.warning(
                                    "⏱️ Synthesis time budget "
                                    "exceeded (%.0fs/%.0fs), "
                                    "skipping deep-dive: %s",
                                    _elapsed,
                                    self.SYNTHESIS_TIME_BUDGET_SECS,
                                    eval_task.topic[:60],
                                )
                                agent.add_research_result(
                                    {
                                        "status": "skipped",
                                        "reason": (
                                            "Time budget exceeded "
                                            f"({_elapsed:.0f}s/"
                                            f"{self.SYNTHESIS_TIME_BUDGET_SECS}s"
                                            "). Synthesize with "
                                            "available evidence and "
                                            "call finish immediately."
                                        ),
                                        "topic": eval_task.topic,
                                    }
                                )
                                continue

                            # 🛑 Enforce deep-dive budget
                            if self._deep_dive_count >= self._max_deep_dives:
                                self.logger.warning(
                                    "⚠️ Deep-dive budget exhausted "
                                    "(%d/%d), skipping request: %s",
                                    self._deep_dive_count,
                                    self._max_deep_dives,
                                    eval_task.topic[:60],
                                )
                                # 📝 Notify agent that budget is
                                # exhausted so it calls finish
                                agent.add_research_result(
                                    {
                                        "status": "skipped",
                                        "reason": (
                                            "Deep-dive budget exhausted "
                                            f"({self._max_deep_dives}/"
                                            f"{self._max_deep_dives}). "
                                            "Synthesize with available "
                                            "evidence and call finish."
                                        ),
                                        "topic": eval_task.topic,
                                    }
                                )
                                continue

                            self._handle_research_request(eval_task, agent)
                            self._deep_dive_count += 1

                    if should_finish:
                        trajectory.finish("completed")
                        break
                else:
                    trajectory.finish(
                        "failed",
                        {"reason": "max_turns_exceeded"},
                    )

                # 💰 Record LLM costs from trajectory
                self._record_trajectory_costs(trajectory)

                # 📝 Step 6: Extract result from trajectory
                raw_result = self._extract_result(trajectory)

                if not raw_result:
                    self.logger.warning(
                        "⚠️ No result extracted from synthesis trajectory"
                    )
                    if attempt < max_retries:
                        self._safe_transition(ExpPhase.RETRYING)
                        self._handle_retry(
                            raw_result,
                            ["no_result_extracted"],
                            attempt + 1,
                        )
                        continue
                    break

                # 🔄 Phase: quality gate validation
                self._safe_transition(ExpPhase.QG_VALIDATING)

                # ✅ Step 7: Run Quality Gate validation
                qg_result = self._run_quality_gate(raw_result)

                # 📡 Emit QG result event
                self.event_emitter.emit(
                    InquiroEvent.QUALITY_GATE_RESULT,
                    task_id,
                    {
                        "attempt": attempt + 1,
                        "passed": qg_result.passed,
                        "hard_failures": qg_result.hard_failures,
                        "soft_failures": qg_result.soft_failures,
                        "risk_flags": qg_result.risk_flags,
                    },
                )

                # 🎯 Handle QG result
                if qg_result.passed:
                    self._safe_transition(ExpPhase.COMPLETED)
                    self.logger.info(
                        "✅ Quality gate passed on attempt %s",
                        attempt + 1,
                    )
                    break

                # ❌ Hard failure — retry if possible
                if qg_result.hard_failures and attempt < max_retries:
                    self._safe_transition(ExpPhase.RETRYING)
                    self._handle_retry(
                        raw_result,
                        qg_result.hard_failures,
                        attempt + 1,
                    )
                    continue

                # 🏁 No more retries
                self.logger.warning(
                    "⚠️ Quality gate exhausted after %s attempt(s), using best result",
                    attempt + 1,
                )
                break

            except CancelledError:
                self._safe_transition(ExpPhase.CANCELLED)
                raise
            except Exception as e:
                self.logger.error(
                    "❌ Synthesis attempt %s failed: %s",
                    attempt + 1,
                    e,
                )
                if attempt < max_retries:
                    self._safe_transition(ExpPhase.RETRYING)
                    self._handle_retry(
                        raw_result,
                        [f"exception: {str(e)}"],
                        attempt + 1,
                    )
                    continue
                self._safe_transition(ExpPhase.FAILED)
                raise

        # 🔄 Ensure terminal state before building result
        if self._phase not in (
            ExpPhase.COMPLETED,
            ExpPhase.FAILED,
            ExpPhase.CANCELLED,
        ):
            self._safe_transition(ExpPhase.COMPLETED)

        # 📊 Build final result
        warning = None
        if qg_result and not qg_result.passed:
            warning = "quality_gate_exhausted"

        result = self._build_result(raw_result, qg_result, trajectory, warning=warning)

        # 📡 Emit task_completed event
        self.event_emitter.emit(
            InquiroEvent.TASK_COMPLETED,
            task_id,
            {
                "decision": result.decision.value,
                "confidence": result.confidence,
                "source_reports": result.source_reports,
                "deep_dives_count": len(result.deep_dives_triggered),
            },
        )

        # 💾 Persist trajectory to disk (if configured)
        if (
            getattr(self.task, "trajectory_streaming", False)
            and self.task.trajectory_dir
        ):
            self._finalize_trajectory_stream(trajectory)
        else:
            self._persist_trajectory(trajectory)

        # 🧬 Post-execution evolution: collect, extract, fitness update
        if self._evolution_helper:
            self._evolution_helper.post_execution_evolution(trajectory, result)

        return result

    def _create_synthesis_agent(
        self,
    ) -> tuple[SynthesisAgent, Any]:
        """Create a SynthesisAgent with report-handling tools 🎯.

        Sets up the agent with:
            - ReadReportTool (pre-loaded with input reports) 📖
            - RequestResearchTool (if deep-dive enabled) 🔬
            - InquiroFinishTool (schema-validated output) 📋

        Returns:
            Tuple of (SynthesisAgent, RequestResearchTool_or_None).
        """
        from inquiro.agents.synthesis_agent import SynthesisAgent
        from inquiro.tools.finish_tool import InquiroFinishTool
        from inquiro.tools.request_research_tool import (
            RequestResearchTool,
        )

        # 📖 Build report content map for ReadReportTool
        report_content_map: dict[str, dict[str, Any]] = {}
        for report in self.task.input_reports:
            report_content_map[report.report_id] = report.content

        # 🛠️ Build tool registry
        tool_registry = ToolRegistry()

        # 📖 Register ReadReportTool
        read_report_tool = _ReadReportTool(reports=report_content_map)
        tool_registry.register(read_report_tool)

        # 🧠 Register Think tool for explicit reasoning
        try:
            from evomaster.agent.tools.builtin.think import ThinkTool

            tool_registry.register(ThinkTool())
        except Exception as e:
            logger.warning("⚠️ Failed to register ThinkTool: %s", e)

        # 🔬 Register RequestResearchTool (if deep-dive enabled)
        request_research_tool = None
        if self.task.allow_additional_research:
            request_research_tool = RequestResearchTool(
                task_runner=self.task_runner,
            )
            tool_registry.register(request_research_tool)

        # 🎯 Filter skills to SynthesisAgent-relevant subset
        skill_registry = self.skill_registry
        if skill_registry is not None:
            try:
                skill_registry = skill_registry.create_subset([
                    "evidence-grader",
                    "evidence-condenser",
                    "evidence-type-taxonomy",
                    "evidence-scope-model",
                ])
            except Exception:
                pass  # Fall back to full registry

        # 🎯 Register SkillTool if skills are available
        if skill_registry is not None:
            try:
                from evomaster.agent.tools.skill import SkillTool

                skill_tool = SkillTool(skill_registry)
                tool_registry.register(skill_tool)
            except Exception as e:
                logger.warning("⚠️ Failed to register SkillTool: %s", e)

        # 📋 Register InquiroFinishTool with output schema
        finish_tool = InquiroFinishTool(self.task.output_schema)
        tool_registry.register(finish_tool)

        # 📝 Render prompts
        system_prompt = self._render_system_prompt()
        user_prompt = self._render_user_prompt()

        # 🔧 Build EvoMaster AgentConfig with context mapping
        from evomaster.agent.context import (
            ContextConfig as EvoContextConfig,
            TruncationStrategy as EvoTruncationStrategy,
        )

        inquiro_ctx = self.task.agent_config.context
        evo_context_config = EvoContextConfig(
            max_tokens=inquiro_ctx.max_tokens,
            truncation_strategy=EvoTruncationStrategy(
                inquiro_ctx.truncation_strategy.value
            ),
            preserve_system_messages=inquiro_ctx.preserve_system_messages,
            preserve_recent_turns=inquiro_ctx.preserve_recent_turns,
        )

        agent_config = EvoAgentConfig(
            max_turns=self.task.agent_config.max_turns,
            context_config=evo_context_config,
        )

        agent = SynthesisAgent(
            llm=self.llm,
            tools=tool_registry,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=agent_config,
            output_schema=self.task.output_schema,
            task_id=self.task.task_id,
            cost_tracker=self.cost_tracker,
            cancellation_token=self.cancellation_token,
            event_emitter=self.event_emitter,
        )

        return agent, request_research_tool

    def _render_system_prompt(self) -> str:
        """Render synthesis system prompt from template 📝.

        Formats the synthesis_system template with task-specific
        synthesis_rules, report_list, output_schema, and deep-dive setting.

        Returns:
            Rendered system prompt string.
        """
        # 📖 Format report list
        report_lines: list[str] = []
        for report in self.task.input_reports:
            report_lines.append(f"- **{report.label}** (ID: `{report.report_id}`)")
        report_list = "\n".join(report_lines) if report_lines else "No reports."

        # 📋 Format output schema
        output_schema_str = json.dumps(
            self.task.output_schema, indent=2, ensure_ascii=False
        )

        # 🔬 Format deep-dive section and synthesis steps
        if self.task.allow_additional_research:
            deep_dive_section = (
                "You may use the `request_research` tool to trigger "
                "additional research on specific topics where you "
                "identify **critical** evidence gaps.\n\n"
                f"**Budget**: You have a maximum of "
                f"**{self._max_deep_dives}** deep-dive requests. "
                "Use them sparingly — only for gaps that would "
                "materially change the assessment. Minor gaps "
                "should be noted in `gaps_remaining` instead."
            )
            synthesis_steps = (
                "1. Read all input reports using `read_report`.\n"
                "2. Cross-reference findings across reports.\n"
                "3. Identify patterns, contradictions, and gaps.\n"
                "4. If gaps are critical, use `request_research` "
                "for additional evidence.\n"
                "5. Synthesize a comprehensive assessment.\n"
                "6. Use the `finish` tool to submit your "
                "structured result."
            )
        else:
            deep_dive_section = (
                "Deep-dive research is disabled for this task. "
                "Do NOT attempt to use `request_research`."
            )
            synthesis_steps = (
                "1. Read all input reports using `read_report`.\n"
                "2. Cross-reference findings across reports.\n"
                "3. Identify patterns, contradictions, and gaps.\n"
                "4. Synthesize a comprehensive assessment, noting "
                "any remaining gaps.\n"
                "5. Use the `finish` tool to submit your "
                "structured result."
            )

        # 🧬 Enrich prompt with learned experiences (if evolution enabled)
        learned_insights = (
            self._evolution_helper.enrich_with_experiences("")
            if self._evolution_helper
            else ""
        )

        # 🎯 Build available skills summary (Progressive Disclosure)
        available_skills = PromptSectionBuilder.format_available_skills(
            self.skill_registry
        )

        prompt = _prompt_loader.render(
            "synthesis_system",
            synthesis_rules=(
                self.task.synthesis_rules or "No specific synthesis rules provided."
            ),
            report_list=report_list,
            output_schema=output_schema_str,
            deep_dive_section=deep_dive_section,
            synthesis_steps=synthesis_steps,
            learned_insights=learned_insights,
            available_skills=available_skills,
        )

        return prompt

    def _render_user_prompt(self) -> str:
        """Render user prompt for the synthesis task 📝.

        Includes topic, list of input report IDs, and instructions
        for using read_report and finish tools.

        Returns:
            User prompt string.
        """
        report_ids = [r.report_id for r in self.task.input_reports]
        report_list_str = ", ".join(f"`{rid}`" for rid in report_ids)

        return _prompt_loader.render(
            "synthesis_user",
            topic=self.task.topic,
            report_list=report_list_str,
        )

    def _handle_research_request(
        self,
        eval_task: Any,
        agent: SynthesisAgent,
    ) -> None:
        """Handle a pending research request from SynthesisAgent 🔬.

        Dispatches the research task to the task_runner synchronously,
        waits for completion, and injects the result back into the
        agent's context.

        Before dispatching, enriches the EvaluationTask with defaults
        inherited from the parent SynthesisTask (output_schema,
        tools_config, quality_gate, cost_guard) so that the spawned
        SearchExp has a complete configuration.

        Args:
            eval_task: The EvaluationTask created by RequestResearchTool.
            agent: The running SynthesisAgent to inject results into.
        """
        from inquiro.core.types import (
            CostGuardConfig,
        )

        self.logger.info(
            "🔬 Handling deep-dive research request: task_id=%s, topic='%s'",
            eval_task.task_id,
            eval_task.topic[:60],
        )

        # 🔧 Enrich deep-dive task with parent task defaults

        # 🔄 Cap deep-dive max_turns unconditionally.
        # Deep-dives are focused sub-searches — they should NOT
        # 🔄 Cap deep-dive max_turns using class-level constant
        original_turns = eval_task.agent_config.max_turns
        eval_task.agent_config.max_turns = min(
            self.DEEP_DIVE_MAX_TURNS_CAP,
            eval_task.agent_config.max_turns,
        )
        if eval_task.agent_config.max_turns != original_turns:
            self.logger.info(
                "🔄 Deep-dive max_turns capped: %d → %d (focused research)",
                original_turns,
                eval_task.agent_config.max_turns,
            )

        if not eval_task.output_schema:
            eval_task.output_schema = self.task.output_schema

        # 🔌 Inherit tools_config from additional_research_config
        if not eval_task.tools_config.mcp_servers:
            if (
                self.task.additional_research_config
                and self.task.additional_research_config.tools_config.mcp_servers
            ):
                eval_task.tools_config = (
                    self.task.additional_research_config.tools_config
                )

        # 💰 Inherit cost_guard from additional_research_config
        if self.task.additional_research_config:
            eval_task.cost_guard = CostGuardConfig(
                max_cost_per_task=(self.task.additional_research_config.cost_budget),
            )

        # 📡 Emit additional research event
        self.event_emitter.emit(
            InquiroEvent.ADDITIONAL_RESEARCH_REQUESTED,
            self.task.task_id,
            {
                "deep_dive_task_id": eval_task.task_id,
                "topic": eval_task.topic,
            },
        )

        try:
            # 🔄 Run research synchronously via task_runner
            t0 = time.monotonic()
            research_result = self.task_runner.run_research_sync(eval_task)
            elapsed = time.monotonic() - t0
            result_dict = research_result.model_dump()

            # 📝 Record the deep-dive result
            self._additional_research.append(
                {
                    "task_id": eval_task.task_id,
                    "topic": eval_task.topic,
                    "result": result_dict,
                }
            )

            # 📖 Inject result into agent context
            agent.add_research_result(result_dict)

            self.logger.info(
                "✅ Deep-dive %d/%d completed in %.1fs: task_id=%s, decision=%s",
                self._deep_dive_count + 1,
                self._max_deep_dives,
                elapsed,
                eval_task.task_id,
                research_result.decision.value,
            )

        except CancelledError:
            raise
        except Exception as e:
            elapsed = time.monotonic() - t0
            self.logger.error(
                "❌ Deep-dive %d/%d failed after %.1fs for %s: %s",
                self._deep_dive_count + 1,
                self._max_deep_dives,
                elapsed,
                eval_task.task_id,
                e,
            )
            # 📝 Record failure but don't crash the synthesis
            self._additional_research.append(
                {
                    "task_id": eval_task.task_id,
                    "topic": eval_task.topic,
                    "error": str(e),
                }
            )

    def _collect_source_evidence(self) -> list[Any]:
        """Collect all evidence from source reports and deep-dives 🔍.

        Iterates over input reports and additional research results,
        extracting evidence items and tagging each with its
        ``source_report_id`` for full provenance tracking.

        Returns:
            Merged list of Evidence objects from all sources.
        """
        from inquiro.core.types import Evidence

        all_evidence: list[Evidence] = []

        # 📖 Evidence from input reports
        for report in self.task.input_reports:
            report_evidence = report.content.get(
                "evidence_index",
                [],
            )
            for ev_data in report_evidence:
                if isinstance(ev_data, dict):
                    ev_copy = {**ev_data}
                    ev_copy["source_report_id"] = report.report_id
                    try:
                        all_evidence.append(Evidence(**ev_copy))
                    except Exception:
                        pass

        # 🔬 Evidence from deep-dive research
        for dd in self._additional_research:
            dd_result = dd.get("result", {})
            dd_evidence = dd_result.get("evidence_index", [])
            dd_task_id = dd.get("task_id", "")
            for ev_data in dd_evidence:
                if isinstance(ev_data, dict):
                    ev_copy = {**ev_data}
                    ev_copy["source_report_id"] = dd_task_id
                    try:
                        all_evidence.append(Evidence(**ev_copy))
                    except Exception:
                        pass

        return all_evidence

    def _run_quality_gate(self, raw_result: dict[str, Any]) -> QualityGateResult:
        """Run Quality Gate validation on synthesis output ✅.

        Performs deterministic checks including:
            - Schema validation (hard fail)
            - Required field completeness (hard fail)
            - Cross-reference check (soft fail)
            - Coverage of input reports (soft fail)

        Args:
            raw_result: Raw result dictionary from agent trajectory.

        Returns:
            QualityGateResult with pass/fail status and details.
        """
        return self.quality_gate.validate(result=raw_result)

    def _build_result(
        self,
        raw_result: dict[str, Any],
        qg_result: QualityGateResult | None,
        trajectory: Any,
        warning: str | None = None,
    ) -> SynthesisResult:
        """Build final SynthesisResult from components 📊.

        Assembles the final result object from raw agent output,
        quality gate results, trajectory summary, cost data, and
        additional research metadata.

        Args:
            raw_result: Raw result from agent's finish tool.
            qg_result: Quality gate validation result.
            trajectory: Agent execution trajectory.
            warning: Optional warning message.

        Returns:
            Structured SynthesisResult.
        """
        from inquiro.core.types import (
            Contradiction,
            CrossReference,
            DeepDiveRecord,
            SynthesisResult,
        )

        # 🎯 Parse shared fields via base helpers
        decision = self._parse_decision(raw_result)
        confidence = self._parse_confidence(raw_result, qg_result)
        reasoning = self._parse_reasoning(raw_result)
        llm_evidence = self._parse_evidence(raw_result)

        # 🔍 Backfill evidence from source reports + deep-dives.
        # Source evidence is authoritative provenance; LLM-generated
        # evidence is accepted only when it can be traced back.
        source_evidence = self._collect_source_evidence()

        def _ev_key(ev: Any) -> tuple[str, str, str]:
            source = str(getattr(ev, "source", "") or "")
            url = str(getattr(ev, "url", "") or "").strip().lower()
            summary = str(getattr(ev, "summary", "") or "")[:200].strip().lower()
            return (source, url, summary)

        source_ids = {ev.id for ev in source_evidence if getattr(ev, "id", "")}
        source_keys = {_ev_key(ev) for ev in source_evidence}

        validated_llm: list[Any] = []
        dropped_untraceable = 0
        if source_evidence:
            for ev in llm_evidence:
                if ev.id in source_ids or _ev_key(ev) in source_keys:
                    validated_llm.append(ev)
                else:
                    dropped_untraceable += 1
        else:
            # ⚠️ Fallback: if no authoritative evidence exists,
            # keep LLM-provided evidence to avoid empty output.
            validated_llm = list(llm_evidence)

        if dropped_untraceable > 0:
            self.logger.warning(
                "⚠️ Dropped %d untraceable synthesis evidence item(s) "
                "(not found in source reports/deep-dives)",
                dropped_untraceable,
            )

        # Merge with global dedup (ID + content key)
        evidence_index: list[Any] = []
        seen_ids: set[str] = set()
        seen_keys: set[tuple[str, str, str]] = set()
        for ev in [*validated_llm, *source_evidence]:
            ev_id = str(getattr(ev, "id", "") or "")
            key = _ev_key(ev)
            if ev_id and ev_id in seen_ids:
                continue
            if key in seen_keys:
                continue
            if ev_id:
                seen_ids.add(ev_id)
            seen_keys.add(key)
            evidence_index.append(ev)

        # 📖 Source reports
        source_reports = raw_result.get(
            "source_reports",
            [r.report_id for r in self.task.input_reports],
        )

        # 🔗 Parse cross-references
        cross_refs_raw = raw_result.get("cross_references", [])
        cross_references: list[CrossReference] = []
        for cr_data in cross_refs_raw:
            if isinstance(cr_data, dict):
                try:
                    cross_references.append(CrossReference(**cr_data))
                except Exception as exc:
                    logger.debug(
                        "⚠️ Skipped malformed cross-reference: %s",
                        exc,
                    )

        # ⚔️ Parse contradictions
        contradictions_raw = raw_result.get("contradictions", [])
        contradictions: list[Contradiction] = []
        for ct_data in contradictions_raw:
            if isinstance(ct_data, dict):
                try:
                    contradictions.append(Contradiction(**ct_data))
                except Exception as exc:
                    logger.debug(
                        "⚠️ Skipped malformed contradiction: %s",
                        exc,
                    )

        # 🔍 Remaining gaps
        gaps_remaining = raw_result.get("gaps_remaining", [])

        # 🔬 Build deep-dive records from tracked research
        deep_dives: list[DeepDiveRecord] = []
        for dd in self._additional_research:
            result_summary = ""
            dd_evidence_count = 0
            dd_gaps: list[str] = []
            if "result" in dd:
                result_data = dd["result"]
                result_summary = (
                    f"Decision: "
                    f"{result_data.get('decision', 'unknown')}, "
                    f"Confidence: "
                    f"{result_data.get('confidence', 0.0):.2f}"
                )
                dd_evidence_count = len(
                    result_data.get("evidence_index", []),
                )
                dd_gaps = result_data.get(
                    "gaps_remaining",
                    [],
                )
            elif "error" in dd:
                result_summary = f"Failed: {dd['error']}"

            deep_dives.append(
                DeepDiveRecord(
                    topic=dd.get("topic", ""),
                    task_id=dd.get("task_id", ""),
                    result_summary=result_summary,
                    evidence_count=dd_evidence_count,
                    gaps_remaining=dd_gaps,
                )
            )

        return SynthesisResult(
            task_id=self.task.task_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence_index=evidence_index,
            source_reports=source_reports,
            cross_references=cross_references,
            contradictions=contradictions,
            gaps_remaining=gaps_remaining,
            deep_dives_triggered=deep_dives,
            cost=self.cost_tracker.get_total_cost(),
            raw_output=raw_result,  # ✨ Preserve complete LLM output
        )
