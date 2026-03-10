"""Inquiro SynthesisAgent — Evidence synthesis agent 📊.

Reads multiple research reports, cross-references findings,
identifies gaps and contradictions, optionally triggers deep-dive
research via RequestResearchTool, and produces a synthesized result.

Tools available to SynthesisAgent:
    - read_report: Read a completed research report 📖
    - request_research: Trigger additional atomic research (optional) 🔬
    - finish: Submit structured synthesis result (InquiroFinishTool) 📋

Key differences from SearchAgent:
    - No direct MCP search tools — reads pre-loaded reports instead
    - Can trigger SearchAgent internally via RequestResearchTool
    - Focus on cross-referencing and synthesis rather than search
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from evomaster.agent.agent import AgentConfig
from evomaster.utils.types import UserMessage

from inquiro.agents.base import InquiroAgentBase
from inquiro.infrastructure.event_emitter import InquiroEvent

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from evomaster.agent.tools import ToolRegistry
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter
    from inquiro.core.types import InputReport


class SynthesisAgent(InquiroAgentBase):
    """Evidence synthesis agent 📊.

    Reads multiple research reports, cross-references findings,
    identifies gaps and contradictions, optionally triggers deep-dive
    research, and produces a synthesized result.

    Core loop: Read -> Reason -> Identify Gaps -> Synthesize

    Tools available:
        - ``read_report``: Read a completed research report 📖
        - ``request_research``: Trigger additional atomic research 🔬
        - ``finish``: Submit structured synthesis result 📋

    SynthesisAgent-specific features (extends InquiroAgentBase):
        1. No MCP search tools — reads pre-loaded reports instead 📖
        2. Can trigger SearchAgent internally via RequestResearchTool 🔬
        3. Tracks which reports have been read 📚
        4. Stores results from deep-dive research requests 🔬

    Attributes:
        _system_prompt_text: Rendered system prompt with synthesis rules.
        _user_prompt_text: Rendered user prompt with topic and report list.
        _task_id: Task identifier for cost tracking and event emission.
        _output_schema: JSON Schema for output validation.
        _cost_tracker: Optional cost tracking instance.
        _cancellation_token: Optional cancellation signal.
        _event_emitter: Optional event emitter for lifecycle events.
        _reports_read: Tracks which reports have been read.
        _additional_research_results: Stores results from deep-dive research.
    """

    VERSION: str = "1.0"

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str,
        user_prompt: str,
        config: AgentConfig,
        output_schema: dict[str, Any],
        task_id: str = "",
        cost_tracker: CostTracker | None = None,
        cancellation_token: CancellationToken | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize SynthesisAgent 🔧.

        Args:
            llm: LLM instance for inference.
            tools: Tool registry (ReadReport + RequestResearch + Finish).
            system_prompt: Rendered system prompt with synthesis rules.
            user_prompt: Rendered user prompt with topic and report list.
            config: Agent configuration (max_turns, etc.).
            output_schema: JSON Schema for output validation.
            task_id: Unique task identifier for cost tracking and events.
            cost_tracker: Optional cost tracking instance.
            cancellation_token: Optional cancellation signal.
            event_emitter: Optional event emitter for lifecycle events.
        """
        # 🔧 Initialize base infrastructure (prompts, tools, cost, cancellation)
        super().__init__(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            output_schema=output_schema,
            task_id=task_id,
            cost_tracker=cost_tracker,
            cancellation_token=cancellation_token,
            event_emitter=event_emitter,
        )

        # 📋 Output schema for validation (SynthesisAgent-specific)
        self._output_schema = output_schema

        # 📖 Report reading progress (SynthesisAgent-specific)
        self._reports_read: set[str] = set()

        # 🔬 Additional research results from deep-dive requests
        # (SynthesisAgent-specific)
        self._additional_research_results: list[dict[str, Any]] = []

    def _get_no_tool_call_prompt(self) -> str:
        """Get the prompt to use when agent doesn't call a tool 📝.

        Returns:
            The prompt string instructing agent to continue synthesis
            or use the finish tool.
        """
        return (
            "You must continue your synthesis process. "
            "Use `read_report` to read input reports, or if you have "
            "completed your synthesis, use the `finish` tool to submit "
            "your structured result.\n"
            "IMPORTANT: Do not ask for human help. "
            "You must work autonomously."
        )

    def _request_research(
        self,
        topic: str,
        search_focus: str,
        keywords: list[str] | None = None,
        prior_context: str | None = None,
    ) -> dict[str, Any]:
        """Request additional atomic research from SearchAgent 🔬.

        This method is called internally when the SynthesisAgent identifies
        evidence gaps that require deeper investigation. It delegates to the
        RequestResearchTool which queues a research task for SynthesisExp.

        Args:
            topic: The research topic to investigate.
            search_focus: Specific focus areas or rules for the research.
            keywords: Optional list of search keywords.
            prior_context: Optional prior context from existing research.

        Returns:
            Research request info dictionary from the tool execution.
        """
        tool = self.tools.get_tool("request_research")
        if tool is None:
            self.logger.warning("⚠️ RequestResearchTool not available in tool registry")
            return {"error": "RequestResearchTool not available"}

        # 🔬 Build tool arguments
        args = {
            "topic": topic,
            "search_focus": search_focus,
            "keywords": keywords or [],
            "prior_context": prior_context,
            "justification": (f"Gap identified during synthesis: {topic}"),
        }

        observation, info = tool.execute(self.session, json.dumps(args))

        # 📡 Emit event for additional research request
        if self._event_emitter and self._task_id:
            self._event_emitter.emit(
                InquiroEvent.ADDITIONAL_RESEARCH_REQUESTED,
                self._task_id,
                {
                    "topic": topic,
                    "task_id": info.get("task_id", ""),
                },
            )

        self.logger.info(
            "🔬 Requested additional research on: %s",
            topic[:80],
        )
        return info

    def _inject_reports_to_context(
        self,
        reports: list[InputReport],
    ) -> None:
        """Inject report summaries into the agent's context 📖.

        Pre-loads report content as user messages into the dialog so
        the agent can read and cross-reference them. Each report is
        added as a separate user message with structured formatting.

        Args:
            reports: List of input reports to make available.
        """
        if not self.current_dialog:
            self.logger.warning("⚠️ Cannot inject reports: no active dialog")
            return

        for report in reports:
            # 📖 Format report content as a structured user message
            content_str = json.dumps(report.content, indent=2, ensure_ascii=False)
            message_content = (
                f"## Input Report: {report.label} "
                f"(ID: {report.report_id})\n\n"
                f"```json\n{content_str}\n```"
            )
            self.current_dialog.add_message(UserMessage(content=message_content))

            self.logger.debug(
                "📖 Injected report '%s' into context",
                report.report_id,
            )

    def mark_report_read(self, report_id: str) -> None:
        """Mark a report as read by the agent 📖.

        Called by ReadReportTool after successfully returning report content.

        Args:
            report_id: The ID of the report that was read.
        """
        self._reports_read.add(report_id)

    def add_research_result(self, result: dict[str, Any]) -> None:
        """Record an additional research result from deep-dive 🔬.

        Called by SynthesisExp after completing a deep-dive research task
        triggered by RequestResearchTool.

        Args:
            result: The research result dictionary.
        """
        self._additional_research_results.append(result)

        # 📝 Also inject the result into the agent's context
        if self.current_dialog:
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            message_content = (
                f"## Additional Research Result\n\n```json\n{result_str}\n```"
            )
            self.current_dialog.add_message(UserMessage(content=message_content))

    def get_reports_read(self) -> set[str]:
        """Return the set of report IDs that have been read 📖.

        Returns:
            Set of report_id strings.
        """
        return set(self._reports_read)

    def get_additional_research_count(self) -> int:
        """Return number of additional research tasks triggered 🔬.

        Returns:
            Count of deep-dive research results.
        """
        return len(self._additional_research_results)

    def get_additional_research_results(self) -> list[dict[str, Any]]:
        """Return all additional research results 🔬.

        Returns:
            List of research result dictionaries from deep-dive tasks.
        """
        return list(self._additional_research_results)
