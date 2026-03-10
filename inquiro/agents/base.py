"""Base class for all Inquiro agents with shared infrastructure 🏗️.

Provides common functionality for SearchAgent and SynthesisAgent:
    - Custom InquiroFinishTool with configurable output schema 📋
    - Cooperative cancellation via CancellationToken 🛑
    - Cost budget enforcement via CostTracker 💰
    - Lifecycle event emission via EventEmitter 📡
    - Pre-rendered prompt injection 📝
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from evomaster.agent.agent import BaseAgent, AgentConfig
from evomaster.utils.types import UserMessage

from inquiro.infrastructure.cancellation import CancelledError
from inquiro.infrastructure.cost_tracker import CostStatus
from inquiro.infrastructure.event_emitter import InquiroEvent

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from evomaster.agent.session import BaseSession
    from evomaster.agent.tools import ToolRegistry
    from evomaster.skills import SkillRegistry
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter


class InquiroAgentBase(BaseAgent):
    """Common base for Inquiro agents providing cancellation,
    cost guard, and custom tool setup 🤖.

    Subclasses only need to implement agent-specific logic:
    - SearchAgent: MCP tool search, evidence tracking, adaptive reasoning
    - SynthesisAgent: Report reading, research delegation

    Attributes:
        _system_prompt_text: Pre-rendered system prompt.
        _user_prompt_text: Pre-rendered user prompt.
        _task_id: Task identifier for cost tracking and events.
        _cost_tracker: Optional cost tracking instance.
        _cancellation_token: Optional cancellation signal.
        _event_emitter: Optional event emitter for lifecycle events.
    """

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
        skill_registry: SkillRegistry | None = None,
    ):
        """Initialize InquiroAgentBase 🔧.

        Args:
            llm: LLM instance for inference.
            tools: Tool registry (will be copied + customized).
            system_prompt: Pre-rendered system prompt.
            user_prompt: Pre-rendered user prompt.
            config: Agent configuration.
            output_schema: JSON Schema for output validation.
            task_id: Unique task identifier.
            cost_tracker: Optional cost tracking.
            cancellation_token: Optional cancellation signal.
            event_emitter: Optional event emitter.
            skill_registry: Optional SkillRegistry for agent skill access.
        """
        # 🎯 Store skill registry before tool setup (needed by _setup_custom_tools)
        self._skill_registry = skill_registry

        # ✨ Replace default FinishTool with schema-enforcing version
        custom_tools = self._setup_custom_tools(tools, output_schema)

        super().__init__(
            llm=llm,
            session=self._create_local_session(),
            tools=custom_tools,
            config=config,
            skill_registry=skill_registry,
            enable_tools=True,
        )

        # 🏷️ Set agent name for trajectory identification
        self.set_agent_name(self.__class__.__name__)

        # 📝 Direct prompt injection (not from file)
        self._system_prompt_text = system_prompt
        self._user_prompt_text = user_prompt

        # 🎯 Task identification
        self._task_id = task_id

        # 💰 Cost tracking
        self._cost_tracker = cost_tracker

        # 🛑 Cancellation support
        self._cancellation_token = cancellation_token

        # 📡 Event emission
        self._event_emitter = event_emitter

        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _create_local_session() -> BaseSession:
        """Create a minimal local session for agent execution 🔧.

        Returns:
            A LocalSession instance with default configuration.
        """
        from evomaster.agent.session import LocalSession, LocalSessionConfig

        return LocalSession(LocalSessionConfig())

    def _get_system_prompt(self) -> str:
        """Return injected system prompt 📝.

        The system prompt is pre-rendered by the Exp layer with
        domain-specific rules, checklist, and output schema.

        Returns:
            The pre-rendered system prompt string.
        """
        return self._system_prompt_text

    def _get_user_prompt(self, task: Any) -> str:
        """Return injected user prompt 📝.

        Args:
            task: Task instance (unused — prompt already rendered
                by Exp layer with topic and instructions).

        Returns:
            The pre-rendered user prompt string.
        """
        return self._user_prompt_text

    def _setup_custom_tools(
        self,
        base_tools: ToolRegistry,
        output_schema: dict[str, Any],
    ) -> ToolRegistry:
        """Replace default FinishTool with schema-enforcing InquiroFinishTool 🔧.

        Iterates over all tools in the base registry and replaces any tool
        named "finish" with InquiroFinishTool. If no finish tool exists,
        appends one.

        Args:
            base_tools: Original tool registry.
            output_schema: JSON Schema for output validation.

        Returns:
            New ToolRegistry with InquiroFinishTool.
        """
        from evomaster.agent.tools.base import ToolRegistry
        from inquiro.tools.finish_tool import InquiroFinishTool

        new_registry = ToolRegistry()

        # 🔄 Copy all non-finish tools from the base registry
        for tool in base_tools.get_all_tools():
            if tool.name == "finish":
                continue  # ⏭️ Skip default finish tool
            new_registry.register(tool)

        # ✨ Register schema-enforcing finish tool
        finish_tool = InquiroFinishTool(output_schema)
        new_registry.register(finish_tool)

        # 🎯 Register SkillTool for skill-based knowledge access
        if self._skill_registry is not None:
            try:
                from evomaster.agent.tools.skill import SkillTool

                skill_tool = SkillTool(skill_registry=self._skill_registry)
                new_registry.register(skill_tool)
            except Exception as e:
                logging.getLogger(self.__class__.__name__).warning(
                    "⚠️ Failed to register SkillTool: %s", e
                )

        return new_registry

    def _step(self) -> bool:
        """Execute one step with cancellation and cost guard checks ⚡.

        Before each LLM call:
        1. Check whether cancellation has been requested 🛑

        After each LLM call:
        2. Check whether cost budget has been exceeded 💰

        Returns:
            True if the agent should stop.

        Raises:
            CancelledError: If the cancellation token is triggered.
        """
        # 🛑 Check cancellation before each step
        if (
            self._cancellation_token is not None
            and self._cancellation_token.is_cancelled
        ):
            reason = self._cancellation_token.reason or "Operation cancelled"
            self.logger.warning("⚠️ Task cancelled: %s", reason)
            raise CancelledError(reason)

        # 🔄 Execute the core LLM interaction via base class
        should_finish = super()._step()

        # 💰 Check cost budget after LLM call
        if self._cost_tracker is not None and self._task_id:
            cost_status = self._cost_tracker.check_budget(self._task_id)
            if cost_status in (
                CostStatus.TASK_EXCEEDED,
                CostStatus.TOTAL_EXCEEDED,
            ):
                self.logger.warning(
                    "⚠️ Cost budget exceeded (status=%s), stopping agent",
                    cost_status.value,
                )
                # 📡 Emit cost warning event
                if self._event_emitter:
                    self._event_emitter.emit(
                        InquiroEvent.COST_WARNING,
                        self._task_id,
                        {
                            "status": cost_status.value,
                            "task_cost": self._cost_tracker.get_task_cost(self._task_id)
                            if self._cost_tracker
                            else 0,
                        },
                    )
                return True

            if cost_status == CostStatus.BUDGET_CRITICAL:
                self.logger.warning(
                    "🔥 Budget critical (>90%%), finishing current step"
                )
                if self._event_emitter:
                    self._event_emitter.emit(
                        InquiroEvent.COST_WARNING,
                        self._task_id,
                        {
                            "status": cost_status.value,
                            "level": "critical",
                        },
                    )
            elif cost_status == CostStatus.MODEL_DOWNGRADE:
                self.logger.warning("⚠️ Budget >80%%, consider model downgrade")
                if self._event_emitter:
                    self._event_emitter.emit(
                        InquiroEvent.COST_WARNING,
                        self._task_id,
                        {
                            "status": cost_status.value,
                            "level": "downgrade",
                        },
                    )
            elif cost_status == CostStatus.WARNING:
                self.logger.info("⚠️ Approaching cost budget limit")

        return should_finish

    def _handle_no_tool_call(self) -> None:
        """Handle case where LLM responds without a tool call 🔄.

        Prompts the agent to continue the task process or use
        the finish tool to submit results. Subclasses can override
        to provide agent-specific prompts.
        """
        prompt = self._get_no_tool_call_prompt()
        self.current_dialog.add_message(UserMessage(content=prompt))

    @abstractmethod
    def _get_no_tool_call_prompt(self) -> str:
        """Get the prompt to use when agent doesn't call a tool 📝.

        Subclasses must implement this to provide agent-specific
        instructions for continuing the task.

        Returns:
            The prompt string to add as a user message.
        """
        pass
