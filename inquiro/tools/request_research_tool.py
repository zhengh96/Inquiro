"""Inquiro RequestResearchTool — deep-dive research trigger 🔬.

Used by SynthesisAgent to request additional atomic research on a
specific topic. Uses a callback pattern: the tool stores pending
requests, and SynthesisExp picks them up after each agent step.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

from inquiro.core.types import (
    AgentConfig,
    Checklist,
    ChecklistItem,
    EvaluationTask,
)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class RequestResearchToolParams(BaseToolParams):
    """Trigger additional evidence-based research on a specific topic.

    Use this tool when you identify gaps or contradictions in the input
    reports that require deeper investigation. A new SearchAgent will
    be spawned to gather additional evidence on the specified topic.

    **Important**: Only use for critical gaps — do not trigger deep dives
    for minor or tangential questions.
    """

    name: ClassVar[str] = "request_research"

    topic: str = Field(
        description=(
            "The specific research topic to investigate. Be precise "
            "and focused — broad topics waste budget."
        )
    )
    search_focus: str | None = Field(
        default=None,
        description=(
            "Optional rules or focus instructions for the research agent. "
            "E.g. 'Focus on Phase 2/3 clinical trial data only.'"
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=("Suggested search keywords for the research agent."),
    )
    prior_context: str | None = Field(
        default=None,
        description=(
            "Summary of what is already known from the input reports. "
            "Helps the research agent avoid redundant searches."
        ),
    )
    justification: str = Field(
        description=(
            "Why this additional research is needed. Must reference "
            "a specific gap or contradiction in the input reports."
        )
    )
    max_turns: int = Field(
        default=10, description="Maximum turns for the spawned research agent."
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class RequestResearchTool(BaseTool):
    """Trigger additional atomic research from within SynthesisAgent 🔬.

    Uses a **callback pattern**: the tool creates an ``EvaluationTask``
    from the parameters and stores it in ``_pending_requests``.
    SynthesisExp reads ``_pending_requests`` after each agent step
    to dispatch the actual research.

    Example (used by SynthesisExp when building agent tools)::

        tool = RequestResearchTool(
            task_runner=eval_task_runner,
            base_config=agent_request_config,
        )
        registry.register(tool)

    Attributes:
        name: ``"request_research"``
        _pending_requests: List of ``EvaluationTask`` objects waiting
            to be picked up by SynthesisExp.
    """

    name: ClassVar[str] = "request_research"
    params_class: ClassVar[type[BaseToolParams]] = RequestResearchToolParams

    def __init__(
        self,
        task_runner: Any | None = None,
        base_config: Any | None = None,
    ) -> None:
        """Initialize with optional references to the task runner 🔧.

        Args:
            task_runner: ``EvalTaskRunner`` instance used to execute
                the spawned research task synchronously. Can be None
                when using the callback pattern.
            base_config: Base ``AgentRequestConfig`` used to configure
                the spawned ``SearchAgent`` (LLM settings, tool
                filter, etc.). Can be None when using the callback pattern.
        """
        super().__init__()
        self._task_runner = task_runner
        self._base_config = base_config
        # 📦 Pending research requests for the callback pattern
        self._pending_requests: list[EvaluationTask] = []
        # 🔒 Lock protecting concurrent access to _pending_requests
        self._lock = threading.Lock()

    def drain_pending_requests(self) -> list[EvaluationTask]:
        """Drain and return all pending research requests 📤.

        Called by SynthesisExp after each agent step to pick up
        newly requested research tasks.

        Returns:
            List of EvaluationTask objects. The internal list is
            cleared after draining.
        """
        with self._lock:
            requests = list(self._pending_requests)
            self._pending_requests.clear()
            return requests

    def execute(
        self,
        session: "BaseSession",
        args_json: str,
    ) -> tuple[str, dict[str, Any]]:
        """Create a research request and store it for SynthesisExp 🔬.

        Steps:
        1. Parse ``RequestResearchToolParams`` from *args_json*.
        2. Build an ``EvaluationTask`` from the params.
        3. Store in ``_pending_requests`` for SynthesisExp to pick up.
        4. Return a confirmation observation.

        Args:
            session: Session instance (unused by this tool).
            args_json: JSON string containing tool parameters.

        Returns:
            Tuple of ``(observation, info)``:
            - On success: confirmation message + task info dict
            - On error: error message + error info dict
        """
        logger = logging.getLogger(self.__class__.__name__)

        # 🎯 Step 1: Parse parameters
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            msg = f"Parameter validation error: {e}"
            logger.warning("❌ %s", msg)
            return msg, {"error": msg}

        assert isinstance(params, RequestResearchToolParams)

        # 🎯 Step 2: Build an EvaluationTask
        task_id = f"deepdive-{uuid.uuid4().hex[:12]}"

        # 📌 Build checklist from keywords
        checklist_items = []
        if params.keywords:
            checklist_items.append(
                ChecklistItem(
                    id="deepdive_focus",
                    description=params.topic,
                    keywords=params.keywords,
                )
            )

        eval_task = EvaluationTask(
            task_id=task_id,
            topic=params.topic,
            rules=params.search_focus or "",
            checklist=Checklist(
                required=checklist_items,
                coverage_threshold=0.5,
            ),
            prior_context=params.prior_context,
            agent_config=AgentConfig(
                max_turns=params.max_turns,
            ),
        )

        # 🎯 Step 3: Store in pending requests (thread-safe)
        with self._lock:
            self._pending_requests.append(eval_task)
        logger.info(
            "🔬 Research request queued: task_id=%s, topic='%s...'",
            task_id,
            params.topic[:60],
        )

        # ✅ Step 4: Return confirmation
        observation = (
            f"Research request accepted (task_id={task_id}). "
            f"Topic: {params.topic}. "
            f"Justification: {params.justification}. "
            "The research will be executed and results integrated."
        )
        return observation, {
            "task_id": task_id,
            "status": "queued",
            "topic": params.topic,
            "justification": params.justification,
        }
