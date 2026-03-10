"""Inquiro SearchExp -- lifecycle manager for one search round 🔍.

Wraps SearchAgent for a single search round in the DISCOVERY pipeline.
Manages prompt rendering, agent creation, execution, evidence cleaning,
and structured result assembly.

Architecture position in DISCOVERY mode:
    DiscoveryLoop
        -> SearchExp (lifecycle + prompt + cleanup)    <-- this module
            -> SearchAgent (LLM + MCP tools)
            -> EvidencePipeline (deterministic cleaning, zero LLM)
        -> AnalysisExp (3 LLM parallel analysis + voting)
        -> GapAnalysis (LLM coverage judgment + deterministic convergence)
        -> SynthesisExp (3 LLM voting)

Key design:
    - Does NOT hold LLM directly -- delegates to SearchAgent.
    - Does NOT know about domain -- receives EvaluationTask with
      injected rules, checklist, and output schema.
    - Does NOT manage multi-round logic -- that is DiscoveryLoop's job.
    - Renders prompts from templates and injects focus_prompt for
      subsequent rounds (round > 1).
    - Runs EvidencePipeline.clean() on raw results (zero LLM cost).
    - Returns a structured SearchRoundResult.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from inquiro.core.evidence_pipeline import CleaningStats, EvidencePipeline
from inquiro.core.types import (
    CleanedEvidence,
    Evidence,
    RawEvidence,
)
from inquiro.prompts.loader import PromptLoader
from inquiro.prompts.section_builder import PromptSectionBuilder

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from evomaster.agent.tools import ToolRegistry
    from evomaster.skills import SkillRegistry
    from inquiro.core.discovery_loop import FeedbackProvider
    from inquiro.core.types import DiscoveryConfig, EvaluationTask
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

# 📝 Module-level PromptLoader instance (shared across all SearchExp)
_prompt_loader = PromptLoader()

# 🔍 Default search output schema for SearchAgent finish tool
_SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["raw_evidence", "total_collected", "search_gaps"],
    "properties": {
        "raw_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "mcp_server",
                    "source_query",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "mcp_server": {"type": "string"},
                    "source_query": {"type": "string"},
                    "observation": {"type": "string"},
                    "url": {"type": ["string", "null"]},
                },
            },
        },
        "total_collected": {
            "type": "integer",
            "minimum": 0,
        },
        "search_gaps": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

# 🔬 SRDR reasoning protocol injected in adaptive search mode
_ADAPTIVE_REASONING_PROTOCOL = """

# REASONING PROTOCOL (Adaptive Mode)

You are now operating in **adaptive search mode**: you perform both
evidence collection AND in-context reasoning within the same conversation.

## Mandatory Reasoning Checkpoints

You **MUST** call the `think` tool at these checkpoints:

1. **After your first 2-3 search tool calls**: Analyze initial findings,
   identify which checklist items are already covered, spot gaps, and plan
   remaining searches.

2. **Before calling `finish`**: Assess overall evidence quality, note any
   contradictions or gaps, and confirm your decision rationale.

## Confidence Calibration

Your `confidence` score should reflect evidence completeness relative to
the search checklist:
- **>= 0.80**: All required checklist items covered with primary-source
  evidence from >= 2 different tools.
- **0.60-0.79**: Most required items covered, but some rely on secondary
  sources or a single tool.
- **< 0.60**: Significant checklist gaps or reliance on weak evidence.

## Evidence Rules

1. Every factual claim **MUST** reference at least one evidence item.
2. Do **NOT** fabricate evidence. If unavailable, record as a gap.
3. Prefer primary sources over secondary.
4. When evidence conflicts, note the conflict explicitly.
"""

# 🔍 Discovery reflection protocol injected in discovery intensity mode
_DISCOVERY_REFLECTION_PROTOCOL = """

# REFLECTION PROTOCOL (Discovery Mode)

You have access to the `think` tool for self-reflection checkpoints.

## When to Reflect

Call `think` at these moments:
1. **After 4-5 search tool calls**: Assess checklist coverage and source diversity.
2. **Before calling `finish`**: Mandatory pre-finish checkpoint.

## Pre-Finish Checkpoint

Before calling `finish`, use `think` to verify:
```
think:
  COVERAGE: X/Y required items have evidence
  UNCOVERED: [list missing items]
  SOURCE DIVERSITY: used N/M available tools
  DECISION: FINISH / CONTINUE
```

If required coverage < 75% and you still have tool budget, continue searching.
Do NOT over-reflect — keep think calls brief (under 150 words).
"""


# ============================================================================
# 📊 SearchRoundResult model
# ============================================================================


class SearchRoundResult(BaseModel):
    """Result of one search round in the DISCOVERY pipeline 📊.

    Contains both raw and cleaned evidence, along with metadata
    about query execution, tool usage, timing, and cost.

    Attributes:
        raw_evidence: Unprocessed evidence from MCP search.
        cleaned_evidence: Evidence after pipeline cleaning.
        cleaning_stats: Statistics from the cleaning pipeline.
        queries_executed: List of search queries that were run.
        mcp_tools_used: List of MCP tool names used in this round.
        duration_seconds: Wall-clock duration of the round.
        cost_usd: Estimated LLM cost for this round.
        agent_trajectory_ref: Path to agent JSONL trajectory file.
        error: Error message if the round failed.
    """

    raw_evidence: list[RawEvidence] = Field(
        default_factory=list,
        description="Unprocessed evidence from MCP search",
    )
    cleaned_evidence: list[CleanedEvidence] = Field(
        default_factory=list,
        description="Evidence after pipeline cleaning",
    )
    cleaning_stats: CleaningStats = Field(
        default_factory=lambda: CleaningStats(
            input_count=0,
            output_count=0,
        ),
        description="Statistics from the cleaning pipeline",
    )
    queries_executed: list[str] = Field(
        default_factory=list,
        description="Search queries that were executed",
    )
    mcp_tools_used: list[str] = Field(
        default_factory=list,
        description="MCP tool names used during search",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Wall-clock duration of this round in seconds",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Estimated LLM cost for this round in USD",
    )
    agent_trajectory_ref: str | None = Field(
        default=None,
        description="Path to agent JSONL trajectory file",
    )
    error: str | None = Field(
        default=None,
        description="Error message if the round failed",
    )


# ============================================================================
# 🔍 SearchExp
# ============================================================================


class SearchExp:
    """Lifecycle manager for one search round in DISCOVERY pipeline 🔍.

    Responsibilities:
        - Render search prompts from templates (system + user).
        - Create and configure SearchAgent with MCP tools.
        - Execute search with timeout and error handling.
        - Run EvidencePipeline cleaning on raw results.
        - Return structured SearchRoundResult.

    Does NOT:
        - Hold LLM directly (delegates to SearchAgent).
        - Know about domain (receives EvaluationTask).
        - Manage multi-round logic (that is DiscoveryLoop's job).

    Attributes:
        llm: LLM instance passed to SearchAgent.
        tools: ToolRegistry with MCP search tools.
        evidence_pipeline: Deterministic cleaning pipeline.
        event_emitter: SSE event emitter for progress updates.
        cost_tracker: Cost tracking instance.
        cancellation_token: Cancellation signal.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        event_emitter: EventEmitter | None = None,
        cost_tracker: CostTracker | None = None,
        cancellation_token: CancellationToken | None = None,
        feedback_provider: FeedbackProvider | None = None,
        evolution_enrichment: str | None = None,
        skill_registry: SkillRegistry | None = None,
        adaptive_search: bool = False,
        mcp_server_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize SearchExp 🔧.

        Args:
            llm: LLM instance for SearchAgent inference.
            tools: ToolRegistry containing MCP search tools.
            event_emitter: SSE event emitter for progress updates.
                Creates a no-op emitter if None.
            cost_tracker: Cost tracking instance for budget enforcement.
                Creates a no-op tracker if None.
            cancellation_token: Cancellation signal for cooperative stop.
                Creates a fresh token if None.
            feedback_provider: Optional FeedbackProvider for injecting
                historical search hints into prompts.
            evolution_enrichment: Optional markdown text from the
                evolution system to post-append to the system prompt.
                When None, prompt rendering is unchanged.
            skill_registry: Optional SkillRegistry for SearchAgent skill
                access during evidence collection.
            adaptive_search: When True, SearchAgent operates in SRDR
                mode — performing both search and in-context reasoning
                within the same conversation turn.
            mcp_server_configs: Optional dict of MCP server name to
                config dict, used to render the Tool Selection Guide
                section in the system prompt.
        """
        from inquiro.infrastructure.cancellation import (
            CancellationToken as CT,
        )
        from inquiro.infrastructure.cost_tracker import (
            CostTracker as CTr,
        )
        from inquiro.infrastructure.event_emitter import (
            EventEmitter as EE,
        )

        self.llm = llm
        self.tools = tools
        self.evidence_pipeline = EvidencePipeline()

        self.event_emitter = event_emitter if event_emitter is not None else EE()
        self.cost_tracker = (
            cost_tracker
            if cost_tracker is not None
            else CTr(max_per_task=10.0, max_total=100.0)
        )
        self.cancellation_token = (
            cancellation_token if cancellation_token is not None else CT()
        )

        # 🔄 Trajectory feedback for historical search hints
        self._feedback_provider = feedback_provider

        # 🧬 Evolution enrichment for system prompt injection
        self._evolution_enrichment = evolution_enrichment

        # 🎯 Skill registry for agent knowledge access
        self._skill_registry = skill_registry

        # 🔄 Adaptive search mode (SRDR-like search+reason in one Agent)
        self._adaptive_search = adaptive_search

        # 🗺️ MCP server configs for Tool Selection Guide
        self._mcp_server_configs = mcp_server_configs or {}

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    async def run_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
        previous_evidence: list[Evidence] | None = None,
        agent_trajectory_dir: str | None = None,
    ) -> SearchRoundResult:
        """Execute one search round 🔍.

        Steps:
            1. Render prompts from templates (inject focus if round > 1).
            2. Create SearchAgent with MCP tool config.
            3. Wire agent trajectory file path if agent_trajectory_dir given.
            4. Execute agent search (with timeout from config).
            5. Extract raw evidence from agent results.
            6. Run EvidencePipeline.clean() on raw evidence.
            7. Build and return SearchRoundResult.

        Args:
            task: Evaluation task with topic, rules, checklist, and schema.
            config: Discovery configuration (timeout, etc.).
            round_number: Current round (1-based).
            focus_prompt: Search focus for subsequent rounds (round > 1).
                None for initial broad search.
            previous_evidence: Evidence from prior rounds (for context).
            agent_trajectory_dir: Directory for writing the SearchAgent's
                JSONL trajectory file. When provided, the agent writes
                each step to ``{agent_trajectory_dir}/search_agent_r
                {round_number}_{task_id}.jsonl`` and the path is stored
                in ``SearchRoundResult.agent_trajectory_ref``.

        Returns:
            SearchRoundResult with raw/cleaned evidence and metadata.
        """
        from inquiro.infrastructure.event_emitter import InquiroEvent

        start_time = time.monotonic()
        task_id = task.task_id

        logger.info(
            "🔍 Starting search round %d for task %s",
            round_number,
            task_id,
        )

        # 📡 Emit search start event
        self.event_emitter.emit(
            InquiroEvent.TASK_STARTED,
            task_id,
            {
                "phase": "search",
                "round_number": round_number,
                "has_focus": focus_prompt is not None,
            },
        )

        try:
            # 📝 Step 1: Render prompts
            system_prompt = self._render_system_prompt(
                task, intensity=config.intensity,
            )
            user_prompt = self._render_user_prompt(
                task,
                round_number,
                focus_prompt,
            )

            # 🔧 Step 2: Create SearchAgent
            agent = self._create_search_agent(
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                intensity=config.intensity,
            )

            # 📝 Step 3: Wire agent trajectory path (instance-level only)
            agent_traj_path: str | None = None
            if agent_trajectory_dir:
                safe_task_id = task_id.replace("/", "_")[:16]
                traj_filename = f"search_agent_r{round_number}_{safe_task_id}.jsonl"
                traj_path = Path(agent_trajectory_dir) / traj_filename
                traj_path.parent.mkdir(parents=True, exist_ok=True)
                # 📝 Set on instance to avoid polluting the class-level var
                agent._trajectory_file_path = traj_path
                agent_traj_path = str(traj_path)
                logger.debug(
                    "📝 Agent trajectory path set: %s",
                    agent_traj_path,
                )

            # 🚀 Step 4: Execute agent with timeout
            timeout = config.timeout_per_round
            trajectory = await self._execute_agent_with_timeout(
                agent,
                timeout,
                task_id=task_id,
            )

            # 📥 Step 4: Extract raw evidence from agent
            raw_evidence = self._extract_raw_evidence(agent)
            queries_executed = self._extract_queries(agent)
            mcp_tools_used = list(agent.get_servers_used())

            # 🧹 Step 5: Run EvidencePipeline
            evidence_for_pipeline = self._convert_raw_to_evidence(
                raw_evidence,
            )
            cleaned_list, cleaning_stats = self.evidence_pipeline.clean(
                evidence_for_pipeline
            )
            cleaned_evidence = self._convert_evidence_to_cleaned(
                cleaned_list,
            )

            # 💰 Step 6: Estimate cost
            cost_usd = self._estimate_cost(trajectory)

            duration = time.monotonic() - start_time

            result = SearchRoundResult(
                raw_evidence=raw_evidence,
                cleaned_evidence=cleaned_evidence,
                cleaning_stats=cleaning_stats,
                queries_executed=queries_executed,
                mcp_tools_used=mcp_tools_used,
                duration_seconds=duration,
                cost_usd=cost_usd,
                agent_trajectory_ref=agent_traj_path,
            )

            logger.info(
                "✅ Search round %d completed for task %s: "
                "%d raw -> %d cleaned evidence items in %.1fs",
                round_number,
                task_id,
                len(raw_evidence),
                len(cleaned_evidence),
                duration,
            )

            # 📡 Emit search completion event
            self.event_emitter.emit(
                InquiroEvent.TASK_COMPLETED,
                task_id,
                {
                    "phase": "search",
                    "round_number": round_number,
                    "raw_count": len(raw_evidence),
                    "cleaned_count": len(cleaned_evidence),
                    "cost_usd": cost_usd,
                    "duration_seconds": duration,
                },
            )

            return result

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            logger.warning(
                "⏰ Search round %d timed out for task %s after %.1fs",
                round_number,
                task_id,
                duration,
            )

            # 🔄 Attempt to salvage partial evidence from timed-out agent
            salvaged_raw: list[Any] = []
            salvaged_clean: list[Any] = []
            try:
                salvaged_raw = self._extract_raw_evidence(agent)
                if salvaged_raw:
                    ev_for_pipe = self._convert_raw_to_evidence(
                        salvaged_raw,
                    )
                    cl_list, _ = self.evidence_pipeline.clean(
                        ev_for_pipe,
                    )
                    salvaged_clean = (
                        self._convert_evidence_to_cleaned(cl_list)
                    )
                    logger.info(
                        "🔄 Salvaged %d raw -> %d cleaned "
                        "from timed-out round %d",
                        len(salvaged_raw),
                        len(salvaged_clean),
                        round_number,
                    )
            except Exception as salvage_exc:
                logger.error(
                    "❌ Evidence salvage failed for round %d: %s",
                    round_number,
                    salvage_exc,
                    exc_info=True,
                )
                salvaged_raw = []
                salvaged_clean = []

            timeout_msg = (
                f"Search round {round_number} timed out "
                f"after {config.timeout_per_round}s"
            )
            if salvaged_raw:
                timeout_msg += (
                    f" (salvaged {len(salvaged_raw)} items)"
                )

            return SearchRoundResult(
                raw_evidence=salvaged_raw,
                cleaned_evidence=salvaged_clean,
                duration_seconds=duration,
                error=timeout_msg,
            )

        except Exception as exc:
            duration = time.monotonic() - start_time
            logger.error(
                "❌ Search round %d failed for task %s: %s",
                round_number,
                task_id,
                exc,
            )

            # 📡 Emit failure event
            self.event_emitter.emit(
                InquiroEvent.TASK_FAILED,
                task_id,
                {
                    "phase": "search",
                    "round_number": round_number,
                    "error": str(exc),
                },
            )

            return SearchRoundResult(
                duration_seconds=duration,
                error=str(exc),
            )

    # ====================================================================
    # 📝 Prompt rendering
    # ====================================================================

    def _render_system_prompt(
        self, task: EvaluationTask, intensity: str = "standard",
    ) -> str:
        """Render the search system prompt from template 📝.

        Loads ``search_system.md`` and fills in placeholders for
        alias expansion, available tools, checklist, focus prompt,
        and output schema.

        Args:
            task: Evaluation task with checklist and output schema.
            intensity: Intensity level ("standard" or "discovery").

        Returns:
            Rendered system prompt string.
        """
        # 📋 Format checklist
        checklist_md = PromptSectionBuilder.format_checklist(
            task.checklist,
        )

        # 🔧 Format available tools
        available_tools = PromptSectionBuilder.format_available_tools(
            self.tools,
        )

        # 🗺️ Format Tool Selection Guide (from MCP server configs)
        tool_selection_guide = PromptSectionBuilder.format_tool_selection_guide(
            self._mcp_server_configs,
        ) if self._mcp_server_configs else ""

        # 🎯 Format available skills
        available_skills = PromptSectionBuilder.format_available_skills(
            self._skill_registry,
        )

        # 📊 Format output schema
        schema_str = PromptSectionBuilder.format_output_schema(
            _SEARCH_OUTPUT_SCHEMA,
        )

        # 📋 Load query template from query-templates Skill
        query_template = self._load_query_template(task)

        # 📋 Extract alias expansion and query section guide from strategy
        alias_expansion, query_section_guide = (
            PromptSectionBuilder.format_query_strategy(task.query_strategy)
        )

        # 🔍 Focus prompt: inject historical search patterns if available
        focus_prompt = ""
        if self._feedback_provider:
            focus_prompt = self._feedback_provider.get_system_prompt_hints(
                task_id=task.task_id,
            )

        rendered = _prompt_loader.render(
            "search_system",
            alias_expansion=alias_expansion,
            query_section_guide=query_section_guide,
            available_tools=available_tools,
            tool_selection_guide=tool_selection_guide,
            available_skills=available_skills,
            query_template=query_template,
            search_checklist=checklist_md,
            focus_prompt=focus_prompt,
            output_schema=schema_str,
        )

        # 🔬 Adaptive mode: inject SRDR reasoning protocol
        if self._adaptive_search:
            rendered += _ADAPTIVE_REASONING_PROTOCOL

        # 🔍 Discovery mode: inject reflection protocol
        if intensity == "discovery" and not self._adaptive_search:
            rendered += _DISCOVERY_REFLECTION_PROTOCOL

        # 💡 Inject evolution hints preamble (structured YAML-based hints)
        if task.evolution_hints_preamble:
            rendered += "\n\n" + task.evolution_hints_preamble

        # 🧬 Append evolution enrichment (ExperienceStore free-text experiences)
        if self._evolution_enrichment:
            rendered += "\n\n" + self._evolution_enrichment

        return rendered

    def _render_user_prompt(
        self,
        task: EvaluationTask,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> str:
        """Render the search user prompt from template 📝.

        Loads ``search_user.md`` and fills in the topic, checklist,
        alias expansion, and focus section.

        For round 1: uses generic "Broad initial search" guidance.
        For round > 1: injects the provided focus_prompt.

        Args:
            task: Evaluation task with topic and checklist.
            round_number: Current round in the discovery loop (1-based).
            focus_prompt: Search focus text for subsequent rounds.

        Returns:
            Rendered user prompt string.
        """
        # 📋 Format checklist
        checklist_md = PromptSectionBuilder.format_checklist(
            task.checklist,
        )

        # 🎯 Enrich focus with trajectory gap hints
        gap_hints = ""
        if self._feedback_provider and focus_prompt:
            gap_hints = self._feedback_provider.get_focus_hints(
                gap_descriptions=[focus_prompt],
            )

        # 📝 Build focus section
        focus_section = self._build_focus_section(
            round_number,
            focus_prompt,
            gap_hints,
        )

        # 📋 Extract alias expansion from strategy (guide not used in user prompt)
        alias_expansion, _ = PromptSectionBuilder.format_query_strategy(
            task.query_strategy
        )

        return _prompt_loader.render(
            "search_user",
            topic=task.topic,
            checklist=checklist_md,
            alias_expansion=alias_expansion,
            focus_section=focus_section,
        )

    @staticmethod
    def _build_focus_section(
        round_number: int,
        focus_prompt: str | None = None,
        gap_hints: str = "",
    ) -> str:
        """Build the focus section for the user prompt 🎯.

        Round 1 receives broad search guidance.
        Subsequent rounds receive the focus_prompt from GapAnalysis,
        optionally enriched with historical gap-closing strategies.

        Args:
            round_number: Current round (1-based).
            focus_prompt: Focus text from GapAnalysis (may be None).
            gap_hints: Optional Markdown with historical gap hints.

        Returns:
            Markdown text for the focus section of the user prompt.
        """
        if round_number <= 1 or focus_prompt is None:
            return (
                "## Search Focus\n\n"
                "This is the **initial search round**. "
                "Conduct a broad, comprehensive search across all "
                "available tools to maximize evidence coverage. "
                "Do not narrow your focus prematurely."
            )
        result = (
            f"## Search Focus (Round {round_number})\n\n"
            f"Previous rounds have identified gaps. "
            f"Focus your search on the following areas:\n\n"
            f"{focus_prompt}"
        )
        if gap_hints:
            result += f"\n\n{gap_hints}"
        return result

    # ====================================================================
    # 📋 Query template loading
    # ====================================================================

    def _load_query_template(self, task: EvaluationTask) -> str:
        """Auto-load query template from Skill registry 📋.

        Mirrors the original query template loading logic.
        When the ``query-templates`` Skill has a reference matching
        the task's sub_item_id, injects it into the system prompt
        as a pre-loaded query sequence guide.

        Args:
            task: Evaluation task with sub_item_id for template matching.

        Returns:
            Markdown section with query template, or empty string.
        """
        if self._skill_registry is None:
            return ""

        topic_key = self._resolve_template_topic(task)
        if not topic_key:
            return ""

        try:
            skill = self._skill_registry.get_skill("query-templates")
            if skill is None:
                return ""

            reference_name = f"{topic_key}.md"
            content = skill.get_reference(reference_name)

            logger.info(
                "📋 Auto-loaded query template for SearchExp: %s (%d chars)",
                reference_name,
                len(content),
            )

            return (
                "## Query Template\n\n"
                "The following expert query template has been pre-loaded "
                "for your research topic. **Follow its query sequence and "
                "tool allocation strategy.** Substitute `{variable}` "
                "placeholders with your task-specific values.\n\n"
                f"{content}"
            )
        except FileNotFoundError:
            logger.debug(
                "📋 No query template found for topic: %s",
                topic_key,
            )
            return ""
        except Exception as exc:
            logger.warning(
                "⚠️ Failed to load query template for %s: %s",
                topic_key,
                exc,
            )
            return ""

    @staticmethod
    def _resolve_template_topic(task: EvaluationTask) -> str:
        """Resolve the topic key for query template matching 🔍.

        Priority:
        1. task.sub_item_id (set by upper layer)
        2. First required checklist item id
        3. First checklist item id (any)

        Args:
            task: Evaluation task.

        Returns:
            Topic key string, or empty string if unresolvable.
        """
        if hasattr(task, "sub_item_id") and task.sub_item_id:
            return task.sub_item_id

        checklist = task.checklist
        if hasattr(checklist, "required") and checklist.required:
            return checklist.required[0].id
        if hasattr(checklist, "optional") and checklist.optional:
            return checklist.optional[0].id

        return ""

    # ====================================================================
    # 🔧 Agent creation and execution
    # ====================================================================

    def _create_search_agent(
        self,
        task: EvaluationTask,
        system_prompt: str,
        user_prompt: str,
        intensity: str = "standard",
    ) -> Any:
        """Create a configured SearchAgent instance 🔧.

        Args:
            task: Evaluation task (for config and task_id).
            system_prompt: Pre-rendered system prompt.
            user_prompt: Pre-rendered user prompt.
            intensity: Intensity level ("standard" or "discovery").

        Returns:
            Configured SearchAgent instance ready for execution.
        """
        from evomaster.agent.agent import AgentConfig
        from evomaster.agent.context import (
            ContextConfig as EvoContextConfig,
            TruncationStrategy as EvoTruncationStrategy,
        )
        from inquiro.agents.search_agent import SearchAgent

        # 🔧 Map Inquiro ContextConfig → EvoMaster ContextConfig
        evo_context: EvoContextConfig | None = None
        if (
            hasattr(task, "agent_config")
            and task.agent_config
            and hasattr(task.agent_config, "context")
            and task.agent_config.context
        ):
            ctx = task.agent_config.context
            evo_context = EvoContextConfig(
                max_tokens=ctx.max_tokens,
                truncation_strategy=EvoTruncationStrategy(
                    ctx.truncation_strategy.value
                ),
                preserve_system_messages=ctx.preserve_system_messages,
                preserve_recent_turns=ctx.preserve_recent_turns,
            )

        # 🔧 Cap search agent turns to prevent timeout without finish.
        # Default AgentConfig.max_turns=30 is too many for a search task
        # with 12+ MCP servers. 15 turns is sufficient for diverse search.
        search_max_turns = min(task.agent_config.max_turns, 15)
        agent_config = AgentConfig(
            max_turns=search_max_turns,
            context_config=evo_context or EvoContextConfig(),
        )

        # 🧠 Register ThinkTool for adaptive SRDR or discovery reflection
        if self._adaptive_search or intensity == "discovery":
            from evomaster.agent.tools.builtin.think import ThinkTool

            self.tools.register(ThinkTool())

        # 🎯 Filter skills to SearchAgent-relevant subset
        skill_registry = self._skill_registry
        if skill_registry is not None:
            try:
                skill_registry = skill_registry.create_subset([
                    "query-templates",
                    "alias-expansion",
                    "evidence-grader",
                    "search-reflection",
                    "evidence-type-taxonomy",
                    "dimension-routing",
                ])
            except Exception:
                pass  # Fall back to full registry

        # 📊 Use full evaluation output_schema in adaptive mode
        output_schema = (
            task.output_schema if self._adaptive_search
            else _SEARCH_OUTPUT_SCHEMA
        )

        return SearchAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=agent_config,
            output_schema=output_schema,
            task_id=task.task_id,
            cost_tracker=self.cost_tracker,
            cancellation_token=self.cancellation_token,
            event_emitter=self.event_emitter,
            skill_registry=skill_registry,
            adaptive_search=self._adaptive_search,
        )

    async def _execute_agent_with_timeout(
        self,
        agent: Any,
        timeout_seconds: int,
        task_id: str = "",
    ) -> Any:
        """Execute SearchAgent with timeout in a thread pool 🚀.

        The agent's ``run()`` method is blocking, so it is run in
        a thread pool via ``asyncio.to_thread()``.

        Args:
            agent: SearchAgent instance to execute.
            timeout_seconds: Maximum allowed execution time.
            task_id: Task identifier for creating TaskInstance.

        Returns:
            Agent trajectory after execution.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        from evomaster.utils.types import TaskInstance

        task_instance = TaskInstance(
            task_id=task_id or "search",
            task_type="search",
            description="Evidence search task",
        )

        async def _run_agent() -> Any:
            return await asyncio.to_thread(agent.run, task_instance)

        await asyncio.wait_for(
            _run_agent(),
            timeout=timeout_seconds,
        )
        return getattr(agent, "trajectory", None)

    # ====================================================================
    # 📥 Evidence extraction and conversion
    # ====================================================================

    @staticmethod
    def _extract_raw_evidence(
        agent: Any,
    ) -> list[RawEvidence]:
        """Extract raw evidence records from SearchAgent 📥.

        Converts the agent's internal record dicts into typed
        RawEvidence model instances.

        Args:
            agent: SearchAgent instance after execution.

        Returns:
            List of RawEvidence items from the agent.
        """
        records = agent.get_raw_evidence_records()
        raw_evidence: list[RawEvidence] = []

        for record in records:
            raw_evidence.append(
                RawEvidence(
                    id=record.get("id", ""),
                    source_query=record.get("source_query", ""),
                    mcp_server=record.get("mcp_server", ""),
                    observation=record.get("observation", ""),
                    url=record.get("url"),
                )
            )

        return raw_evidence

    @staticmethod
    def _extract_queries(agent: Any) -> list[str]:
        """Extract the list of executed queries from SearchAgent 📊.

        Args:
            agent: SearchAgent instance after execution.

        Returns:
            List of query strings that were executed.
        """
        queries_data = agent.get_queries_executed()
        return [q.get("query", "") for q in queries_data]

    @staticmethod
    def _convert_raw_to_evidence(
        raw_evidence: list[RawEvidence],
    ) -> list[Evidence]:
        """Convert RawEvidence items to Evidence for pipeline 🔄.

        The EvidencePipeline expects ``Evidence`` objects. This method
        maps RawEvidence fields to the Evidence model.

        Args:
            raw_evidence: List of raw evidence from the agent.

        Returns:
            List of Evidence items suitable for EvidencePipeline.clean().
        """
        evidence_items: list[Evidence] = []
        for raw in raw_evidence:
            evidence_items.append(
                Evidence(
                    id=raw.id,
                    source=raw.mcp_server,
                    url=raw.url,
                    query=raw.source_query,
                    summary=raw.observation,
                )
            )
        return evidence_items

    @staticmethod
    def _convert_evidence_to_cleaned(
        evidence_list: list[Evidence],
    ) -> list[CleanedEvidence]:
        """Convert cleaned Evidence items to CleanedEvidence model 🔄.

        After EvidencePipeline.clean(), the items are still Evidence
        objects. This converts them to the CleanedEvidence model
        used in SearchRoundResult.

        Args:
            evidence_list: Evidence items after pipeline cleaning.

        Returns:
            List of CleanedEvidence items.
        """
        from inquiro.core.evidence_pipeline import EvidencePipeline

        cleaned: list[CleanedEvidence] = []
        for ev in evidence_list:
            # 🏷️ Classify evidence tag from URL (no monkey-patching)
            tag_value = EvidencePipeline.classify_url(ev.url or "")

            cleaned.append(
                CleanedEvidence(
                    id=ev.id,
                    summary=ev.summary,
                    url=ev.url,
                    tag=tag_value.value,
                    source_query=ev.query,
                    mcp_server=ev.source,
                )
            )
        return cleaned

    @staticmethod
    def _estimate_cost(trajectory: Any) -> float:
        """Estimate LLM cost from agent trajectory metadata 💰.

        Iterates through trajectory steps and sums up token usage
        to estimate cost.

        Args:
            trajectory: Agent execution trajectory (may be None).

        Returns:
            Estimated cost in USD.
        """
        if not trajectory:
            return 0.0

        steps = getattr(trajectory, "steps", []) or []
        total_cost = 0.0

        for step in steps:
            msg = getattr(step, "assistant_message", None)
            if msg is None:
                continue

            meta = getattr(msg, "meta", {}) or {}
            usage = meta.get("usage", {})

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # 💰 Conservative average cost estimate
            input_cost = input_tokens * 3.0 / 1_000_000
            output_cost = output_tokens * 15.0 / 1_000_000
            total_cost += input_cost + output_cost

        return total_cost
