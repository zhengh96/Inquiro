"""DiscoveryLoop — multi-round search-analyze-gap orchestrator 🔄.

Core orchestrator for the DISCOVERY pipeline.  Coordinates iterative
rounds of search, evidence cleaning, analysis, and gap assessment until
a convergence condition is met.

Architecture position::

    Runner → DiscoveryLoop (this module)
                ├── SearchExp  → SearchAgent (MCP search)
                ├── EvidencePipeline (zero-LLM cleaning)
                ├── AnalysisExp (3-model parallel analysis)
                ├── GapAnalysis (coverage + convergence)
                └── SynthesisExp (3-model final synthesis)

Key design decisions:
    - Domain-agnostic: no pharma terms.  All domain knowledge injected
      via ``EvaluationTask.rules / checklist / output_schema``.
    - Does NOT hold LLM directly — delegates to Exp/Agent layers.
    - Trajectory recording: writes JSONL after each phase completion.
    - Focus prompt generation: lightweight LLM call to guide next round.
    - Accumulative evidence: each round adds to the shared evidence pool.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from inquiro.core.evidence_condenser import (
    CondensedEvidence,
    CondenserConfig,
    EvidenceCondenser,
    GroupSummarizer,
)
from inquiro.core.evidence_pipeline import CleaningStats, EvidencePipeline
from inquiro.core.metadata_enricher import MetadataEnricher
from inquiro.core.evidence_pool import SharedEvidencePool
from inquiro.core.gap_analysis import (
    CoverageJudge,
    CoverageResult,
    GapAnalysis,
    MockCoverageJudge,
)
from inquiro.infrastructure.event_emitter import EventEmitter, InquiroEvent
from inquiro.core.trajectory.models import (
    AnalysisPhaseRecord,
    CleaningPhaseRecord,
    ConsensusRecord,
    DiscoveryRoundRecord,
    DiscoverySummary,
    GapPhaseRecord,
    ModelAnalysisRecord,
    SearchPhaseRecord,
    ServerStats,
    TrajectoryEventType,
)
from inquiro.core.trajectory.writer import TrajectoryWriter
from inquiro.core.types import (
    ChecklistCoverage,
    CleanedEvidence,
    DiscoveryConfig,
    DiscoveryResult,
    DiscoveryRoundSummary,
    Evidence,
    EvidenceTag,
    EvolutionProvider,
    FocusPrompt,
    GapReport,
    RoundMetrics,
)

if TYPE_CHECKING:
    from inquiro.core.types import EvaluationTask

logger = logging.getLogger(__name__)


# ============================================================================
# 🤖 Protocols for injectable components
# ============================================================================


class SearchExecutor(Protocol):
    """Protocol for search round execution 🔍.

    Allows DiscoveryLoop to run search without knowing the concrete
    SearchExp implementation.
    """

    async def execute_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> SearchRoundOutput:
        """Execute one search round and return raw evidence 🔍.

        Args:
            task: The evaluation task with rules and checklist.
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            focus_prompt: Optional focus guidance for targeted search.

        Returns:
            SearchRoundOutput with raw evidence and metadata.
        """
        ...


class AnalysisExecutor(Protocol):
    """Protocol for evidence analysis execution 🔬.

    Allows DiscoveryLoop to run analysis without knowing the concrete
    AnalysisExp implementation.
    """

    async def execute_analysis(
        self,
        task: EvaluationTask,
        evidence: list[Evidence],
        config: DiscoveryConfig,
        round_number: int,
        supplementary_context: str | None = None,
    ) -> AnalysisRoundOutput:
        """Analyze evidence with multi-model consensus 🔬.

        Args:
            task: The evaluation task with rules and checklist.
            evidence: Cleaned evidence to analyze.
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            supplementary_context: Optional text summaries of evidence
                excluded by Tier-2 condensation. Appended to user prompt
                so the LLM is aware of out-of-window evidence groups.

        Returns:
            AnalysisRoundOutput with claims and consensus info.
        """
        ...


class FeedbackProvider(Protocol):
    """Protocol for trajectory feedback providers 🔄.

    Provides formatted historical hints for prompt injection
    and structured feedback for focus prompt enrichment.
    """

    def get_system_prompt_hints(
        self,
        task_id: str | None = None,
    ) -> str:
        """Get formatted query templates for system prompt 📊."""
        ...

    def get_focus_hints(
        self,
        gap_descriptions: list[str],
    ) -> str:
        """Get formatted gap-closing strategies 🎯."""
        ...

    def get_feedback(
        self,
        gap_descriptions: list[str],
    ) -> Any:
        """Get structured feedback with hints and text 📊."""
        ...


class FocusPromptGenerator(Protocol):
    """Protocol for focus prompt generation 🎯.

    Generates targeted search guidance based on gap analysis results.
    """

    async def generate_focus(
        self,
        gap_report: GapReport,
        config: DiscoveryConfig,
        round_number: int,
    ) -> FocusPrompt:
        """Generate a focus prompt for the next round 🎯.

        Args:
            gap_report: Gap analysis result from current round.
            config: Discovery pipeline configuration.
            round_number: Current round number (just completed).

        Returns:
            FocusPrompt with targeted search guidance.
        """
        ...


# ============================================================================
# 📊 Data models for round I/O
# ============================================================================


class SearchRoundOutput(BaseModel):
    """Output from one search round 📊.

    Attributes:
        evidence: Raw evidence items collected.
        queries_executed: Search queries that were run.
        mcp_tools_used: MCP tools that were invoked.
        cost_usd: Search cost in USD.
        duration_seconds: Time taken for search.
    """

    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Raw evidence items collected",
    )
    queries_executed: list[str] = Field(
        default_factory=list,
        description="Search queries that were run",
    )
    mcp_tools_used: list[str] = Field(
        default_factory=list,
        description="MCP tools that were invoked",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Search cost in USD",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Search duration in seconds",
    )
    agent_trajectory_ref: str | None = Field(
        default=None,
        description="Path to the SearchAgent JSONL trajectory file",
    )
    agent_trajectory_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Paths to all SearchAgent JSONL trajectory files (parallel sections)"
        ),
    )
    section_errors: list[str] = Field(
        default_factory=list,
        description="Error messages from failed parallel sections",
    )


class AnalysisRoundOutput(BaseModel):
    """Output from one analysis round 📊.

    Attributes:
        claims: Consensus claims from multi-model analysis.
        model_decisions: Per-model decisions.
        consensus_decision: Majority vote decision.
        consensus_ratio: Fraction of models agreeing.
        cost_usd: Analysis cost in USD.
        duration_seconds: Time taken for analysis.
    """

    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Consensus claims with evidence references",
    )
    model_decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-model analysis decisions",
    )
    consensus_decision: str = Field(
        default="cautious",
        description="Majority vote decision",
    )
    consensus_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model agreement ratio",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Analysis cost in USD",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Analysis duration in seconds",
    )
    checklist_coverage: ChecklistCoverage | None = Field(
        default=None,
        description=(
            "Merged checklist coverage from analysis agent "
            "(ChecklistCoverage or None if not available)"
        ),
    )
    coverage_conflicts: list[str] = Field(
        default_factory=list,
        description="Checklist items where models disagreed on coverage",
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Information gaps identified by analysis models",
    )
    doubts_remaining: list[str] = Field(
        default_factory=list,
        description="Evidence contradictions or doubts from analysis",
    )


# ============================================================================
# 🧪 Mock implementations for testing
# ============================================================================


class MockSearchExecutor:
    """Mock search executor for testing 🧪.

    Returns pre-configured evidence items without calling any MCP tools.

    Attributes:
        mock_evidence: Evidence items to return from execute_search().
        call_count: Number of times execute_search() has been called.
    """

    def __init__(
        self,
        mock_evidence: list[Evidence] | None = None,
    ) -> None:
        """Initialize with optional pre-configured evidence 🔧.

        Args:
            mock_evidence: Evidence items to return. Defaults to empty list.
        """
        self.mock_evidence = mock_evidence or []
        self.call_count = 0

    async def execute_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> SearchRoundOutput:
        """Return pre-configured evidence 🧪.

        Args:
            task: Evaluation task (unused in mock).
            config: Discovery config (unused in mock).
            round_number: Round number (unused in mock).
            focus_prompt: Focus prompt (unused in mock).

        Returns:
            SearchRoundOutput with mock evidence.
        """
        self.call_count += 1
        return SearchRoundOutput(
            evidence=self.mock_evidence,
            queries_executed=[f"mock-query-{round_number}"],
            mcp_tools_used=["mock-tool"],
            cost_usd=0.10,
            duration_seconds=1.0,
        )


class MockAnalysisExecutor:
    """Mock analysis executor for testing 🧪.

    Returns pre-configured claims without calling any LLM.

    Attributes:
        mock_claims: Claims to return from execute_analysis().
        mock_decision: Decision to return.
        call_count: Number of times execute_analysis() has been called.
    """

    def __init__(
        self,
        mock_claims: list[dict[str, Any]] | None = None,
        mock_decision: str = "positive",
    ) -> None:
        """Initialize with optional pre-configured claims 🔧.

        Args:
            mock_claims: Claims to return. Defaults to empty list.
            mock_decision: Decision string. Defaults to "positive".
        """
        self.mock_claims = mock_claims or []
        self.mock_decision = mock_decision
        self.call_count = 0

    async def execute_analysis(
        self,
        task: EvaluationTask,
        evidence: list[Evidence],
        config: DiscoveryConfig,
        round_number: int,
        supplementary_context: str | None = None,
    ) -> AnalysisRoundOutput:
        """Return pre-configured analysis result 🧪.

        Args:
            task: Evaluation task (unused in mock).
            evidence: Evidence list (unused in mock).
            config: Discovery config (unused in mock).
            round_number: Round number (unused in mock).
            supplementary_context: Tier-2 group summaries (unused in mock).

        Returns:
            AnalysisRoundOutput with mock claims and decision.
        """
        self.call_count += 1
        return AnalysisRoundOutput(
            claims=self.mock_claims,
            consensus_decision=self.mock_decision,
            consensus_ratio=0.80,
            cost_usd=0.50,
            duration_seconds=2.0,
        )


class DefaultFocusPromptGenerator:
    """Default focus prompt generator used when no feedback provider exists 🎯.

    Produces structured, multi-part search guidance from a GapReport,
    including explicit lists of uncovered items, suggested search queries,
    and a "do not re-search" section for already-covered items to prevent
    redundant searches in subsequent rounds.

    Attributes:
        call_count: Number of times generate_focus() has been called.
    """

    # 📝 High-quality source types to prioritize in searches
    _PRIORITY_SOURCES = (
        "clinical trials",
        "peer-reviewed journals",
        "systematic reviews",
        "meta-analyses",
    )

    def __init__(self, follow_up_rules: str = "") -> None:
        """Initialize the default focus prompt generator 🔧.

        Args:
            follow_up_rules: Raw markdown of follow-up / gap-closing
                guidance from the query template.  When non-empty,
                appended to every focus prompt as an additional section.
        """
        self.call_count = 0
        self._follow_up_rules = follow_up_rules

    async def generate_focus(
        self,
        gap_report: GapReport,
        config: DiscoveryConfig,
        round_number: int,
    ) -> FocusPrompt:
        """Generate an enhanced multi-part focus prompt from gap data 🎯.

        Builds a structured prompt that:
        1. Enumerates specific uncovered items with per-item search guidance.
        2. Instructs the agent to use DIFFERENT search terms than previous
           rounds to avoid redundant queries.
        3. Includes a "do NOT re-search" exclusion list for covered items.
        4. Requests prioritization of high-quality evidence sources.

        Falls back to a broadening instruction when all items are covered.

        Args:
            gap_report: Gap analysis result from the current round.
            config: Discovery configuration (controls max focus items).
            round_number: Round number just completed (used for logging).

        Returns:
            FocusPrompt with detailed search guidance and gap targeting.
        """
        self.call_count += 1
        uncovered = gap_report.uncovered_items[: config.gap_focus_max_items]
        covered = gap_report.covered_items

        if not uncovered:
            # 🔍 All gaps closed — broaden scope to strengthen evidence depth
            return FocusPrompt(
                prompt_text=self._build_broadening_prompt(covered),
                target_gaps=[],
                suggested_queries=[],
            )

        prompt_text = self._build_focus_prompt(
            uncovered=uncovered,
            covered=covered,
            round_number=round_number,
        )
        suggested_queries = self._build_suggested_queries(uncovered)

        logger.debug(
            "🎯 DefaultFocusPromptGenerator: round=%d, uncovered=%d, covered=%d",
            round_number,
            len(uncovered),
            len(covered),
        )
        return FocusPrompt(
            prompt_text=prompt_text,
            target_gaps=uncovered,
            suggested_queries=suggested_queries,
        )

    # ------------------------------------------------------------------ #
    # 🔧 Private helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_focus_prompt(
        self,
        uncovered: list[str],
        covered: list[str],
        round_number: int,
    ) -> str:
        """Assemble the full multi-section focus prompt string 📝.

        Args:
            uncovered: Items that still lack sufficient evidence.
            covered: Items already covered (to be excluded).
            round_number: Completed round number for context.

        Returns:
            Formatted multi-section prompt text.
        """
        sections: list[str] = []

        # 🎯 Section 1: primary directive
        sections.append(
            "SEARCH FOCUS — UNCOVERED ITEMS\n"
            "You MUST search specifically for the following uncovered "
            "checklist items. Each item requires dedicated, targeted "
            "queries using DIFFERENT search terms than those used in "
            f"previous rounds (completed: {round_number})."
        )

        # 📋 Section 2: enumerated uncovered items with per-item guidance
        item_lines: list[str] = []
        for idx, item in enumerate(uncovered, start=1):
            # 🔧 Generate diverse search angle suggestions per item
            item_lines.append(
                f"  {idx}. {item}\n"
                f"     → Try: synonyms, alternative terminology, "
                f"related concepts, and different keyword combinations."
            )
        sections.append("UNCOVERED ITEMS:\n" + "\n".join(item_lines))

        # 🔑 Section 3: source quality guidance
        source_list = ", ".join(self._PRIORITY_SOURCES)
        sections.append(
            f"SOURCE PRIORITY: Prioritize {source_list}. "
            "Avoid re-fetching sources already consulted."
        )

        # 🚫 Section 4: exclusion list (only when there are covered items)
        if covered:
            exclusion_lines = "\n".join(f"  - {item}" for item in covered)
            sections.append(
                "ALREADY COVERED — DO NOT RE-SEARCH:\n"
                "The following items already have sufficient evidence. "
                "Do NOT spend search budget re-investigating them:\n" + exclusion_lines
            )

        # 📋 Section 5: follow-up rules from query template (when available)
        if self._follow_up_rules:
            sections.append(
                "FOLLOW-UP GUIDANCE (from query template):\n" + self._follow_up_rules
            )

        return "\n\n".join(sections)

    def _build_broadening_prompt(self, covered: list[str]) -> str:
        """Build a prompt instructing the agent to broaden search scope 🔍.

        Used when all checklist items are covered; directs the agent to
        deepen existing evidence rather than searching for new items.

        Args:
            covered: Items already covered.

        Returns:
            Broadening focus prompt text.
        """
        if not covered:
            return "Broaden search scope to gather additional evidence."

        covered_list = "; ".join(covered[:5])
        suffix = f" and {len(covered) - 5} more" if len(covered) > 5 else ""
        return (
            "All checklist items are covered. Broaden search scope to "
            "strengthen evidence depth and quality for: "
            f"{covered_list}{suffix}. "
            "Seek additional corroborating sources, especially "
            + ", ".join(self._PRIORITY_SOURCES)
            + "."
        )

    def _build_suggested_queries(self, uncovered: list[str]) -> list[str]:
        """Generate diverse suggested search queries for uncovered items 💡.

        Creates two query variants per item — a direct query and a
        broader conceptual query — to encourage search diversity.

        Args:
            uncovered: Uncovered checklist item descriptions.

        Returns:
            List of suggested search query strings.
        """
        queries: list[str] = []
        for item in uncovered:
            # 🔍 Direct query
            queries.append(f"search: {item}")
            # 🔍 Broader alternative framing
            queries.append(f"evidence for: {item} (alternative terminology)")
        return queries


# ⬇️ Backward-compatibility alias — existing code using MockFocusPromptGenerator
# continues to work without modification.
MockFocusPromptGenerator = DefaultFocusPromptGenerator


class TrajectoryAwareFocusGenerator:
    """Focus prompt generator enriched with trajectory history 🎯.

    Wraps the DefaultFocusPromptGenerator logic with historical gap-closing
    strategies from TrajectoryFeedbackProvider.  Adds two additional sections
    to the base prompt:

    1. **Search diversity hints**: suggests alternative MCP tools when
       previous rounds over-relied on certain tools.
    2. **"DO NOT search for" exclusion list**: explicitly prohibits
       re-searching already-covered items to save search budget.

    Falls back to DefaultFocusPromptGenerator output when no historical
    data is available.

    Attributes:
        _feedback: FeedbackProvider for historical hints.
        _base_generator: DefaultFocusPromptGenerator for base prompt.
    """

    def __init__(
        self,
        feedback_provider: FeedbackProvider,
        follow_up_rules: str = "",
    ) -> None:
        """Initialize with a FeedbackProvider 🔧.

        Args:
            feedback_provider: FeedbackProvider instance for
                historical gap-closing strategies.
            follow_up_rules: Raw markdown of follow-up / gap-closing
                guidance.  Forwarded to the base generator.
        """
        self._feedback = feedback_provider
        self._base_generator = DefaultFocusPromptGenerator(
            follow_up_rules=follow_up_rules,
        )

    async def generate_focus(
        self,
        gap_report: GapReport,
        config: DiscoveryConfig,
        round_number: int,
    ) -> FocusPrompt:
        """Generate a focus prompt enriched with trajectory history 🎯.

        Builds on DefaultFocusPromptGenerator and then appends:
        - Historical gap-closing strategy hints from the FeedbackProvider.
        - A search diversity section that recommends alternative MCP tools
          when historical data shows tool over-reliance.
        - An explicit "DO NOT search for" block for covered items.

        Falls back gracefully if the FeedbackProvider call fails.

        Args:
            gap_report: Gap analysis result from current round.
            config: Discovery pipeline configuration.
            round_number: Current round number (just completed).

        Returns:
            FocusPrompt with targeted search guidance and historical hints.
        """
        uncovered = gap_report.uncovered_items[: config.gap_focus_max_items]
        covered = gap_report.covered_items

        # 🏗️ Start from the enhanced base prompt
        base = await self._base_generator.generate_focus(
            gap_report, config, round_number
        )
        prompt_parts: list[str] = [base.prompt_text]
        suggested_queries: list[str] = list(base.suggested_queries)
        suggested_tools: list[str] = []

        # 🎯 Enrich with historical gap-closing strategies
        if uncovered:
            try:
                # 📊 Single call to get both text and structured data
                feedback = self._feedback.get_feedback(uncovered)
                if feedback.gap_hints:
                    from inquiro.core.trajectory.feedback import (
                        TrajectoryFeedbackProvider,
                    )

                    hints_text = TrajectoryFeedbackProvider._format_gap_hints(
                        feedback.gap_hints,
                    )
                    prompt_parts.append("HISTORICAL SEARCH STRATEGIES:\n" + hints_text)
                    for hint in feedback.gap_hints:
                        suggested_queries.extend(hint.effective_queries)
                        suggested_tools.extend(hint.recommended_tools)
            except Exception as exc:
                logger.warning(
                    "⚠️ Trajectory feedback failed at round %d: %s",
                    round_number,
                    exc,
                )

        # 🔀 Search diversity section
        diversity_section = self._build_diversity_section(suggested_tools)
        if diversity_section:
            prompt_parts.append(diversity_section)

        # 🚫 Explicit "DO NOT search for" block for covered items
        if covered:
            exclusion_section = self._build_covered_exclusion_section(covered)
            prompt_parts.append(exclusion_section)

        return FocusPrompt(
            prompt_text="\n\n".join(prompt_parts),
            target_gaps=uncovered,
            suggested_queries=list(dict.fromkeys(suggested_queries)),
            suggested_tools=list(dict.fromkeys(suggested_tools)),
        )

    # ------------------------------------------------------------------ #
    # 🔧 Private helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_diversity_section(suggested_tools: list[str]) -> str:
        """Build a search-diversity guidance section 🔀.

        When historical data identifies tools that were used before,
        encourages the agent to try alternative tools for broader
        coverage.  Returns an empty string when no tools are known.

        Args:
            suggested_tools: Tool names from historical hints.

        Returns:
            Diversity guidance text, or empty string if not applicable.
        """
        if not suggested_tools:
            return ""
        unique_tools = list(dict.fromkeys(suggested_tools))
        tool_list = ", ".join(unique_tools)
        return (
            "SEARCH DIVERSITY — TOOL SELECTION:\n"
            f"Previously effective tools include: {tool_list}. "
            "If you have already used these tools in prior rounds, "
            "CONSIDER using alternative tools to reach different data "
            "sources and avoid redundant results."
        )

    @staticmethod
    def _build_covered_exclusion_section(covered: list[str]) -> str:
        """Build an explicit exclusion block for already-covered items 🚫.

        Instructs the agent not to spend search budget on checklist items
        that already have sufficient evidence.

        Args:
            covered: Checklist items already marked as covered.

        Returns:
            Formatted exclusion guidance text.
        """
        item_lines = "\n".join(f"  - {item}" for item in covered)
        return (
            "DO NOT SEARCH FOR — ALREADY COVERED:\n"
            "The following checklist items already have sufficient "
            "evidence. Do NOT re-search or re-fetch sources for them:\n" + item_lines
        )


# ============================================================================
# 📦 Loop state container
# ============================================================================


@dataclass
class _LoopState:
    """Mutable state carried across discovery rounds 📦.

    Consolidates all local variables that the discovery loop accumulates
    across rounds into a single container, making the ``run()`` method
    simpler and the per-round helper signatures cleaner.

    Attributes:
        all_evidence: Accumulated cleaned evidence across rounds.
        all_claims: Accumulated consensus claims across rounds.
        gap_reports: Gap analysis results per round.
        round_summaries: Summary records per round.
        total_cost: Running cost total in USD.
        previous_coverage: Coverage ratio from the prior round.
        coverage_curve: Coverage ratio after each round.
        gap_report: Most recent gap report (None before first round).
        focus_prompt: Focus prompt text for the next round.
        termination_reason: Why the loop stopped.
        seeded_mode: Whether KB seeded evidence was provided.
        loop_start: Monotonic timestamp when the loop began.
    """

    all_evidence: list[Evidence] = field(default_factory=list)
    all_claims: list[dict[str, Any]] = field(default_factory=list)
    gap_reports: list[GapReport] = field(default_factory=list)
    round_summaries: list[DiscoveryRoundSummary] = field(
        default_factory=list,
    )
    total_cost: float = 0.0
    previous_coverage: float = 0.0
    coverage_curve: list[float] = field(default_factory=list)
    gap_report: GapReport | None = None
    focus_prompt: str | None = None
    termination_reason: str = ""
    seeded_mode: bool = False
    loop_start: float = 0.0
    prev_covered_items: set[str] = field(default_factory=set)


# ============================================================================
# 🔄 DiscoveryLoop
# ============================================================================


class DiscoveryLoop:
    """Multi-round search-analyze-gap orchestrator 🔄.

    Coordinates iterative rounds of search, evidence cleaning, analysis,
    and gap assessment until a convergence condition is met.  Each round
    adds to the shared evidence and claims pools.

    Convergence is determined by GapAnalysis, which checks five stopping
    conditions in priority order: coverage threshold → budget → max rounds
    → diminishing returns → search exhaustion.

    Trajectory recording writes JSONL after each phase completion for
    crash-safe streaming persistence.

    Attributes:
        search_executor: Pluggable search component.
        analysis_executor: Pluggable analysis component.
        gap_analysis: Coverage and convergence checker.
        evidence_pipeline: Deterministic evidence cleaner.
        focus_generator: Focus prompt generator for next rounds.
        trajectory_writer: JSONL trajectory writer (optional).
    """

    def __init__(
        self,
        search_executor: SearchExecutor,
        analysis_executor: AnalysisExecutor,
        gap_analysis: GapAnalysis | None = None,
        evidence_pipeline: EvidencePipeline | None = None,
        focus_generator: FocusPromptGenerator | None = None,
        trajectory_dir: str | None = None,
        coverage_judge: CoverageJudge | None = None,
        event_emitter: EventEmitter | None = None,
        feedback_provider: FeedbackProvider | None = None,
        evolution_provider: EvolutionProvider | None = None,
        condenser: EvidenceCondenser | None = None,
        group_summarizer: GroupSummarizer | None = None,
        metadata_enricher: MetadataEnricher | None = None,
    ) -> None:
        """Initialize DiscoveryLoop with injectable components 🔧.

        Args:
            search_executor: Component that runs search rounds.
            analysis_executor: Component that runs analysis rounds.
            gap_analysis: Gap analysis instance. Defaults to GapAnalysis
                with MockCoverageJudge.
            evidence_pipeline: Evidence cleaner. Defaults to
                EvidencePipeline().
            focus_generator: Focus prompt generator. Defaults to
                DefaultFocusPromptGenerator().
            trajectory_dir: Directory for JSONL trajectory files. When
                None, trajectory recording is disabled.
            coverage_judge: Custom coverage judge for GapAnalysis.
                Ignored if gap_analysis is provided.
            event_emitter: Optional SSE event emitter for real-time
                progress streaming. When None, SSE events are not emitted.
            feedback_provider: Optional FeedbackProvider for enriching
                focus prompts with historical data.  When provided and
                focus_generator is the default, auto-upgrades to
                TrajectoryAwareFocusGenerator.
            evolution_provider: Optional EvolutionProvider for injecting
                learned experiences into prompts and collecting trajectory
                data after each round.
            condenser: Optional EvidenceCondenser for three-tier evidence
                condensation before analysis.  When None a default instance
                is created using config thresholds from DiscoveryConfig.
            group_summarizer: Optional GroupSummarizer for LLM-based
                enrichment of Tier 2 group summaries.  When provided,
                template-based summaries are replaced with LLM-generated
                text.  Falls back to template text on failure.
            metadata_enricher: Optional MetadataEnricher for LLM-based
                evidence metadata extraction.  When None, metadata
                enrichment is skipped.
        """
        self.search_executor = search_executor
        self.analysis_executor = analysis_executor
        self.gap_analysis = gap_analysis or GapAnalysis(
            coverage_judge=coverage_judge or MockCoverageJudge()
        )
        self.evidence_pipeline = evidence_pipeline or EvidencePipeline()
        self.focus_generator = focus_generator or DefaultFocusPromptGenerator()

        # 🔄 Auto-upgrade focus generator when feedback is available
        self._feedback_provider = feedback_provider
        if feedback_provider and isinstance(
            self.focus_generator, DefaultFocusPromptGenerator
        ):
            self.focus_generator = TrajectoryAwareFocusGenerator(
                feedback_provider,
            )
            logger.info("🔄 Focus generator upgraded to TrajectoryAwareFocusGenerator")

        # 🧬 Evolution provider for experience injection/collection
        self._evolution = evolution_provider
        # 📊 Track enrichment injection counts for hit rate
        self._enrichment_injected_count: int = 0
        self._enrichment_total_rounds: int = 0

        # 📊 Trajectory recording
        self._trajectory_dir = trajectory_dir
        self._trajectory_writer: TrajectoryWriter | None = None

        # 📡 SSE event emitter for real-time progress
        self._event_emitter = event_emitter

        # 🗜️ Evidence condenser for token-overflow prevention
        self._condenser = condenser

        # 📝 LLM group summarizer for Tier 2 enrichment
        self._group_summarizer = group_summarizer

        # 🏷️ Optional metadata enricher for LLM-based extraction
        self._metadata_enricher = metadata_enricher

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    @property
    def trajectory_writer(self) -> TrajectoryWriter | None:
        """Expose the underlying TrajectoryWriter for post-loop writes 📊.

        Allows callers (e.g. runner.py) to append synthesis records to
        the same JSONL file after the discovery loop has completed.

        Returns:
            TrajectoryWriter instance if trajectory_dir was configured,
            None otherwise.
        """
        return self._trajectory_writer

    @property
    def enrichment_hit_rate(self) -> float:
        """Calculate enrichment injection rate across rounds 📊.

        Returns:
            Fraction of rounds where enrichment was injected (0.0-1.0).
            Returns 0.0 if no rounds have been recorded.
        """
        if self._enrichment_total_rounds == 0:
            return 0.0
        return self._enrichment_injected_count / self._enrichment_total_rounds

    async def run(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig | None = None,
        shared_evidence_pool: SharedEvidencePool | None = None,
        *,
        seeded_evidence: list[Evidence] | None = None,
        seeded_gap_hints: list[str] | None = None,
    ) -> DiscoveryResult:
        """Execute the full discovery loop 🔄.

        Runs iterative rounds of search → clean → analyze → gap until
        convergence.  After the loop, returns accumulated results.

        When a ``shared_evidence_pool`` is provided, the loop will:
        1. Pre-fill local evidence with relevant items from the pool
           at the start of each round (skipped when seeded_evidence
           is provided — seeded evidence takes priority).
        2. Contribute newly cleaned evidence back to the pool after
           each round's cleaning phase.

        When ``seeded_evidence`` is provided (Knowledge Base pre-fill):
        - Evidence items are injected BEFORE round 1 as a starting set.
        - ``shared_evidence_pool`` pre-fill is skipped (seeded takes
          priority), but contribution back to the pool is unchanged.
        - ``all_claims`` is reset at the start of each round
          (seeded mode) to prevent stale claim accumulation across
          re-analysis cycles.

        When ``seeded_gap_hints`` is provided:
        - Hints are prepended to the focus prompt for round 1 only,
          directing the agent toward known gaps from prior KB knowledge.

        Args:
            task: Evaluation task with rules, checklist, and output_schema.
            config: Discovery configuration. If None, parses from
                task.discovery_config or uses defaults.
            shared_evidence_pool: Optional shared pool for cross-task
                evidence reuse.  When None, evidence is not shared.
            seeded_evidence: Optional pre-filled evidence from the
                Knowledge Base.  When provided, takes priority over
                shared_evidence_pool pre-fill.  Must be keyword-only.
            seeded_gap_hints: Optional known gap descriptions from the
                Knowledge Base.  Injected into round 1 focus prompt to
                guide search toward uncovered areas.  Must be
                keyword-only.

        Returns:
            DiscoveryResult with all accumulated evidence, claims, gap
            reports, and round summaries.
        """
        # 🔧 Resolve configuration
        config = self._resolve_config(task, config)
        task_id = task.task_id
        trajectory_id = str(uuid.uuid4())

        logger.info(
            "🔄 DiscoveryLoop starting for task %s "
            "(max_rounds=%d, coverage=%.2f, budget=$%.2f)",
            task_id,
            config.max_rounds,
            config.coverage_threshold,
            config.max_cost_per_subitem,
        )

        # 📊 Initialize trajectory writer
        self._init_trajectory(task_id, trajectory_id, config, task)

        # 📝 Extract checklist items for gap analysis
        checklist = self._extract_checklist(task)

        # 📡 Emit DISCOVERY_STARTED SSE event
        self._emit_sse(
            InquiroEvent.DISCOVERY_STARTED,
            task_id,
            {
                "max_rounds": config.max_rounds,
                "coverage_threshold": config.coverage_threshold,
            },
        )

        # 📦 Prepare loop state (seeds, focus hints, follow-up rules)
        state = self._prepare_loop_state(
            task,
            config,
            checklist,
            seeded_evidence=seeded_evidence,
            seeded_gap_hints=seeded_gap_hints,
            shared_evidence_pool=shared_evidence_pool,
        )

        # 🔄 Main discovery loop
        for round_num in range(1, config.max_rounds + 1):
            if await self._execute_round(
                task,
                config,
                checklist,
                round_num,
                state,
                shared_evidence_pool,
            ):
                break
        else:
            # 🔧 Loop completed without convergence
            state.termination_reason = "max_rounds_reached"

        return self._build_discovery_result(
            task_id,
            trajectory_id,
            state,
        )

    # ====================================================================
    # 🔧 Extracted loop helpers
    # ====================================================================

    def _prepare_loop_state(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        checklist: list[str],
        *,
        seeded_evidence: list[Evidence] | None = None,
        seeded_gap_hints: list[str] | None = None,
        shared_evidence_pool: SharedEvidencePool | None = None,
    ) -> _LoopState:
        """Initialize all loop accumulators and inject seeds 📦.

        Sets up the ``_LoopState`` container, injects seeded evidence /
        gap hints, applies follow-up rules to the focus generator, and
        pre-fills from the shared evidence pool when appropriate.

        Args:
            task: Evaluation task with rules, checklist, and strategies.
            config: Resolved discovery configuration.
            checklist: Extracted checklist item descriptions.
            seeded_evidence: Optional KB pre-filled evidence.
            seeded_gap_hints: Optional KB gap hint descriptions.
            shared_evidence_pool: Optional shared cross-task pool.

        Returns:
            Fully initialized ``_LoopState`` ready for the round loop.
        """
        state = _LoopState(loop_start=time.monotonic())

        # 🌱 Inject seeded gap hints into round-1 focus prompt
        if seeded_gap_hints:
            hint_lines = "\n".join(
                f"  {i + 1}. {hint}" for i, hint in enumerate(seeded_gap_hints)
            )
            state.focus_prompt = (
                "KB GAP HINTS — KNOWN COVERAGE GAPS\n"
                "The following gaps were identified from prior "
                "knowledge. Prioritise searching for evidence "
                "that addresses them:\n" + hint_lines
            )
            logger.info(
                "🌱 Injected %d KB gap hints into round-1 focus prompt",
                len(seeded_gap_hints),
            )

        # 📋 Extract follow-up rules from query strategy
        follow_up_rules = ""
        if task.query_strategy and isinstance(task.query_strategy, dict):
            follow_up_rules = task.query_strategy.get(
                "follow_up_rules",
                "",
            )

        # 🔧 Inject follow-up rules into focus generator
        if follow_up_rules:
            if isinstance(
                self.focus_generator,
                TrajectoryAwareFocusGenerator,
            ):
                self.focus_generator = TrajectoryAwareFocusGenerator(
                    feedback_provider=self._feedback_provider,  # type: ignore[arg-type]
                    follow_up_rules=follow_up_rules,
                )
            elif isinstance(
                self.focus_generator,
                DefaultFocusPromptGenerator,
            ):
                self.focus_generator = DefaultFocusPromptGenerator(
                    follow_up_rules=follow_up_rules,
                )
            logger.info(
                "📋 Injected follow-up rules into focus generator (%d chars)",
                len(follow_up_rules),
            )

        # 🧬 Evolution enrichment state (read by adapters)
        self._current_search_enrichment: str | None = None
        self._current_analysis_enrichment: str | None = None

        # 🌱 Store seeded gap hints for potential downstream use
        self._seeded_gap_hints = seeded_gap_hints

        # ── KB: Pre-fill seeded evidence ──
        if seeded_evidence:
            state.all_evidence.extend(seeded_evidence)
            state.seeded_mode = True
            logger.info(
                "📦 Pre-filled %d evidence from knowledge base",
                len(seeded_evidence),
            )
        elif shared_evidence_pool is not None and shared_evidence_pool.size > 0:
            shared_items = shared_evidence_pool.get_relevant(
                checklist,
                limit=config.shared_pool_prefill_limit,
            )
            if shared_items:
                state.all_evidence.extend(shared_items)
                logger.info(
                    "🔄 Pre-filled %d evidence items from shared pool for task %s",
                    len(shared_items),
                    task.task_id,
                )

        return state

    async def _execute_round(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        checklist: list[str],
        round_num: int,
        state: _LoopState,
        shared_pool: SharedEvidencePool | None,
    ) -> bool:
        """Execute one complete discovery round 🔄.

        Runs search → clean → analyze → gap → trajectory → evolution
        for a single round.  Updates ``state`` in-place.

        Args:
            task: Evaluation task.
            config: Discovery configuration.
            checklist: Checklist item descriptions.
            round_num: Current round number (1-based).
            state: Mutable loop state container.
            shared_pool: Optional shared evidence pool.

        Returns:
            True if the loop should break (converged), False otherwise.
        """
        task_id = task.task_id
        round_start = time.monotonic()

        logger.info(
            "🔄 === Round %d/%d ===  evidence=%d claims=%d coverage=%.2f cost=$%.2f",
            round_num,
            config.max_rounds,
            len(state.all_evidence),
            len(state.all_claims),
            state.previous_coverage,
            state.total_cost,
        )

        # ✏️ Emit round start events
        self._emit_event(
            TrajectoryEventType.ROUND_STARTED,
            {"round_number": round_num},
        )
        self._emit_sse(
            InquiroEvent.DISCOVERY_ROUND_STARTED,
            task_id,
            {"round_number": round_num},
        )

        # 🧬 Get enrichment from evolution provider
        search_enrichment, analysis_enrichment = self._fetch_evolution_enrichment(
            round_num, state.gap_report
        )
        self._current_analysis_enrichment = analysis_enrichment
        self._enrichment_total_rounds += 1
        if search_enrichment or analysis_enrichment:
            self._enrichment_injected_count += 1

        # ── 1-2. Search + Clean ────────────────────────────
        # 📦 Skip search on round 1 when KB seeded evidence is available:
        # the evidence pool is already pre-filled from the knowledge base.
        # Subsequent rounds (if gap analysis triggers them) search normally.
        if state.seeded_mode and round_num == 1:
            logger.info(
                "⏩ Skipping search for round 1 (KB seeded mode, "
                "%d evidence pre-filled)",
                len(state.all_evidence),
            )
            search_output = SearchRoundOutput()
            cleaned: list[Evidence] = []
            cleaning_stats = CleaningStats(
                input_count=0,
                output_count=0,
                dedup_removed=0,
                noise_removed=0,
            )
        else:
            search_output, cleaned, cleaning_stats = await self._search_and_clean(
                task,
                config,
                round_num,
                state,
                shared_pool,
                search_enrichment,
            )
        search_cost = search_output.cost_usd
        new_evidence_count = len(cleaned)

        # ── 3. Analyze ─────────────────────────────────────
        if state.seeded_mode:
            state.all_claims = []

        analysis_output = await self._run_analysis(
            task,
            state.all_evidence,
            config,
            round_num,
        )
        analysis_cost = analysis_output.cost_usd
        state.total_cost += analysis_cost
        state.all_claims.extend(analysis_output.claims)

        # ── 4. Gap Analysis ────────────────────────────────
        # ✅ Convert analysis agent coverage (IDs) → CoverageResult (descriptions)
        pre_coverage = self._build_pre_computed_coverage(
            analysis_output, task, checklist
        )

        gap_report = await self._run_gap_analysis(
            checklist_items=checklist,
            claims=state.all_claims,
            evidence=state.all_evidence,
            previous_coverage=state.previous_coverage,
            round_number=round_num,
            config=config,
            cost_spent=state.total_cost,
            pre_computed_coverage=pre_coverage,
        )
        state.gap_report = gap_report
        state.gap_reports.append(gap_report)
        state.total_cost += gap_report.judge_cost_usd

        # 📊 Build round summary + emit SSE + trajectory
        round_duration = time.monotonic() - round_start
        round_cost = search_cost + analysis_cost + gap_report.judge_cost_usd
        self._finalize_round(
            round_num,
            task_id,
            search_output,
            cleaning_stats,
            analysis_output,
            gap_report,
            round_cost,
            round_duration,
            new_evidence_count,
            state,
        )

        # 📊 Persist round record + evolution hook
        await self._build_and_persist_round_record(
            round_num,
            search_output,
            cleaning_stats,
            analysis_output,
            gap_report,
            round_cost,
            round_duration,
            state,
            new_evidence_count,
        )

        # 🛑 Check convergence and generate focus
        return await self._check_convergence_and_focus(
            gap_report,
            config,
            round_num,
            task_id,
            state,
        )

    def _fetch_evolution_enrichment(
        self,
        round_num: int,
        gap_report: GapReport | None,
    ) -> tuple[str | None, str | None]:
        """Fetch search and analysis enrichment from evolution provider 🧬.

        Args:
            round_num: Current round number.
            gap_report: Most recent gap report (may be None).

        Returns:
            Tuple of (search_enrichment, analysis_enrichment).
        """
        search_enrichment: str | None = None
        analysis_enrichment: str | None = None
        if not self._evolution:
            return search_enrichment, analysis_enrichment

        try:
            gap_items = gap_report.uncovered_items if gap_report else []
            search_enrichment = self._evolution.get_search_enrichment(
                round_num,
                gap_items,
            )
        except Exception:
            logger.warning(
                "⚠️ Evolution search enrichment failed for round %d",
                round_num,
                exc_info=True,
            )
        try:
            analysis_enrichment = self._evolution.get_analysis_enrichment()
        except Exception:
            logger.warning(
                "⚠️ Evolution analysis enrichment failed for round %d",
                round_num,
                exc_info=True,
            )
        return search_enrichment, analysis_enrichment

    async def _search_and_clean(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_num: int,
        state: _LoopState,
        shared_pool: SharedEvidencePool | None,
        search_enrichment: str | None,
    ) -> tuple[SearchRoundOutput, list[Evidence], CleaningStats]:
        """Run search + cleaning and update state accumulators 🔍🧹.

        Args:
            task: Evaluation task.
            config: Discovery configuration.
            round_num: Current round number.
            state: Mutable loop state container.
            shared_pool: Optional shared evidence pool.
            search_enrichment: Optional evolution search enrichment.

        Returns:
            Tuple of (search_output, cleaned_evidence, cleaning_stats).
        """
        task_id = task.task_id

        # ── Search ────────────────────────────────────────
        search_output = await self._run_search(
            task,
            config,
            round_num,
            state.focus_prompt,
            search_enrichment=search_enrichment,
        )
        state.total_cost += search_output.cost_usd

        # ⚠️ I7: Detect and report search section errors
        if search_output.section_errors:
            for err in search_output.section_errors:
                logger.warning(
                    "⚠️ Search section error (round %d): %s",
                    round_num,
                    err,
                )
            self._emit_sse(
                InquiroEvent.DISCOVERY_SEARCH_WARNING,
                task_id,
                {
                    "round_number": round_num,
                    "section_errors": search_output.section_errors,
                },
            )

        # ── Clean ─────────────────────────────────────────
        cleaned, cleaning_stats = self._run_cleaning(
            search_output.evidence,
        )
        state.all_evidence.extend(cleaned)

        # 🏷️ Metadata enrichment (optional, async LLM)
        if self._metadata_enricher and cleaned:
            cleaned = await self._metadata_enricher.enrich(cleaned)

        # 🔄 Contribute new evidence to shared pool
        if shared_pool is not None and cleaned:
            pool_added = shared_pool.add(cleaned)
            if pool_added > 0:
                logger.info(
                    "🔄 Contributed %d new items to shared pool (round %d, task %s)",
                    pool_added,
                    round_num,
                    task_id,
                )

        return search_output, cleaned, cleaning_stats

    def _finalize_round(
        self,
        round_num: int,
        task_id: str,
        search_output: SearchRoundOutput,
        cleaning_stats: CleaningStats,
        analysis_output: AnalysisRoundOutput,
        gap_report: GapReport,
        round_cost: float,
        round_duration: float,
        new_evidence_count: int,
        state: _LoopState,
    ) -> None:
        """Build round summary and emit SSE progress events 📊.

        Args:
            round_num: Round number (1-based).
            task_id: Task identifier for SSE events.
            search_output: Search phase results.
            cleaning_stats: Cleaning statistics.
            analysis_output: Analysis phase results.
            gap_report: Gap analysis results.
            round_cost: Total cost for this round in USD.
            round_duration: Wall-clock duration for this round.
            new_evidence_count: Count of newly cleaned evidence.
            state: Mutable loop state (updated in place).
        """
        coverage_delta = gap_report.coverage_ratio - state.previous_coverage
        round_summary = DiscoveryRoundSummary(
            round_number=round_num,
            queries_executed=len(search_output.queries_executed),
            raw_evidence_count=len(search_output.evidence),
            cleaned_evidence_count=new_evidence_count,
            coverage_ratio=gap_report.coverage_ratio,
            coverage_delta=coverage_delta,
            round_cost_usd=round_cost,
            converged=gap_report.converged,
            convergence_reason=gap_report.convergence_reason,
        )
        state.round_summaries.append(round_summary)
        state.coverage_curve.append(gap_report.coverage_ratio)

        self._emit_sse(
            InquiroEvent.DISCOVERY_ROUND_COMPLETED,
            task_id,
            {
                "round_number": round_num,
                "coverage": gap_report.coverage_ratio,
                "cost_usd": state.total_cost,
                "evidence_count": len(state.all_evidence),
            },
        )
        self._emit_sse(
            InquiroEvent.DISCOVERY_COVERAGE_UPDATED,
            task_id,
            {"coverage_curve": list(state.coverage_curve)},
        )

    async def _build_and_persist_round_record(
        self,
        round_num: int,
        search_output: SearchRoundOutput,
        cleaning_stats: CleaningStats,
        analysis_output: AnalysisRoundOutput,
        gap_report: GapReport,
        round_cost: float,
        round_duration: float,
        state: _LoopState,
        new_evidence_count: int,
    ) -> None:
        """Build round record, persist to trajectory, call evolution hook 📊.

        Args:
            round_num: Round number (1-based).
            search_output: Search phase results.
            cleaning_stats: Cleaning phase statistics.
            analysis_output: Analysis phase results.
            gap_report: Gap analysis results.
            round_cost: Total cost for this round in USD.
            round_duration: Wall-clock duration for this round.
            state: Loop state for evolution metrics.
            new_evidence_count: Count of newly cleaned evidence.
        """
        round_record = self._build_round_record(
            round_num=round_num,
            search_output=search_output,
            cleaning_stats=cleaning_stats,
            analysis_output=analysis_output,
            gap_report=gap_report,
            round_cost=round_cost,
            round_duration=round_duration,
            state=state,
        )

        # 📊 Persist to trajectory file when writer is configured
        if self._trajectory_writer:
            self._trajectory_writer.write_round(round_record)

        # 🧬 Evolution hook — extract and store experiences
        if self._evolution:
            if not self._trajectory_writer:
                logger.warning(
                    "⚠️ Evolution hook called without trajectory "
                    "recording — round record built from in-memory "
                    "data",
                )
            try:
                round_metrics = RoundMetrics(
                    evidence_count=len(state.all_evidence),
                    new_evidence_count=new_evidence_count,
                    coverage=gap_report.coverage_ratio,
                    cost_usd=state.total_cost,
                    round_index=round_num - 1,
                )
                await self._evolution.on_round_complete(
                    round_num,
                    round_record,
                    round_metrics,
                )
            except Exception:
                logger.warning(
                    "⚠️ Evolution on_round_complete failed for "
                    "round %d, continuing without evolution",
                    round_num,
                    exc_info=True,
                )

    async def _check_convergence_and_focus(
        self,
        gap_report: GapReport,
        config: DiscoveryConfig,
        round_num: int,
        task_id: str,
        state: _LoopState,
    ) -> bool:
        """Check convergence and generate focus prompt for next round 🛑.

        Updates ``state.previous_coverage``, ``state.termination_reason``,
        and ``state.focus_prompt`` as side effects.

        Args:
            gap_report: Gap analysis result from current round.
            config: Discovery configuration.
            round_num: Current round number.
            task_id: Task identifier for SSE events.
            state: Mutable loop state container.

        Returns:
            True if converged (loop should break), False otherwise.
        """
        # 🏁 Single-round mode: always converge after round 1.
        # Avoids unnecessary gap analysis → focus prompt generation
        # when max_rounds=1 is explicitly configured.
        if config.max_rounds == 1:
            state.termination_reason = "single_round_complete"
            state.previous_coverage = gap_report.coverage_ratio
            logger.info(
                "🏁 Single-round mode — converging after round %d",
                round_num,
            )
            self._emit_event(
                TrajectoryEventType.CONVERGENCE_REACHED,
                {
                    "round_number": round_num,
                    "reason": state.termination_reason,
                },
            )
            self._emit_sse(
                InquiroEvent.DISCOVERY_CONVERGED,
                task_id,
                {
                    "reason": state.termination_reason,
                    "total_rounds": round_num,
                },
            )
            return True

        state.previous_coverage = gap_report.coverage_ratio

        if gap_report.converged:
            state.termination_reason = gap_report.convergence_reason or ""
            logger.info(
                "🛑 DiscoveryLoop converged at round %d: %s",
                round_num,
                state.termination_reason,
            )
            self._emit_event(
                TrajectoryEventType.CONVERGENCE_REACHED,
                {
                    "round_number": round_num,
                    "reason": state.termination_reason,
                },
            )
            self._emit_sse(
                InquiroEvent.DISCOVERY_CONVERGED,
                task_id,
                {
                    "reason": state.termination_reason,
                    "total_rounds": round_num,
                },
            )
            return True

        # ── Generate focus prompt for next round ──────────
        focus_prompt_obj = await self._generate_focus(
            gap_report,
            config,
            round_num,
        )
        state.focus_prompt = focus_prompt_obj.prompt_text

        self._emit_event(
            TrajectoryEventType.ROUND_COMPLETED,
            {"round_number": round_num},
        )
        return False

    def _build_discovery_result(
        self,
        task_id: str,
        trajectory_id: str,
        state: _LoopState,
    ) -> DiscoveryResult:
        """Build the final DiscoveryResult and finalize trajectory 📊.

        Args:
            task_id: Task identifier.
            trajectory_id: Trajectory identifier.
            state: Loop state with accumulated data.

        Returns:
            Complete DiscoveryResult.
        """
        total_duration = time.monotonic() - state.loop_start

        logger.info(
            "🔄 DiscoveryLoop completed for task %s: "
            "rounds=%d, coverage=%.2f, cost=$%.2f, reason=%s",
            task_id,
            len(state.round_summaries),
            state.previous_coverage,
            state.total_cost,
            state.termination_reason,
        )

        # 📡 Emit DISCOVERY_COMPLETED SSE event
        self._emit_sse(
            InquiroEvent.DISCOVERY_COMPLETED,
            task_id,
            {
                "total_rounds": len(state.round_summaries),
                "final_coverage": state.previous_coverage,
                "total_cost_usd": state.total_cost,
                "total_evidence": len(state.all_evidence),
                "termination_reason": state.termination_reason,
            },
        )

        # 📊 Write summary and finalize trajectory
        self._finalize_trajectory(
            total_rounds=len(state.round_summaries),
            final_coverage=state.previous_coverage,
            total_cost=state.total_cost,
            total_evidence=len(state.all_evidence),
            total_claims=len(state.all_claims),
            total_duration=total_duration,
            termination_reason=state.termination_reason,
        )

        return DiscoveryResult(
            task_id=task_id,
            pipeline_mode="discovery",
            total_rounds=len(state.round_summaries),
            final_coverage=state.previous_coverage,
            total_cost_usd=state.total_cost,
            termination_reason=state.termination_reason,
            evidence=[self._to_cleaned_evidence(e) for e in state.all_evidence],
            claims=state.all_claims,
            gap_reports=state.gap_reports,
            round_summaries=state.round_summaries,
            trajectory_id=trajectory_id,
        )

    # ====================================================================
    # 🔧 Internal methods
    # ====================================================================

    def _resolve_config(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig | None,
    ) -> DiscoveryConfig:
        """Resolve DiscoveryConfig from task or explicit parameter 🔧.

        Args:
            task: Evaluation task (may contain discovery_config dict).
            config: Explicitly passed config (takes priority).

        Returns:
            Resolved DiscoveryConfig instance.
        """
        if config is not None:
            return config
        if task.discovery_config:
            return DiscoveryConfig.model_validate(task.discovery_config)
        return DiscoveryConfig()

    def _extract_checklist(self, task: EvaluationTask) -> list[str]:
        """Extract checklist item descriptions from task 📋.

        Extracts descriptions from required checklist items only,
        since coverage is measured against required items.

        Args:
            task: Evaluation task with checklist.

        Returns:
            List of checklist item description strings.
        """
        if not task.checklist or not task.checklist.required:
            return []
        return [item.description for item in task.checklist.required]

    def _init_trajectory(
        self,
        task_id: str,
        trajectory_id: str,
        config: DiscoveryConfig,
        task: EvaluationTask,
    ) -> None:
        """Initialize trajectory writer if trajectory_dir is set 📊.

        Args:
            task_id: Task identifier.
            trajectory_id: Unique trajectory identifier.
            config: Discovery configuration.
            task: Evaluation task.
        """
        if self._trajectory_dir:
            self._trajectory_writer = TrajectoryWriter(
                output_dir=self._trajectory_dir,
                task_id=task_id,
            )
            self._trajectory_writer.write_meta(
                trajectory_id=trajectory_id,
                config_snapshot=config.model_dump(),
                task_snapshot={
                    "topic": task.topic,
                    "rules_length": len(task.rules) if task.rules else 0,
                    "checklist_count": (
                        len(task.checklist.required)
                        if task.checklist and task.checklist.required
                        else 0
                    ),
                },
            )

    def _emit_event(
        self,
        event_type: TrajectoryEventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit a trajectory event 📊.

        Args:
            event_type: Type of trajectory event.
            data: Optional event data payload.
        """
        if self._trajectory_writer:
            event = TrajectoryWriter.emit_event(event_type, data)
            self._trajectory_writer.write_event(event)

    def _emit_sse(
        self,
        event_type: InquiroEvent,
        task_id: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an SSE progress event via the event emitter 📡.

        Args:
            event_type: InquiroEvent type for the SSE event.
            task_id: Task identifier to tag the event.
            data: Optional event-specific payload.
        """
        if self._event_emitter:
            self._event_emitter.emit(event_type, task_id, data)

    async def _run_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None,
        search_enrichment: str | None = None,
    ) -> SearchRoundOutput:
        """Execute search phase with timeout 🔍.

        Args:
            task: Evaluation task.
            config: Discovery configuration.
            round_number: Current round number.
            focus_prompt: Optional focus guidance.
            search_enrichment: Optional evolution enrichment for search prompts.

        Returns:
            SearchRoundOutput from the search executor.
        """
        self._emit_event(
            TrajectoryEventType.ROUND_STARTED,
            {"phase": "search", "round_number": round_number},
        )

        # 🔧 Dynamic timeout: scale with section count and parallelism
        n_sections = 1
        if config.enable_parallel_search and task.query_strategy:
            sections = task.query_strategy.get("query_sections", [])
            n_sections = max(1, len(sections))
        search_timeout = config.timeout_per_round * math.ceil(
            n_sections / max(1, config.max_parallel_agents)
        ) + max(30, config.timeout_per_round // 5)  # Buffer for salvage

        # 🧬 Set enrichment for adapter access (scoped to this call)
        self._current_search_enrichment = search_enrichment

        try:
            result = await asyncio.wait_for(
                self.search_executor.execute_search(
                    task=task,
                    config=config,
                    round_number=round_number,
                    focus_prompt=focus_prompt,
                ),
                timeout=search_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "⏰ Search timeout at round %d (limit=%ds)",
                round_number,
                search_timeout,
            )
            result = SearchRoundOutput(cost_usd=0.0)
        except Exception as exc:
            logger.error(
                "❌ Search failed at round %d: %s",
                round_number,
                exc,
            )
            result = SearchRoundOutput(cost_usd=0.0)
        finally:
            # 🧹 Clear enrichment after search completes
            self._current_search_enrichment = None

        self._emit_event(
            TrajectoryEventType.SEARCH_COMPLETED,
            {
                "round_number": round_number,
                "evidence_count": len(result.evidence),
            },
        )
        return result

    def _run_cleaning(
        self,
        raw_evidence: list[Evidence],
    ) -> tuple[list[Evidence], CleaningStats]:
        """Run EvidencePipeline cleaning 🧹.

        Args:
            raw_evidence: Raw evidence from search.

        Returns:
            Tuple of (cleaned evidence list, cleaning stats).
        """
        cleaned, stats = self.evidence_pipeline.clean(raw_evidence)

        self._emit_event(
            TrajectoryEventType.CLEANING_COMPLETED,
            {
                "input_count": stats.input_count,
                "output_count": stats.output_count,
                "dedup_removed": stats.dedup_removed,
                "noise_removed": stats.noise_removed,
            },
        )
        return cleaned, stats

    async def _run_analysis(
        self,
        task: EvaluationTask,
        evidence: list[Evidence],
        config: DiscoveryConfig,
        round_number: int,
    ) -> AnalysisRoundOutput:
        """Execute analysis phase with error handling 🔬.

        Applies EvidenceCondenser before analysis when evidence exceeds
        the Tier 0 threshold, preventing LLM token-overflow errors.

        Args:
            task: Evaluation task.
            evidence: All accumulated evidence.
            config: Discovery configuration.
            round_number: Current round number.

        Returns:
            AnalysisRoundOutput from the analysis executor.
        """
        # 🗜️ Condense evidence to stay within LLM context limits
        condenser = self._condenser
        if condenser is None and len(evidence) > config.condenser_tier1_threshold:
            # Build tag quality map: merge caller overrides on top of defaults
            _tag_map = dict(CondenserConfig().tag_quality_map)
            if config.condenser_tag_quality_map:
                _tag_map.update(config.condenser_tag_quality_map)
            condenser = EvidenceCondenser(
                CondenserConfig(
                    tier1_threshold=config.condenser_tier1_threshold,
                    tier2_threshold=config.condenser_tier2_threshold,
                    doi_prefix_quality_map=config.condenser_doi_prefix_map,
                    doi_prefix_default_score=config.condenser_doi_prefix_default,
                    tag_quality_map=_tag_map,
                )
            )

        supplementary_context: str | None = None
        if condenser is not None:
            # Extract checklist item descriptions as plain strings
            checklist_strs: list[str] = []
            if task.checklist and hasattr(task.checklist, "required"):
                checklist_strs = [
                    item.description
                    for item in task.checklist.required
                    if hasattr(item, "description")
                ]
            condensed = condenser.condense(evidence, checklist_strs)
            evidence_for_analysis = condensed.evidence
            if condensed.meta.tier > 0:
                logger.info(
                    "🗜️ Condenser Tier %d applied at round %d: "
                    "%d → %d evidence items. %s",
                    condensed.meta.tier,
                    round_number,
                    condensed.meta.original_count,
                    condensed.meta.condensed_count,
                    condensed.meta.transparency_footer,
                )
            # 📝 Enrich Tier-2 summaries with LLM if summarizer available
            if (
                condensed.excluded_groups
                and self._group_summarizer
                and condensed.meta.group_summaries
            ):
                await self._enrich_group_summaries(condensed)

            # 📋 Build supplementary context from Tier-2 group summaries
            if condensed.meta.group_summaries:
                parts = [condensed.meta.transparency_footer, ""]
                for gs in condensed.meta.group_summaries:
                    parts.append(gs.summary_text)
                supplementary_context = "\n".join(parts)
        else:
            evidence_for_analysis = evidence

        try:
            result = await self.analysis_executor.execute_analysis(
                task=task,
                evidence=evidence_for_analysis,
                config=config,
                round_number=round_number,
                supplementary_context=supplementary_context,
            )
        except Exception as exc:
            logger.error(
                "❌ Analysis failed at round %d: %s",
                round_number,
                exc,
            )
            result = AnalysisRoundOutput()

        self._emit_event(
            TrajectoryEventType.ANALYSIS_COMPLETED,
            {
                "round_number": round_number,
                "claims_count": len(result.claims),
                "consensus": result.consensus_decision,
            },
        )
        return result

    async def _enrich_group_summaries(
        self,
        condensed: CondensedEvidence,
    ) -> None:
        """Enrich Tier 2 group summaries with LLM-generated text 📝.

        Calls the group_summarizer concurrently for each tag group.
        On failure, the original template text is preserved (graceful
        fallback).  Modifies condensed.meta.group_summaries in place.

        Args:
            condensed: CondensedEvidence with excluded_groups populated.
        """
        if not self._group_summarizer:
            return

        tasks: list[asyncio.Task[str]] = []
        gs_list = condensed.meta.group_summaries
        for gs in gs_list:
            items = condensed.excluded_groups.get(gs.tag, [])
            if not items:
                tasks.append(asyncio.ensure_future(asyncio.sleep(0, result="")))
                continue
            tasks.append(
                asyncio.ensure_future(
                    self._group_summarizer.summarize(
                        tag=gs.tag,
                        items=items,
                        included_count=gs.included_count,
                    )
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        enriched_count = 0
        for gs, result in zip(gs_list, results):
            if isinstance(result, str) and result:
                gs.summary_text = result
                enriched_count += 1
            elif isinstance(result, Exception):
                logger.warning(
                    "⚠️ Group summarizer failed for tag '%s': %s — "
                    "using template fallback.",
                    gs.tag,
                    result,
                )
            # else: empty string → keep template text

        if enriched_count > 0:
            logger.info(
                "📝 Enriched %d/%d group summaries with LLM.",
                enriched_count,
                len(gs_list),
            )

    def _build_pre_computed_coverage(
        self,
        analysis_output: AnalysisRoundOutput,
        task: EvaluationTask,
        checklist: list[str],
    ) -> CoverageResult | None:
        """Convert analysis agent coverage (IDs) to CoverageResult (descriptions) ✅.

        Maps checklist item IDs from the analysis agent's ChecklistCoverage
        to the description strings expected by GapAnalysis/CoverageResult.

        Args:
            analysis_output: Output from multi-model analysis.
            task: Evaluation task with checklist items.
            checklist: Checklist description strings (used as fallback).

        Returns:
            CoverageResult with descriptions, or None if no coverage data.
        """
        cov = analysis_output.checklist_coverage
        if cov is None:
            return None

        # 🗺️ Build ID → description mapping from task checklist
        id_to_desc: dict[str, str] = {}
        if task.checklist and task.checklist.required:
            for item in task.checklist.required:
                id_to_desc[item.id] = item.description

        if not id_to_desc:
            return None

        # 🔄 Map IDs to descriptions
        covered_descs: list[str] = []
        uncovered_descs: list[str] = []
        conflict_descs: list[str] = []

        for item_id in cov.required_covered:
            desc = id_to_desc.get(item_id)
            if desc:
                covered_descs.append(desc)
            else:
                logger.debug(
                    "⚠️ Coverage ID %s not found in checklist, skipping",
                    item_id,
                )

        for item_id in cov.required_missing:
            desc = id_to_desc.get(item_id)
            if desc:
                uncovered_descs.append(desc)
            else:
                logger.debug(
                    "⚠️ Coverage ID %s not found in checklist, skipping",
                    item_id,
                )

        # 🔍 Map conflict IDs to descriptions
        for item_id in analysis_output.coverage_conflicts:
            desc = id_to_desc.get(item_id)
            if desc:
                conflict_descs.append(desc)

        # 🛡️ Sanity check: all checklist items should be accounted for
        mapped_total = len(covered_descs) + len(uncovered_descs)
        if mapped_total == 0:
            logger.warning(
                "⚠️ Analysis coverage mapped 0 items — falling back to judge"
            )
            return None

        logger.info(
            "✅ Pre-computed coverage: %d covered, %d uncovered, "
            "%d conflicts (from %d checklist items)",
            len(covered_descs),
            len(uncovered_descs),
            len(conflict_descs),
            len(checklist),
        )

        return CoverageResult(
            covered=covered_descs,
            uncovered=uncovered_descs,
            conflict_signals=conflict_descs,
        )

    async def _run_gap_analysis(
        self,
        checklist_items: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Evidence],
        previous_coverage: float,
        round_number: int,
        config: DiscoveryConfig,
        cost_spent: float,
        pre_computed_coverage: Any = None,
    ) -> GapReport:
        """Execute gap analysis 🎯.

        Args:
            checklist_items: Checklist item descriptions.
            claims: All accumulated claims.
            evidence: All accumulated evidence.
            previous_coverage: Coverage from previous round.
            round_number: Current round number.
            config: Discovery configuration.
            cost_spent: Total cost spent so far.
            pre_computed_coverage: Optional pre-computed CoverageResult
                from analysis agent multi-model voting.

        Returns:
            GapReport with coverage and convergence assessment.
        """
        report = await self.gap_analysis.analyze(
            checklist=checklist_items,
            claims=claims,
            evidence=evidence,
            previous_coverage=previous_coverage,
            round_number=round_number,
            config=config,
            cost_spent=cost_spent,
            pre_computed_coverage=pre_computed_coverage,
        )

        self._emit_event(
            TrajectoryEventType.GAP_COMPLETED,
            {
                "round_number": round_number,
                "coverage_ratio": report.coverage_ratio,
                "converged": report.converged,
                "convergence_reason": report.convergence_reason,
            },
        )
        return report

    async def _generate_focus(
        self,
        gap_report: GapReport,
        config: DiscoveryConfig,
        round_number: int,
    ) -> FocusPrompt:
        """Generate focus prompt for the next round 🎯.

        Args:
            gap_report: Gap analysis result from current round.
            config: Discovery configuration.
            round_number: Round number just completed.

        Returns:
            FocusPrompt with targeted search guidance.
        """
        try:
            focus = await self.focus_generator.generate_focus(
                gap_report=gap_report,
                config=config,
                round_number=round_number,
            )
        except Exception as exc:
            logger.warning(
                "⚠️ Focus prompt generation failed at round %d: %s",
                round_number,
                exc,
            )
            # 🔧 Fallback: simple focus from uncovered items
            uncovered = gap_report.uncovered_items[: config.gap_focus_max_items]
            focus = FocusPrompt(
                prompt_text=(
                    f"Focus on: {', '.join(uncovered)}"
                    if uncovered
                    else "Broaden search scope"
                ),
                target_gaps=uncovered,
            )

        self._emit_event(
            TrajectoryEventType.FOCUS_PROMPT_GENERATED,
            {
                "round_number": round_number,
                "target_gaps": focus.target_gaps,
            },
        )
        return focus

    def _build_round_record(
        self,
        round_num: int,
        search_output: SearchRoundOutput,
        cleaning_stats: CleaningStats,
        analysis_output: AnalysisRoundOutput,
        gap_report: GapReport,
        round_cost: float,
        round_duration: float,
        state: _LoopState | None = None,
    ) -> DiscoveryRoundRecord:
        """Build a DiscoveryRoundRecord from in-memory round data 📊.

        Always constructs and returns a record regardless of whether
        trajectory writing is enabled.  Callers can pass the returned
        record to evolution hooks without needing a trajectory writer.

        Args:
            round_num: Round number (1-based).
            search_output: Search phase results.
            cleaning_stats: Cleaning phase statistics.
            analysis_output: Analysis phase results.
            gap_report: Gap analysis results.
            round_cost: Total cost for this round in USD.
            round_duration: Total wall-clock duration for this round.
            state: Loop state for tracking previously covered items.

        Returns:
            A fully-populated DiscoveryRoundRecord.
        """
        # 📡 Compute per-server effectiveness from raw evidence
        server_effectiveness = self._compute_server_effectiveness(
            search_output.evidence,
        )

        # 🔀 Compute query diversity score
        query_diversity = self._compute_query_diversity(
            search_output.queries_executed,
        )

        # 📊 Compute evidence quality distribution from raw evidence
        quality_dist = self._compute_quality_distribution(
            search_output.evidence,
        )

        # 🆕 Compute newly covered items (delta from previous round)
        prev_covered = state.prev_covered_items if state else set()
        current_covered = set(gap_report.covered_items)
        newly_covered = sorted(current_covered - prev_covered)
        if state is not None:
            state.prev_covered_items = current_covered

        return DiscoveryRoundRecord(
            round_number=round_num,
            search_phase=SearchPhaseRecord(
                total_raw_evidence=len(search_output.evidence),
                agent_trajectory_ref=search_output.agent_trajectory_ref,
                agent_trajectory_refs=(search_output.agent_trajectory_refs),
                duration_seconds=search_output.duration_seconds,
                server_effectiveness=server_effectiveness,
                query_diversity_score=query_diversity,
            ),
            cleaning_phase=CleaningPhaseRecord(
                input_count=cleaning_stats.input_count,
                output_count=cleaning_stats.output_count,
                dedup_removed=cleaning_stats.dedup_removed,
                noise_removed=cleaning_stats.noise_removed,
                tag_distribution=cleaning_stats.tag_distribution,
            ),
            analysis_phase=AnalysisPhaseRecord(
                model_results=[
                    ModelAnalysisRecord(
                        model_name=md.get("model", ""),
                        decision=md.get("decision", ""),
                        confidence=md.get("confidence", 0.0),
                        claims_count=md.get("claims_count", 0),
                        cost_usd=md.get("cost_usd", 0.0),
                    )
                    for md in analysis_output.model_decisions
                ],
                consensus=ConsensusRecord(
                    consensus_decision=analysis_output.consensus_decision,
                    consensus_ratio=analysis_output.consensus_ratio,
                    total_claims=len(analysis_output.claims),
                ),
                gaps_remaining=analysis_output.gaps_remaining,
                doubts_remaining=analysis_output.doubts_remaining,
                evidence_quality_distribution=quality_dist,
                duration_seconds=analysis_output.duration_seconds,
            ),
            gap_phase=GapPhaseRecord(
                coverage_ratio=gap_report.coverage_ratio,
                covered_items=gap_report.covered_items,
                uncovered_items=gap_report.uncovered_items,
                conflict_signals=gap_report.conflict_signals,
                newly_covered_items=newly_covered,
                convergence_reason=(
                    gap_report.convergence_reason if gap_report.converged else None
                ),
            ),
            round_cost_usd=round_cost,
            round_duration_seconds=round_duration,
        )

    @staticmethod
    def _compute_server_effectiveness(
        evidence: list[Evidence],
    ) -> dict[str, ServerStats]:
        """Compute per-MCP-server hit rates from raw evidence 📡.

        Groups evidence by source server and calculates query-level
        hit rates for each server.

        Args:
            evidence: Raw evidence items from the search phase.

        Returns:
            Mapping of server name to ServerStats.
        """
        if not evidence:
            return {}

        # Group by source server
        server_queries: dict[str, set[str]] = {}
        server_results: dict[str, int] = {}
        for ev in evidence:
            server = ev.source or "unknown"
            if server not in server_queries:
                server_queries[server] = set()
                server_results[server] = 0
            server_queries[server].add(ev.query)
            server_results[server] += 1

        stats: dict[str, ServerStats] = {}
        for server, queries in server_queries.items():
            n_queries = len(queries)
            n_results = server_results[server]
            # All queries in this group returned results (evidence exists)
            stats[server] = ServerStats(
                queries_sent=n_queries,
                results_returned=n_results,
                hit_rate=1.0 if n_queries > 0 else 0.0,
            )
        return stats

    @staticmethod
    def _compute_query_diversity(
        queries: list[str],
    ) -> float:
        """Compute query diversity score using normalized unique ratio 🔀.

        Returns a 0-1 score where 1.0 means all queries are unique
        and values closer to 0 indicate many duplicates.

        Args:
            queries: List of query strings executed in this round.

        Returns:
            Diversity score between 0.0 and 1.0.
        """
        if not queries:
            return 0.0
        unique = len(set(queries))
        return unique / len(queries)

    @staticmethod
    def _compute_quality_distribution(
        evidence: list[Evidence],
    ) -> dict[str, int]:
        """Compute evidence quality tier distribution 📊.

        Counts how many evidence items fall into each quality_label
        bucket.  Items without a quality_label are counted under
        "unknown".

        Args:
            evidence: Raw evidence items from the search phase.

        Returns:
            Mapping of quality label to count.
        """
        dist: dict[str, int] = {}
        for ev in evidence:
            label = ev.quality_label or "unknown"
            dist[label] = dist.get(label, 0) + 1
        return dist

    def _finalize_trajectory(
        self,
        total_rounds: int,
        final_coverage: float,
        total_cost: float,
        total_evidence: int,
        total_claims: int,
        total_duration: float,
        termination_reason: str,
    ) -> None:
        """Write summary and finalize trajectory 📊.

        Args:
            total_rounds: Number of rounds executed.
            final_coverage: Final coverage ratio.
            total_cost: Total cost in USD.
            total_evidence: Total evidence items.
            total_claims: Total consensus claims.
            total_duration: Total wall-clock time.
            termination_reason: Why the loop stopped.
        """
        if not self._trajectory_writer:
            return

        summary = DiscoverySummary(
            total_rounds=total_rounds,
            final_coverage=final_coverage,
            total_cost_usd=total_cost,
            total_evidence=total_evidence,
            total_claims=total_claims,
            total_duration_seconds=total_duration,
            termination_reason=termination_reason,
        )
        self._trajectory_writer.write_summary(summary)

        status = "completed" if termination_reason != "error" else "failed"
        self._trajectory_writer.finalize(
            status=status,
            termination_reason=termination_reason,
        )

    @staticmethod
    def _to_cleaned_evidence(evidence: Evidence) -> CleanedEvidence:
        """Convert Evidence to CleanedEvidence model 🔄.

        Args:
            evidence: Standard Evidence model instance.

        Returns:
            CleanedEvidence model with mapped fields.
        """
        return CleanedEvidence(
            id=evidence.id,
            summary=evidence.summary,
            url=evidence.url,
            tag=(
                EvidenceTag(evidence.evidence_tag)
                if evidence.evidence_tag
                else EvidenceTag.OTHER
            ),
            source_query=evidence.query,
            mcp_server=evidence.source,
            doi=evidence.doi,
            clinical_trial_id=evidence.clinical_trial_id,
            quality_label=getattr(evidence, "quality_label", None),
        )
