"""Inquiro DiscoverySynthesisExp -- Multi-LLM synthesis for DISCOVERY pipeline 🧬.

Runs evidence synthesis across multiple LLM models in parallel, then
aggregates their findings using consensus voting.  This is the final
stage of the DISCOVERY pipeline, invoked after all search-analysis
rounds have converged.

Architecture position in DISCOVERY mode:
    DiscoveryLoop
        -> SearchExp -> SearchAgent (search, MCP tools)
        -> EvidencePipeline (deterministic cleaning, zero LLM)
        -> AnalysisExp (3 LLM parallel analysis + voting)
        -> GapAnalysis (LLM coverage judgment + deterministic convergence)
        -> DiscoverySynthesisExp (3 LLM synthesis + voting)   <-- this module

Inheritance chain: DiscoverySynthesisExp -> InquiroBaseExp -> BaseExp (EvoMaster)

Key design:
    - Takes ALL accumulated evidence and claims as input.
    - Calls N LLM models in parallel for independent synthesis.
    - Each model receives the same evidence set, claims, and checklist.
    - Aggregates results via majority vote consensus.
    - No MCP tools -- pure reasoning passes.
    - Differs from AnalysisExp: synthesis prompts, richer input context,
      final EvaluationResult construction with full evidence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from inquiro.core.aggregation import AggregationEngine
from inquiro.core.llm_utils import extract_cost_from_response, extract_json_from_text
from inquiro.core.trajectory.models import (
    ModelAnalysisRecord,
    SynthesisRecord,
)
from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.infrastructure.cancellation import CancelledError
from inquiro.infrastructure.event_emitter import InquiroEvent
from inquiro.prompts.loader import PromptLoader
from inquiro.prompts.section_builder import PromptSectionBuilder

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from inquiro.core.types import (
        DiscoveryConfig,
        Evidence,
        EvaluationResult,
        EvaluationTask,
        ReasoningClaim,
    )
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter
    from inquiro.infrastructure.llm_pool import LLMProviderPool

logger = logging.getLogger(__name__)

# 📝 Module-level PromptLoader instance (shared across all instances)
_prompt_loader = PromptLoader()

# 🔢 Minimum successful models needed for valid consensus
_MIN_MODELS_FOR_CONSENSUS = 2


# ============================================================================
# 📊 Result Models
# ============================================================================


class ModelSynthesisOutput(BaseModel):
    """Output from a single LLM model synthesis 🧠.

    Attributes:
        model_name: LLM model identifier.
        decision: Model's decision (positive/cautious/negative).
        confidence: Model's confidence score.
        claims: List of structured reasoning claims.
        summary_text: Narrative summary text.
        cost_usd: Estimated cost for this model's synthesis.
        raw_response: Raw LLM response text for audit trail.
    """

    model_name: str = Field(description="LLM model identifier")
    decision: str = Field(
        default="cautious",
        description="Model decision: positive / cautious / negative",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model confidence score",
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured reasoning claims from this model",
    )
    summary_text: str = Field(
        default="",
        description="Narrative summary text from this model",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD for this model",
    )
    raw_response: str = Field(
        default="",
        description="Raw LLM response text for audit",
    )


class SynthesisResult(BaseModel):
    """Result of multi-LLM discovery synthesis 📊.

    Attributes:
        evaluation_result: Final structured EvaluationResult.
        model_results: Per-model synthesis outputs.
        consensus_decision: Majority-voted decision string.
        consensus_ratio: Fraction of models agreeing with majority.
        total_claims: Total merged claims across models.
        cost_usd: Total synthesis cost.
        duration_seconds: Wall-clock time for synthesis.
        synthesis_record: Trajectory record for discovery trajectory.
    """

    evaluation_result: Any = Field(
        description="Final EvaluationResult from synthesis",
    )
    model_results: list[ModelSynthesisOutput] = Field(
        default_factory=list,
        description="Per-model synthesis outputs",
    )
    consensus_decision: str = Field(
        default="cautious",
        description="Consensus decision: positive / cautious / negative",
    )
    consensus_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of models agreeing with majority",
    )
    total_claims: int = Field(
        default=0,
        description="Total merged claims across models",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total synthesis cost in USD",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock synthesis duration in seconds",
    )
    synthesis_record: SynthesisRecord = Field(
        default_factory=SynthesisRecord,
        description="Trajectory record for the synthesis phase",
    )


# ============================================================================
# 🧬 DiscoverySynthesisExp
# ============================================================================


class DiscoverySynthesisExp(InquiroBaseExp):
    """Multi-LLM synthesis experiment for DISCOVERY pipeline 🧬.

    Runs N (default 3) LLM models in parallel to synthesize
    all accumulated evidence into a final assessment.  Uses
    consensus voting to determine the final decision.

    Pipeline:
    1. Render synthesis prompts with all evidence + claims
    2. Fan out to N models via asyncio.gather()
    3. Parse structured responses (claims, decision, confidence)
    4. Determine consensus decision by majority vote
    5. Merge and deduplicate claims across models
    6. Build final EvaluationResult
    7. Return SynthesisResult with trajectory record

    Inherits: DiscoverySynthesisExp -> InquiroBaseExp -> BaseExp

    Attributes:
        task: The evaluation task with rules and checklist.
        llm_pool: Provider pool for obtaining model-specific LLM instances.
        aggregation_engine: Stateless engine for evidence merging.
    """

    def __init__(
        self,
        task: EvaluationTask,
        llm: BaseLLM,
        llm_pool: LLMProviderPool | None = None,
        quality_gate_config: Any = None,
        cost_tracker: CostTracker | None = None,
        event_emitter: EventEmitter | None = None,
        cancellation_token: CancellationToken | None = None,
        evolution_enrichment: str | None = None,
    ) -> None:
        """Initialize DiscoverySynthesisExp 🔧.

        Args:
            task: Evaluation task definition containing topic, rules,
                checklist, output_schema, and agent configuration.
            llm: Default LLM instance (used as fallback when llm_pool
                is unavailable for a given model key).
            llm_pool: Provider pool for obtaining model-specific LLM
                instances. When None, all models use the default llm.
            quality_gate_config: Quality validation configuration.
                Falls back to task.quality_gate if None.
            cost_tracker: Cost tracking instance for budget enforcement.
                Creates a no-op tracker if None.
            event_emitter: SSE event emitter for progress updates.
                Creates a no-op emitter if None.
            cancellation_token: Cancellation signal for cooperative stop.
                Creates a fresh token if None.
            evolution_enrichment: Optional markdown text from the
                evolution system to post-append to the system prompt.
                When None, prompt rendering is unchanged.
        """
        from inquiro.infrastructure.cancellation import CancellationToken as CT
        from inquiro.infrastructure.cost_tracker import CostTracker as CTr
        from inquiro.infrastructure.event_emitter import EventEmitter as EE

        qg_config = quality_gate_config or task.quality_gate

        self._init_base(
            task=task,
            llm=llm,
            quality_gate_config=qg_config,
            cost_tracker=cost_tracker
            if cost_tracker is not None
            else CTr(
                max_per_task=10.0,
                max_total=100.0,
            ),
            event_emitter=event_emitter if event_emitter is not None else EE(),
            cancellation_token=(
                cancellation_token if cancellation_token is not None else CT()
            ),
        )

        self.llm_pool = llm_pool
        self.aggregation_engine = AggregationEngine()

        # 🧬 Evolution enrichment for system prompt injection
        self._evolution_enrichment = evolution_enrichment

    @property
    def exp_name(self) -> str:
        """Return experiment name 🏷️.

        Returns:
            "DiscoverySynthesis" as the experiment type name.
        """
        return "DiscoverySynthesis"

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    async def run_synthesis(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        all_evidence: list[Evidence],
        all_claims: list[ReasoningClaim],
        coverage_ratio: float,
        round_summaries: list[dict[str, Any]],
    ) -> SynthesisResult:
        """Run multi-LLM synthesis 🧬.

        Orchestrates the multi-phase synthesis pipeline:
            1. Render synthesis prompts with all evidence + claims.
            2. Fan out to N models in parallel via asyncio.gather().
            3. Parse each model's response (JSON extraction).
            4. Determine consensus decision by majority vote.
            5. Merge and deduplicate claims across models.
            6. Build final EvaluationResult with all evidence.
            7. Return SynthesisResult with trajectory record.

        Args:
            task: The evaluation task with rules and checklist.
            config: Discovery configuration with analysis_models list.
            all_evidence: All evidence items accumulated across rounds.
            all_claims: All claims from analysis rounds.
            coverage_ratio: Final coverage ratio from gap analysis.
            round_summaries: Summary dicts for each discovery round.

        Returns:
            SynthesisResult with consensus decision, merged claims,
            and full EvaluationResult.

        Raises:
            CancelledError: If the cancellation token is signalled.
            ValueError: If config.analysis_models is empty.
        """
        synthesis_models = config.analysis_models
        if not synthesis_models:
            raise ValueError("config.analysis_models must contain at least one model")

        task_id = task.task_id
        start_time = time.monotonic()

        # 📡 Emit synthesis start event
        self.event_emitter.emit(
            InquiroEvent.TASK_STARTED,
            task_id,
            {
                "phase": "discovery_synthesis",
                "models": synthesis_models,
                "evidence_count": len(all_evidence),
                "claims_count": len(all_claims),
                "coverage_ratio": coverage_ratio,
            },
        )

        logger.info(
            "🧬 Starting discovery synthesis for task %s "
            "with %d models, %d evidence items, %d claims",
            task_id,
            len(synthesis_models),
            len(all_evidence),
            len(all_claims),
        )

        # 🛑 Check cancellation before launching
        if self.cancellation_token.is_cancelled:
            raise CancelledError(
                self.cancellation_token.reason or "Cancelled before synthesis"
            )

        # 📝 Step 1: Render prompts
        system_prompt = self._render_system_prompt(task)
        user_prompt = self._render_user_prompt(
            task,
            all_evidence,
            all_claims,
            coverage_ratio,
            round_summaries,
        )

        # 🚀 Step 2: Fan out to models in parallel
        synthesis_coros = [
            self._run_single_synthesis_model(
                model_key=model_key,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            for model_key in synthesis_models
        ]

        raw_outcomes = await asyncio.gather(
            *synthesis_coros,
            return_exceptions=True,
        )

        # 📊 Step 3: Collect results
        successful: list[ModelSynthesisOutput] = []
        failed_models: list[str] = []

        for model_key, outcome in zip(synthesis_models, raw_outcomes):
            if isinstance(outcome, BaseException):
                logger.error(
                    "❌ Synthesis model '%s' failed: %s",
                    model_key,
                    outcome,
                )
                failed_models.append(model_key)
            else:
                successful.append(outcome)
                logger.info(
                    "✅ Synthesis model '%s' completed: "
                    "decision=%s confidence=%.2f cost=$%.4f",
                    model_key,
                    outcome.decision,
                    outcome.confidence,
                    outcome.cost_usd,
                )

        # ⏱️ Compute duration
        duration = time.monotonic() - start_time

        # 🔄 Handle total failure
        if not successful:
            logger.error(
                "❌ All synthesis models failed for task %s. Failed: %s",
                task_id,
                failed_models,
            )
            self.event_emitter.emit(
                InquiroEvent.TASK_FAILED,
                task_id,
                {
                    "phase": "discovery_synthesis",
                    "error": "All synthesis models failed",
                    "failed_models": failed_models,
                },
            )
            return self._build_error_result(
                task,
                all_evidence,
                duration,
            )

        # ⚖️ Step 4: Determine consensus decision
        consensus_decision, consensus_ratio = self._compute_consensus(successful)

        # 🧠 Step 5: Merge and deduplicate claims
        merged_claims = self._merge_claims(successful)

        # 📊 Step 6: Compute aggregate confidence
        consensus_confidence = self._compute_consensus_confidence(
            successful,
            consensus_decision,
        )

        # 💰 Compute total cost
        total_cost = sum(m.cost_usd for m in successful)

        # 🏗️ Step 7: Build EvaluationResult
        evaluation_result = self._build_evaluation_result(
            task=task,
            all_evidence=all_evidence,
            merged_claims=merged_claims,
            decision=consensus_decision,
            confidence=consensus_confidence,
            coverage_ratio=coverage_ratio,
            total_cost=total_cost,
        )

        # 📊 Build trajectory record
        synthesis_record = self._build_synthesis_record(
            successful=successful,
            consensus_decision=consensus_decision,
            consensus_ratio=consensus_ratio,
            total_cost=total_cost,
            duration=duration,
        )

        # 📡 Emit synthesis completion event
        self.event_emitter.emit(
            InquiroEvent.TASK_COMPLETED,
            task_id,
            {
                "phase": "discovery_synthesis",
                "decision": consensus_decision,
                "confidence": consensus_confidence,
                "consensus_ratio": consensus_ratio,
                "successful_models": [m.model_name for m in successful],
                "failed_models": failed_models,
                "total_claims": len(merged_claims),
                "cost_usd": total_cost,
            },
        )

        logger.info(
            "📊 Discovery synthesis complete for task %s: "
            "decision=%s confidence=%.2f consensus=%.2f "
            "claims=%d cost=$%.4f duration=%.1fs",
            task_id,
            consensus_decision,
            consensus_confidence,
            consensus_ratio,
            len(merged_claims),
            total_cost,
            duration,
        )

        return SynthesisResult(
            evaluation_result=evaluation_result,
            model_results=successful,
            consensus_decision=consensus_decision,
            consensus_ratio=consensus_ratio,
            total_claims=len(merged_claims),
            cost_usd=total_cost,
            duration_seconds=duration,
            synthesis_record=synthesis_record,
        )

    # ====================================================================
    # 🧠 Single model synthesis
    # ====================================================================

    async def _run_single_synthesis_model(
        self,
        model_key: str,
        task: EvaluationTask,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelSynthesisOutput:
        """Run synthesis with a single LLM model 🧠.

        Creates a Dialog with system + user prompt and calls the LLM
        for a single reasoning pass. No MCP tools -- pure reasoning.

        Args:
            model_key: LLM model identifier / provider key.
            task: Evaluation task (for parsing context).
            system_prompt: Pre-rendered system prompt.
            user_prompt: Pre-rendered user prompt with evidence.

        Returns:
            ModelSynthesisOutput for this model.
        """
        from evomaster.utils.types import Dialog, SystemMessage, UserMessage

        logger.info(
            "🧠 Launching synthesis model '%s' for task %s",
            model_key,
            task.task_id,
        )

        # 🛑 Check cancellation
        if self.cancellation_token.is_cancelled:
            raise CancelledError(
                self.cancellation_token.reason or "Cancelled before model synthesis"
            )

        # 🔧 Get LLM instance for this model
        llm = self._get_llm_for_model(model_key)

        # 🏗️ Create Dialog (no tools -- pure reasoning)
        dialog = Dialog(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            tools=[],
        )

        # 🚀 Call LLM in thread pool (query() is blocking)
        response = await asyncio.to_thread(llm.query, dialog)

        # 📊 Parse response
        raw_content = getattr(response, "content", "") or ""
        parsed = extract_json_from_text(raw_content)

        # 🎯 Extract decision
        decision = parsed.get("decision", "cautious")
        if decision not in ("positive", "cautious", "negative"):
            decision = "cautious"

        # 📊 Extract confidence
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # 🧠 Extract claims
        claims = self._extract_claims(parsed)

        # 📝 Extract summary
        summary = parsed.get("summary", "")

        # 💰 Estimate cost
        cost = extract_cost_from_response(response)

        return ModelSynthesisOutput(
            model_name=model_key,
            decision=decision,
            confidence=confidence,
            claims=claims,
            summary_text=summary,
            cost_usd=cost,
            raw_response=raw_content,
        )

    def _get_llm_for_model(self, model_key: str) -> Any:
        """Retrieve the LLM instance for a given model key 🔧.

        Attempts to use the llm_pool first. Falls back to the default
        llm if the pool is unavailable or the key is not found.

        Args:
            model_key: LLM model identifier / provider key.

        Returns:
            LLM instance suitable for calling query().
        """
        if self.llm_pool is not None:
            try:
                return self.llm_pool.get_llm(model_key)
            except Exception:
                logger.warning(
                    "⚠️ Failed to get LLM for model '%s' from pool, using default LLM",
                    model_key,
                )
        return self.llm

    # ====================================================================
    # 📝 Prompt rendering
    # ====================================================================

    def _render_system_prompt(self, task: EvaluationTask) -> str:
        """Render the synthesis system prompt from template 📝.

        Loads the discovery_synthesis_system template and fills in
        rules, checklist, and output schema from the task definition.

        Args:
            task: Evaluation task with rules, checklist, and schema.

        Returns:
            Rendered system prompt string.
        """
        # 📋 Format checklist
        checklist_md = PromptSectionBuilder.format_checklist(task.checklist)

        # 📊 Format output schema
        schema_str = PromptSectionBuilder.format_output_schema(
            task.output_schema,
        )

        # 📝 Rules
        rules = task.rules or "No specific evaluation rules provided."

        rendered = _prompt_loader.render(
            "discovery_synthesis_system",
            rules=rules,
            checklist=checklist_md,
            output_schema=schema_str,
        )

        # 🧬 Append evolution enrichment (if available)
        if self._evolution_enrichment:
            rendered += "\n\n" + self._evolution_enrichment

        return rendered

    def _render_user_prompt(
        self,
        task: EvaluationTask,
        all_evidence: list[Evidence],
        all_claims: list[ReasoningClaim],
        coverage_ratio: float,
        round_summaries: list[dict[str, Any]],
    ) -> str:
        """Render the synthesis user prompt with all evidence 📝.

        Loads the discovery_synthesis_user template and fills in
        evidence, claims, coverage info, and round context.

        Args:
            task: Evaluation task with rules and checklist.
            all_evidence: All accumulated evidence items.
            all_claims: All claims from analysis rounds.
            coverage_ratio: Final coverage ratio.
            round_summaries: Summary of each discovery round.

        Returns:
            Rendered user prompt string.
        """
        # 📋 Format evidence list
        evidence_list = self._format_evidence_list(all_evidence)

        # 🧠 Format claims summary
        claims_summary = self._format_claims_summary(all_claims)

        # 📊 Format coverage info
        coverage_info = self._format_coverage_info(coverage_ratio)

        # 🔄 Format round context
        round_context = self._format_round_context(round_summaries)

        # 📋 Format checklist
        checklist_md = PromptSectionBuilder.format_checklist(task.checklist)

        # 📝 Rules
        rules = task.rules or "No specific evaluation rules provided."

        return _prompt_loader.render(
            "discovery_synthesis_user",
            evidence_list=evidence_list,
            claims_summary=claims_summary,
            coverage_info=coverage_info,
            round_context=round_context,
            checklist=checklist_md,
            rules=rules,
        )

    @staticmethod
    def _format_evidence_list(
        all_evidence: list[Evidence],
    ) -> str:
        """Format evidence items into a numbered Markdown list 📋.

        Produces a human-readable block suitable for injection
        into the user prompt.

        Args:
            all_evidence: All accumulated evidence items.

        Returns:
            Formatted evidence string, or fallback message if empty.
        """
        if not all_evidence:
            return "No evidence items available."

        lines: list[str] = []
        for ev in all_evidence:
            lines.append(f"### [{ev.id}] (Source: {ev.source})")
            if ev.summary:
                lines.append(f"- **Summary**: {ev.summary}")
            if ev.url:
                lines.append(f"- **URL**: {ev.url}")
            if ev.query:
                lines.append(f"- **Query**: {ev.query}")
            if ev.quality_label:
                lines.append(f"- **Quality**: {ev.quality_label}")
            if ev.round_number is not None:
                lines.append(f"- **Round**: {ev.round_number}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_claims_summary(
        all_claims: list[ReasoningClaim],
    ) -> str:
        """Format pre-existing claims for prompt injection 🧠.

        Args:
            all_claims: All claims from analysis rounds.

        Returns:
            Formatted claims summary string.
        """
        if not all_claims:
            return "No prior claims available."

        lines: list[str] = []
        for i, claim in enumerate(all_claims, 1):
            evidence_refs = ", ".join(claim.evidence_ids)
            lines.append(
                f"{i}. [{claim.strength.value}] {claim.claim} "
                f"(evidence: {evidence_refs})"
            )

        return "\n".join(lines)

    @staticmethod
    def _format_coverage_info(coverage_ratio: float) -> str:
        """Format coverage ratio information 📊.

        Args:
            coverage_ratio: Coverage ratio from gap analysis.

        Returns:
            Formatted coverage info string.
        """
        pct = coverage_ratio * 100
        if coverage_ratio >= 0.80:
            status = "Good coverage achieved"
        elif coverage_ratio >= 0.60:
            status = "Moderate coverage with some gaps"
        else:
            status = "Limited coverage with significant gaps"

        return (
            f"## Coverage Status\n\n"
            f"Current checklist coverage: **{pct:.0f}%** "
            f"({status}). Consider this when calibrating your "
            f"confidence score."
        )

    @staticmethod
    def _format_round_context(
        round_summaries: list[dict[str, Any]],
    ) -> str:
        """Format round summaries for prompt injection 🔄.

        Args:
            round_summaries: Summary dicts for each discovery round.

        Returns:
            Formatted round context string.
        """
        if not round_summaries:
            return "No round summaries available."

        lines: list[str] = []
        for summary in round_summaries:
            round_num = summary.get("round_number", "?")
            evidence_count = summary.get("evidence_count", 0)
            coverage = summary.get("coverage_ratio", 0.0)
            lines.append(
                f"- **Round {round_num}**: "
                f"{evidence_count} evidence items, "
                f"coverage {coverage:.0%}"
            )

        return "\n".join(lines)

    # ====================================================================
    # 📊 Response parsing
    # ====================================================================

    # ====================================================================
    # 🔗 Backward-compatible static method delegates (for tests and callers)
    # ====================================================================

    @staticmethod
    def _extract_json_from_text(text: str) -> dict[str, Any]:
        """Delegate to shared llm_utils.extract_json_from_text 🔍.

        Args:
            text: Raw text from LLM response.

        Returns:
            Parsed JSON dict. Empty dict on parse failure.
        """
        return extract_json_from_text(text)

    @staticmethod
    def _extract_cost_from_response(
        response: Any,
        input_cost_per_token: float = 3.0 / 1_000_000,
        output_cost_per_token: float = 15.0 / 1_000_000,
    ) -> float:
        """Delegate to shared llm_utils.extract_cost_from_response 💰.

        Args:
            response: AssistantMessage with meta dict.
            input_cost_per_token: Cost per input token in USD.
            output_cost_per_token: Cost per output token in USD.

        Returns:
            Estimated cost in USD.
        """
        return extract_cost_from_response(
            response, input_cost_per_token, output_cost_per_token
        )

    @staticmethod
    def _extract_claims(parsed: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract structured claims from parsed JSON response 🧠.

        Handles multiple field names that LLMs might use:
        'claims', 'reasoning', or both.

        Args:
            parsed: Parsed JSON response dict.

        Returns:
            List of claim dicts with claim/evidence_ids/strength keys.
        """
        # 🔍 Try 'claims' first, then 'reasoning'
        raw_claims = parsed.get("claims", parsed.get("reasoning", []))
        if not isinstance(raw_claims, list):
            return []

        claims: list[dict[str, Any]] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            claim_text = item.get("claim", "")
            if not claim_text:
                continue
            claims.append(
                {
                    "claim": claim_text,
                    "evidence_ids": item.get("evidence_ids", []),
                    "strength": item.get("strength", "moderate"),
                }
            )

        return claims

    # ====================================================================
    # ⚖️ Consensus voting
    # ====================================================================

    @staticmethod
    def _compute_consensus(
        successful: list[ModelSynthesisOutput],
    ) -> tuple[str, float]:
        """Compute consensus decision by majority vote ⚖️.

        If there is a tie (e.g., 3 different decisions), "cautious"
        is used as the tie-breaking default.

        Args:
            successful: List of successful model outputs.

        Returns:
            Tuple of (consensus_decision, consensus_ratio).
        """
        if not successful:
            return "cautious", 0.0

        # ⚠️ M6: Warn when below minimum models for reliable consensus
        if len(successful) < _MIN_MODELS_FOR_CONSENSUS:
            logger.warning(
                "⚠️ Only %d model(s) succeeded — below minimum %d "
                "for reliable consensus; result may be unreliable",
                len(successful),
                _MIN_MODELS_FOR_CONSENSUS,
            )

        # 🗳️ Count decisions
        decision_counts: Counter[str] = Counter()
        for model_output in successful:
            decision_counts[model_output.decision] += 1

        # 🏆 Find majority decision
        most_common = decision_counts.most_common()
        top_decision = most_common[0][0]
        top_count = most_common[0][1]

        # 🔄 Tie-breaking: if multiple decisions have same count, use cautious
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # ⚖️ True tie -- prefer cautious as safe default
            if "cautious" in decision_counts:
                top_decision = "cautious"
                top_count = decision_counts["cautious"]
            # 🔄 If cautious not in tie, keep the first most_common

        consensus_ratio = top_count / len(successful)
        return top_decision, consensus_ratio

    @staticmethod
    def _compute_consensus_confidence(
        successful: list[ModelSynthesisOutput],
        consensus_decision: str,
    ) -> float:
        """Compute weighted consensus confidence 📊.

        Averages confidence scores from models that agree with
        the consensus decision.  If no models agree (should not
        happen), falls back to the overall average.

        Args:
            successful: List of successful model outputs.
            consensus_decision: The majority-voted decision.

        Returns:
            Consensus confidence value clamped to [0.0, 1.0].
        """
        if not successful:
            return 0.0

        # 📊 Average confidence from agreeing models
        agreeing = [
            m.confidence for m in successful if m.decision == consensus_decision
        ]

        if agreeing:
            confidence = sum(agreeing) / len(agreeing)
        else:
            # 🔄 Fallback: average all models
            confidence = sum(m.confidence for m in successful) / len(successful)

        return max(0.0, min(1.0, confidence))

    # ====================================================================
    # 🧠 Claim merging
    # ====================================================================

    @staticmethod
    def _merge_claims(
        successful: list[ModelSynthesisOutput],
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate claims across models 🧠.

        Uses simple substring deduplication: if claim A is a
        substring of claim B, only B is kept.

        Args:
            successful: List of successful model outputs.

        Returns:
            Deduplicated list of claim dicts.
        """
        all_claims: list[dict[str, Any]] = []
        for model_output in successful:
            all_claims.extend(model_output.claims)

        if not all_claims:
            return []

        # 🧹 Exact + substring dedup: remove exact duplicates first,
        # then drop any claim A that is a proper substring of claim B.
        seen_texts: set[str] = set()
        unique_claims: list[dict[str, Any]] = []
        for item in all_claims:
            claim_text = item.get("claim", "")
            if not claim_text or claim_text in seen_texts:
                continue
            seen_texts.add(claim_text)
            unique_claims.append(item)

        deduped: list[dict[str, Any]] = []
        for i, item in enumerate(unique_claims):
            claim_text = item.get("claim", "")
            is_substring = any(
                claim_text in other.get("claim", "")
                and claim_text != other.get("claim", "")
                for j, other in enumerate(unique_claims)
                if i != j
            )
            if not is_substring:
                deduped.append(item)

        logger.debug(
            "🧠 Merged synthesis claims: %d raw -> %d deduped",
            len(all_claims),
            len(deduped),
        )
        return deduped

    # ====================================================================
    # 🏗️ Result construction
    # ====================================================================

    def _build_evaluation_result(
        self,
        task: EvaluationTask,
        all_evidence: list[Evidence],
        merged_claims: list[dict[str, Any]],
        decision: str,
        confidence: float,
        coverage_ratio: float,
        total_cost: float,
    ) -> EvaluationResult:
        """Build the final EvaluationResult from synthesis 🏗️.

        Args:
            task: Original evaluation task.
            all_evidence: All accumulated evidence items.
            merged_claims: Deduplicated claims from all models.
            decision: Consensus decision string.
            confidence: Consensus confidence score.
            coverage_ratio: Final coverage ratio.
            total_cost: Total synthesis cost.

        Returns:
            Fully constructed EvaluationResult.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            Decision,
            EvaluationResult,
            EvidenceStrength,
            ReasoningClaim,
        )

        # 🎯 Parse decision enum
        try:
            decision_enum = Decision(decision)
        except ValueError:
            decision_enum = Decision.CAUTIOUS

        # 🧠 Convert claim dicts to ReasoningClaim objects
        reasoning: list[ReasoningClaim] = []
        for claim_dict in merged_claims:
            strength_raw = claim_dict.get("strength", "moderate")
            try:
                strength = EvidenceStrength(strength_raw)
            except ValueError:
                strength = EvidenceStrength.MODERATE

            reasoning.append(
                ReasoningClaim(
                    claim=claim_dict.get("claim", ""),
                    evidence_ids=claim_dict.get("evidence_ids", []),
                    strength=strength,
                )
            )

        # 📊 Build checklist coverage from evidence
        checklist_coverage = ChecklistCoverage()

        return EvaluationResult(
            task_id=f"{task.task_id}::synthesis",
            decision=decision_enum,
            confidence=confidence,
            reasoning=reasoning,
            evidence_index=all_evidence,
            search_rounds=0,
            round_logs=[],
            checklist_coverage=checklist_coverage,
            gaps_remaining=[],
            doubts_remaining=[],
            cost=total_cost,
            pipeline_mode="discovery",
            discovery_coverage=coverage_ratio,
        )

    def _build_synthesis_record(
        self,
        successful: list[ModelSynthesisOutput],
        consensus_decision: str,
        consensus_ratio: float,
        total_cost: float,
        duration: float,
    ) -> SynthesisRecord:
        """Build trajectory SynthesisRecord 📊.

        Args:
            successful: Successful model outputs.
            consensus_decision: Final consensus decision.
            consensus_ratio: Model agreement ratio.
            total_cost: Total synthesis cost.
            duration: Synthesis duration in seconds.

        Returns:
            SynthesisRecord for trajectory recording.
        """
        model_records = [
            ModelAnalysisRecord(
                model_name=m.model_name,
                claims_count=len(m.claims),
                decision=m.decision,
                confidence=m.confidence,
                cost_usd=m.cost_usd,
            )
            for m in successful
        ]

        return SynthesisRecord(
            model_results=model_records,
            consensus_decision=consensus_decision,
            consensus_ratio=consensus_ratio,
            cost_usd=total_cost,
            duration_seconds=duration,
        )

    def _build_error_result(
        self,
        task: EvaluationTask,
        all_evidence: list[Evidence],
        duration: float,
    ) -> SynthesisResult:
        """Build error result when all models fail 🔄.

        Args:
            task: Original evaluation task.
            all_evidence: All evidence items (preserved in result).
            duration: Elapsed time in seconds.

        Returns:
            SynthesisResult with error state.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            Decision,
            EvaluationResult,
        )

        evaluation_result = EvaluationResult(
            task_id=f"{task.task_id}::synthesis::error",
            decision=Decision.CAUTIOUS,
            confidence=0.0,
            reasoning=[],
            evidence_index=all_evidence,
            search_rounds=0,
            round_logs=[],
            checklist_coverage=ChecklistCoverage(),
            gaps_remaining=["All synthesis models failed"],
            doubts_remaining=[],
            cost=0.0,
            pipeline_mode="discovery",
        )

        return SynthesisResult(
            evaluation_result=evaluation_result,
            model_results=[],
            consensus_decision="error",
            consensus_ratio=0.0,
            total_claims=0,
            cost_usd=0.0,
            duration_seconds=duration,
            synthesis_record=SynthesisRecord(),
        )

    # ====================================================================
    # 🔄 BaseExp interface
    # ====================================================================

    def run_sync(self) -> Any:
        """Synchronous run -- not used for DiscoverySynthesisExp 🔄.

        DiscoverySynthesisExp uses the async ``run_synthesis()`` method.
        This method exists to satisfy the InquiroBaseExp interface.

        Returns:
            Empty dict (DiscoverySynthesisExp is async-only).
        """
        logger.warning(
            "⚠️ DiscoverySynthesisExp.run_sync() called -- use run_synthesis() instead"
        )
        return {}
