"""Inquiro AnalysisExp -- Three-model parallel analysis experiment 🔬.

Runs evidence analysis across multiple LLM models in parallel, then
aggregates their findings using voting and quality weighting.

Architecture position in DISCOVERY mode:
    DiscoveryLoop
        -> SearchExp → SearchAgent (search, MCP tools)
        -> EvidencePipeline (deterministic cleaning, zero LLM)
        -> AnalysisExp (3 LLM parallel analysis + voting)   <-- this module
        -> GapAnalysis (LLM coverage judgment + deterministic convergence)
        -> SynthesisExp (3 LLM voting)

Inheritance chain: AnalysisExp -> InquiroBaseExp -> BaseExp (EvoMaster)

Key design:
    - Takes *cleaned* evidence (NOT raw evidence) as input.
    - Calls N LLM models in parallel for independent analysis.
    - Each model receives the same evidence set and checklist.
    - Aggregates results via AggregationEngine (weighted voting).
    - No MCP tools -- pure reasoning passes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from inquiro.core.aggregation import AggregatedResult, AggregationEngine
from inquiro.core.llm_utils import extract_cost_from_response, extract_json_from_text
from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.infrastructure.cancellation import CancelledError
from inquiro.infrastructure.event_emitter import InquiroEvent
from inquiro.prompts.loader import PromptLoader
from inquiro.prompts.section_builder import PromptSectionBuilder

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from inquiro.core.types import (
        CleanedEvidence,
        DiscoveryConfig,
        EvaluationResult,
        EvaluationTask,
    )
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter
    from inquiro.infrastructure.llm_pool import LLMProviderPool

logger = logging.getLogger(__name__)

# 📝 Module-level PromptLoader instance (shared across all AnalysisExp)
_prompt_loader = PromptLoader()

# 💰 Conservative average cost estimates per token (re-exported for config override)
_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000


class AnalysisExp(InquiroBaseExp):
    """Three-model parallel analysis experiment 🔬.

    Runs evidence analysis across multiple LLM models in parallel,
    then aggregates their findings using voting and quality weighting.

    Inherits: AnalysisExp -> InquiroBaseExp -> BaseExp (EvoMaster)

    Attributes:
        task: The evaluation task with rules and checklist.
        llm_pool: Provider pool for obtaining model-specific LLM instances.
        aggregation_engine: Stateless engine for result aggregation.
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
        discovery_config: DiscoveryConfig | None = None,
        evolution_enrichment: str | None = None,
    ) -> None:
        """Initialize AnalysisExp 🔧.

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
            discovery_config: Optional discovery configuration carrying
                custom token cost rates.  When None, module-level
                defaults (_INPUT_COST_PER_TOKEN, _OUTPUT_COST_PER_TOKEN)
                are used for cost estimation.
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
        self._discovery_config = discovery_config

        # 🧬 Evolution enrichment for system prompt injection
        self._evolution_enrichment = evolution_enrichment

    @property
    def exp_name(self) -> str:
        """Return experiment name 🏷️.

        Returns:
            "Analysis" as the experiment type name.
        """
        return "Analysis"

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    async def run_analysis(
        self,
        task: EvaluationTask,
        cleaned_evidence: list[CleanedEvidence],
        analysis_models: list[str],
        round_number: int = 1,
        supplementary_context: str | None = None,
    ) -> tuple[AggregatedResult, float]:
        """Run parallel analysis on cleaned evidence 🔬.

        Orchestrates the three-phase analysis pipeline:
            1. Render analysis prompts with evidence and checklist.
            2. Fan out to N models in parallel for independent analysis.
            3. Aggregate results via AggregationEngine.

        Args:
            task: The evaluation task with rules and checklist.
            cleaned_evidence: Evidence already cleaned by EvidencePipeline.
            analysis_models: List of LLM model names/keys for parallel
                analysis. Typically 3 models from DiscoveryConfig.
            round_number: Current round in the discovery loop (1-based).
            supplementary_context: Optional text summaries of evidence
                excluded by Tier-2 condensation. Appended to user prompt.

        Returns:
            Tuple of (AggregatedResult, total_cost_usd).

        Raises:
            CancelledError: If the cancellation token is signalled.
            ValueError: If analysis_models is empty.
        """
        if not analysis_models:
            raise ValueError("analysis_models must contain at least one model")

        task_id = task.task_id

        # 📡 Emit analysis start event
        self.event_emitter.emit(
            InquiroEvent.TASK_STARTED,
            task_id,
            {
                "phase": "analysis",
                "round_number": round_number,
                "models": analysis_models,
                "evidence_count": len(cleaned_evidence),
            },
        )

        logger.info(
            "🔬 Starting parallel analysis for task %s round %d "
            "with %d models and %d evidence items",
            task_id,
            round_number,
            len(analysis_models),
            len(cleaned_evidence),
        )

        # 🛑 Check cancellation before launching
        if self.cancellation_token.is_cancelled:
            raise CancelledError(
                self.cancellation_token.reason or "Cancelled before analysis"
            )

        # 📝 Step 1: Render prompts
        system_prompt = self._render_system_prompt(task)
        user_prompt = self._render_user_prompt(
            task,
            cleaned_evidence,
            round_number,
            supplementary_context=supplementary_context,
        )

        # 🚀 Step 2: Fan out to models in parallel
        analysis_coros = [
            self._run_single_analysis_model(
                model_key=model_key,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            for model_key in analysis_models
        ]

        raw_outcomes = await asyncio.gather(
            *analysis_coros,
            return_exceptions=True,
        )

        # 📊 Step 3: Collect results and accumulate costs
        successful: list[tuple[str, EvaluationResult]] = []
        failed_models: list[str] = []
        total_analysis_cost = 0.0

        for model_key, outcome in zip(analysis_models, raw_outcomes):
            if isinstance(outcome, BaseException):
                logger.error(
                    "❌ Analysis model '%s' failed: %s",
                    model_key,
                    outcome,
                )
                failed_models.append(model_key)
            else:
                result, cost = outcome
                successful.append((model_key, result))
                total_analysis_cost += cost
                logger.info(
                    "✅ Analysis model '%s' completed: "
                    "decision=%s confidence=%.2f cost=$%.4f",
                    model_key,
                    result.decision.value,
                    result.confidence,
                    cost,
                )

        if not successful:
            logger.error(
                "❌ All analysis models failed for task %s. Failed: %s",
                task_id,
                failed_models,
            )
            # 📡 Emit failure event
            self.event_emitter.emit(
                InquiroEvent.TASK_FAILED,
                task_id,
                {
                    "phase": "analysis",
                    "error": "All analysis models failed",
                    "failed_models": failed_models,
                },
            )
            # 🔄 Return a fallback cautious result
            return self._build_fallback_result(task_id, cleaned_evidence), 0.0

        # ⚖️ Step 4: Aggregate via AggregationEngine
        weights = {model: 1.0 for model in analysis_models}
        aggregated = self.aggregation_engine.aggregate(
            results=successful,
            weights=weights,
            strategy="weighted_voting",
            consensus_threshold=0.7,
        )

        # 🔍 Step 4b: Merge gaps/doubts from individual model results
        all_gaps: list[str] = []
        all_doubts: list[str] = []
        for _, result in successful:
            all_gaps.extend(result.gaps_remaining or [])
            all_doubts.extend(result.doubts_remaining or [])
        aggregated.gaps_remaining = list(dict.fromkeys(all_gaps))
        aggregated.doubts_remaining = list(dict.fromkeys(all_doubts))

        # 📡 Emit analysis completion event
        self.event_emitter.emit(
            InquiroEvent.TASK_COMPLETED,
            task_id,
            {
                "phase": "analysis",
                "round_number": round_number,
                "decision": aggregated.decision.value,
                "confidence": aggregated.confidence,
                "consensus_ratio": aggregated.consensus_ratio,
                "successful_models": [n for n, _ in successful],
                "failed_models": failed_models,
            },
        )

        logger.info(
            "📊 Analysis aggregation complete for task %s round %d: "
            "decision=%s confidence=%.2f consensus=%.2f "
            "cost=$%.4f",
            task_id,
            round_number,
            aggregated.decision.value,
            aggregated.confidence,
            aggregated.consensus_ratio,
            total_analysis_cost,
        )

        return aggregated, total_analysis_cost

    # ====================================================================
    # 🧠 Single model analysis
    # ====================================================================

    async def _run_single_analysis_model(
        self,
        model_key: str,
        task: EvaluationTask,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[EvaluationResult, float]:
        """Run analysis with a single LLM model 🧠.

        Creates a Dialog with system + user prompt and calls the LLM
        for a single reasoning pass. No MCP tools -- pure reasoning.

        Args:
            model_key: LLM model identifier / provider key.
            task: Evaluation task (for parsing and result construction).
            system_prompt: Pre-rendered system prompt.
            user_prompt: Pre-rendered user prompt with evidence.

        Returns:
            Tuple of (EvaluationResult, cost_usd) for this model.
        """
        from evomaster.utils.types import Dialog, SystemMessage, UserMessage

        logger.info(
            "🧠 Launching analysis model '%s' for task %s",
            model_key,
            task.task_id,
        )

        # 🛑 Check cancellation
        if self.cancellation_token.is_cancelled:
            raise CancelledError(
                self.cancellation_token.reason or "Cancelled before model analysis"
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

        # 📊 Parse response into EvaluationResult
        result = self._parse_analysis_response(
            response,
            model_key,
            task,
        )

        # 💰 Estimate cost from response metadata (use config rates if set)
        cfg = self._discovery_config
        cost = extract_cost_from_response(
            response,
            input_cost_per_token=(
                cfg.input_cost_per_token if cfg is not None else _INPUT_COST_PER_TOKEN
            ),
            output_cost_per_token=(
                cfg.output_cost_per_token if cfg is not None else _OUTPUT_COST_PER_TOKEN
            ),
        )

        return result, cost

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
        """Render the analysis system prompt from template 📝.

        Loads the analysis_system template and fills in rules,
        checklist, and output schema from the task definition.

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
            "analysis_system",
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
        cleaned_evidence: list[CleanedEvidence],
        round_number: int,
        supplementary_context: str | None = None,
    ) -> str:
        """Render the analysis user prompt with evidence 📝.

        Loads the analysis_user template and fills in formatted
        evidence, checklist, rules, and round context. When
        supplementary_context is provided (Tier-2 condensation was
        applied), it is appended so the LLM is aware of out-of-window
        evidence groups.

        Args:
            task: Evaluation task with rules and checklist.
            cleaned_evidence: List of cleaned evidence items.
            round_number: Current round in the discovery loop.
            supplementary_context: Optional Tier-2 group summaries to
                append after the main evidence list.

        Returns:
            Rendered user prompt string with evidence list.
        """
        # 📋 Format evidence list
        evidence_list = self._format_evidence_list(cleaned_evidence)

        # 📋 Format checklist
        checklist_md = PromptSectionBuilder.format_checklist(task.checklist)

        # 📝 Rules
        rules = task.rules or "No specific evaluation rules provided."

        # 🔄 Round context
        round_context = self._format_round_context(round_number)

        rendered = _prompt_loader.render(
            "analysis_user",
            evidence_list=evidence_list,
            checklist=checklist_md,
            rules=rules,
            round_context=round_context,
        )

        # 📊 Append Tier-2 group summaries when condenser excluded evidence
        if supplementary_context:
            rendered += (
                "\n\n## Additional Evidence (Summarised)\n\n"
                + supplementary_context
            )

        return rendered

    @staticmethod
    def _format_evidence_list(
        cleaned_evidence: list[CleanedEvidence],
    ) -> str:
        """Format cleaned evidence into a numbered Markdown list 📋.

        Produces a human-readable block of evidence items suitable
        for injection into the user prompt.  Includes DOI,
        clinical_trial_id, and quality_label when available.

        Args:
            cleaned_evidence: List of CleanedEvidence items.

        Returns:
            Formatted evidence string, or fallback message if empty.
        """
        if not cleaned_evidence:
            return "No evidence items available."

        lines: list[str] = []
        for ev in cleaned_evidence:
            lines.append(f"### [{ev.id}]")
            if ev.summary:
                lines.append(f"- **Summary**: {ev.summary}")
            if ev.url:
                lines.append(f"- **URL**: {ev.url}")
            if ev.tag:
                lines.append(f"- **Type**: {ev.tag.value}")
            if ev.doi:
                lines.append(f"- **DOI**: {ev.doi}")
            if ev.clinical_trial_id:
                lines.append(f"- **Clinical Trial**: {ev.clinical_trial_id}")
            if ev.quality_label:
                lines.append(f"- **Quality**: {ev.quality_label}")
            if ev.mcp_server:
                lines.append(f"- **Source**: {ev.mcp_server}")
            if ev.source_query:
                lines.append(f"- **Query**: {ev.source_query}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_round_context(round_number: int) -> str:
        """Format round context information 🔄.

        Args:
            round_number: Current round in the discovery loop.

        Returns:
            Context string indicating the current round.
        """
        if round_number <= 1:
            return "## Round Context\n\nThis is the **initial analysis** round."
        return (
            f"## Round Context\n\n"
            f"This is analysis round **{round_number}**. Previous rounds "
            f"may have identified gaps. Focus on newly available evidence "
            f"and any previously uncovered checklist items."
        )

    # ====================================================================
    # 📊 Response parsing
    # ====================================================================

    def _parse_analysis_response(
        self,
        response: Any,
        model_name: str,
        task: EvaluationTask,
    ) -> EvaluationResult:
        """Parse an LLM analysis response into EvaluationResult 📊.

        Extracts a JSON object from the response text, then converts
        it into a structured EvaluationResult.

        Args:
            response: AssistantMessage from the LLM.
            model_name: Name of the analysis model.
            task: Original evaluation task.

        Returns:
            Parsed EvaluationResult.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            Decision,
            EvaluationResult,
            Evidence,
            EvidenceStrength,
            ReasoningClaim,
        )

        content = getattr(response, "content", "") or ""
        raw_result = extract_json_from_text(content)

        # 🎯 Parse decision
        decision_str = raw_result.get("decision", "cautious")
        try:
            decision = Decision(decision_str)
        except ValueError:
            decision = Decision.CAUTIOUS

        # 📊 Parse confidence
        confidence = float(raw_result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        # 🧠 Parse reasoning claims
        reasoning: list[ReasoningClaim] = []
        for claim_data in raw_result.get("reasoning", []):
            if isinstance(claim_data, dict):
                try:
                    reasoning.append(ReasoningClaim(**claim_data))
                except Exception as exc:
                    # 🔧 Lenient fallback: fill missing fields
                    logger.debug("⚠️ Skipped malformed ReasoningClaim: %s", exc)
                    claim_text = claim_data.get("claim", "")
                    if claim_text:
                        strength_raw = claim_data.get("strength", "moderate")
                        try:
                            strength = EvidenceStrength(strength_raw)
                        except ValueError:
                            strength = EvidenceStrength.MODERATE
                        reasoning.append(
                            ReasoningClaim(
                                claim=claim_text,
                                evidence_ids=claim_data.get(
                                    "evidence_ids",
                                    [],
                                ),
                                strength=strength,
                            )
                        )

        # 🔗 Parse evidence index from response
        evidence_index: list[Evidence] = []
        for ev_data in raw_result.get("evidence_index", []):
            if isinstance(ev_data, dict):
                try:
                    evidence_index.append(Evidence(**ev_data))
                except Exception as exc:
                    logger.debug("⚠️ Skipped malformed Evidence item: %s", exc)

        # 📋 Parse gaps
        gaps_remaining = raw_result.get("gaps_remaining", [])
        if not isinstance(gaps_remaining, list):
            gaps_remaining = []

        # 🧩 Parse checklist coverage
        coverage_raw = raw_result.get("checklist_coverage", {})
        if isinstance(coverage_raw, dict):
            checklist_coverage = ChecklistCoverage(
                required_covered=coverage_raw.get(
                    "required_covered",
                    [],
                ),
                required_missing=coverage_raw.get(
                    "required_missing",
                    [],
                ),
            )
        else:
            checklist_coverage = ChecklistCoverage()

        # 📝 Build model-specific task ID
        model_task_id = f"{task.task_id}::analysis::{model_name}"

        return EvaluationResult(
            task_id=model_task_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence_index=evidence_index,
            search_rounds=0,
            round_logs=[],
            checklist_coverage=checklist_coverage,
            gaps_remaining=gaps_remaining,
            doubts_remaining=raw_result.get("doubts_remaining", []),
            cost=0.0,
            pipeline_mode="discovery",
        )

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
        input_cost_per_token: float = _INPUT_COST_PER_TOKEN,
        output_cost_per_token: float = _OUTPUT_COST_PER_TOKEN,
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

    # ====================================================================
    # 🔄 Fallback and BaseExp interface
    # ====================================================================

    def _build_fallback_result(
        self,
        task_id: str,
        cleaned_evidence: list[CleanedEvidence],
    ) -> AggregatedResult:
        """Build a cautious fallback when all models fail 🔄.

        Args:
            task_id: Task identifier.
            cleaned_evidence: Evidence items (used for evidence count
                in the fallback result).

        Returns:
            AggregatedResult with cautious decision and zero confidence.
        """
        from inquiro.core.types import Decision

        return AggregatedResult(
            decision=Decision.CAUTIOUS,
            confidence=0.0,
            consensus_ratio=0.0,
            model_decisions={},
            model_confidences={},
            reasoning_summary="All analysis models failed.",
            evidence_index=[],
            individual_results=[],
            structured_reasoning=[],
            conflict_info=None,
        )

    def run_sync(self) -> Any:
        """Synchronous run -- not used for AnalysisExp 🔄.

        AnalysisExp uses the async ``run_analysis()`` method.
        This method exists to satisfy the InquiroBaseExp interface.

        Returns:
            Empty dict (AnalysisExp is async-only).
        """
        logger.warning("⚠️ AnalysisExp.run_sync() called -- use run_analysis() instead")
        return {}
