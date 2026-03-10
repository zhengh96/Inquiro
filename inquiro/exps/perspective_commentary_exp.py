"""Inquiro PerspectiveCommentaryExp -- Multi-perspective commentary 🎭.

Runs N analyst perspectives in parallel, each analyzing the same synthesis
data from a different viewpoint, then synthesizes a committee consensus.
This Exp is domain-agnostic: all role definitions and prompts are injected
by the calling layer (e.g., TargetMaster ExpertPanel).

Architecture position:
    TargetMaster Orchestrator
        -> PerspectiveCommentaryExp (per-panel commentary)  <-- this module
            -> N LLM calls in parallel (one per perspective)
            -> 1 consensus aggregation LLM call

Inheritance chain:
    PerspectiveCommentaryExp -> InquiroBaseExp -> BaseExp (EvoMaster)

Key design:
    - Perspectives run in parallel via asyncio.gather().
    - Each perspective call is a lightweight Dialog (system + user, no tools).
    - Consensus is a single additional LLM call aggregating all outputs.
    - All failures degrade gracefully: missing perspectives produce fallback
      PerspectiveOutput; consensus falls back to majority-vote decision.
    - Zero domain knowledge: role names, prompts, weights are caller-supplied.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from inquiro.core.llm_utils import extract_cost_from_response, extract_json_from_text
from inquiro.exps.base_exp import InquiroBaseExp

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM
    from inquiro.core.types import PerspectiveConfig, PerspectiveOutput, ConsensusOutput
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter

logger = logging.getLogger(__name__)

# 🔢 Minimum narrative length for a valid ConsensusOutput
_MIN_CONSENSUS_NARRATIVE_LEN = 50

# 🎭 Valid perspective decision values
_VALID_DECISIONS = frozenset({"GO", "CONDITIONAL_GO", "NO_GO"})

# 🔄 Fallback decision when parsing fails
_FALLBACK_DECISION = "CONDITIONAL_GO"


# ============================================================================
# 📊 Result model
# ============================================================================


class CommentaryResult(BaseModel):
    """Combined output from PerspectiveCommentaryExp 🎭.

    Attributes:
        perspectives: Per-perspective analysis outputs.
        consensus: Aggregated committee consensus.
        cost_usd: Total LLM cost for all calls.
        duration_seconds: Wall-clock duration.
        successful_perspective_ids: IDs of perspectives that succeeded.
        failed_perspective_ids: IDs of perspectives that failed (degraded).
    """

    perspectives: list[Any] = Field(
        default_factory=list,
        description="Per-perspective PerspectiveOutput objects.",
    )
    consensus: Any = Field(
        description="Aggregated ConsensusOutput.",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total LLM cost in USD.",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration in seconds.",
    )
    successful_perspective_ids: list[str] = Field(
        default_factory=list,
        description="IDs of perspectives that ran successfully.",
    )
    failed_perspective_ids: list[str] = Field(
        default_factory=list,
        description="IDs of perspectives that used fallback output.",
    )


# ============================================================================
# 🎭 PerspectiveCommentaryExp
# ============================================================================


class PerspectiveCommentaryExp(InquiroBaseExp):
    """Multi-perspective commentary experiment 🎭.

    Runs N perspectives in parallel, each analyzing the same evaluation
    data from a different viewpoint, then synthesizes a consensus.

    Pipeline:
    1. Fan out N perspective coroutines via asyncio.gather().
    2. Parse each LLM response into PerspectiveOutput (with fallback).
    3. Aggregate all outputs into a ConsensusOutput via one extra LLM call.
    4. Return CommentaryResult with all outputs and cost/duration metadata.

    Inherits: PerspectiveCommentaryExp -> InquiroBaseExp -> BaseExp

    Attributes:
        task: Minimal task definition (only task_id + output_schema used).
        llm: Default LLM instance for all perspective and consensus calls.
    """

    def __init__(
        self,
        task: Any,
        llm: BaseLLM,
        cost_tracker: CostTracker | None = None,
        event_emitter: EventEmitter | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Initialize PerspectiveCommentaryExp 🔧.

        Args:
            task: Task definition. Must have ``task_id`` and
                ``output_schema`` attributes.  ``quality_gate`` is
                optional (a permissive default is used if absent).
            llm: LLM instance used for all LLM calls.  A single shared
                instance is used for both perspective and consensus calls.
            cost_tracker: Cost tracking instance.  A no-op tracker is
                created when None.
            event_emitter: SSE event emitter.  A no-op emitter is created
                when None.
            cancellation_token: Cooperative cancellation signal.  A fresh
                token is created when None.
        """
        from inquiro.infrastructure.cancellation import CancellationToken as CT
        from inquiro.infrastructure.cost_tracker import CostTracker as CTr
        from inquiro.infrastructure.event_emitter import EventEmitter as EE
        from inquiro.core.types import QualityGateConfig

        qg_config = getattr(task, "quality_gate", None) or QualityGateConfig()

        self._init_base(
            task=task,
            llm=llm,
            quality_gate_config=qg_config,
            cost_tracker=cost_tracker if cost_tracker is not None else CTr(
                max_per_task=10.0,
                max_total=100.0,
            ),
            event_emitter=event_emitter if event_emitter is not None else EE(),
            cancellation_token=(
                cancellation_token if cancellation_token is not None else CT()
            ),
        )

    @property
    def exp_name(self) -> str:
        """Return experiment name 🏷️.

        Returns:
            "PerspectiveCommentary" as the experiment type name.
        """
        return "PerspectiveCommentary"

    def run_sync(self) -> Any:
        """Synchronous run -- not used for PerspectiveCommentaryExp 🔄.

        This Exp is async-only.  Use ``run_commentary()`` instead.

        Returns:
            Empty dict (async-only Exp).
        """
        logger.warning(
            "⚠️ PerspectiveCommentaryExp.run_sync() called "
            "-- use run_commentary() instead"
        )
        return {}

    # ====================================================================
    # 🚀 Public API
    # ====================================================================

    async def run_commentary(
        self,
        perspectives: list[PerspectiveConfig],
        synthesis_data: dict[str, Any],
        evidence_summary: str,
        consensus_prompt: str,
    ) -> CommentaryResult:
        """Run multi-perspective commentary and consensus aggregation 🎭.

        Orchestrates the full commentary pipeline:
            1. Fan out N perspective LLM calls in parallel.
            2. Collect results, applying fallback on failures.
            3. Run one consensus aggregation LLM call.
            4. Validate outputs via quality checks.
            5. Return CommentaryResult.

        Args:
            perspectives: List of PerspectiveConfig objects defining each
                analyst role.  Must be non-empty.
            synthesis_data: Structured evaluation data dict passed to each
                perspective as context (e.g., sub-item score, evidence).
            evidence_summary: Human-readable evidence digest injected into
                each perspective's user prompt.
            consensus_prompt: System prompt for the consensus aggregation
                LLM call.  Domain-specific framing is caller-supplied.

        Returns:
            CommentaryResult with per-perspective PerspectiveOutput objects,
            a ConsensusOutput, and cost/duration metadata.

        Raises:
            ValueError: If ``perspectives`` is empty.
        """
        if not perspectives:
            raise ValueError("perspectives must contain at least one PerspectiveConfig")

        task_id = self.task.task_id
        start_time = time.monotonic()

        logger.info(
            "🎭 Starting perspective commentary for task %s "
            "with %d perspectives",
            task_id,
            len(perspectives),
        )

        # 🚀 Step 1: Fan out all perspective calls in parallel
        perspective_coros = [
            self._run_single_perspective(
                config=config,
                synthesis_data=synthesis_data,
                evidence_summary=evidence_summary,
            )
            for config in perspectives
        ]

        raw_outcomes = await asyncio.gather(
            *perspective_coros,
            return_exceptions=True,
        )

        # 📊 Step 2: Collect results with fallback handling
        perspective_outputs: list[PerspectiveOutput] = []
        successful_ids: list[str] = []
        failed_ids: list[str] = []
        total_cost: float = 0.0

        for config, outcome in zip(perspectives, raw_outcomes):
            if isinstance(outcome, BaseException):
                logger.error(
                    "❌ Perspective '%s' failed: %s",
                    config.perspective_id,
                    outcome,
                )
                failed_ids.append(config.perspective_id)
                perspective_outputs.append(
                    self._build_fallback_perspective(config.perspective_id)
                )
            else:
                output, cost = outcome
                perspective_outputs.append(output)
                successful_ids.append(config.perspective_id)
                total_cost += cost
                logger.info(
                    "✅ Perspective '%s' completed: decision=%s confidence=%.2f",
                    config.perspective_id,
                    output.decision,
                    output.confidence,
                )

        # 🤝 Step 3: Synthesize consensus
        consensus, consensus_cost = await self._synthesize_consensus(
            outputs=perspective_outputs,
            synthesis_data=synthesis_data,
            consensus_prompt=consensus_prompt,
            perspective_configs=perspectives,
        )
        total_cost += consensus_cost

        # ✅ Step 4: Quality checks (non-blocking warnings)
        self._validate_outputs(perspective_outputs, consensus)

        duration = time.monotonic() - start_time

        logger.info(
            "📊 Commentary complete for task %s: "
            "consensus=%s perspectives=%d/%d cost=$%.4f duration=%.1fs",
            task_id,
            consensus.unified_recommendation,
            len(successful_ids),
            len(perspectives),
            total_cost,
            duration,
        )

        return CommentaryResult(
            perspectives=perspective_outputs,
            consensus=consensus,
            cost_usd=total_cost,
            duration_seconds=duration,
            successful_perspective_ids=successful_ids,
            failed_perspective_ids=failed_ids,
        )

    # ====================================================================
    # 🧠 Single perspective LLM call
    # ====================================================================

    async def _run_single_perspective(
        self,
        config: PerspectiveConfig,
        synthesis_data: dict[str, Any],
        evidence_summary: str,
    ) -> tuple[PerspectiveOutput, float]:
        """Run a single perspective LLM call 🧠.

        Builds a Dialog with:
            - system: config.system_prompt_context
            - user: formatted synthesis_data + evidence_summary

        Parses the JSON response into a PerspectiveOutput.

        Args:
            config: Perspective configuration with role prompt and ID.
            synthesis_data: Structured evaluation context dict.
            evidence_summary: Pre-formatted evidence digest string.

        Returns:
            Tuple of (PerspectiveOutput, cost_usd).
        """
        from evomaster.utils.types import Dialog, SystemMessage, UserMessage

        logger.debug(
            "🎭 Launching perspective '%s'",
            config.perspective_id,
        )

        user_prompt = self._build_perspective_user_prompt(
            synthesis_data=synthesis_data,
            evidence_summary=evidence_summary,
            perspective_id=config.perspective_id,
        )

        dialog = Dialog(
            messages=[
                SystemMessage(content=config.system_prompt_context),
                UserMessage(content=user_prompt),
            ],
            tools=[],
        )

        response = await asyncio.to_thread(self.llm.query, dialog)

        raw_content = getattr(response, "content", "") or ""
        parsed = extract_json_from_text(raw_content)
        cost = extract_cost_from_response(response)

        output = self._parse_perspective_output(
            parsed=parsed,
            perspective_id=config.perspective_id,
            raw_response=raw_content,
        )

        return output, cost

    # ====================================================================
    # 🤝 Consensus aggregation LLM call
    # ====================================================================

    async def _synthesize_consensus(
        self,
        outputs: list[PerspectiveOutput],
        synthesis_data: dict[str, Any],
        consensus_prompt: str,
        perspective_configs: list[PerspectiveConfig] | None = None,
    ) -> tuple[ConsensusOutput, float]:
        """Aggregate all perspective outputs into a committee consensus 🤝.

        Builds a Dialog with:
            - system: consensus_prompt (caller-supplied domain framing)
            - user: formatted all perspective outputs + synthesis_data summary

        Parses the JSON response into a ConsensusOutput.  Falls back to a
        majority-vote result if the LLM call fails.

        Args:
            outputs: All collected PerspectiveOutput objects.
            synthesis_data: Structured evaluation context dict.
            consensus_prompt: Caller-supplied system prompt for aggregation.
            perspective_configs: Optional configs with weight information.

        Returns:
            Tuple of (ConsensusOutput, cost_usd).
        """
        from evomaster.utils.types import Dialog, SystemMessage, UserMessage

        user_prompt = self._build_consensus_user_prompt(
            outputs=outputs,
            synthesis_data=synthesis_data,
            perspective_configs=perspective_configs,
        )

        dialog = Dialog(
            messages=[
                SystemMessage(content=consensus_prompt),
                UserMessage(content=user_prompt),
            ],
            tools=[],
        )

        try:
            response = await asyncio.to_thread(self.llm.query, dialog)
            raw_content = getattr(response, "content", "") or ""
            parsed = extract_json_from_text(raw_content)
            cost = extract_cost_from_response(response)
            consensus = self._parse_consensus_output(parsed=parsed, outputs=outputs)
        except Exception as exc:
            logger.warning(
                "⚠️ Consensus LLM call failed (%s), using majority-vote fallback",
                exc,
            )
            consensus = self._majority_vote_consensus(outputs)
            cost = 0.0

        return consensus, cost

    # ====================================================================
    # 📝 Prompt building
    # ====================================================================

    @staticmethod
    def _build_perspective_user_prompt(
        synthesis_data: dict[str, Any],
        evidence_summary: str,
        perspective_id: str,
    ) -> str:
        """Build the user prompt for a single perspective call 📝.

        Args:
            synthesis_data: Structured evaluation context dict.
            evidence_summary: Pre-formatted evidence digest.
            perspective_id: Identifier for logging context.

        Returns:
            Formatted user prompt string.
        """
        import json as _json

        data_block = _json.dumps(synthesis_data, ensure_ascii=False, indent=2)

        return (
            f"# Evaluation Data\n\n"
            f"```json\n{data_block}\n```\n\n"
            f"# Evidence Summary\n\n"
            f"{evidence_summary}\n\n"
            f"# Your Task\n\n"
            f"The evaluation data above includes `sub_item_results` with "
            f"per-dimension decision, confidence, and evidence count. "
            f"Use these to understand the evidence strength for each "
            f"assessment dimension.\n\n"
            f"Provide your expert perspective as '{perspective_id}'. "
            f"Respond in JSON with this exact structure:\n"
            f"{{\n"
            f'  "decision": "GO | CONDITIONAL_GO | NO_GO",\n'
            f'  "key_insight": "<1-2 sentences>",\n'
            f'  "concern": "<1-2 sentences>",\n'
            f'  "confidence": <0.0-1.0>,\n'
            f'  "recommendation": "<1 sentence>"\n'
            f"}}"
        )

    @staticmethod
    def _build_consensus_user_prompt(
        outputs: list[PerspectiveOutput],
        synthesis_data: dict[str, Any],
        perspective_configs: list[PerspectiveConfig] | None = None,
    ) -> str:
        """Build the user prompt for the consensus aggregation call 📝.

        Args:
            outputs: All collected PerspectiveOutput objects.
            synthesis_data: Structured evaluation context dict.
            perspective_configs: Optional configs with weight information.

        Returns:
            Formatted consensus user prompt string.
        """
        import json as _json

        # 📊 Build weight lookup from configs
        weight_map: dict[str, float] = {}
        if perspective_configs:
            for pc in perspective_configs:
                weight_map[pc.perspective_id] = pc.weight

        lines: list[str] = ["# Expert Panel Outputs\n"]
        for i, output in enumerate(outputs, 1):
            weight = weight_map.get(output.perspective_id, 1.0)
            lines.append(
                f"## Perspective {i}: {output.perspective_id} "
                f"(weight: {weight:.1f})\n"
                f"- **Decision**: {output.decision}\n"
                f"- **Confidence**: {output.confidence:.2f}\n"
                f"- **Key Insight**: {output.key_insight}\n"
                f"- **Concern**: {output.concern}\n"
                f"- **Recommendation**: {output.recommendation}\n"
            )

        # 📋 Include a concise data summary (top-level keys only)
        summary_keys = {
            k: v
            for k, v in synthesis_data.items()
            if isinstance(v, (str, int, float, bool))
        }
        data_summary = _json.dumps(summary_keys, ensure_ascii=False, indent=2)

        lines.append(
            f"\n# Evaluation Summary\n\n"
            f"```json\n{data_summary}\n```\n\n"
            f"# Your Task\n\n"
            f"Synthesize the above expert perspectives into a committee consensus. "
            f"Respond in JSON with this exact structure:\n"
            f"{{\n"
            f'  "unified_recommendation": "GO | CONDITIONAL_GO | NO_GO",\n'
            f'  "consensus_narrative": "<~200 word summary>",\n'
            f'  "dissent_notes": ["<note1>", "<note2>"],\n'
            f'  "key_action_items": ["<action1>", "<action2>"]\n'
            f"}}"
        )

        return "\n".join(lines)

    # ====================================================================
    # 🔍 Response parsing
    # ====================================================================

    @staticmethod
    def _parse_perspective_output(
        parsed: dict[str, Any],
        perspective_id: str,
        raw_response: str,
    ) -> PerspectiveOutput:
        """Parse LLM JSON response into a PerspectiveOutput 🔍.

        Applies safe defaults for all fields.  An invalid decision value
        is silently replaced with ``CONDITIONAL_GO``.

        Args:
            parsed: Pre-extracted JSON dict from LLM response.
            perspective_id: Perspective identifier to embed in output.
            raw_response: Raw LLM text (for debug logging on parse issues).

        Returns:
            PerspectiveOutput with validated field values.
        """
        from inquiro.core.types import PerspectiveOutput

        decision = parsed.get("decision", _FALLBACK_DECISION)
        if decision not in _VALID_DECISIONS:
            logger.warning(
                "⚠️ Perspective '%s' returned invalid decision '%s', "
                "using '%s'",
                perspective_id,
                decision,
                _FALLBACK_DECISION,
            )
            decision = _FALLBACK_DECISION

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        key_insight = parsed.get("key_insight", "")
        concern = parsed.get("concern", "")
        recommendation = parsed.get("recommendation", "")

        if not key_insight or not concern:
            logger.warning(
                "⚠️ Perspective '%s' returned empty key_insight or concern "
                "(raw len=%d)",
                perspective_id,
                len(raw_response),
            )

        return PerspectiveOutput(
            perspective_id=perspective_id,
            decision=decision,
            key_insight=key_insight or "Analysis unavailable.",
            concern=concern or "No specific concern identified.",
            confidence=confidence,
            recommendation=recommendation,
        )

    @staticmethod
    def _parse_consensus_output(
        parsed: dict[str, Any],
        outputs: list[PerspectiveOutput],
    ) -> ConsensusOutput:
        """Parse LLM JSON response into a ConsensusOutput 🔍.

        Falls back to majority-vote result when the narrative is too short.

        Args:
            parsed: Pre-extracted JSON dict from consensus LLM response.
            outputs: All perspective outputs (used for fallback vote).

        Returns:
            ConsensusOutput with validated field values.
        """
        from inquiro.core.types import ConsensusOutput

        recommendation = parsed.get("unified_recommendation", _FALLBACK_DECISION)
        if recommendation not in _VALID_DECISIONS:
            recommendation = _FALLBACK_DECISION

        narrative = parsed.get("consensus_narrative", "")
        dissent_notes = parsed.get("dissent_notes", [])
        key_action_items = parsed.get("key_action_items", [])

        if len(narrative) < _MIN_CONSENSUS_NARRATIVE_LEN:
            logger.warning(
                "⚠️ Consensus narrative too short (%d chars < %d), "
                "using majority-vote fallback",
                len(narrative),
                _MIN_CONSENSUS_NARRATIVE_LEN,
            )
            return PerspectiveCommentaryExp._majority_vote_consensus(outputs)

        if not isinstance(dissent_notes, list):
            dissent_notes = []
        if not isinstance(key_action_items, list):
            key_action_items = []

        return ConsensusOutput(
            unified_recommendation=recommendation,
            consensus_narrative=narrative,
            dissent_notes=[str(n) for n in dissent_notes],
            key_action_items=[str(a) for a in key_action_items],
        )

    # ====================================================================
    # 🔄 Fallback helpers
    # ====================================================================

    @staticmethod
    def _build_fallback_perspective(perspective_id: str) -> PerspectiveOutput:
        """Build a degraded PerspectiveOutput when LLM call fails 🔄.

        Args:
            perspective_id: Identifier for the failed perspective.

        Returns:
            PerspectiveOutput with safe fallback values.
        """
        from inquiro.core.types import PerspectiveOutput

        return PerspectiveOutput(
            perspective_id=perspective_id,
            decision=_FALLBACK_DECISION,
            key_insight="Analysis unavailable due to a processing error.",
            concern="Perspective could not be generated; treat as missing data.",
            confidence=0.0,
            recommendation="Re-run or consult domain expert directly.",
        )

    @staticmethod
    def _majority_vote_consensus(
        outputs: list[PerspectiveOutput],
    ) -> ConsensusOutput:
        """Compute a simple majority-vote ConsensusOutput 🗳️.

        Used as fallback when the consensus LLM call fails or produces
        an inadequate response.

        Args:
            outputs: All collected PerspectiveOutput objects.

        Returns:
            ConsensusOutput derived from majority decision vote.
        """
        from inquiro.core.types import ConsensusOutput

        if not outputs:
            return ConsensusOutput(
                unified_recommendation=_FALLBACK_DECISION,
                consensus_narrative=(
                    "No perspective outputs available for consensus aggregation."
                ),
            )

        decision_counts: Counter[str] = Counter(o.decision for o in outputs)
        most_common_decision = decision_counts.most_common(1)[0][0]

        # 🗳️ Collect insights from all perspectives for narrative
        insight_lines = [
            f"- {o.perspective_id}: {o.decision} — {o.key_insight}"
            for o in outputs
        ]
        narrative = (
            f"Majority vote consensus: {most_common_decision}. "
            f"Panel summary:\n" + "\n".join(insight_lines)
        )

        return ConsensusOutput(
            unified_recommendation=most_common_decision,
            consensus_narrative=narrative,
            dissent_notes=[
                f"{o.perspective_id}: {o.concern}"
                for o in outputs
                if o.decision != most_common_decision
            ],
            key_action_items=[],
        )

    # ====================================================================
    # ✅ Quality checks
    # ====================================================================

    def _validate_outputs(
        self,
        outputs: list[PerspectiveOutput],
        consensus: ConsensusOutput,
    ) -> None:
        """Validate perspective outputs and consensus quality ✅.

        Logs warnings for quality issues but does NOT block execution.

        Args:
            outputs: All perspective outputs to validate.
            consensus: Aggregated consensus to validate.
        """
        for output in outputs:
            if not output.decision or not output.key_insight:
                logger.warning(
                    "⚠️ Perspective '%s' has empty decision or key_insight",
                    output.perspective_id,
                )

        if len(consensus.consensus_narrative) < _MIN_CONSENSUS_NARRATIVE_LEN:
            logger.warning(
                "⚠️ ConsensusOutput narrative is too short (%d chars)",
                len(consensus.consensus_narrative),
            )
        else:
            logger.debug(
                "✅ Quality check passed: %d perspectives, "
                "consensus narrative %d chars",
                len(outputs),
                len(consensus.consensus_narrative),
            )
