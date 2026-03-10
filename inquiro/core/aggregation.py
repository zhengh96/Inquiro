"""Inquiro AggregationEngine -- multi-model result aggregation 📊.

Aggregates EvaluationResult instances from multiple LLM models into a
single consensus result.  Supports weighted voting and majority voting
strategies with configurable consensus thresholds.

Architecture position:
    EnsembleRunner
        -> (parallel) SearchExp per model
        -> AggregationEngine.aggregate()   <-- this module
        -> EnsembleResult

The engine is domain-agnostic: it works on Decision enums and numeric
confidence scores without any knowledge of the underlying research topic.
"""

from __future__ import annotations

import enum
import logging
from typing import Any

from pydantic import BaseModel, Field

import math

from inquiro.core.types import ChecklistCoverage, Decision, Evidence, EvaluationResult

logger = logging.getLogger(__name__)


# ============================================================================
# 🤝 Conflict Resolution Models
# ============================================================================

# 🔢 Evidence quality label → numeric score for weighting
_QUALITY_SCORES: dict[str | None, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    None: 1.0,
}


class ConflictResolution(str, enum.Enum):
    """Resolution status for ensemble decision conflicts 🤝.

    Attributes:
        UNANIMOUS: All models agree on the decision.
        RESOLVED_BY_WEIGHT: Conflict resolved by evidence quality
            weighting.
        RESOLVED_BY_VOTE: Conflict resolved by majority vote.
        UNRESOLVED: Conflict could not be resolved, both viewpoints
            preserved.
    """

    UNANIMOUS = "unanimous"
    RESOLVED_BY_WEIGHT = "resolved_by_weight"
    RESOLVED_BY_VOTE = "resolved_by_vote"
    UNRESOLVED = "unresolved"


class ConflictInfo(BaseModel):
    """Details about a detected conflict and its resolution 🔍.

    Attributes:
        resolution: How the conflict was resolved.
        majority_decision: The decision held by the majority.
        minority_decisions: Decisions held by minority models.
        majority_models: Models agreeing with majority.
        minority_models: Models disagreeing.
        evidence_weight_score: Score from evidence quality weighting.
    """

    resolution: ConflictResolution = Field(
        description="How the conflict was resolved",
    )
    majority_decision: str = Field(
        default="",
        description="Decision held by the majority of models",
    )
    minority_decisions: list[str] = Field(
        default_factory=list,
        description="Decisions held by minority models",
    )
    majority_models: list[str] = Field(
        default_factory=list,
        description="Model names agreeing with majority decision",
    )
    minority_models: list[str] = Field(
        default_factory=list,
        description="Model names with minority decisions",
    )
    evidence_weight_score: float = Field(
        default=0.0,
        description=("Aggregate evidence quality score for the winning side"),
    )


# ============================================================================
# 📊 Result Models
# ============================================================================


class AggregatedResult(BaseModel):
    """Aggregated result from ensemble evaluation 📊.

    Contains the consensus decision, merged evidence, and per-model
    breakdown produced by the AggregationEngine.
    """

    decision: Decision = Field(
        description="Final consensus decision from ensemble voting",
    )
    confidence: float = Field(
        description="Weighted average confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    consensus_ratio: float = Field(
        description="Fraction of models that agree with the final decision",
        ge=0.0,
        le=1.0,
    )
    model_decisions: dict[str, str] = Field(
        default_factory=dict,
        description="Per-model decision map {model_name: decision_value}",
    )
    model_confidences: dict[str, float] = Field(
        default_factory=dict,
        description="Per-model confidence map {model_name: confidence}",
    )
    reasoning_summary: str = Field(
        default="",
        description="Merged reasoning narrative from all models",
    )
    evidence_index: list[Evidence] = Field(
        default_factory=list,
        description="Merged and deduplicated evidence from all models",
    )
    individual_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw result dicts per model for audit trail",
    )
    structured_reasoning: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Flat list of structured claims with remapped evidence IDs. "
            "Each dict: {claim: str, evidence_ids: list[str], strength: str}"
        ),
    )
    conflict_info: ConflictInfo | None = Field(
        default=None,
        description=(
            "Conflict resolution details when model decisions disagree. "
            "None if all models are unanimous."
        ),
    )
    checklist_coverage: ChecklistCoverage | None = Field(
        default=None,
        description=(
            "Merged checklist coverage from multi-model voting. "
            "None if no model produced coverage data."
        ),
    )
    coverage_conflicts: list[str] = Field(
        default_factory=list,
        description=(
            "Checklist item IDs where models disagreed on coverage status"
        ),
    )
    gaps_remaining: list[str] = Field(
        default_factory=list,
        description="Merged information gaps from all models",
    )
    doubts_remaining: list[str] = Field(
        default_factory=list,
        description="Merged evidence contradictions from all models",
    )


class EnsembleResult(BaseModel):
    """Full ensemble evaluation result 🎭.

    Top-level envelope returned by EnsembleRunner.run_ensemble().
    Wraps the aggregated result with metadata about the ensemble run.
    """

    task_id: str = Field(description="ID of the evaluated task")
    aggregated: AggregatedResult = Field(
        description="Aggregated consensus result from all models",
    )
    successful_models: list[str] = Field(
        default_factory=list,
        description="Names of models that completed successfully",
    )
    failed_models: list[str] = Field(
        default_factory=list,
        description="Names of models that failed during evaluation",
    )
    total_cost_usd: float = Field(
        default=0.0,
        description="Total cost in USD across all model runs",
        ge=0.0,
    )


# ============================================================================
# 🎯 Decision-to-score mapping
# ============================================================================

# 🔢 Numeric encoding: positive=+1, cautious=0, negative=-1
_DECISION_SCORES: dict[Decision, float] = {
    Decision.POSITIVE: 1.0,
    Decision.CAUTIOUS: 0.0,
    Decision.NEGATIVE: -1.0,
}

# 🎯 Thresholds for mapping weighted score back to Decision
_POSITIVE_THRESHOLD = 0.3
_NEGATIVE_THRESHOLD = -0.3


# ============================================================================
# 📊 AggregationEngine
# ============================================================================


class AggregationEngine:
    """Aggregate results from multiple research agents 📊.

    Stateless engine that combines EvaluationResult instances using
    configurable voting strategies.  Each model result carries a weight
    that influences the final decision and confidence.

    Supported strategies:
        - ``weighted_voting``: decision derived from weighted score
        - ``majority_voting``: each model gets equal weight regardless
          of the configured weights

    Example::

        engine = AggregationEngine()
        aggregated = engine.aggregate(
            results=[("claude", result_a), ("gpt", result_b)],
            weights={"claude": 0.6, "gpt": 0.4},
        )
    """

    def aggregate(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
        strategy: str = "weighted_voting",
        consensus_threshold: float = 0.7,
    ) -> AggregatedResult:
        """Aggregate multiple model results into a single consensus 🎯.

        Args:
            results: List of (model_name, EvaluationResult) tuples from
                successful model runs.
            weights: Voting weight map {model_name: weight}.  Weights are
                normalised internally so they sum to 1.0.
            strategy: Aggregation strategy.  ``"weighted_voting"`` uses
                the provided weights; ``"majority_voting"`` assigns
                equal weight to every model.
            consensus_threshold: Minimum fraction of models that must
                agree for the result to be considered consensus.

        Returns:
            AggregatedResult with consensus decision, merged evidence,
            and per-model breakdowns.

        Raises:
            ValueError: If results is empty.
        """
        if not results:
            raise ValueError("Cannot aggregate zero results")

        # 🔄 Normalise strategy name
        effective_strategy = strategy.lower().strip()

        # ⚖️ Build effective weights
        effective_weights = self._build_effective_weights(
            results, weights, effective_strategy
        )

        # 🎯 Step 1: Compute weighted score
        weighted_score = self._compute_weighted_score(results, effective_weights)

        # 🎯 Step 2: Map score back to Decision
        final_decision = self._score_to_decision(weighted_score)

        # 📊 Step 3: Compute weighted average confidence
        confidence = self._compute_weighted_confidence(results, effective_weights)

        # 🤝 Step 4: Compute consensus ratio
        consensus_ratio = self._compute_consensus_ratio(results, final_decision)

        # 🤝 Step 4b: Resolve conflicts between models
        conflict_info = self._resolve_conflicts(
            results, effective_weights, final_decision, consensus_ratio
        )

        # 📝 Step 5: Build per-model maps
        model_decisions: dict[str, str] = {}
        model_confidences: dict[str, float] = {}
        for model_name, result in results:
            model_decisions[model_name] = result.decision.value
            model_confidences[model_name] = result.confidence

        # 🔗 Step 6: Merge evidence
        merged_evidence, remap = self._merge_evidence(results)

        # 🧠 Step 7: Merge reasoning summaries
        reasoning_summary = self._merge_reasoning(results)

        # 🧠 Step 7b: Build structured reasoning with remapped IDs
        structured_reasoning = self._merge_structured_reasoning(results, remap)

        # 📦 Step 8: Build individual results for audit
        individual_results = self._build_individual_results(results)

        # ✅ Step 9: Merge checklist coverage from multi-model voting
        merged_coverage, cov_conflicts = self._merge_checklist_coverage(results)

        logger.info(
            "📊 Ensemble aggregation complete: decision=%s "
            "confidence=%.2f consensus=%.2f strategy=%s "
            "models=%d coverage=%s",
            final_decision.value,
            confidence,
            consensus_ratio,
            effective_strategy,
            len(results),
            "yes" if merged_coverage else "none",
        )

        return AggregatedResult(
            decision=final_decision,
            confidence=confidence,
            consensus_ratio=consensus_ratio,
            model_decisions=model_decisions,
            model_confidences=model_confidences,
            reasoning_summary=reasoning_summary,
            evidence_index=merged_evidence,
            individual_results=individual_results,
            structured_reasoning=structured_reasoning,
            conflict_info=conflict_info,
            checklist_coverage=merged_coverage,
            coverage_conflicts=cov_conflicts,
        )

    # -- Private helpers -----------------------------------------------------

    def _build_effective_weights(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
        strategy: str,
    ) -> dict[str, float]:
        """Build normalised weight map based on strategy 📐.

        For ``majority_voting`` all models receive equal weight.
        For ``weighted_voting`` the supplied weights are normalised
        to sum to 1.0.

        Args:
            results: List of (model_name, result) tuples.
            weights: Raw weight map from configuration.
            strategy: Aggregation strategy name.

        Returns:
            Normalised {model_name: weight} dict summing to ~1.0.
        """
        model_names = [name for name, _ in results]

        if strategy == "majority_voting":
            equal_weight = 1.0 / len(model_names) if model_names else 0.0
            return {name: equal_weight for name in model_names}

        # ⚖️ weighted_voting: normalise supplied weights
        raw = {name: weights.get(name, 1.0) for name in model_names}
        total = sum(raw.values())
        if total <= 0.0:
            # 🔄 Fallback to equal weights
            equal_weight = 1.0 / len(model_names) if model_names else 0.0
            return {name: equal_weight for name in model_names}

        return {name: w / total for name, w in raw.items()}

    def _compute_weighted_score(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
    ) -> float:
        """Compute weighted decision score from all results 🔢.

        Maps decisions to numeric scores (positive=+1, cautious=0,
        negative=-1), then returns the weighted sum.

        Args:
            results: List of (model_name, result) tuples.
            weights: Normalised weight map.

        Returns:
            Weighted score in [-1.0, +1.0].
        """
        score = 0.0
        for model_name, result in results:
            decision_score = _DECISION_SCORES.get(result.decision, 0.0)
            model_weight = weights.get(model_name, 0.0)
            score += decision_score * model_weight
        return score

    def _score_to_decision(self, score: float) -> Decision:
        """Map a numeric weighted score to a Decision enum 🎯.

        Args:
            score: Weighted score in [-1.0, +1.0].

        Returns:
            Decision.POSITIVE if score > 0.3,
            Decision.NEGATIVE if score < -0.3,
            Decision.CAUTIOUS otherwise.
        """
        if score > _POSITIVE_THRESHOLD:
            return Decision.POSITIVE
        if score < _NEGATIVE_THRESHOLD:
            return Decision.NEGATIVE
        return Decision.CAUTIOUS

    def _compute_weighted_confidence(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
    ) -> float:
        """Compute weighted average confidence 📊.

        Args:
            results: List of (model_name, result) tuples.
            weights: Normalised weight map.

        Returns:
            Weighted average confidence clamped to [0.0, 1.0].
        """
        confidence = 0.0
        for model_name, result in results:
            model_weight = weights.get(model_name, 0.0)
            confidence += result.confidence * model_weight
        return max(0.0, min(1.0, confidence))

    def _compute_consensus_ratio(
        self,
        results: list[tuple[str, EvaluationResult]],
        final_decision: Decision,
    ) -> float:
        """Compute the fraction of models agreeing with the decision 🤝.

        Args:
            results: List of (model_name, result) tuples.
            final_decision: The aggregated consensus Decision.

        Returns:
            Ratio in [0.0, 1.0].
        """
        if not results:
            return 0.0
        agree_count = sum(
            1 for _, result in results if result.decision == final_decision
        )
        return agree_count / len(results)

    def _resolve_conflicts(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
        final_decision: Decision,
        consensus_ratio: float,
    ) -> ConflictInfo | None:
        """Analyze and resolve conflicts between model decisions 🤝.

        Resolution order:
        1. Unanimous -- all models agree, return None (no conflict).
        2. Evidence weight -- stronger evidence side wins if its
           normalised evidence quality score >= 0.6.
        3. Majority vote -- majority decision wins if the majority
           group holds >= 2/3 of participating models.
        4. Unresolved -- mark as conflicting, preserving all viewpoints.

        Args:
            results: List of (model_name, EvaluationResult) tuples.
            weights: Normalised weight map {model_name: weight}.
            final_decision: The aggregated consensus Decision.
            consensus_ratio: Fraction of models agreeing with
                final_decision.

        Returns:
            ConflictInfo with resolution details, or None if all
            models are unanimous.
        """
        # 🎯 Step 1: Unanimous -- no conflict
        if consensus_ratio == 1.0:
            return None

        # 📊 Step 2: Group models by their decision
        decision_groups: dict[str, list[str]] = {}
        for model_name, result in results:
            dec_val = result.decision.value
            decision_groups.setdefault(dec_val, []).append(model_name)

        # 🏆 Find the majority group (largest group)
        sorted_groups = sorted(
            decision_groups.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        majority_decision_val = sorted_groups[0][0]
        majority_models = sorted_groups[0][1]

        # 📋 Collect minority groups
        minority_models: list[str] = []
        minority_decisions: list[str] = []
        for dec_val, models in sorted_groups[1:]:
            minority_models.extend(models)
            if dec_val not in minority_decisions:
                minority_decisions.append(dec_val)

        total_models = len(results)

        # ⚖️ Step 3: Evidence weight resolution
        # Calculate evidence quality score for each group
        majority_score = self._compute_group_evidence_score(
            results,
            weights,
            majority_models,
        )
        minority_score = self._compute_group_evidence_score(
            results,
            weights,
            minority_models,
        )
        total_score = majority_score + minority_score

        if total_score > 0.0:
            normalised_majority = majority_score / total_score
            if normalised_majority >= 0.6:
                logger.info(
                    "🤝 Conflict resolved by evidence weight: "
                    "majority=%s (score=%.2f/%.2f=%.2f)",
                    majority_decision_val,
                    majority_score,
                    total_score,
                    normalised_majority,
                )
                return ConflictInfo(
                    resolution=ConflictResolution.RESOLVED_BY_WEIGHT,
                    majority_decision=majority_decision_val,
                    minority_decisions=minority_decisions,
                    majority_models=majority_models,
                    minority_models=minority_models,
                    evidence_weight_score=normalised_majority,
                )

        # 🗳️ Step 4: Majority vote resolution (>= 2/3 threshold)
        majority_ratio = len(majority_models) / total_models
        if majority_ratio >= 2.0 / 3.0:
            logger.info(
                "🤝 Conflict resolved by majority vote: majority=%s ratio=%.2f (%d/%d)",
                majority_decision_val,
                majority_ratio,
                len(majority_models),
                total_models,
            )
            return ConflictInfo(
                resolution=ConflictResolution.RESOLVED_BY_VOTE,
                majority_decision=majority_decision_val,
                minority_decisions=minority_decisions,
                majority_models=majority_models,
                minority_models=minority_models,
                evidence_weight_score=(
                    majority_score / total_score if total_score > 0.0 else 0.0
                ),
            )

        # ❓ Step 5: Unresolved -- preserve all viewpoints
        logger.warning(
            "🤝 Conflict unresolved: decisions=%s models=%d",
            list(decision_groups.keys()),
            total_models,
        )
        return ConflictInfo(
            resolution=ConflictResolution.UNRESOLVED,
            majority_decision=majority_decision_val,
            minority_decisions=minority_decisions,
            majority_models=majority_models,
            minority_models=minority_models,
            evidence_weight_score=(
                majority_score / total_score if total_score > 0.0 else 0.0
            ),
        )

    def _compute_group_evidence_score(
        self,
        results: list[tuple[str, EvaluationResult]],
        weights: dict[str, float],
        group_models: list[str],
    ) -> float:
        """Compute aggregate evidence quality score for a model group 📊.

        Sums the evidence quality scores for all evidence items from
        models in the specified group, weighted by each model's voting
        weight.

        Args:
            results: List of (model_name, EvaluationResult) tuples.
            weights: Normalised weight map {model_name: weight}.
            group_models: Model names in this group.

        Returns:
            Aggregate evidence quality score (higher is better).
        """
        group_set = set(group_models)
        score = 0.0
        for model_name, result in results:
            if model_name not in group_set:
                continue
            model_weight = weights.get(model_name, 0.0)
            for ev in result.evidence_index:
                quality = _QUALITY_SCORES.get(ev.quality_label, 1.0)
                score += quality * model_weight
        return score

    def _merge_evidence(
        self,
        results: list[tuple[str, EvaluationResult]],
    ) -> tuple[list[Evidence], dict[tuple[str, str], str]]:
        """Merge evidence from all models with deduplication 🔗.

        Evidence items are prefixed with the model name to ensure
        unique IDs across models.  Two levels of deduplication are
        applied:

        1. **ID-based**: same prefixed ID (``model:E1``) is never
           added twice.
        2. **Content-based**: evidence items with identical
           ``(summary[:200], source)`` from different models are
           treated as duplicates.  This prevents 3x bloat in
           ``search_once_reason_many`` mode where every reasoning
           model carries the same search-phase evidence forward.

        In ``parallel_full`` mode different models typically find
        different evidence, so content-based dedup has no effect.

        Args:
            results: List of (model_name, result) tuples.

        Returns:
            Tuple of (merged_evidence_list, remap_dict) where
            remap_dict maps (model_name, original_eid) to the
            surviving prefixed evidence ID.
        """
        seen_ids: set[str] = set()
        # 🔑 Content-based dedup key: (summary prefix, source)
        seen_content: dict[tuple[str, str], str] = {}
        merged: list[Evidence] = []
        # 🗺️ Remap: (model_name, original_eid) → surviving prefixed ID
        remap: dict[tuple[str, str], str] = {}

        for model_name, result in results:
            for ev in result.evidence_index:
                # 🏷️ Prefix evidence ID with model name
                prefixed_id = f"{model_name}:{ev.id}"

                # 🔍 Content-based dedup to avoid bloat in
                # search_once_reason_many mode
                content_key = (ev.summary[:200], ev.source)
                if content_key in seen_content:
                    # 🔄 Map to the surviving duplicate's prefixed ID
                    remap[(model_name, ev.id)] = seen_content[content_key]
                    continue

                if prefixed_id in seen_ids:
                    remap[(model_name, ev.id)] = prefixed_id
                    continue
                seen_ids.add(prefixed_id)
                seen_content[content_key] = prefixed_id

                # 🗺️ Track remap for this evidence
                remap[(model_name, ev.id)] = prefixed_id

                # ✨ Create a new Evidence with prefixed ID
                merged.append(
                    Evidence(
                        id=prefixed_id,
                        source=ev.source,
                        url=ev.url,
                        query=ev.query,
                        summary=ev.summary,
                        quality_label=ev.quality_label,
                        round_number=ev.round_number,
                        timestamp=ev.timestamp,
                        source_report_id=ev.source_report_id,
                    )
                )

        logger.debug(
            "🔗 Merged %d unique evidence items from %d models",
            len(merged),
            len(results),
        )
        return merged, remap

    def _merge_reasoning(
        self,
        results: list[tuple[str, EvaluationResult]],
    ) -> str:
        """Merge reasoning claims from all models into a summary 🧠.

        Produces a structured text summary with each model's key
        claims grouped under a heading.

        Args:
            results: List of (model_name, result) tuples.

        Returns:
            Multi-line reasoning summary string.
        """
        sections: list[str] = []

        for model_name, result in results:
            if not result.reasoning:
                continue
            claims = []
            for claim in result.reasoning:
                evidence_refs = ", ".join(claim.evidence_ids)
                claims.append(
                    f"  - [{claim.strength.value}] {claim.claim} "
                    f"(evidence: {evidence_refs})"
                )
            section = f"[{model_name}]\n" + "\n".join(claims)
            sections.append(section)

        return "\n\n".join(sections)

    def _merge_structured_reasoning(
        self,
        results: list[tuple[str, EvaluationResult]],
        remap: dict[tuple[str, str], str],
    ) -> list[dict[str, Any]]:
        """Merge structured reasoning claims from all models 🧠.

        Produces a flat list of claim dicts with evidence IDs remapped
        to their surviving prefixed IDs.  Simple substring dedup is
        applied: if claim A is a substring of claim B, only B is kept.

        Args:
            results: List of (model_name, result) tuples.
            remap: Evidence ID remap from ``_merge_evidence()``.

        Returns:
            Flat list of dicts, each with keys:
            ``claim``, ``evidence_ids``, ``strength``.
        """
        raw_claims: list[dict[str, Any]] = []

        for model_name, result in results:
            if not result.reasoning:
                continue
            for claim in result.reasoning:
                # 🗺️ Remap evidence IDs to surviving prefixed IDs
                remapped_ids: list[str] = []
                for eid in claim.evidence_ids:
                    mapped = remap.get((model_name, eid))
                    if mapped:
                        remapped_ids.append(mapped)
                    else:
                        # 🔄 Fallback: use model:original_id
                        remapped_ids.append(f"{model_name}:{eid}")

                raw_claims.append(
                    {
                        "claim": claim.claim,
                        "evidence_ids": list(dict.fromkeys(remapped_ids)),
                        "strength": claim.strength.value,
                        **({"direction": claim.direction} if claim.direction else {}),
                    }
                )

        # 🧹 Simple substring dedup: if claim A ⊂ claim B, drop A
        deduped: list[dict[str, Any]] = []
        for i, item in enumerate(raw_claims):
            is_substring = False
            for j, other in enumerate(raw_claims):
                if i != j and item["claim"] in other["claim"]:
                    is_substring = True
                    break
            if not is_substring:
                deduped.append(item)

        logger.debug(
            "🧠 Merged structured reasoning: %d raw → %d deduped",
            len(raw_claims),
            len(deduped),
        )
        return deduped

    def _merge_checklist_coverage(
        self,
        results: list[tuple[str, EvaluationResult]],
    ) -> tuple[ChecklistCoverage | None, list[str]]:
        """Merge checklist coverage from multiple models via majority vote ✅.

        For each checklist item ID reported by any model, counts how many
        models mark it as "covered".  An item is merged as covered if
        >= ceil(n/2) models agree.  Items where models disagree are
        reported as conflicts.

        Args:
            results: List of (model_name, EvaluationResult) tuples.

        Returns:
            Tuple of (merged_coverage, conflict_item_ids).
            merged_coverage is None if no model produced coverage data.
        """
        # 📊 Collect coverage data from all models
        coverages: list[ChecklistCoverage] = []
        for _, result in results:
            if result.checklist_coverage and (
                result.checklist_coverage.required_covered
                or result.checklist_coverage.required_missing
            ):
                coverages.append(result.checklist_coverage)

        if not coverages:
            return None, []

        n = len(coverages)
        threshold = math.ceil(n / 2)

        # 🗳️ Count votes per item: how many models say "covered"
        covered_votes: dict[str, int] = {}
        total_seen: dict[str, int] = {}

        for cov in coverages:
            for item_id in cov.required_covered:
                covered_votes[item_id] = covered_votes.get(item_id, 0) + 1
                total_seen[item_id] = total_seen.get(item_id, 0) + 1
            for item_id in cov.required_missing:
                total_seen[item_id] = total_seen.get(item_id, 0) + 1
                # Not incrementing covered_votes => defaults to 0

        # 🎯 Apply majority vote
        merged_covered: list[str] = []
        merged_missing: list[str] = []
        conflicts: list[str] = []

        for item_id in sorted(total_seen.keys()):
            votes = covered_votes.get(item_id, 0)
            if votes >= threshold:
                merged_covered.append(item_id)
            else:
                merged_missing.append(item_id)
            # 🔍 Detect conflict: not unanimous
            if 0 < votes < n:
                conflicts.append(item_id)

        logger.info(
            "✅ Checklist coverage merged: %d covered, %d missing, "
            "%d conflicts (from %d models, threshold=%d)",
            len(merged_covered),
            len(merged_missing),
            len(conflicts),
            n,
            threshold,
        )

        return (
            ChecklistCoverage(
                required_covered=merged_covered,
                required_missing=merged_missing,
            ),
            conflicts,
        )

    def _build_individual_results(
        self,
        results: list[tuple[str, EvaluationResult]],
    ) -> list[dict[str, Any]]:
        """Build individual result dicts for audit trail 📋.

        Args:
            results: List of (model_name, result) tuples.

        Returns:
            List of dicts with model_name and serialised result.
        """
        individual: list[dict[str, Any]] = []
        for model_name, result in results:
            individual.append(
                {
                    "model_name": model_name,
                    "decision": result.decision.value,
                    "confidence": result.confidence,
                    "evidence_count": len(result.evidence_index),
                    "search_rounds": result.search_rounds,
                    "round_logs": result.round_logs,
                    "checklist_coverage": result.checklist_coverage.model_dump(),
                    "gaps_remaining": result.gaps_remaining,
                    "doubts_remaining": result.doubts_remaining,
                }
            )
        return individual
