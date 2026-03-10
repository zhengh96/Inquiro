"""GapAnalysis — checklist coverage and convergence analysis 🎯.

Provides hybrid gap analysis for the DISCOVERY pipeline:
1. **LLM coverage judgment** — lightweight LLM call to assess which
   checklist items are covered by the current claims/evidence pool.
2. **Deterministic convergence check** — rule-based logic that decides
   whether the discovery loop should stop iterating.

The module is domain-agnostic: it knows nothing about pharma, targets,
or dimensions. All domain context arrives via the checklist and claims.

Example::

    gap = GapAnalysis()
    report = await gap.analyze(
        checklist=["Item A", "Item B", "Item C"],
        claims=[{"claim": "A is true", "evidence_ids": ["E1"]}],
        evidence=[{"id": "E1", "summary": "..."}],
        previous_coverage=0.0,
        round_number=1,
        config=DiscoveryConfig(),
        cost_spent=0.10,
    )
    print(report.converged, report.convergence_reason)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from inquiro.core.types import DiscoveryConfig, GapReport

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = [
    "GapAnalysis",
    "StopReason",
    "CoverageJudge",
    "MockCoverageJudge",
]


# ============================================================================
# 🏷️ Stop Reason Enum
# ============================================================================


class StopReason(str, Enum):
    """Reason the discovery loop should stop iterating 🛑.

    Each value maps to one of the five deterministic convergence
    conditions checked by GapAnalysis.
    """

    CONVERGED = "coverage_threshold_reached"
    BUDGET_EXHAUSTED = "max_cost_per_subitem_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    DIMINISHING_RETURNS = "diminishing_returns"
    SEARCH_EXHAUSTED = "search_exhausted"


# ============================================================================
# 🤖 Coverage Judge Protocol + Mock
# ============================================================================


class CoverageJudge(Protocol):
    """Protocol for LLM-based checklist coverage assessment 🤖.

    Implementations take a checklist and the current claims/evidence,
    then return which items are covered vs. uncovered.  The protocol
    allows swapping between mock and real LLM implementations.
    """

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Assess which checklist items are covered by claims 🔍.

        Args:
            checklist: Checklist item descriptions to assess.
            claims: Current analysis claims with evidence references.
            evidence: Current evidence pool items.

        Returns:
            CoverageResult with covered/uncovered item lists.
        """
        ...


class CoverageResult(BaseModel):
    """Result of LLM coverage judgment 📊.

    Holds the partitioned checklist items after the LLM (or mock)
    decides which are covered and which remain uncovered.
    """

    covered: list[str] = Field(
        default_factory=list,
        description="Checklist items deemed covered by existing evidence",
    )
    uncovered: list[str] = Field(
        default_factory=list,
        description="Checklist items lacking sufficient evidence",
    )
    conflict_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Items with contradictory evidence signals detected via "
            "keyword-level heuristic or LLM-based coverage judgment."
        ),
    )
    judge_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost of this coverage judgment call in USD.",
    )


class MockCoverageJudge:
    """Mock LLM coverage judge for testing and development 🧪.

    Uses simple keyword matching between checklist items and
    claim text to simulate LLM coverage judgment.  Each checklist
    item is marked as "covered" if any claim text contains at
    least one word from the checklist item (case-insensitive,
    words longer than 3 characters).

    Also detects conflict signals when claims about the same
    checklist item contain both positive and negative indicators,
    using a keyword-level heuristic.

    This is intentionally naive — real LLM judgment will be
    semantically richer.  Replace with an LLMCoverageJudge
    backed by LLMProviderPool when infrastructure is ready.
    """

    # 📝 Minimum word length to use for keyword matching
    _MIN_KEYWORD_LENGTH = 4

    # ⚡ Heuristic keywords for positive/negative signal detection
    _POSITIVE_KEYWORDS = frozenset(
        {
            "effective",
            "significant",
            "positive",
            "confirmed",
            "demonstrated",
            "supports",
            "approved",
            "successful",
            "promising",
            "validated",
            "beneficial",
        }
    )
    _NEGATIVE_KEYWORDS = frozenset(
        {
            "ineffective",
            "insignificant",
            "negative",
            "failed",
            "contradicted",
            "refuted",
            "rejected",
            "unsuccessful",
            "adverse",
            "disproven",
            "harmful",
            "no effect",
            "no evidence",
        }
    )

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Assess coverage via keyword matching heuristic 🔍.

        For each checklist item, extracts significant words (length >= 4)
        and checks if any claim's text contains at least one keyword.
        Also detects conflict signals when claims about the same item
        contain both positive and negative indicators.

        Args:
            checklist: Checklist item descriptions to assess.
            claims: Current analysis claims with evidence references.
            evidence: Current evidence pool items (unused in mock).

        Returns:
            CoverageResult with keyword-matched coverage assessment
            and conflict signals.
        """
        if not checklist:
            return CoverageResult(covered=[], uncovered=[])

        covered: list[str] = []
        uncovered: list[str] = []
        conflict_signals: list[str] = []

        for item in checklist:
            matching_claims = self._find_matching_claims(item, claims)

            if matching_claims:
                covered.append(item)
                # 🔍 Detect conflicting signals in matching claims
                if self._has_conflicting_signals(matching_claims):
                    conflict_signals.append(item)
            else:
                uncovered.append(item)

        logger.debug(
            "🧪 MockCoverageJudge: %d/%d items covered, "
            "%d conflict signals",
            len(covered),
            len(checklist),
            len(conflict_signals),
        )
        return CoverageResult(
            covered=covered,
            uncovered=uncovered,
            conflict_signals=conflict_signals,
        )

    def _find_matching_claims(
        self,
        item: str,
        claims: list[dict[str, Any]],
    ) -> list[str]:
        """Find claim texts that match a checklist item 🔎.

        Args:
            item: Single checklist item description.
            claims: List of claim dicts with 'claim' key.

        Returns:
            List of matching claim text strings.
        """
        keywords = [
            word.lower()
            for word in item.split()
            if len(word) >= self._MIN_KEYWORD_LENGTH
        ]
        if not keywords:
            return []

        matching: list[str] = []
        for claim in claims:
            claim_text = claim.get("claim", "")
            if not isinstance(claim_text, str):
                continue
            claim_lower = claim_text.lower()
            if any(kw in claim_lower for kw in keywords):
                matching.append(claim_lower)
        return matching

    def _has_conflicting_signals(
        self,
        claim_texts: list[str],
    ) -> bool:
        """Detect conflicting positive/negative signals in claims ⚡.

        Returns True when the matching claims contain both positive
        and negative indicator keywords.

        Args:
            claim_texts: Lowercased claim text strings.

        Returns:
            True if both positive and negative signals are present.
        """
        combined = " ".join(claim_texts)
        has_positive = any(kw in combined for kw in self._POSITIVE_KEYWORDS)
        has_negative = any(kw in combined for kw in self._NEGATIVE_KEYWORDS)
        return has_positive and has_negative

    def _build_claims_text(self, claims: list[dict[str, Any]]) -> str:
        """Concatenate all claim text into a single searchable string 📝.

        Args:
            claims: List of claim dicts, each may have a 'claim' key.

        Returns:
            Concatenated text from all claims.
        """
        parts: list[str] = []
        for claim in claims:
            claim_text = claim.get("claim", "")
            if isinstance(claim_text, str):
                parts.append(claim_text)
        return " ".join(parts)

    def _item_is_covered(
        self,
        item: str,
        combined_claims_text: str,
    ) -> bool:
        """Check if a checklist item has keyword overlap with claims 🔎.

        Args:
            item: Single checklist item description.
            combined_claims_text: Lowercased concatenation of all claims.

        Returns:
            True if at least one significant keyword from the item
            appears in the combined claims text.
        """
        keywords = [
            word.lower()
            for word in item.split()
            if len(word) >= self._MIN_KEYWORD_LENGTH
        ]
        if not keywords:
            # ⚠️ No significant keywords — conservatively mark uncovered
            return False
        return any(kw in combined_claims_text for kw in keywords)


# ============================================================================
# 🎯 GapAnalysis
# ============================================================================


class GapAnalysis:
    """Analyze checklist coverage and determine convergence 🎯.

    Hybrid approach: an LLM-based CoverageJudge assesses which checklist
    items are covered, then deterministic code checks five convergence
    conditions to decide whether the loop should stop.

    The five stopping conditions (checked in order):
        1. ``coverage_threshold`` reached → CONVERGED
        2. ``max_cost_per_subitem`` exhausted → BUDGET_EXHAUSTED
        3. ``max_rounds`` reached → MAX_ROUNDS_REACHED
        4. ``convergence_delta`` too small for ``convergence_patience``
           consecutive rounds → DIMINISHING_RETURNS
        5. ``min_evidence_per_round`` not met → SEARCH_EXHAUSTED

    Attributes:
        coverage_judge: Pluggable component for coverage assessment.
    """

    def __init__(
        self,
        coverage_judge: CoverageJudge | None = None,
        coverage_judge_mode: str = "always",
    ) -> None:
        """Initialize GapAnalysis with optional coverage judge 🔧.

        Args:
            coverage_judge: LLM or mock judge for coverage assessment.
                Defaults to MockCoverageJudge if not provided.
            coverage_judge_mode: Controls how pre-computed coverage is used.
                ``"always"`` always calls the coverage judge (ignores
                pre_computed_coverage).  ``"fallback"`` uses pre-computed
                coverage when available and skips the judge call.
                Defaults to ``"always"``.
        """
        self.coverage_judge: CoverageJudge = coverage_judge or MockCoverageJudge()
        self.coverage_judge_mode: str = coverage_judge_mode
        # 📈 Track consecutive rounds with below-threshold coverage delta
        self._consecutive_low_delta: int = 0

    async def analyze(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
        previous_coverage: float,
        round_number: int,
        config: DiscoveryConfig,
        cost_spent: float,
        pre_computed_coverage: CoverageResult | None = None,
    ) -> GapReport:
        """Analyze gaps and convergence for one round 🔍.

        Runs LLM coverage judgment, computes coverage ratio, then
        checks the five deterministic convergence conditions in order.

        If ``pre_computed_coverage`` is provided (from analysis agent
        multi-model voting), it is used directly and the coverage_judge
        is skipped.  This avoids redundant keyword-matching when the
        analysis agent has already produced semantic coverage data.

        Args:
            checklist: Checklist item descriptions to assess.
            claims: Current analysis claims with evidence references.
            evidence: Current evidence pool items.
            previous_coverage: Coverage ratio from the previous round
                (0.0 for the first round).
            round_number: Current round number (1-based).
            config: Discovery configuration with thresholds.
            cost_spent: Total cost already spent in USD.
            pre_computed_coverage: Optional pre-computed coverage from
                analysis agent.  When provided, ``coverage_judge`` is
                bypassed entirely.

        Returns:
            GapReport with coverage assessment and convergence status.

        Raises:
            ValueError: If round_number < 1.
        """
        if round_number < 1:
            raise ValueError(f"round_number must be >= 1, got {round_number}")

        # 🤖 Step 1: Use pre-computed coverage or call the judge
        if self.coverage_judge_mode == "fallback" and pre_computed_coverage is not None:
            coverage_result = pre_computed_coverage
            logger.info(
                "✅ Using pre-computed coverage from analysis agent "
                "[mode=fallback] (covered=%d, uncovered=%d, conflicts=%d)",
                len(coverage_result.covered),
                len(coverage_result.uncovered),
                len(coverage_result.conflict_signals),
            )
        else:
            if pre_computed_coverage is not None:
                logger.info(
                    "🔄 Ignoring pre-computed coverage in 'always' mode — "
                    "calling coverage judge for authoritative assessment",
                )
            coverage_result = await self.coverage_judge.judge_coverage(
                checklist=checklist,
                claims=claims,
                evidence=evidence,
            )

        # 📊 Step 2: Compute coverage ratio
        coverage_ratio = self._compute_coverage_ratio(
            checklist=checklist,
            covered=coverage_result.covered,
        )

        logger.info(
            "📊 GapAnalysis round=%d: coverage=%.2f (prev=%.2f), "
            "covered=%d/%d, cost=$%.2f",
            round_number,
            coverage_ratio,
            previous_coverage,
            len(coverage_result.covered),
            len(checklist),
            cost_spent,
        )

        # 🛑 Step 3: Check convergence conditions
        converged, reason = self._check_convergence(
            coverage_ratio=coverage_ratio,
            previous_coverage=previous_coverage,
            round_number=round_number,
            evidence_count=len(evidence),
            config=config,
            cost_spent=cost_spent,
        )

        if converged:
            logger.info(
                "🛑 GapAnalysis converged: %s (round=%d)",
                reason,
                round_number,
            )

        return GapReport(
            round_number=round_number,
            coverage_ratio=coverage_ratio,
            covered_items=coverage_result.covered,
            uncovered_items=coverage_result.uncovered,
            conflict_signals=coverage_result.conflict_signals,
            converged=converged,
            convergence_reason=reason,
            judge_cost_usd=coverage_result.judge_cost_usd,
        )

    def _compute_coverage_ratio(
        self,
        checklist: list[str],
        covered: list[str],
    ) -> float:
        """Compute the fraction of checklist items covered 📊.

        Args:
            checklist: Full list of checklist item descriptions.
            covered: Items judged as covered.

        Returns:
            Coverage ratio in [0.0, 1.0]. Returns 1.0 if checklist
            is empty (vacuously true).
        """
        if not checklist:
            return 1.0
        # 🔧 Clamp to [0, 1] in case of duplicate items
        ratio = len(covered) / len(checklist)
        return min(ratio, 1.0)

    def _check_convergence(
        self,
        coverage_ratio: float,
        previous_coverage: float,
        round_number: int,
        evidence_count: int,
        config: DiscoveryConfig,
        cost_spent: float,
    ) -> tuple[bool, str | None]:
        """Check the five deterministic convergence conditions 🛑.

        Conditions are checked in priority order. The first matching
        condition determines the stop reason.  Condition 4 (diminishing
        returns) respects ``config.convergence_patience``: the loop only
        stops after that many *consecutive* low-delta rounds, preventing
        premature termination from a single unproductive round.

        Args:
            coverage_ratio: Current coverage ratio.
            previous_coverage: Previous round's coverage ratio.
            round_number: Current round number (1-based).
            evidence_count: Number of evidence items in the current pool.
            config: Discovery configuration with thresholds.
            cost_spent: Total cost spent so far in USD.

        Returns:
            Tuple of (converged: bool, reason: str | None).
            reason is None when converged is False.
        """
        # 1️⃣ Coverage threshold reached
        if coverage_ratio >= config.coverage_threshold:
            self._consecutive_low_delta = 0
            return True, StopReason.CONVERGED.value

        # 2️⃣ Budget exhausted
        if cost_spent >= config.max_cost_per_subitem:
            return True, StopReason.BUDGET_EXHAUSTED.value

        # 3️⃣ Max rounds reached
        if round_number >= config.max_rounds:
            return True, StopReason.MAX_ROUNDS_REACHED.value

        # 4️⃣ Diminishing returns (skip on first round — no previous data)
        if round_number > 1:
            delta = coverage_ratio - previous_coverage
            if delta < config.convergence_delta:
                self._consecutive_low_delta += 1
                logger.debug(
                    "📉 Low delta=%.4f (threshold=%.4f), consecutive_low_delta=%d/%d",
                    delta,
                    config.convergence_delta,
                    self._consecutive_low_delta,
                    config.convergence_patience,
                )
                if self._consecutive_low_delta >= config.convergence_patience:
                    return True, StopReason.DIMINISHING_RETURNS.value
            else:
                # 🔄 Delta above threshold — reset the patience counter
                self._consecutive_low_delta = 0

        # 5️⃣ Search exhausted (not enough new evidence)
        if round_number > 1 and evidence_count < config.min_evidence_per_round:
            return True, StopReason.SEARCH_EXHAUSTED.value

        return False, None
