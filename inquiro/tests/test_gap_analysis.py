"""Tests for GapAnalysis — coverage judgment and convergence logic 🧪.

Covers:
- Each of the five convergence conditions independently
- Coverage ratio calculation edge cases
- MockCoverageJudge keyword matching behavior
- Empty/degenerate inputs (empty checklist, empty claims, first round)
- GapReport field correctness

Each test is independent and follows Arrange → Act → Assert.
"""

from __future__ import annotations

from typing import Any

import pytest

from inquiro.core.gap_analysis import (
    CoverageResult,
    GapAnalysis,
    MockCoverageJudge,
    StopReason,
)
from inquiro.core.types import DiscoveryConfig


# ============================================================================
# 🏗️ Helpers
# ============================================================================


def _default_config(**overrides: Any) -> DiscoveryConfig:
    """Build a DiscoveryConfig with optional overrides 🔧.

    Args:
        **overrides: Fields to override on the default config.

    Returns:
        DiscoveryConfig with defaults merged with overrides.
    """
    defaults: dict[str, Any] = {
        "max_rounds": 3,
        "max_cost_per_subitem": 8.0,
        "coverage_threshold": 0.80,
        "convergence_delta": 0.05,
        "min_evidence_per_round": 3,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


def _make_claims(*texts: str) -> list[dict[str, Any]]:
    """Build claim dicts from text strings 📝.

    Args:
        *texts: Claim text strings.

    Returns:
        List of claim dicts with 'claim' and 'evidence_ids' keys.
    """
    return [
        {"claim": text, "evidence_ids": [f"E{i + 1}"]} for i, text in enumerate(texts)
    ]


def _make_evidence(count: int) -> list[dict[str, Any]]:
    """Build a list of minimal evidence items 📄.

    Args:
        count: Number of evidence items to create.

    Returns:
        List of evidence dicts with 'id' and 'summary' keys.
    """
    return [
        {"id": f"E{i + 1}", "summary": f"Evidence item {i + 1}"} for i in range(count)
    ]


# ============================================================================
# 📊 Test Class: Coverage Ratio Calculation
# ============================================================================


class TestCoverageRatio:
    """Tests for coverage ratio computation 📊."""

    async def test_empty_checklist_returns_full_coverage(self) -> None:
        """Empty checklist is vacuously fully covered (ratio=1.0) ✅."""
        # Arrange
        gap = GapAnalysis()
        config = _default_config()

        # Act
        report = await gap.analyze(
            checklist=[],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.coverage_ratio == 1.0
        assert report.converged is True
        assert report.convergence_reason == StopReason.CONVERGED.value

    async def test_full_coverage_with_all_items_matched(self) -> None:
        """All checklist items covered yields coverage=1.0 ✅."""
        # Arrange
        checklist = ["market size analysis", "growth forecast"]
        claims = _make_claims(
            "The market size is $20B globally",
            "Growth forecast shows 8% CAGR",
        )
        gap = GapAnalysis()
        config = _default_config(coverage_threshold=0.80)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(2),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.coverage_ratio == 1.0
        assert report.converged is True

    async def test_partial_coverage_calculation(self) -> None:
        """Only some checklist items covered yields correct ratio 📊."""
        # Arrange — only the first item has keyword overlap
        checklist = [
            "market size analysis",
            "regulatory landscape overview",
            "competitive positioning",
        ]
        claims = _make_claims("The market size is very large")
        gap = GapAnalysis()
        config = _default_config(coverage_threshold=0.80, max_rounds=5)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert — 1 of 3 items covered
        assert report.coverage_ratio == pytest.approx(1 / 3, abs=0.01)
        assert len(report.covered_items) == 1
        assert len(report.uncovered_items) == 2

    async def test_zero_coverage_with_no_claims(self) -> None:
        """No claims yields zero coverage 📊."""
        # Arrange
        checklist = ["item one", "item two"]
        gap = GapAnalysis()
        config = _default_config(max_rounds=5)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.coverage_ratio == 0.0
        assert len(report.covered_items) == 0
        assert len(report.uncovered_items) == 2


# ============================================================================
# 🛑 Test Class: Convergence Conditions
# ============================================================================


class TestConvergenceConditions:
    """Tests for the five deterministic convergence conditions 🛑."""

    async def test_condition1_coverage_threshold_reached(self) -> None:
        """Coverage >= threshold → CONVERGED (condition 1) ✅."""
        # Arrange — all items covered, threshold 0.80
        checklist = ["market analysis", "growth forecast"]
        claims = _make_claims(
            "market analysis shows $20B",
            "growth forecast at 8% CAGR",
        )
        gap = GapAnalysis()
        config = _default_config(coverage_threshold=0.80)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.CONVERGED.value

    async def test_condition1_exact_threshold_converges(self) -> None:
        """Coverage exactly at threshold → CONVERGED (boundary) ✅."""
        # Arrange — 4 of 5 items covered → 0.80 = threshold
        checklist = [
            "aaa analysis",
            "bbb forecast",
            "ccc review",
            "ddd evaluation",
            "eee uncovered",
        ]
        claims = _make_claims(
            "aaa analysis is complete",
            "bbb forecast shows growth",
            "ccc review is positive",
            "ddd evaluation confirms",
        )
        gap = GapAnalysis()
        config = _default_config(coverage_threshold=0.80)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert — 4/5 = 0.80 >= 0.80 → converged
        assert report.coverage_ratio == pytest.approx(0.80)
        assert report.converged is True
        assert report.convergence_reason == StopReason.CONVERGED.value

    async def test_condition2_budget_exhausted(self) -> None:
        """Cost >= max_cost_per_subitem → BUDGET_EXHAUSTED (condition 2) 💰."""
        # Arrange — low coverage but budget is spent
        checklist = ["item that wont match anything at all"]
        gap = GapAnalysis()
        config = _default_config(max_cost_per_subitem=5.0, max_rounds=10)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(10),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=5.0,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.BUDGET_EXHAUSTED.value

    async def test_condition2_budget_slightly_over(self) -> None:
        """Cost slightly over budget → BUDGET_EXHAUSTED 💰."""
        # Arrange
        checklist = ["unmatchable checklist item here"]
        gap = GapAnalysis()
        config = _default_config(max_cost_per_subitem=2.0, max_rounds=10)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(10),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=2.01,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.BUDGET_EXHAUSTED.value

    async def test_condition3_max_rounds_reached(self) -> None:
        """round_number >= max_rounds → MAX_ROUNDS_REACHED (condition 3) 🔄."""
        # Arrange — low coverage, under budget, but at max rounds
        checklist = ["unmatchable checklist item here"]
        gap = GapAnalysis()
        config = _default_config(max_rounds=3)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(10),
            previous_coverage=0.0,
            round_number=3,
            config=config,
            cost_spent=1.0,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.MAX_ROUNDS_REACHED.value

    async def test_condition4_diminishing_returns(self) -> None:
        """Coverage delta < convergence_delta → DIMINISHING_RETURNS (condition 4) 📉."""
        # Arrange — round 2, coverage improved by only 0.01 (< 0.05 delta)
        checklist = [
            "market analysis item",
            "growth forecast item",
            "regulatory landscape item",
        ]
        claims = _make_claims("market analysis is complete")
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.90,
            convergence_delta=0.05,
            max_rounds=10,
        )

        # Act — previous=0.30, current≈0.33 → delta=0.03 < 0.05
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(10),
            previous_coverage=0.30,
            round_number=2,
            config=config,
            cost_spent=1.0,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.DIMINISHING_RETURNS.value

    async def test_condition4_skipped_on_first_round(self) -> None:
        """Diminishing returns is NOT checked on round 1 🔍."""
        # Arrange — round 1, coverage is low but no previous data
        checklist = ["unmatchable checklist item here"]
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.90,
            convergence_delta=0.05,
            max_rounds=10,
        )

        # Act — round 1, delta from 0.0 to 0.0 would be 0 < 0.05
        # but should NOT trigger diminishing returns on round 1
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(10),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert — should NOT have converged (round 1 skips condition 4 & 5)
        assert report.converged is False

    async def test_condition5_search_exhausted(self) -> None:
        """Too few evidence items → SEARCH_EXHAUSTED (condition 5) 🔍."""
        # Arrange — round 2, enough coverage improvement but too few evidence
        checklist = ["market analysis item", "growth forecast item"]
        claims = _make_claims("market analysis is done")
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.90,
            convergence_delta=0.01,
            min_evidence_per_round=3,
            max_rounds=10,
        )

        # Act — only 2 evidence items, min is 3
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(2),
            previous_coverage=0.0,
            round_number=2,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.converged is True
        assert report.convergence_reason == StopReason.SEARCH_EXHAUSTED.value

    async def test_no_convergence_when_all_conditions_pass(self) -> None:
        """Loop continues when no stopping condition is met 🟢."""
        # Arrange — round 1, partial coverage, under budget, under max rounds
        checklist = ["market analysis item", "growth forecast item"]
        claims = _make_claims("market analysis is done")
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.90,
            max_rounds=5,
            max_cost_per_subitem=10.0,
        )

        # Act — 1/2 covered = 0.50 < 0.90, within budget, round 1
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=1.0,
        )

        # Assert
        assert report.converged is False
        assert report.convergence_reason is None


# ============================================================================
# 🏷️ Test Class: Convergence Priority Order
# ============================================================================


class TestConvergencePriority:
    """Tests verifying convergence conditions are checked in priority order 🏷️."""

    async def test_coverage_wins_over_budget(self) -> None:
        """Condition 1 (coverage) takes priority over condition 2 (budget) 🏆."""
        # Arrange — both coverage met AND budget exhausted
        checklist = ["market analysis"]
        claims = _make_claims("market analysis is complete")
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.80,
            max_cost_per_subitem=1.0,
        )

        # Act — coverage=1.0 >= 0.80 AND cost=5.0 >= 1.0
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=5.0,
        )

        # Assert — coverage wins (checked first)
        assert report.converged is True
        assert report.convergence_reason == StopReason.CONVERGED.value

    async def test_budget_wins_over_max_rounds(self) -> None:
        """Condition 2 (budget) takes priority over condition 3 (rounds) 🏆."""
        # Arrange — budget exhausted AND max rounds reached
        checklist = ["unmatchable checklist item here"]
        gap = GapAnalysis()
        config = _default_config(
            max_cost_per_subitem=1.0,
            max_rounds=2,
        )

        # Act — cost=1.0 >= 1.0 AND round=2 >= 2
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=2,
            config=config,
            cost_spent=1.0,
        )

        # Assert — budget wins (checked before max_rounds)
        assert report.converged is True
        assert report.convergence_reason == StopReason.BUDGET_EXHAUSTED.value


# ============================================================================
# 🧪 Test Class: MockCoverageJudge
# ============================================================================


class TestMockCoverageJudge:
    """Tests for the MockCoverageJudge keyword matching 🧪."""

    async def test_keyword_matching_covers_item(self) -> None:
        """Items with matching keywords in claims are covered ✅."""
        # Arrange
        judge = MockCoverageJudge()
        checklist = ["market analysis report"]
        claims = [{"claim": "The market analysis shows strong growth"}]

        # Act
        result = await judge.judge_coverage(checklist, claims, [])

        # Assert — "market" and "analysis" appear in claim
        assert "market analysis report" in result.covered

    async def test_no_matching_keywords(self) -> None:
        """Items without matching keywords remain uncovered ❌."""
        # Arrange
        judge = MockCoverageJudge()
        checklist = ["competitive landscape overview"]
        claims = [{"claim": "The market is growing rapidly"}]

        # Act
        result = await judge.judge_coverage(checklist, claims, [])

        # Assert — no keywords from checklist appear in claims
        assert "competitive landscape overview" in result.uncovered

    async def test_short_words_ignored(self) -> None:
        """Words shorter than 4 characters are ignored in matching ⚠️."""
        # Arrange — checklist has only short words
        judge = MockCoverageJudge()
        checklist = ["a is on"]
        claims = [{"claim": "a is on the table"}]

        # Act
        result = await judge.judge_coverage(checklist, claims, [])

        # Assert — no significant keywords → uncovered
        assert "a is on" in result.uncovered

    async def test_case_insensitive_matching(self) -> None:
        """Keyword matching is case-insensitive 🔤."""
        # Arrange
        judge = MockCoverageJudge()
        checklist = ["GLOBAL Market Overview"]
        claims = [{"claim": "the global market is worth billions"}]

        # Act
        result = await judge.judge_coverage(checklist, claims, [])

        # Assert — "global" and "market" match case-insensitively
        assert "GLOBAL Market Overview" in result.covered

    async def test_empty_checklist_returns_empty(self) -> None:
        """Empty checklist yields empty covered and uncovered lists 📭."""
        # Arrange
        judge = MockCoverageJudge()

        # Act
        result = await judge.judge_coverage([], [{"claim": "data"}], [])

        # Assert
        assert result.covered == []
        assert result.uncovered == []

    async def test_empty_claims_yields_zero_coverage(self) -> None:
        """No claims means nothing is covered 📭."""
        # Arrange
        judge = MockCoverageJudge()
        checklist = ["important analysis item"]

        # Act
        result = await judge.judge_coverage(checklist, [], [])

        # Assert
        assert len(result.covered) == 0
        assert len(result.uncovered) == 1


# ============================================================================
# 📋 Test Class: GapReport Structure
# ============================================================================


class TestGapReportStructure:
    """Tests verifying GapReport fields are populated correctly 📋."""

    async def test_report_contains_round_number(self) -> None:
        """GapReport.round_number matches the input round 📊."""
        # Arrange
        gap = GapAnalysis()
        config = _default_config(max_rounds=10)

        # Act
        report = await gap.analyze(
            checklist=["some item here"],
            claims=[],
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=3,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.round_number == 3

    async def test_report_lists_covered_and_uncovered(self) -> None:
        """GapReport lists both covered and uncovered items 📋."""
        # Arrange
        checklist = ["market analysis item", "unknown thing here"]
        claims = _make_claims("market analysis is complete")
        gap = GapAnalysis()
        config = _default_config(coverage_threshold=0.99, max_rounds=10)

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert "market analysis item" in report.covered_items
        assert "unknown thing here" in report.uncovered_items

    async def test_convergence_reason_none_when_not_converged(self) -> None:
        """convergence_reason is None when loop should continue 📋."""
        # Arrange
        gap = GapAnalysis()
        config = _default_config(
            coverage_threshold=0.99,
            max_rounds=10,
        )

        # Act
        report = await gap.analyze(
            checklist=["unmatchable item here"],
            claims=[],
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert
        assert report.converged is False
        assert report.convergence_reason is None


# ============================================================================
# ⚠️ Test Class: Edge Cases and Validation
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and input validation ⚠️."""

    async def test_invalid_round_number_raises(self) -> None:
        """round_number < 1 raises ValueError ❌."""
        # Arrange
        gap = GapAnalysis()
        config = _default_config()

        # Act & Assert
        with pytest.raises(ValueError, match="round_number must be >= 1"):
            await gap.analyze(
                checklist=["item"],
                claims=[],
                evidence=[],
                previous_coverage=0.0,
                round_number=0,
                config=config,
                cost_spent=0.0,
            )

    async def test_claims_with_missing_claim_key(self) -> None:
        """Claims without 'claim' key are handled gracefully 🔧."""
        # Arrange — claims missing the 'claim' key
        checklist = ["market analysis item"]
        claims = [{"evidence_ids": ["E1"]}, {"other_key": "value"}]
        gap = GapAnalysis()
        config = _default_config(max_rounds=10)

        # Act — should not raise
        report = await gap.analyze(
            checklist=checklist,
            claims=claims,
            evidence=_make_evidence(5),
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert — no keywords matched, all uncovered
        assert report.coverage_ratio == 0.0

    async def test_coverage_result_defaults(self) -> None:
        """CoverageResult defaults to empty lists 📭."""
        # Act
        result = CoverageResult()

        # Assert
        assert result.covered == []
        assert result.uncovered == []
        assert result.conflict_signals == []

    async def test_stop_reason_enum_values(self) -> None:
        """StopReason enum values are descriptive strings 🏷️."""
        # Assert
        assert StopReason.CONVERGED.value == "coverage_threshold_reached"
        assert StopReason.BUDGET_EXHAUSTED.value == ("max_cost_per_subitem_exhausted")
        assert StopReason.MAX_ROUNDS_REACHED.value == "max_rounds_reached"
        assert StopReason.DIMINISHING_RETURNS.value == "diminishing_returns"
        assert StopReason.SEARCH_EXHAUSTED.value == "search_exhausted"


# ============================================================================
# 🔌 Test Class: Custom CoverageJudge
# ============================================================================


class TestCustomCoverageJudge:
    """Tests verifying pluggable CoverageJudge support 🔌."""

    async def test_custom_judge_is_used(self) -> None:
        """GapAnalysis uses the injected CoverageJudge instead of mock 🔌."""

        # Arrange — custom judge that always returns full coverage
        class AlwaysCoveredJudge:
            async def judge_coverage(
                self,
                checklist: list[str],
                claims: list[dict[str, Any]],
                evidence: list[Any],
            ) -> CoverageResult:
                return CoverageResult(
                    covered=list(checklist),
                    uncovered=[],
                )

        gap = GapAnalysis(coverage_judge=AlwaysCoveredJudge())
        config = _default_config(coverage_threshold=0.80)
        checklist = ["item A", "item B", "item C"]

        # Act
        report = await gap.analyze(
            checklist=checklist,
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=config,
            cost_spent=0.0,
        )

        # Assert — custom judge says everything is covered
        assert report.coverage_ratio == 1.0
        assert report.converged is True
        assert report.convergence_reason == StopReason.CONVERGED.value
        assert report.covered_items == checklist
        assert report.uncovered_items == []


# ============================================================================
# ⚡ MockCoverageJudge Conflict Signals Tests
# ============================================================================


class TestMockCoverageJudgeConflictSignals:
    """Tests for conflict signal detection in MockCoverageJudge ⚡."""

    @pytest.mark.asyncio
    async def test_conflict_detected_with_positive_and_negative(
        self,
    ) -> None:
        """Conflict signal when claims have both positive and negative keywords ⚡."""
        judge = MockCoverageJudge()
        checklist = ["Drug efficacy results"]
        claims = _make_claims(
            "Drug efficacy shows significant positive results",
            "Drug efficacy was ineffective in secondary endpoints",
        )
        result = await judge.judge_coverage(checklist, claims, [])
        assert "Drug efficacy results" in result.covered
        assert "Drug efficacy results" in result.conflict_signals

    @pytest.mark.asyncio
    async def test_no_conflict_with_only_positive_claims(self) -> None:
        """No conflict signal when all claims are positive ✅."""
        judge = MockCoverageJudge()
        checklist = ["Binding affinity data"]
        claims = _make_claims(
            "Binding affinity was confirmed and validated",
        )
        result = await judge.judge_coverage(checklist, claims, [])
        assert "Binding affinity data" in result.covered
        assert result.conflict_signals == []

    @pytest.mark.asyncio
    async def test_no_conflict_with_only_negative_claims(self) -> None:
        """No conflict signal when all claims are negative ❌."""
        judge = MockCoverageJudge()
        checklist = ["Treatment outcomes"]
        claims = _make_claims(
            "Treatment outcomes were rejected and failed",
        )
        result = await judge.judge_coverage(checklist, claims, [])
        assert "Treatment outcomes" in result.covered
        assert result.conflict_signals == []

    @pytest.mark.asyncio
    async def test_no_conflict_with_neutral_claims(self) -> None:
        """No conflict signal when claims lack polarity keywords 🔹."""
        judge = MockCoverageJudge()
        checklist = ["Pharmacokinetic profile"]
        claims = _make_claims(
            "Pharmacokinetic profile was measured in plasma",
        )
        result = await judge.judge_coverage(checklist, claims, [])
        assert "Pharmacokinetic profile" in result.covered
        assert result.conflict_signals == []

    @pytest.mark.asyncio
    async def test_uncovered_item_not_in_conflict_signals(self) -> None:
        """Uncovered items should never appear in conflict signals 🔹."""
        judge = MockCoverageJudge()
        checklist = ["Completely unrelated topic"]
        claims = _make_claims(
            "Something positive and something negative here",
        )
        result = await judge.judge_coverage(checklist, claims, [])
        assert "Completely unrelated topic" in result.uncovered
        assert result.conflict_signals == []
