"""Tests for GapAnalysis follow-up rules injection 🧪.

Verifies that follow_up_rules from EvaluationTask.query_strategy are
correctly injected into focus prompt generators:
- DefaultFocusPromptGenerator without rules: no "FOLLOW-UP GUIDANCE" section
- DefaultFocusPromptGenerator with rules: "FOLLOW-UP GUIDANCE" section present
- TrajectoryAwareFocusGenerator with rules: rules appear in base output
- DiscoveryLoop.run() re-creates focus generator when task has follow_up_rules
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from inquiro.core.discovery_loop import (
    DefaultFocusPromptGenerator,
    DiscoveryLoop,
    MockAnalysisExecutor,
    MockSearchExecutor,
    TrajectoryAwareFocusGenerator,
)
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    DiscoveryConfig,
    Evidence,
    EvaluationTask,
    FocusPrompt,
    GapReport,
)


# ============================================================================
# 🔧 Test helpers
# ============================================================================


def _make_gap_report(
    uncovered: list[str] | None = None,
    covered: list[str] | None = None,
    round_number: int = 1,
) -> GapReport:
    """Create a minimal GapReport for testing 🔧.

    Args:
        uncovered: Uncovered checklist item descriptions.
        covered: Already-covered checklist item descriptions.
        round_number: Round number to associate with the report.

    Returns:
        GapReport instance.
    """
    return GapReport(
        round_number=round_number,
        coverage_ratio=0.5,
        covered_items=covered if covered is not None else ["covered_item"],
        uncovered_items=uncovered if uncovered is not None else ["uncovered_A"],
        conflict_signals=[],
        converged=False,
    )


def _make_config(**kwargs: Any) -> DiscoveryConfig:
    """Create a DiscoveryConfig with test-friendly defaults 🔧.

    Args:
        **kwargs: Override any DiscoveryConfig field.

    Returns:
        DiscoveryConfig instance.
    """
    defaults: dict[str, Any] = {
        "max_rounds": 2,
        "max_cost_per_subitem": 10.0,
        "coverage_threshold": 0.80,
        "convergence_delta": 0.05,
        "min_evidence_per_round": 1,
        "timeout_per_round": 60,
        "timeout_total": 300,
    }
    defaults.update(kwargs)
    return DiscoveryConfig(**defaults)


def _make_task(
    task_id: str = "task-followup-001",
    query_strategy: dict[str, Any] | None = None,
) -> EvaluationTask:
    """Create a test EvaluationTask with an optional query_strategy 🔧.

    Args:
        task_id: Unique task identifier.
        query_strategy: Optional query strategy dict; may contain
            a "follow_up_rules" key with markdown guidance text.

    Returns:
        EvaluationTask instance.
    """
    checklist = Checklist(
        required=[
            ChecklistItem(id="C1", description="Assess binding affinity data"),
            ChecklistItem(id="C2", description="Evaluate safety profile"),
        ]
    )
    return EvaluationTask(
        task_id=task_id,
        topic="Test topic for follow-up rules injection",
        rules="Evaluate all checklist items thoroughly.",
        checklist=checklist,
        output_schema={},
        query_strategy=query_strategy,
    )


def _make_evidence(
    eid: str = "E1",
    summary: str = (
        "This is a sufficiently long test evidence summary about "
        "protein interactions and binding affinities in cell cultures"
    ),
) -> Evidence:
    """Create a test Evidence instance with valid length 🔧.

    Args:
        eid: Evidence identifier.
        summary: Evidence summary (must be > 50 chars).

    Returns:
        Evidence instance.
    """
    return Evidence(
        id=eid,
        source="test-mcp",
        query="test query",
        summary=summary,
    )


# ============================================================================
# 🧪 DefaultFocusPromptGenerator — follow_up_rules parameter
# ============================================================================


class TestDefaultFocusPromptGeneratorFollowUpRules:
    """Tests for DefaultFocusPromptGenerator follow_up_rules injection 🧪."""

    @pytest.mark.asyncio
    async def test_without_follow_up_rules_no_section(self) -> None:
        """Focus prompt does NOT contain 'FOLLOW-UP GUIDANCE' when empty 🚫."""
        generator = DefaultFocusPromptGenerator()  # default: no follow_up_rules
        config = _make_config()
        gap_report = _make_gap_report(
            uncovered=["missing clinical data"],
            covered=["binding affinity"],
        )

        result = await generator.generate_focus(gap_report, config, 1)

        assert isinstance(result, FocusPrompt)
        assert "FOLLOW-UP GUIDANCE" not in result.prompt_text

    @pytest.mark.asyncio
    async def test_with_follow_up_rules_section_present(self) -> None:
        """Focus prompt CONTAINS 'FOLLOW-UP GUIDANCE' when rules supplied ✅."""
        rules = "## Gap-Closing Strategy\n- Use alternative MeSH terms\n- Target grey literature"
        generator = DefaultFocusPromptGenerator(follow_up_rules=rules)
        config = _make_config()
        gap_report = _make_gap_report(
            uncovered=["missing clinical data"],
            covered=["binding affinity"],
        )

        result = await generator.generate_focus(gap_report, config, 1)

        assert isinstance(result, FocusPrompt)
        assert "FOLLOW-UP GUIDANCE" in result.prompt_text
        assert "from query template" in result.prompt_text
        assert "Use alternative MeSH terms" in result.prompt_text
        assert "Target grey literature" in result.prompt_text

    @pytest.mark.asyncio
    async def test_follow_up_rules_appear_after_other_sections(self) -> None:
        """Follow-up guidance appears after exclusion section 📋."""
        rules = "UNIQUE_FOLLOW_UP_MARKER"
        generator = DefaultFocusPromptGenerator(follow_up_rules=rules)
        config = _make_config()
        gap_report = _make_gap_report(
            uncovered=["gap_one"],
            covered=["covered_item"],
        )

        result = await generator.generate_focus(gap_report, config, 1)

        # 📝 Verify ordering: "ALREADY COVERED" section must appear before
        # the follow-up rules section
        pos_covered = result.prompt_text.find("ALREADY COVERED")
        pos_followup = result.prompt_text.find("FOLLOW-UP GUIDANCE")

        assert pos_covered != -1, "Expected 'ALREADY COVERED' section missing"
        assert pos_followup != -1, "Expected 'FOLLOW-UP GUIDANCE' section missing"
        assert pos_covered < pos_followup, (
            "Follow-up guidance must appear after the exclusion section"
        )

    @pytest.mark.asyncio
    async def test_follow_up_rules_empty_string_no_section(self) -> None:
        """Explicit empty string behaves identically to default (no section) 🚫."""
        generator = DefaultFocusPromptGenerator(follow_up_rules="")
        config = _make_config()
        gap_report = _make_gap_report(uncovered=["gap_one"])

        result = await generator.generate_focus(gap_report, config, 1)

        assert "FOLLOW-UP GUIDANCE" not in result.prompt_text

    @pytest.mark.asyncio
    async def test_broadening_prompt_does_not_include_follow_up_rules(
        self,
    ) -> None:
        """Broadening prompt path (all covered) skips follow-up section 🔍."""
        rules = "SHOULD_NOT_APPEAR_IN_BROADENING"
        generator = DefaultFocusPromptGenerator(follow_up_rules=rules)
        config = _make_config()
        # 📝 Empty uncovered → broadening path
        gap_report = _make_gap_report(
            uncovered=[],
            covered=["item_a", "item_b"],
        )

        result = await generator.generate_focus(gap_report, config, 1)

        # 📝 Broadening path returns early without calling _build_focus_prompt
        assert "FOLLOW-UP GUIDANCE" not in result.prompt_text
        assert "Broaden search scope" in result.prompt_text

    def test_call_count_increments_normally(self) -> None:
        """call_count attribute still increments as expected 📊."""
        generator = DefaultFocusPromptGenerator(follow_up_rules="some rules")
        assert generator.call_count == 0


# ============================================================================
# 🧪 TrajectoryAwareFocusGenerator — follow_up_rules forwarding
# ============================================================================


class TestTrajectoryAwareFocusGeneratorFollowUpRules:
    """Tests for TrajectoryAwareFocusGenerator follow_up_rules forwarding 🧪."""

    @pytest.mark.asyncio
    async def test_follow_up_rules_appear_in_output(self) -> None:
        """Rules forwarded to base generator appear in final prompt 📋."""
        rules = "## Additional Search Strategy\n- Prefer systematic reviews"
        mock_provider = MagicMock()
        # 📝 Return empty feedback so only base output is produced
        from inquiro.core.trajectory.feedback import FeedbackResult

        mock_provider.get_feedback.return_value = FeedbackResult(gap_hints=[])

        generator = TrajectoryAwareFocusGenerator(
            feedback_provider=mock_provider,
            follow_up_rules=rules,
        )
        config = _make_config()
        gap_report = _make_gap_report(
            uncovered=["gap_item"],
            covered=["covered_item"],
        )

        result = await generator.generate_focus(gap_report, config, 1)

        assert isinstance(result, FocusPrompt)
        assert "FOLLOW-UP GUIDANCE" in result.prompt_text
        assert "Prefer systematic reviews" in result.prompt_text

    @pytest.mark.asyncio
    async def test_no_follow_up_rules_by_default(self) -> None:
        """TrajectoryAwareFocusGenerator defaults to no follow-up section 🚫."""
        mock_provider = MagicMock()
        from inquiro.core.trajectory.feedback import FeedbackResult

        mock_provider.get_feedback.return_value = FeedbackResult(gap_hints=[])

        generator = TrajectoryAwareFocusGenerator(
            feedback_provider=mock_provider,
        )
        config = _make_config()
        gap_report = _make_gap_report(uncovered=["gap_item"])

        result = await generator.generate_focus(gap_report, config, 1)

        assert "FOLLOW-UP GUIDANCE" not in result.prompt_text

    def test_base_generator_receives_follow_up_rules(self) -> None:
        """Internal base generator stores the forwarded follow_up_rules 🔧."""
        rules = "MY_TEST_RULES"
        mock_provider = MagicMock()

        generator = TrajectoryAwareFocusGenerator(
            feedback_provider=mock_provider,
            follow_up_rules=rules,
        )

        # 📝 Verify the base generator has the rules stored
        assert generator._base_generator._follow_up_rules == rules


# ============================================================================
# 🧪 DiscoveryLoop.run() — follow_up_rules injection via query_strategy
# ============================================================================


class TestDiscoveryLoopFollowUpRulesInjection:
    """Tests for DiscoveryLoop.run() follow_up_rules injection 🧪."""

    @pytest.mark.asyncio
    async def test_focus_generator_recreated_with_follow_up_rules(
        self,
    ) -> None:
        """When task.query_strategy has follow_up_rules, generator is re-created ✅."""
        follow_up_rules_text = "DISTINCT_FOLLOW_UP_RULES_TEXT_XYZ"
        task = _make_task(query_strategy={"follow_up_rules": follow_up_rules_text})
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        evidence = [_make_evidence()]
        claims = [
            {
                "claim": "Binding affinity confirmed in binding assay",
                "evidence_ids": ["E1"],
            },
            {
                "claim": "Safety profile evaluated in toxicology study",
                "evidence_ids": ["E1"],
            },
        ]
        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )

        # 🔍 Before run(), the generator has no follow_up_rules
        assert isinstance(loop.focus_generator, DefaultFocusPromptGenerator)
        assert loop.focus_generator._follow_up_rules == ""

        await loop.run(task, config)

        # 📋 After run(), the generator has been re-created with the rules
        assert isinstance(loop.focus_generator, DefaultFocusPromptGenerator)
        assert loop.focus_generator._follow_up_rules == follow_up_rules_text

    @pytest.mark.asyncio
    async def test_no_injection_when_query_strategy_absent(self) -> None:
        """Without query_strategy, generator is NOT re-created 🚫."""
        task = _make_task(query_strategy=None)
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )
        original_generator = loop.focus_generator

        await loop.run(task, config)

        # 📝 Same instance, not re-created
        assert loop.focus_generator is original_generator

    @pytest.mark.asyncio
    async def test_no_injection_when_follow_up_rules_empty(self) -> None:
        """Empty follow_up_rules string skips injection 🚫."""
        task = _make_task(query_strategy={"follow_up_rules": ""})
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )
        original_generator = loop.focus_generator

        await loop.run(task, config)

        # 📝 Same instance, not re-created (empty string is falsy)
        assert loop.focus_generator is original_generator

    @pytest.mark.asyncio
    async def test_trajectory_aware_generator_recreated_with_follow_up_rules(
        self,
    ) -> None:
        """TrajectoryAwareFocusGenerator is also re-created with rules ✅."""
        follow_up_rules_text = "TRAJECTORY_FOLLOW_UP_MARKER"
        task = _make_task(query_strategy={"follow_up_rules": follow_up_rules_text})
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        evidence = [_make_evidence()]
        claims: list[dict[str, Any]] = []
        search = MockSearchExecutor(mock_evidence=evidence)
        analysis = MockAnalysisExecutor(mock_claims=claims)

        mock_feedback = MagicMock()
        from inquiro.core.trajectory.feedback import FeedbackResult

        mock_feedback.get_feedback.return_value = FeedbackResult(gap_hints=[])

        # 🔄 Providing feedback_provider triggers auto-upgrade to
        # TrajectoryAwareFocusGenerator inside __init__
        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
            feedback_provider=mock_feedback,
        )

        assert isinstance(loop.focus_generator, TrajectoryAwareFocusGenerator)

        await loop.run(task, config)

        # 📋 After run(), still a TrajectoryAwareFocusGenerator, but re-created
        assert isinstance(loop.focus_generator, TrajectoryAwareFocusGenerator)
        assert (
            loop.focus_generator._base_generator._follow_up_rules
            == follow_up_rules_text
        )

    @pytest.mark.asyncio
    async def test_query_strategy_missing_follow_up_rules_key(self) -> None:
        """query_strategy without follow_up_rules key skips injection 🚫."""
        task = _make_task(query_strategy={"other_key": "other_value"})
        config = _make_config(max_rounds=1, coverage_threshold=0.0)

        search = MockSearchExecutor(mock_evidence=[_make_evidence()])
        analysis = MockAnalysisExecutor(mock_claims=[])

        loop = DiscoveryLoop(
            search_executor=search,
            analysis_executor=analysis,
        )
        original_generator = loop.focus_generator

        await loop.run(task, config)

        # 📝 Same instance — no follow_up_rules key means no injection
        assert loop.focus_generator is original_generator
