"""Tests for evolution enrichment injection in Exp classes 🧬.

Verifies that SearchExp, AnalysisExp, and DiscoverySynthesisExp
correctly post-append evolution enrichment text to their rendered
system prompts.

Tests cover:
    - SearchExp with enrichment appends text to system prompt
    - SearchExp without enrichment produces normal (unchanged) prompt
    - AnalysisExp with enrichment appends text to system prompt
    - DiscoverySynthesisExp with enrichment appends text to system prompt
    - Enrichment text appears at the END of the prompt (post-appended)
"""

from __future__ import annotations


from inquiro.exps.analysis_exp import AnalysisExp
from inquiro.exps.discovery_synthesis_exp import DiscoverySynthesisExp
from inquiro.exps.search_exp import SearchExp
from inquiro.tests.mock_helpers import (
    MockLLM,
    build_sample_evaluation_task,
)


# ============================================================================
# 🧬 Sample enrichment text
# ============================================================================

_SAMPLE_ENRICHMENT = (
    "# LEARNED INSIGHTS\n"
    "- Kinase targets: focus on selectivity data\n"
    "- Check Phase III trial results for safety signals"
)


# ============================================================================
# 🏭 Helpers
# ============================================================================


def _build_search_exp(
    enrichment: str | None = None,
) -> SearchExp:
    """Build a SearchExp with optional enrichment 🔍.

    Args:
        enrichment: Evolution enrichment text or None.

    Returns:
        Configured SearchExp instance.
    """
    from evomaster.agent.tools.base import ToolRegistry

    return SearchExp(
        llm=MockLLM(),
        tools=ToolRegistry(),
        evolution_enrichment=enrichment,
    )


def _build_analysis_exp(
    enrichment: str | None = None,
) -> AnalysisExp:
    """Build an AnalysisExp with optional enrichment 🔬.

    Args:
        enrichment: Evolution enrichment text or None.

    Returns:
        Configured AnalysisExp instance.
    """
    task = build_sample_evaluation_task()
    return AnalysisExp(
        task=task,
        llm=MockLLM(),
        evolution_enrichment=enrichment,
    )


def _build_synthesis_exp(
    enrichment: str | None = None,
) -> DiscoverySynthesisExp:
    """Build a DiscoverySynthesisExp with optional enrichment 🧬.

    Args:
        enrichment: Evolution enrichment text or None.

    Returns:
        Configured DiscoverySynthesisExp instance.
    """
    task = build_sample_evaluation_task()
    return DiscoverySynthesisExp(
        task=task,
        llm=MockLLM(),
        evolution_enrichment=enrichment,
    )


# ============================================================================
# 🔍 SearchExp enrichment tests
# ============================================================================


class TestSearchExpEnrichment:
    """Tests for evolution enrichment in SearchExp 🔍."""

    def test_with_enrichment_appends_to_system_prompt(self) -> None:
        """SearchExp with enrichment appends text to system prompt 🧬."""
        exp = _build_search_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert _SAMPLE_ENRICHMENT in prompt
        assert "LEARNED INSIGHTS" in prompt

    def test_without_enrichment_no_change(self) -> None:
        """SearchExp without enrichment produces normal prompt 📝."""
        exp_with = _build_search_exp(enrichment=_SAMPLE_ENRICHMENT)
        exp_without = _build_search_exp(enrichment=None)
        task = build_sample_evaluation_task()

        prompt_with = exp_with._render_system_prompt(task)
        prompt_without = exp_without._render_system_prompt(task)

        # 🧬 Enrichment version must contain the extra text
        assert _SAMPLE_ENRICHMENT in prompt_with
        # 📝 Non-enrichment version must NOT contain the extra text
        assert _SAMPLE_ENRICHMENT not in prompt_without
        # ✅ The base prompt content should be present in both
        assert len(prompt_without) > 0
        assert prompt_without in prompt_with

    def test_enrichment_appears_at_end(self) -> None:
        """Enrichment text appears at the END of the prompt 📌."""
        exp = _build_search_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert prompt.endswith(_SAMPLE_ENRICHMENT)

    def test_empty_string_enrichment_not_appended(self) -> None:
        """Empty string enrichment is treated as falsy (not appended) ⚡."""
        exp = _build_search_exp(enrichment="")
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        # 📝 Empty enrichment should not add trailing newlines
        exp_none = _build_search_exp(enrichment=None)
        prompt_none = exp_none._render_system_prompt(task)
        assert prompt == prompt_none


# ============================================================================
# 🔬 AnalysisExp enrichment tests
# ============================================================================


class TestAnalysisExpEnrichment:
    """Tests for evolution enrichment in AnalysisExp 🔬."""

    def test_with_enrichment_appends_to_system_prompt(self) -> None:
        """AnalysisExp with enrichment appends text to system prompt 🧬."""
        exp = _build_analysis_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert _SAMPLE_ENRICHMENT in prompt
        assert "LEARNED INSIGHTS" in prompt

    def test_without_enrichment_no_change(self) -> None:
        """AnalysisExp without enrichment produces normal prompt 📝."""
        exp_with = _build_analysis_exp(enrichment=_SAMPLE_ENRICHMENT)
        exp_without = _build_analysis_exp(enrichment=None)
        task = build_sample_evaluation_task()

        prompt_with = exp_with._render_system_prompt(task)
        prompt_without = exp_without._render_system_prompt(task)

        assert _SAMPLE_ENRICHMENT in prompt_with
        assert _SAMPLE_ENRICHMENT not in prompt_without
        assert prompt_without in prompt_with

    def test_enrichment_appears_at_end(self) -> None:
        """Enrichment text appears at the END of the prompt 📌."""
        exp = _build_analysis_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert prompt.endswith(_SAMPLE_ENRICHMENT)


# ============================================================================
# 🧬 DiscoverySynthesisExp enrichment tests
# ============================================================================


class TestDiscoverySynthesisExpEnrichment:
    """Tests for evolution enrichment in DiscoverySynthesisExp 🧬."""

    def test_with_enrichment_appends_to_system_prompt(self) -> None:
        """DiscoverySynthesisExp with enrichment appends text 🧬."""
        exp = _build_synthesis_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert _SAMPLE_ENRICHMENT in prompt
        assert "LEARNED INSIGHTS" in prompt

    def test_without_enrichment_no_change(self) -> None:
        """DiscoverySynthesisExp without enrichment is unchanged 📝."""
        exp_with = _build_synthesis_exp(enrichment=_SAMPLE_ENRICHMENT)
        exp_without = _build_synthesis_exp(enrichment=None)
        task = build_sample_evaluation_task()

        prompt_with = exp_with._render_system_prompt(task)
        prompt_without = exp_without._render_system_prompt(task)

        assert _SAMPLE_ENRICHMENT in prompt_with
        assert _SAMPLE_ENRICHMENT not in prompt_without
        assert prompt_without in prompt_with

    def test_enrichment_appears_at_end(self) -> None:
        """Enrichment text appears at the END of the prompt 📌."""
        exp = _build_synthesis_exp(enrichment=_SAMPLE_ENRICHMENT)
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task)

        assert prompt.endswith(_SAMPLE_ENRICHMENT)
