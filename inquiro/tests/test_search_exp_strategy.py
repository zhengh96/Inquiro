"""Tests for SearchExp query-strategy injection 🧪.

Validates that:
- PromptSectionBuilder.format_query_strategy handles all edge cases
- SearchExp._render_system_prompt injects alias_expansion and
  query_section_guide from a populated query_strategy dict
- Backward compatibility is preserved when query_strategy is None
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


from inquiro.prompts.section_builder import PromptSectionBuilder


# ============================================================================
# 🧪 PromptSectionBuilder.format_query_strategy unit tests
# ============================================================================


class TestFormatQueryStrategy:
    """Tests for PromptSectionBuilder.format_query_strategy 🔬."""

    def test_none_returns_empty_strings(self) -> None:
        """None strategy MUST return ('', '') 🚫."""
        alias, guide = PromptSectionBuilder.format_query_strategy(None)
        assert alias == ""
        assert guide == ""

    def test_empty_dict_returns_empty_strings(self) -> None:
        """Empty dict strategy MUST return ('', '') 🚫."""
        alias, guide = PromptSectionBuilder.format_query_strategy({})
        assert alias == ""
        assert guide == ""

    def test_alias_expansion_returned_when_no_sections(self) -> None:
        """alias_expansion MUST be returned even when query_sections is empty 📋."""
        strategy: dict[str, Any] = {
            "alias_expansion": "EGFR aliases: ErbB1, HER1, ...",
            "query_sections": [],
        }
        alias, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert alias == "EGFR aliases: ErbB1, HER1, ..."
        assert guide == ""

    def test_alias_expansion_without_sections_key(self) -> None:
        """alias_expansion returned even when query_sections key is absent 📋."""
        strategy: dict[str, Any] = {
            "alias_expansion": "Target aliases: FOO, BAR",
        }
        alias, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert alias == "Target aliases: FOO, BAR"
        assert guide == ""

    def test_full_strategy_returns_alias_and_guide(self) -> None:
        """Full strategy MUST produce non-empty alias and guide 📋."""
        strategy: dict[str, Any] = {
            "alias_expansion": "EGFR: ErbB1, HER1",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "bohrium",
                    "description": "Literature review",
                    "content": "Search for genetic evidence studies.",
                },
            ],
            "tool_allocations": [
                {"tool_name": "bohrium", "percentage": 60},
                {"tool_name": "perplexity", "percentage": 40},
            ],
        }
        alias, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert alias == "EGFR: ErbB1, HER1"
        assert "Query Section Guide" in guide
        assert "Q1" in guide
        assert "Priority 1" in guide
        assert "bohrium" in guide

    def test_query_section_guide_header_present(self) -> None:
        """Guide MUST include the standard header text 📝."""
        strategy: dict[str, Any] = {
            "alias_expansion": "",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "perplexity",
                    "description": "",
                    "content": "Search body content.",
                },
            ],
            "tool_allocations": [],
        }
        _, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert "Query Section Guide (from template)" in guide
        assert "SHOULD execute at least one query per section" in guide

    def test_tool_allocation_table_rendered(self) -> None:
        """Tool allocation table MUST appear when allocations are present 📊."""
        strategy: dict[str, Any] = {
            "alias_expansion": "",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "bohrium",
                    "description": "",
                    "content": "Query content",
                },
            ],
            "tool_allocations": [
                {"tool_name": "bohrium", "percentage": 70},
                {"tool_name": "perplexity", "percentage": 30},
            ],
        }
        _, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert "Tool Allocation Guidance" in guide
        assert "70%" in guide
        assert "30%" in guide

    def test_no_tool_allocations_no_table(self) -> None:
        """Guide MUST NOT include allocation table when list is empty 🚫."""
        strategy: dict[str, Any] = {
            "alias_expansion": "",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "perplexity",
                    "description": "",
                    "content": "Query content",
                },
            ],
            "tool_allocations": [],
        }
        _, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert "Tool Allocation Guidance" not in guide

    def test_multiple_sections_all_rendered(self) -> None:
        """All query sections MUST appear in the guide 📋."""
        strategy: dict[str, Any] = {
            "alias_expansion": "",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "bohrium",
                    "description": "First",
                    "content": "Body one",
                },
                {
                    "id": "Q2",
                    "priority": 2,
                    "tool_name": "perplexity",
                    "description": "Second",
                    "content": "Body two",
                },
                {
                    "id": "Q3",
                    "priority": 3,
                    "tool_name": "brave",
                    "description": "",
                    "content": "Body three",
                },
            ],
            "tool_allocations": [],
        }
        _, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert "Q1" in guide
        assert "Q2" in guide
        assert "Q3" in guide
        assert "Body one" in guide
        assert "Body two" in guide
        assert "Body three" in guide

    def test_section_description_rendered_when_present(self) -> None:
        """Non-empty description MUST appear in the guide 📝."""
        strategy: dict[str, Any] = {
            "alias_expansion": "",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "bohrium",
                    "description": "end-to-end feasibility synthesis",
                    "content": "Search content",
                },
            ],
            "tool_allocations": [],
        }
        _, guide = PromptSectionBuilder.format_query_strategy(strategy)
        assert "end-to-end feasibility synthesis" in guide

    def test_returns_tuple_type(self) -> None:
        """Return value MUST be a 2-tuple of strings 🔑."""
        result = PromptSectionBuilder.format_query_strategy(None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ============================================================================
# 🧪 SearchExp integration tests
# ============================================================================


class TestSearchExpStrategyInjection:
    """Tests for SearchExp._render_system_prompt strategy injection 🔍."""

    def _build_search_exp(self) -> Any:
        """Create a minimal SearchExp for prompt rendering tests 🔧.

        Returns:
            SearchExp instance with mocked LLM and tools.
        """
        from inquiro.exps.search_exp import SearchExp

        llm = MagicMock()
        tools = MagicMock()
        tools.__len__ = MagicMock(return_value=0)
        return SearchExp(llm=llm, tools=tools)

    def _build_task_with_strategy(
        self,
        query_strategy: dict[str, Any] | None,
    ) -> Any:
        """Build an EvaluationTask with the given query_strategy 🔬.

        Args:
            query_strategy: Query strategy dict or None.

        Returns:
            EvaluationTask instance.
        """
        from inquiro.tests.mock_helpers import build_sample_evaluation_task

        task = build_sample_evaluation_task()
        # 🔧 Inject query_strategy directly (bypasses Pydantic validation)
        object.__setattr__(task, "query_strategy", query_strategy)
        return task

    def test_render_system_prompt_with_none_strategy_succeeds(self) -> None:
        """Prompt rendering MUST succeed when query_strategy is None 📝."""
        exp = self._build_search_exp()
        task = build_task_no_strategy()
        # Should not raise
        prompt = exp._render_system_prompt(task)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_render_system_prompt_with_strategy_injects_alias(
        self,
    ) -> None:
        """alias_expansion from strategy MUST appear in rendered prompt 📋."""
        exp = self._build_search_exp()
        task = build_task_with_query_strategy()
        prompt = exp._render_system_prompt(task)
        assert "EGFR aliases test" in prompt

    def test_render_system_prompt_with_strategy_injects_guide(
        self,
    ) -> None:
        """query_section_guide MUST appear in rendered prompt when sections present 📋."""
        exp = self._build_search_exp()
        task = build_task_with_query_strategy()
        prompt = exp._render_system_prompt(task)
        assert "Query Section Guide" in prompt
        assert "Q1" in prompt

    def test_render_user_prompt_with_none_strategy_succeeds(self) -> None:
        """User prompt rendering MUST succeed when query_strategy is None 📝."""
        exp = self._build_search_exp()
        task = build_task_no_strategy()
        prompt = exp._render_user_prompt(task, round_number=1)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_render_user_prompt_with_strategy_injects_alias(
        self,
    ) -> None:
        """alias_expansion from strategy MUST appear in user prompt 📋."""
        exp = self._build_search_exp()
        task = build_task_with_query_strategy()
        prompt = exp._render_user_prompt(task, round_number=1)
        assert "EGFR aliases test" in prompt


# ============================================================================
# 🏭 Helpers for task construction
# ============================================================================


def build_task_no_strategy() -> Any:
    """Build a task with query_strategy=None 🔧.

    Returns:
        EvaluationTask with no query strategy.
    """
    from inquiro.tests.mock_helpers import build_sample_evaluation_task

    return build_sample_evaluation_task()


def build_task_with_query_strategy() -> Any:
    """Build a task with a populated query_strategy dict 🔧.

    Returns:
        EvaluationTask with query_strategy containing sections and alias.
    """
    from inquiro.tests.mock_helpers import build_sample_evaluation_task

    task = build_sample_evaluation_task()
    strategy: dict[str, Any] = {
        "sub_item_id": "genetic_evidence",
        "alias_expansion": "EGFR aliases test",
        "query_sections": [
            {
                "id": "Q1",
                "priority": 1,
                "tool_name": "bohrium",
                "description": "Literature review",
                "content": "Search for EGFR genetic evidence in academic papers.",
            },
        ],
        "tool_allocations": [
            {"tool_name": "bohrium", "percentage": 60},
        ],
        "follow_up_rules": "",
        "evidence_tiers": "",
    }
    # 🔧 Return new task with query_strategy set via model_copy
    return task.model_copy(update={"query_strategy": strategy})
