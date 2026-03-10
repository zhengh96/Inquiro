"""Tests for SkillRegistry wiring into DISCOVERY pipeline 🧪.

Verifies that:
- InquiroAgentBase accepts skill_registry and registers SkillTool.
- SearchAgent passes skill_registry through to InquiroAgentBase.
- SearchExp stores skill_registry and passes it to SearchAgent.

Uses mocks for all external dependencies — no real LLM or MCP calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evomaster.agent.agent import AgentConfig
from evomaster.agent.tools.base import ToolRegistry
from inquiro.tests.mock_helpers import (
    MockLLM,
    make_finish_response,
)


# ============================================================
# 🧰 Shared test fixtures
# ============================================================

_SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["raw_evidence", "total_collected", "search_gaps"],
    "properties": {
        "raw_evidence": {"type": "array", "items": {"type": "object"}},
        "total_collected": {"type": "integer", "minimum": 0},
        "search_gaps": {"type": "array", "items": {"type": "string"}},
    },
}


@pytest.fixture
def mock_tools() -> ToolRegistry:
    """Empty ToolRegistry for agent initialization 🔧."""
    return ToolRegistry()


@pytest.fixture
def mock_llm() -> MockLLM:
    """MockLLM that immediately finishes with a valid search result 🤖."""
    return MockLLM(
        responses=[
            make_finish_response(
                {
                    "raw_evidence": [],
                    "total_collected": 0,
                    "search_gaps": [],
                }
            )
        ]
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Minimal AgentConfig for testing 🔧."""
    return AgentConfig(max_turns=1)


# ============================================================
# 🧪 InquiroAgentBase skill_registry wiring
# ============================================================


class TestInquiroAgentBaseSkillRegistry:
    """Tests that InquiroAgentBase correctly wires SkillRegistry 🎯."""

    def test_accepts_skill_registry_none(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """Should accept skill_registry=None without errors ✅."""
        from inquiro.agents.search_agent import SearchAgent

        agent = SearchAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_prompt="test system",
            user_prompt="test user",
            config=agent_config,
            output_schema=_SEARCH_OUTPUT_SCHEMA,
            skill_registry=None,
        )
        # 🎯 skill_registry stored as None
        assert agent._skill_registry is None

    def test_stores_skill_registry(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """Should store skill_registry on self._skill_registry ✅."""
        from inquiro.agents.search_agent import SearchAgent

        mock_registry = MagicMock()
        agent = SearchAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_prompt="test system",
            user_prompt="test user",
            config=agent_config,
            output_schema=_SEARCH_OUTPUT_SCHEMA,
            skill_registry=mock_registry,
        )
        # 🎯 skill_registry correctly stored
        assert agent._skill_registry is mock_registry

    def test_registers_skill_tool_when_registry_present(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """Should register SkillTool when skill_registry is provided ✅."""
        from inquiro.agents.search_agent import SearchAgent

        mock_registry = MagicMock()
        mock_skill_tool = MagicMock()
        mock_skill_tool.name = "get_reference"

        with patch(
            "evomaster.agent.tools.skill.SkillTool",
            return_value=mock_skill_tool,
        ) as mock_skill_tool_cls:
            agent = SearchAgent(
                llm=mock_llm,
                tools=mock_tools,
                system_prompt="test system",
                user_prompt="test user",
                config=agent_config,
                output_schema=_SEARCH_OUTPUT_SCHEMA,
                skill_registry=mock_registry,
            )
            # 🎯 SkillTool constructor called with the registry
            mock_skill_tool_cls.assert_called_once_with(
                skill_registry=mock_registry
            )
            # 🎯 SkillTool registered in the agent's tool registry
            registered_names = {
                t.name for t in agent.tools.get_all_tools()
            }
            assert "get_reference" in registered_names

    def test_no_skill_tool_without_registry(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """Should NOT register SkillTool when skill_registry is None ✅."""
        from inquiro.agents.search_agent import SearchAgent

        with patch(
            "evomaster.agent.tools.skill.SkillTool"
        ) as mock_skill_tool_cls:
            SearchAgent(
                llm=mock_llm,
                tools=mock_tools,
                system_prompt="test system",
                user_prompt="test user",
                config=agent_config,
                output_schema=_SEARCH_OUTPUT_SCHEMA,
                skill_registry=None,
            )
            # 🎯 SkillTool constructor should NOT be called
            mock_skill_tool_cls.assert_not_called()


# ============================================================
# 🧪 SearchAgent skill_registry pass-through
# ============================================================


class TestSearchAgentSkillRegistry:
    """Tests that SearchAgent correctly passes skill_registry to base 🎯."""

    def test_pass_through_to_base(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """skill_registry should propagate from SearchAgent to base ✅."""
        from inquiro.agents.search_agent import SearchAgent

        mock_registry = MagicMock()
        agent = SearchAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_prompt="test system",
            user_prompt="test user",
            config=agent_config,
            output_schema=_SEARCH_OUTPUT_SCHEMA,
            skill_registry=mock_registry,
        )
        # 🎯 Verify the base class stored it via InquiroAgentBase.__init__
        assert agent._skill_registry is mock_registry

    def test_default_none_when_not_provided(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        agent_config: AgentConfig,
    ) -> None:
        """skill_registry defaults to None when omitted ✅."""
        from inquiro.agents.search_agent import SearchAgent

        agent = SearchAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_prompt="test system",
            user_prompt="test user",
            config=agent_config,
            output_schema=_SEARCH_OUTPUT_SCHEMA,
        )
        assert agent._skill_registry is None


# ============================================================
# 🧪 SearchExp skill_registry storage and injection
# ============================================================


class TestSearchExpSkillRegistry:
    """Tests that SearchExp stores and injects skill_registry ✅."""

    def test_stores_skill_registry(self) -> None:
        """SearchExp should store skill_registry on self._skill_registry ✅."""
        from inquiro.exps.search_exp import SearchExp

        mock_registry = MagicMock()
        exp = SearchExp(
            llm=MagicMock(),
            tools=MagicMock(),
            skill_registry=mock_registry,
        )
        assert exp._skill_registry is mock_registry

    def test_defaults_to_none(self) -> None:
        """SearchExp._skill_registry should default to None ✅."""
        from inquiro.exps.search_exp import SearchExp

        exp = SearchExp(
            llm=MagicMock(),
            tools=MagicMock(),
        )
        assert exp._skill_registry is None

    def test_creates_agent_with_skill_registry(self) -> None:
        """_create_search_agent should pass skill_registry to SearchAgent ✅."""
        from inquiro.exps.search_exp import SearchExp
        from inquiro.core.types import EvaluationTask

        mock_registry = MagicMock()
        exp = SearchExp(
            llm=MagicMock(),
            tools=ToolRegistry(),
            skill_registry=mock_registry,
        )

        task = EvaluationTask(
            task_id="test-wiring",
            topic="Test topic",
            rules="Test rules",
            output_schema={"type": "object", "properties": {}},
        )

        with patch("inquiro.agents.search_agent.SearchAgent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            exp._create_search_agent(
                task=task,
                system_prompt="sys",
                user_prompt="usr",
            )
            # 🎯 skill_registry must be passed to SearchAgent constructor
            # After create_subset() filtering, the passed registry is
            # the subset returned by mock_registry.create_subset()
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert "skill_registry" in call_kwargs
            mock_registry.create_subset.assert_called_once()
            assert call_kwargs["skill_registry"] is mock_registry.create_subset.return_value

    def test_creates_agent_without_skill_registry(self) -> None:
        """_create_search_agent passes None skill_registry when not set ✅."""
        from inquiro.exps.search_exp import SearchExp
        from inquiro.core.types import EvaluationTask

        exp = SearchExp(
            llm=MagicMock(),
            tools=ToolRegistry(),
        )

        task = EvaluationTask(
            task_id="test-wiring-none",
            topic="Test topic",
            rules="Test rules",
            output_schema={"type": "object", "properties": {}},
        )

        with patch("inquiro.agents.search_agent.SearchAgent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            exp._create_search_agent(
                task=task,
                system_prompt="sys",
                user_prompt="usr",
            )
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert call_kwargs.get("skill_registry") is None
