"""Unit tests for SearchAgent 🔍.

Tests initialization, prompt rendering, evidence collection,
MCP tool handling, and public accessor methods.

Uses MockLLM and MagicMock for all external dependencies —
no real LLM or MCP calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from evomaster.agent.agent import AgentConfig
from evomaster.agent.tools.base import ToolRegistry
from inquiro.agents.base import InquiroAgentBase
from inquiro.agents.search_agent import SearchAgent
from inquiro.prompts.loader import PromptLoader
from inquiro.tests.mock_helpers import (
    MockLLM,
    create_tool_call,
    make_finish_response,
)


# ============================================================
# 🏭 Fixtures
# ============================================================


SAMPLE_SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["raw_evidence", "total_collected", "search_gaps"],
    "properties": {
        "raw_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "mcp_server", "source_query"],
                "properties": {
                    "id": {"type": "string"},
                    "mcp_server": {"type": "string"},
                    "source_query": {"type": "string"},
                    "observation": {"type": "string"},
                    "url": {"type": ["string", "null"]},
                },
            },
        },
        "total_collected": {"type": "integer", "minimum": 0},
        "search_gaps": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@pytest.fixture
def sample_output_schema() -> dict[str, Any]:
    """Search output JSON Schema for testing 📋."""
    return SAMPLE_SEARCH_OUTPUT_SCHEMA.copy()


@pytest.fixture
def mock_tools() -> ToolRegistry:
    """Empty ToolRegistry for agent initialization 🔧."""
    return ToolRegistry()


@pytest.fixture
def mock_llm() -> MockLLM:
    """MockLLM that immediately finishes 🤖."""
    return MockLLM(
        responses=[
            make_finish_response(
                {
                    "raw_evidence": [],
                    "total_collected": 0,
                    "search_gaps": [],
                }
            ),
        ]
    )


@pytest.fixture
def sample_system_prompt() -> str:
    """Pre-rendered system prompt for testing 📝."""
    return (
        "# AGENT IDENTITY\n\n"
        "You are a search execution specialist.\n\n"
        "# OUTPUT FORMAT\n\n"
        "Submit raw evidence via finish tool."
    )


@pytest.fixture
def sample_user_prompt() -> str:
    """Pre-rendered user prompt for testing 📝."""
    return (
        "Execute a comprehensive evidence search on: Test Topic\n\n"
        "Collect ALL raw evidence from every available search tool."
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Agent configuration for testing ⚙️."""
    return AgentConfig(max_turns=10)


@pytest.fixture
def search_agent(
    mock_llm: MockLLM,
    mock_tools: ToolRegistry,
    sample_system_prompt: str,
    sample_user_prompt: str,
    agent_config: AgentConfig,
    sample_output_schema: dict[str, Any],
) -> SearchAgent:
    """Fully constructed SearchAgent for testing 🔍."""
    return SearchAgent(
        llm=mock_llm,
        tools=mock_tools,
        system_prompt=sample_system_prompt,
        user_prompt=sample_user_prompt,
        config=agent_config,
        output_schema=sample_output_schema,
        task_id="test-search-001",
    )


# ============================================================
# 🧪 Test: Initialization
# ============================================================


class TestSearchAgentInit:
    """Tests for SearchAgent initialization 🔧."""

    def test_inherits_from_inquiro_agent_base(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """SearchAgent must inherit from InquiroAgentBase."""
        assert isinstance(search_agent, InquiroAgentBase)

    def test_initial_evidence_counter_is_zero(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Evidence counter starts at zero."""
        assert search_agent.get_evidence_count() == 0

    def test_initial_raw_evidence_is_empty(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Raw evidence list starts empty."""
        assert search_agent.get_raw_evidence_records() == []

    def test_initial_servers_used_is_empty(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Servers used set starts empty."""
        assert search_agent.get_servers_used() == set()

    def test_initial_queries_executed_is_empty(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Queries executed list starts empty."""
        assert search_agent.get_queries_executed() == []

    def test_initial_search_gaps_is_empty(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Search gaps list starts empty."""
        assert search_agent.get_search_gaps() == []

    def test_system_prompt_stored(
        self,
        search_agent: SearchAgent,
        sample_system_prompt: str,
    ) -> None:
        """System prompt is correctly stored via base class."""
        assert search_agent._system_prompt_text == sample_system_prompt

    def test_user_prompt_stored(
        self,
        search_agent: SearchAgent,
        sample_user_prompt: str,
    ) -> None:
        """User prompt is correctly stored via base class."""
        assert search_agent._user_prompt_text == sample_user_prompt

    def test_task_id_stored(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Task ID is correctly stored."""
        assert search_agent._task_id == "test-search-001"

    def test_version_is_set(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """VERSION class attribute is set."""
        assert SearchAgent.VERSION == "1.0"

    def test_optional_infrastructure_defaults_to_none(
        self,
        mock_llm: MockLLM,
        mock_tools: ToolRegistry,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Cost tracker, cancellation, and event emitter default to None."""
        agent = SearchAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_prompt="test",
            user_prompt="test",
            config=AgentConfig(max_turns=5),
            output_schema=sample_output_schema,
        )
        assert agent._cost_tracker is None
        assert agent._cancellation_token is None
        assert agent._event_emitter is None


# ============================================================
# 🧪 Test: Prompt Rendering
# ============================================================


class TestSearchAgentPromptRendering:
    """Tests for search prompt template loading and rendering 📝."""

    def test_system_prompt_template_exists(self) -> None:
        """search_system.md template file exists and loads."""
        loader = PromptLoader()
        content = loader.load("search_system")
        assert len(content) > 100

    def test_system_prompt_has_required_placeholders(self) -> None:
        """search_system.md contains all required placeholders."""
        loader = PromptLoader()
        content = loader.load("search_system")
        required_placeholders = [
            "{alias_expansion}",
            "{available_tools}",
            "{search_checklist}",
            "{focus_prompt}",
            "{output_schema}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"

    def test_system_prompt_is_domain_agnostic(self) -> None:
        """search_system.md must not contain domain-specific terms."""
        loader = PromptLoader()
        content = loader.load("search_system").lower()
        # ❌ These pharma terms must NOT appear in Inquiro code/prompts
        forbidden_terms = [
            "drug",
            "pharma",
            "target",
            "indication",
            "clinical trial",
            "fda",
        ]
        for term in forbidden_terms:
            assert term not in content, (
                f"Domain-specific term found in search prompt: '{term}'"
            )

    def test_system_prompt_emphasizes_no_analysis(self) -> None:
        """search_system.md must instruct agent NOT to analyze."""
        loader = PromptLoader()
        content = loader.load("search_system").lower()
        # ✅ The prompt must contain anti-analysis instructions
        assert "not" in content and "analy" in content

    def test_user_prompt_template_exists(self) -> None:
        """search_user.md template file exists and loads."""
        loader = PromptLoader()
        content = loader.load("search_user")
        assert len(content) > 50

    def test_user_prompt_has_required_placeholders(self) -> None:
        """search_user.md contains all required placeholders."""
        loader = PromptLoader()
        content = loader.load("search_user")
        required_placeholders = [
            "{topic}",
            "{checklist}",
            "{alias_expansion}",
            "{focus_section}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in content, f"Missing placeholder: {placeholder}"

    def test_system_prompt_render_succeeds(self) -> None:
        """search_system.md renders without error with all kwargs."""
        loader = PromptLoader()
        rendered = loader.render(
            "search_system",
            alias_expansion="No aliases available.",
            query_section_guide="",
            available_tools="No tools configured.",
            tool_selection_guide="",
            available_skills="",
            query_template="",
            search_checklist="- [R1] Check item 1",
            focus_prompt="No focus instructions.",
            output_schema='{"type": "object"}',
        )
        assert "No aliases available" in rendered
        assert "Check item 1" in rendered

    def test_user_prompt_render_succeeds(self) -> None:
        """search_user.md renders without error with all kwargs."""
        loader = PromptLoader()
        rendered = loader.render(
            "search_user",
            topic="Evidence-based testing",
            checklist="- [R1] Test checklist item",
            alias_expansion="No aliases.",
            focus_section="",
        )
        assert "Evidence-based testing" in rendered
        assert "Test checklist item" in rendered


# ============================================================
# 🧪 Test: Evidence Collection via _execute_tool
# ============================================================


class TestSearchAgentEvidenceCollection:
    """Tests for MCP tool evidence collection 🏷️."""

    def test_mcp_tool_call_increments_counter(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """MCP tool calls increment the evidence counter."""
        # 🔧 Set up a mock MCP tool in the registry
        mock_tool = MagicMock()
        mock_tool.name = "mcp__perplexity__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "perplexity"
        mock_tool.execute.return_value = (
            "Some search results about testing",
            {},
        )
        search_agent.tools.register(mock_tool)

        # 🏷️ Create and execute tool call
        tool_call = create_tool_call(
            name="mcp__perplexity__search",
            arguments={"query": "evidence testing"},
        )
        observation, info = search_agent._execute_tool(tool_call)

        assert search_agent.get_evidence_count() == 1
        assert info["evidence_id"] == "E1"
        assert info["source_server"] == "perplexity"

    def test_mcp_tool_call_records_raw_evidence(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """MCP tool calls produce raw evidence records."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__brave__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "brave"
        mock_tool.execute.return_value = (
            "Results from brave search engine",
            {},
        )
        search_agent.tools.register(mock_tool)

        tool_call = create_tool_call(
            name="mcp__brave__search",
            arguments={"query": "test query"},
        )
        search_agent._execute_tool(tool_call)

        records = search_agent.get_raw_evidence_records()
        assert len(records) == 1
        assert records[0]["id"] == "E1"
        assert records[0]["mcp_server"] == "brave"
        assert records[0]["observation"] == "Results from brave search engine"

    def test_non_mcp_tool_call_does_not_record_evidence(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Non-MCP tool calls (e.g., think) do not produce evidence."""
        mock_tool = MagicMock()
        mock_tool.name = "think"
        mock_tool._is_mcp_tool = False
        mock_tool.execute.return_value = ("Thinking...", {})
        search_agent.tools.register(mock_tool)

        tool_call = create_tool_call(
            name="think",
            arguments={"thought": "planning next search"},
        )
        search_agent._execute_tool(tool_call)

        assert search_agent.get_evidence_count() == 0
        assert search_agent.get_raw_evidence_records() == []

    def test_multiple_mcp_calls_increment_sequentially(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Multiple MCP calls produce E1, E2, E3, ... IDs."""
        for server in ["perplexity", "brave", "bohrium"]:
            mock_tool = MagicMock()
            mock_tool.name = f"mcp__{server}__search"
            mock_tool._is_mcp_tool = True
            mock_tool._mcp_server = server
            mock_tool.execute.return_value = (
                f"Results from {server}",
                {},
            )
            search_agent.tools.register(mock_tool)

        for i, server in enumerate(
            ["perplexity", "brave", "bohrium"],
            start=1,
        ):
            tool_call = create_tool_call(
                name=f"mcp__{server}__search",
                arguments={"query": f"query {i}"},
            )
            search_agent._execute_tool(tool_call)

        assert search_agent.get_evidence_count() == 3
        records = search_agent.get_raw_evidence_records()
        assert [r["id"] for r in records] == ["E1", "E2", "E3"]

    def test_observation_prefixed_with_evidence_tag(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """MCP tool observations are prefixed with evidence metadata."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__perplexity__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "perplexity"
        mock_tool.execute.return_value = ("Raw result text", {})
        search_agent.tools.register(mock_tool)

        tool_call = create_tool_call(
            name="mcp__perplexity__search",
            arguments={"query": "test"},
        )
        observation, _ = search_agent._execute_tool(tool_call)

        assert observation.startswith("[Evidence E1 | Source: perplexity]")
        assert "Raw result text" in observation

    def test_empty_observation_recorded_as_gap(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Empty tool responses are recorded as search gaps."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__perplexity__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "perplexity"
        mock_tool.execute.return_value = ("", {})
        search_agent.tools.register(mock_tool)

        tool_call = create_tool_call(
            name="mcp__perplexity__search",
            arguments={"query": "empty query"},
        )
        search_agent._execute_tool(tool_call)

        gaps = search_agent.get_search_gaps()
        assert len(gaps) == 1
        assert gaps[0]["server"] == "perplexity"
        assert gaps[0]["reason"] == "empty_or_error"

    def test_error_observation_recorded_as_gap(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Error responses from tools are recorded as search gaps."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__brave__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "brave"
        mock_tool.execute.return_value = (
            "Error: connection timeout",
            {},
        )
        search_agent.tools.register(mock_tool)

        tool_call = create_tool_call(
            name="mcp__brave__search",
            arguments={"query": "failing query"},
        )
        search_agent._execute_tool(tool_call)

        gaps = search_agent.get_search_gaps()
        assert len(gaps) == 1
        assert gaps[0]["server"] == "brave"


# ============================================================
# 🧪 Test: Server Diversity Tracking
# ============================================================


class TestSearchAgentDiversity:
    """Tests for server diversity tracking 📊."""

    def test_servers_used_tracks_unique_servers(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Servers used set tracks unique MCP server names."""
        for server in ["perplexity", "brave"]:
            mock_tool = MagicMock()
            mock_tool.name = f"mcp__{server}__search"
            mock_tool._is_mcp_tool = True
            mock_tool._mcp_server = server
            mock_tool.execute.return_value = (f"{server} results", {})
            search_agent.tools.register(mock_tool)

        for server in ["perplexity", "brave", "perplexity"]:
            tool_call = create_tool_call(
                name=f"mcp__{server}__search",
                arguments={"query": "test"},
            )
            search_agent._execute_tool(tool_call)

        assert search_agent.get_servers_used() == {
            "perplexity",
            "brave",
        }

    def test_queries_executed_tracks_all_calls(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Queries executed list records every MCP tool call."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__perplexity__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "perplexity"
        mock_tool.execute.return_value = ("results", {})
        search_agent.tools.register(mock_tool)

        for i in range(3):
            tool_call = create_tool_call(
                name="mcp__perplexity__search",
                arguments={"query": f"query_{i}"},
            )
            search_agent._execute_tool(tool_call)

        queries = search_agent.get_queries_executed()
        assert len(queries) == 3
        assert all(q["server"] == "perplexity" for q in queries)

    def test_diversity_ratio_with_no_queries(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Diversity ratio is 1.0 when no queries have been executed."""
        assert search_agent.get_server_diversity_ratio() == 1.0

    def test_diversity_ratio_single_server(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """Diversity ratio reflects single-server usage."""
        mock_tool = MagicMock()
        mock_tool.name = "mcp__perplexity__search"
        mock_tool._is_mcp_tool = True
        mock_tool._mcp_server = "perplexity"
        mock_tool.execute.return_value = ("results", {})
        search_agent.tools.register(mock_tool)

        for _ in range(4):
            tool_call = create_tool_call(
                name="mcp__perplexity__search",
                arguments={"query": "test"},
            )
            search_agent._execute_tool(tool_call)

        # 📊 1 unique server / 4 total queries = 0.25
        assert search_agent.get_server_diversity_ratio() == 0.25


# ============================================================
# 🧪 Test: URL Extraction
# ============================================================


class TestSearchAgentURLExtraction:
    """Tests for URL extraction from tool observations 🔗."""

    def test_extract_doi_url(self) -> None:
        """Extracts DOI URL from observation."""
        obs = "See https://doi.org/10.1234/test.2024 for details."
        assert SearchAgent._extract_url(obs) == ("https://doi.org/10.1234/test.2024")

    def test_extract_raw_doi(self) -> None:
        """Converts raw DOI to URL."""
        obs = "Reference: doi:10.5678/example.2024"
        assert SearchAgent._extract_url(obs) == ("https://doi.org/10.5678/example.2024")

    def test_extract_pmid(self) -> None:
        """Converts PMID to PubMed URL."""
        obs = "Published in PMID: 12345678"
        assert SearchAgent._extract_url(obs) == (
            "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )

    def test_extract_generic_https(self) -> None:
        """Falls back to generic HTTPS URL extraction."""
        obs = "Found at https://example.com/article/123"
        assert SearchAgent._extract_url(obs) == ("https://example.com/article/123")

    def test_extract_none_from_empty(self) -> None:
        """Returns None for empty observation."""
        assert SearchAgent._extract_url("") is None
        assert SearchAgent._extract_url(None) is None

    def test_extract_none_from_no_url(self) -> None:
        """Returns None when no URL is present."""
        obs = "This observation has no URLs at all."
        assert SearchAgent._extract_url(obs) is None


# ============================================================
# 🧪 Test: No-tool-call Prompt
# ============================================================


class TestSearchAgentNoToolCallPrompt:
    """Tests for the no-tool-call fallback prompt 📝."""

    def test_prompt_instructs_to_continue_searching(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """No-tool-call prompt tells agent to keep searching."""
        prompt = search_agent._get_no_tool_call_prompt()
        assert "search" in prompt.lower()
        assert "finish" in prompt.lower()

    def test_prompt_discourages_analysis(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """No-tool-call prompt tells agent NOT to analyze."""
        prompt = search_agent._get_no_tool_call_prompt()
        assert "not" in prompt.lower()
        assert "analy" in prompt.lower() or "evaluat" in prompt.lower()

    def test_prompt_requires_autonomous_work(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """No-tool-call prompt forbids asking for human help."""
        prompt = search_agent._get_no_tool_call_prompt()
        assert "human help" in prompt.lower() or "autonomous" in prompt.lower()


# ============================================================
# 🧪 Test: Empty/Error Detection
# ============================================================


class TestSearchAgentEmptyOrErrorDetection:
    """Tests for _is_empty_or_error static method 🔍."""

    def test_empty_string_is_empty(self) -> None:
        """Empty string is detected."""
        assert SearchAgent._is_empty_or_error("") is True

    def test_whitespace_only_is_empty(self) -> None:
        """Whitespace-only string is detected."""
        assert SearchAgent._is_empty_or_error("   \n\t  ") is True

    def test_none_is_empty(self) -> None:
        """None is detected as empty."""
        assert SearchAgent._is_empty_or_error(None) is True

    def test_error_prefix_detected(self) -> None:
        """Error: prefix triggers detection."""
        assert SearchAgent._is_empty_or_error("Error: timeout") is True

    def test_tool_execution_error_detected(self) -> None:
        """Tool execution error prefix triggers detection."""
        assert (
            SearchAgent._is_empty_or_error(
                "Tool execution error: failed",
            )
            is True
        )

    def test_valid_result_not_detected(self) -> None:
        """Normal results are not flagged as empty/error."""
        assert (
            SearchAgent._is_empty_or_error(
                "Found 10 results about the topic.",
            )
            is False
        )

    def test_no_results_found_detected(self) -> None:
        """'no results found' prefix triggers detection."""
        assert (
            SearchAgent._is_empty_or_error(
                "No results found for the query.",
            )
            is True
        )


# ============================================================
# 🧪 Test: Accessor Immutability
# ============================================================


class TestSearchAgentAccessorImmutability:
    """Tests that public accessors return copies, not references 🔒."""

    def test_raw_evidence_records_returns_copy(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """get_raw_evidence_records returns a new list each time."""
        records1 = search_agent.get_raw_evidence_records()
        records2 = search_agent.get_raw_evidence_records()
        assert records1 is not records2

    def test_servers_used_returns_copy(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """get_servers_used returns a new set each time."""
        set1 = search_agent.get_servers_used()
        set2 = search_agent.get_servers_used()
        assert set1 is not set2

    def test_queries_executed_returns_copy(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """get_queries_executed returns a new list each time."""
        list1 = search_agent.get_queries_executed()
        list2 = search_agent.get_queries_executed()
        assert list1 is not list2

    def test_search_gaps_returns_copy(
        self,
        search_agent: SearchAgent,
    ) -> None:
        """get_search_gaps returns a new list each time."""
        list1 = search_agent.get_search_gaps()
        list2 = search_agent.get_search_gaps()
        assert list1 is not list2
