"""Unit and integration tests for SearchExp lifecycle manager 🧪.

Tests cover:
    - SearchRoundResult model construction and defaults
    - Prompt rendering (system prompt, user prompt, focus section)
    - Round 1 vs Round N behavior (broad vs focused search)
    - SearchAgent creation with correct configuration
    - Raw evidence extraction from agent records
    - EvidencePipeline integration (cleaning, stats)
    - Evidence conversion (RawEvidence -> Evidence -> CleanedEvidence)
    - Timeout handling (returns partial/empty result)
    - Error handling (agent failure returns empty result with error)
    - Cost estimation from trajectory metadata
    - MCP tool tracking (servers used)
    - Query extraction from agent
    - Empty results handling
    - Event emission during search lifecycle
    - Cancellation token propagation
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inquiro.core.evidence_pipeline import (
    CleaningStats,
    EvidencePipeline,
)
from inquiro.core.types import (
    CleanedEvidence,
    DiscoveryConfig,
    Evidence,
    RawEvidence,
)
from inquiro.exps.search_exp import (
    SearchExp,
    SearchRoundResult,
    _SEARCH_OUTPUT_SCHEMA,
)
from inquiro.tests.mock_helpers import (
    MockLLM,
    build_sample_evaluation_task,
)


# ============================================================================
# 🏭 Test Fixtures and Helpers
# ============================================================================


def _build_discovery_config(**overrides: Any) -> DiscoveryConfig:
    """Build a DiscoveryConfig with optional overrides 🔧.

    Args:
        **overrides: Fields to override from defaults.

    Returns:
        DiscoveryConfig instance.
    """
    defaults: dict[str, Any] = {
        "max_rounds": 3,
        "timeout_per_round": 300,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


def _build_search_exp(
    llm: Any | None = None,
    tools: Any | None = None,
    event_emitter: Any | None = None,
    cost_tracker: Any | None = None,
    cancellation_token: Any | None = None,
) -> SearchExp:
    """Create a SearchExp with sensible test defaults 🔍.

    Args:
        llm: LLM instance (MockLLM if None).
        tools: ToolRegistry mock (MagicMock if None).
        event_emitter: EventEmitter (MagicMock if None).
        cost_tracker: CostTracker (real if None).
        cancellation_token: CancellationToken (real if None).

    Returns:
        Configured SearchExp instance.
    """
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker

    if tools is None:
        tools = MagicMock()
        tools.__len__ = MagicMock(return_value=0)
        tools.get_mcp_server_names = MagicMock(return_value=[])

    if event_emitter is None:
        event_emitter = MagicMock()
        event_emitter.emit = MagicMock()

    return SearchExp(
        llm=llm if llm is not None else MockLLM(),
        tools=tools,
        event_emitter=event_emitter,
        cost_tracker=(
            cost_tracker
            if cost_tracker is not None
            else CostTracker(max_per_task=10.0, max_total=100.0)
        ),
        cancellation_token=(
            cancellation_token
            if cancellation_token is not None
            else CancellationToken()
        ),
    )


def _make_mock_agent(
    raw_evidence_records: list[dict[str, Any]] | None = None,
    queries_executed: list[dict[str, str]] | None = None,
    servers_used: set[str] | None = None,
) -> MagicMock:
    """Build a mock SearchAgent with configurable results 🤖.

    Args:
        raw_evidence_records: Raw evidence record dicts.
        queries_executed: Query tracking dicts.
        servers_used: Set of MCP server names.

    Returns:
        Configured mock SearchAgent.
    """
    agent = MagicMock()
    agent.get_raw_evidence_records.return_value = raw_evidence_records or []
    agent.get_queries_executed.return_value = queries_executed or []
    agent.get_servers_used.return_value = servers_used or set()
    agent.trajectory = None
    agent.run = MagicMock(return_value=None)
    return agent


def _sample_raw_evidence_records(count: int = 3) -> list[dict[str, Any]]:
    """Build sample raw evidence records for testing 📄.

    Ensures summaries exceed MIN_EVIDENCE_LENGTH (50 chars).

    Args:
        count: Number of records to generate.

    Returns:
        List of raw evidence record dicts.
    """
    records: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        records.append(
            {
                "id": f"E{i}",
                "mcp_server": f"server_{(i % 2) + 1}",
                "source_query": f"evidence search query number {i}",
                "observation": (
                    f"This is a detailed observation for evidence item "
                    f"number {i} that contains enough text to pass the "
                    f"minimum evidence length filter in the pipeline "
                    f"and provides meaningful research context."
                ),
                "url": f"https://doi.org/10.1234/test{i}",
            }
        )
    return records


# ============================================================================
# 📊 SearchRoundResult model tests
# ============================================================================


class TestSearchRoundResult:
    """Tests for SearchRoundResult Pydantic model 📊."""

    def test_default_construction(self) -> None:
        """Default SearchRoundResult has empty lists and zero values ✅."""
        result = SearchRoundResult()
        assert result.raw_evidence == []
        assert result.cleaned_evidence == []
        assert result.queries_executed == []
        assert result.mcp_tools_used == []
        assert result.duration_seconds == 0.0
        assert result.cost_usd == 0.0
        assert result.agent_trajectory_ref is None
        assert result.error is None

    def test_construction_with_data(self) -> None:
        """SearchRoundResult stores provided data correctly ✅."""
        raw = [RawEvidence(id="E1", observation="test observation")]
        cleaned = [CleanedEvidence(id="E1", summary="test summary")]
        stats = CleaningStats(input_count=1, output_count=1)

        result = SearchRoundResult(
            raw_evidence=raw,
            cleaned_evidence=cleaned,
            cleaning_stats=stats,
            queries_executed=["query one"],
            mcp_tools_used=["server_a"],
            duration_seconds=5.0,
            cost_usd=0.50,
            agent_trajectory_ref="/tmp/traj.jsonl",
        )

        assert len(result.raw_evidence) == 1
        assert result.raw_evidence[0].id == "E1"
        assert len(result.cleaned_evidence) == 1
        assert result.cleaning_stats.input_count == 1
        assert result.queries_executed == ["query one"]
        assert result.mcp_tools_used == ["server_a"]
        assert result.duration_seconds == 5.0
        assert result.cost_usd == 0.50
        assert result.agent_trajectory_ref == "/tmp/traj.jsonl"

    def test_error_field(self) -> None:
        """SearchRoundResult can carry an error message ✅."""
        result = SearchRoundResult(error="Agent timed out")
        assert result.error == "Agent timed out"
        assert result.raw_evidence == []

    def test_serialization_roundtrip(self) -> None:
        """SearchRoundResult survives model_dump/model_validate ✅."""
        result = SearchRoundResult(
            raw_evidence=[RawEvidence(id="E1")],
            duration_seconds=3.14,
        )
        data = result.model_dump()
        restored = SearchRoundResult.model_validate(data)
        assert restored.raw_evidence[0].id == "E1"
        assert restored.duration_seconds == pytest.approx(3.14)


# ============================================================================
# 📝 Prompt rendering tests
# ============================================================================


class TestPromptRendering:
    """Tests for SearchExp prompt rendering methods 📝."""

    def test_render_system_prompt_contains_checklist(self) -> None:
        """System prompt includes formatted checklist items ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        prompt = exp._render_system_prompt(task)

        assert "item_1" in prompt or "market size" in prompt.lower()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_render_system_prompt_contains_output_schema(self) -> None:
        """System prompt includes the search output schema ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        prompt = exp._render_system_prompt(task)

        assert "raw_evidence" in prompt
        assert "total_collected" in prompt

    def test_render_user_prompt_contains_topic(self) -> None:
        """User prompt includes the research topic ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task(topic="KRAS signaling pathway")
        prompt = exp._render_user_prompt(task, round_number=1)

        assert "KRAS signaling pathway" in prompt

    def test_render_user_prompt_round_1_broad_search(self) -> None:
        """Round 1 user prompt includes broad search guidance ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        prompt = exp._render_user_prompt(
            task,
            round_number=1,
            focus_prompt=None,
        )

        assert "initial search round" in prompt.lower()
        assert "broad" in prompt.lower()

    def test_render_user_prompt_round_n_with_focus(self) -> None:
        """Round > 1 user prompt includes injected focus text ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        focus = "Search for patent landscape data and IP filings."
        prompt = exp._render_user_prompt(
            task,
            round_number=2,
            focus_prompt=focus,
        )

        assert "patent landscape" in prompt.lower()
        assert "Round 2" in prompt

    def test_render_user_prompt_round_n_without_focus_falls_back(
        self,
    ) -> None:
        """Round > 1 without focus_prompt uses broad search text ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        prompt = exp._render_user_prompt(
            task,
            round_number=3,
            focus_prompt=None,
        )

        # ✅ Falls back to initial broad search guidance
        assert "initial search round" in prompt.lower()


class TestBuildFocusSection:
    """Tests for _build_focus_section static method 🎯."""

    def test_round_1_returns_broad_guidance(self) -> None:
        """Round 1 produces broad initial search guidance ✅."""
        section = SearchExp._build_focus_section(1, None)
        assert "initial search round" in section.lower()
        assert "broad" in section.lower()

    def test_round_1_ignores_focus_prompt(self) -> None:
        """Round 1 ignores any focus_prompt and uses broad text ✅."""
        section = SearchExp._build_focus_section(
            1,
            "Focus on patents only",
        )
        assert "initial search round" in section.lower()
        assert "patents" not in section.lower()

    def test_round_2_with_focus(self) -> None:
        """Round 2 with focus_prompt includes the focus text ✅."""
        focus = "Search for regulatory filing data from FDA."
        section = SearchExp._build_focus_section(2, focus)
        assert "Round 2" in section
        assert "regulatory filing data" in section.lower()
        assert "gaps" in section.lower()

    def test_round_n_without_focus_broad(self) -> None:
        """Round > 1 without focus falls back to broad guidance ✅."""
        section = SearchExp._build_focus_section(5, None)
        assert "initial search round" in section.lower()


# ============================================================================
# 🔧 Agent creation tests
# ============================================================================


class TestAgentCreation:
    """Tests for SearchExp agent creation 🔧."""

    def test_create_search_agent_returns_agent(self) -> None:
        """_create_search_agent returns a SearchAgent instance ✅."""
        from inquiro.agents.search_agent import SearchAgent

        exp = _build_search_exp()
        task = build_sample_evaluation_task()

        agent = exp._create_search_agent(
            task=task,
            system_prompt="System prompt text for test agent",
            user_prompt="User prompt text for test agent",
        )

        assert isinstance(agent, SearchAgent)

    def test_create_search_agent_passes_task_id(self) -> None:
        """Agent receives the correct task_id from task ✅."""

        exp = _build_search_exp()
        task = build_sample_evaluation_task(task_id="custom-task-42")

        agent = exp._create_search_agent(
            task=task,
            system_prompt="test system prompt content here",
            user_prompt="test user prompt content here",
        )

        assert agent._task_id == "custom-task-42"


# ============================================================================
# 📥 Evidence extraction tests
# ============================================================================


class TestEvidenceExtraction:
    """Tests for evidence extraction from agent 📥."""

    def test_extract_raw_evidence_from_records(self) -> None:
        """Extracts RawEvidence from agent record dicts ✅."""
        records = _sample_raw_evidence_records(2)
        agent = _make_mock_agent(raw_evidence_records=records)

        raw = SearchExp._extract_raw_evidence(agent)

        assert len(raw) == 2
        assert raw[0].id == "E1"
        assert raw[1].id == "E2"
        assert raw[0].mcp_server == "server_2"
        assert raw[1].mcp_server == "server_1"

    def test_extract_raw_evidence_empty_agent(self) -> None:
        """Empty agent records produce empty list ✅."""
        agent = _make_mock_agent()
        raw = SearchExp._extract_raw_evidence(agent)
        assert raw == []

    def test_extract_queries_from_agent(self) -> None:
        """Extracts query strings from agent metadata ✅."""
        queries = [
            {"server": "s1", "query": "query alpha", "evidence_id": "E1"},
            {"server": "s2", "query": "query beta", "evidence_id": "E2"},
        ]
        agent = _make_mock_agent(queries_executed=queries)

        result = SearchExp._extract_queries(agent)

        assert result == ["query alpha", "query beta"]

    def test_extract_queries_empty(self) -> None:
        """No queries returns empty list ✅."""
        agent = _make_mock_agent()
        result = SearchExp._extract_queries(agent)
        assert result == []


# ============================================================================
# 🔄 Evidence conversion tests
# ============================================================================


class TestEvidenceConversion:
    """Tests for evidence model conversions 🔄."""

    def test_convert_raw_to_evidence(self) -> None:
        """RawEvidence items convert to Evidence objects ✅."""
        raw = [
            RawEvidence(
                id="E1",
                mcp_server="perplexity",
                source_query="market size query for testing",
                observation=(
                    "The global market is estimated at 22 billion "
                    "dollars with compound annual growth rate of 8.5%"
                ),
                url="https://doi.org/10.1234/test1",
            ),
        ]
        evidence = SearchExp._convert_raw_to_evidence(raw)

        assert len(evidence) == 1
        assert evidence[0].id == "E1"
        assert evidence[0].source == "perplexity"
        assert evidence[0].query == "market size query for testing"
        assert "22 billion" in evidence[0].summary

    def test_convert_raw_to_evidence_preserves_url(self) -> None:
        """URL is preserved during conversion ✅."""
        raw = [
            RawEvidence(
                id="E1",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                observation="test observation with enough text for filter",
            ),
        ]
        evidence = SearchExp._convert_raw_to_evidence(raw)
        assert evidence[0].url == ("https://pubmed.ncbi.nlm.nih.gov/12345/")

    def test_convert_raw_to_evidence_handles_none_url(self) -> None:
        """None URL is preserved as None during conversion ✅."""
        raw = [
            RawEvidence(
                id="E1",
                url=None,
                observation="test observation text",
            ),
        ]
        evidence = SearchExp._convert_raw_to_evidence(raw)
        assert evidence[0].url is None

    def test_convert_evidence_to_cleaned(self) -> None:
        """Evidence items convert to CleanedEvidence correctly ✅."""
        ev = Evidence(
            id="E1",
            source="perplexity",
            url="https://doi.org/10.1234/abc",
            query="search query text",
            summary="Detailed evidence summary text for testing purposes",
        )
        cleaned = SearchExp._convert_evidence_to_cleaned([ev])

        assert len(cleaned) == 1
        assert cleaned[0].id == "E1"
        assert cleaned[0].mcp_server == "perplexity"
        # 🏷️ Tag is now classified via classify_url(), not monkey-patched
        assert cleaned[0].tag == "academic"
        assert cleaned[0].source_query == "search query text"

    def test_convert_evidence_to_cleaned_default_tag(self) -> None:
        """Evidence without recognized URL gets OTHER tag ✅."""
        ev = Evidence(
            id="E1",
            source="unknown",
            url=None,
            query="test",
            summary="Test evidence content",
        )
        # ✅ No recognized URL -- should classify as OTHER

        cleaned = SearchExp._convert_evidence_to_cleaned([ev])
        assert cleaned[0].tag == "other"


# ============================================================================
# 🧹 EvidencePipeline integration tests
# ============================================================================


class TestEvidencePipelineIntegration:
    """Tests for EvidencePipeline integration within SearchExp 🧹."""

    def test_pipeline_cleans_raw_evidence(self) -> None:
        """Pipeline deduplicates and filters evidence ✅."""
        pipeline = EvidencePipeline()

        # 📊 3 unique + 1 duplicate + 1 too short
        evidence = [
            Evidence(
                id="E1",
                source="s1",
                url="https://doi.org/10.1234/a",
                query="q1",
                summary=(
                    "First unique evidence about novel treatment "
                    "efficacy in randomized controlled trials"
                ),
            ),
            Evidence(
                id="E2",
                source="s1",
                url="https://doi.org/10.1234/b",
                query="q2",
                summary=(
                    "Second unique evidence about molecular "
                    "mechanism of action and binding affinity data"
                ),
            ),
            Evidence(
                id="E3",
                source="s2",
                url="https://doi.org/10.1234/c",
                query="q3",
                summary=(
                    "Third unique evidence regarding clinical "
                    "trial outcomes and safety profile results"
                ),
            ),
            Evidence(
                id="E4",
                source="s1",
                url="https://doi.org/10.1234/a",
                query="q4",
                # 🔑 Duplicate of E1 (same URL + summary → same hash)
                summary=(
                    "First unique evidence about novel treatment "
                    "efficacy in randomized controlled trials"
                ),
            ),
            Evidence(
                id="E5",
                source="s2",
                url=None,
                query="q5",
                summary="Short",  # ❌ Below MIN_EVIDENCE_LENGTH
            ),
        ]

        cleaned, stats = pipeline.clean(evidence)

        assert stats.input_count == 5
        assert stats.dedup_removed == 1
        assert stats.noise_removed == 1
        assert stats.output_count == 3
        assert len(cleaned) == 3

    def test_pipeline_tags_academic_sources(self) -> None:
        """Pipeline classifies academic URLs in tag_distribution ✅."""
        pipeline = EvidencePipeline()
        evidence = [
            Evidence(
                id="E1",
                source="s1",
                url="https://pubmed.ncbi.nlm.nih.gov/12345/",
                query="q1",
                summary=(
                    "Academic evidence from PubMed about treatment "
                    "efficacy analysis in a systematic review study"
                ),
            ),
        ]
        cleaned, stats = pipeline.clean(evidence)
        assert len(cleaned) == 1
        # 🏷️ Tag distribution reflects the classification
        assert stats.tag_distribution.get("academic", 0) == 1


# ============================================================================
# 💰 Cost estimation tests
# ============================================================================


class TestCostEstimation:
    """Tests for cost estimation from trajectory 💰."""

    def test_estimate_cost_none_trajectory(self) -> None:
        """None trajectory returns zero cost ✅."""
        assert SearchExp._estimate_cost(None) == 0.0

    def test_estimate_cost_empty_trajectory(self) -> None:
        """Trajectory with no steps returns zero cost ✅."""
        traj = MagicMock()
        traj.steps = []
        assert SearchExp._estimate_cost(traj) == 0.0

    def test_estimate_cost_with_usage(self) -> None:
        """Cost is calculated from token usage metadata ✅."""
        step = MagicMock()
        step.assistant_message = MagicMock()
        step.assistant_message.meta = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
        }
        traj = MagicMock()
        traj.steps = [step]

        cost = SearchExp._estimate_cost(traj)

        # 📊 1000 * 3/1M + 500 * 15/1M = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105)

    def test_estimate_cost_multiple_steps(self) -> None:
        """Cost accumulates across multiple trajectory steps ✅."""
        step1 = MagicMock()
        step1.assistant_message = MagicMock()
        step1.assistant_message.meta = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        step2 = MagicMock()
        step2.assistant_message = MagicMock()
        step2.assistant_message.meta = {
            "usage": {"prompt_tokens": 200, "completion_tokens": 100},
        }
        traj = MagicMock()
        traj.steps = [step1, step2]

        cost = SearchExp._estimate_cost(traj)

        # 📊 Step1: 100*3/1M + 50*15/1M = 0.0003 + 0.00075 = 0.00105
        # 📊 Step2: 200*3/1M + 100*15/1M = 0.0006 + 0.0015 = 0.0021
        assert cost == pytest.approx(0.00315)

    def test_estimate_cost_missing_meta(self) -> None:
        """Steps without meta return zero cost contribution ✅."""
        step = MagicMock()
        step.assistant_message = MagicMock()
        step.assistant_message.meta = None
        traj = MagicMock()
        traj.steps = [step]

        cost = SearchExp._estimate_cost(traj)
        assert cost == 0.0


# ============================================================================
# 🚀 Full run_search integration tests (mocked agent)
# ============================================================================


class TestRunSearch:
    """Integration tests for run_search lifecycle 🚀."""

    @pytest.mark.asyncio
    async def test_run_search_round_1_success(self) -> None:
        """Successful round 1 returns evidence and metrics ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        records = _sample_raw_evidence_records(3)
        mock_agent = _make_mock_agent(
            raw_evidence_records=records,
            queries_executed=[
                {"query": "q1", "server": "s1", "evidence_id": "E1"},
                {"query": "q2", "server": "s2", "evidence_id": "E2"},
            ],
            servers_used={"server_1", "server_2"},
        )

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.error is None
        assert len(result.raw_evidence) == 3
        assert len(result.queries_executed) == 2
        assert "server_1" in result.mcp_tools_used
        assert "server_2" in result.mcp_tools_used
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_search_round_n_with_focus(self) -> None:
        """Round > 1 with focus_prompt uses focused search ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        records = _sample_raw_evidence_records(1)
        mock_agent = _make_mock_agent(
            raw_evidence_records=records,
            servers_used={"server_1"},
        )

        # ✅ Capture rendered prompts
        rendered_user_prompts: list[str] = []
        original_render = exp._render_user_prompt

        def capture_render(t: Any, rn: int, fp: str | None = None) -> str:
            result = original_render(t, rn, fp)
            rendered_user_prompts.append(result)
            return result

        with (
            patch.object(
                exp,
                "_render_user_prompt",
                side_effect=capture_render,
            ),
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=3,
                focus_prompt="Search for patent data and IP filings.",
            )

        assert result.error is None
        assert len(rendered_user_prompts) == 1
        assert "patent data" in rendered_user_prompts[0].lower()

    @pytest.mark.asyncio
    async def test_run_search_timeout_returns_empty_result(self) -> None:
        """Timeout returns SearchRoundResult with error message ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config(timeout_per_round=1)

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=MagicMock(),
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError(),
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.error is not None
        assert "timed out" in result.error.lower()
        assert result.raw_evidence == []
        assert result.cleaned_evidence == []

    @pytest.mark.asyncio
    async def test_run_search_agent_failure_returns_error(self) -> None:
        """Agent failure returns SearchRoundResult with error ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        with patch.object(
            exp,
            "_create_search_agent",
            side_effect=RuntimeError("LLM connection failed"),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.error is not None
        assert "LLM connection failed" in result.error
        assert result.raw_evidence == []

    @pytest.mark.asyncio
    async def test_run_search_emits_start_event(self) -> None:
        """Search emits TASK_STARTED event at beginning ✅."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        exp = _build_search_exp(event_emitter=emitter)
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()
        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        # 📡 Check that TASK_STARTED was emitted
        calls = emitter.emit.call_args_list
        started_calls = [
            c for c in calls if len(c.args) >= 1 and c.args[0].value == "task_started"
        ]
        assert len(started_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_search_emits_completed_event(self) -> None:
        """Successful search emits TASK_COMPLETED event ✅."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        exp = _build_search_exp(event_emitter=emitter)
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()
        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        calls = emitter.emit.call_args_list
        completed_calls = [
            c for c in calls if len(c.args) >= 1 and c.args[0].value == "task_completed"
        ]
        assert len(completed_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_search_emits_failed_event_on_error(self) -> None:
        """Failed search emits TASK_FAILED event ✅."""
        emitter = MagicMock()
        emitter.emit = MagicMock()
        exp = _build_search_exp(event_emitter=emitter)
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        with patch.object(
            exp,
            "_create_search_agent",
            side_effect=ValueError("bad config"),
        ):
            await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        calls = emitter.emit.call_args_list
        failed_calls = [
            c for c in calls if len(c.args) >= 1 and c.args[0].value == "task_failed"
        ]
        assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_search_empty_evidence(self) -> None:
        """Agent with no evidence returns empty result ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent(
            raw_evidence_records=[],
            servers_used=set(),
        )

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.error is None
        assert result.raw_evidence == []
        assert result.cleaned_evidence == []
        assert result.cleaning_stats.input_count == 0
        assert result.cleaning_stats.output_count == 0

    @pytest.mark.asyncio
    async def test_run_search_pipeline_cleans_noise(self) -> None:
        """Pipeline removes noise from raw evidence in full flow ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        # 📊 2 valid + 1 noise record
        records = [
            {
                "id": "E1",
                "mcp_server": "s1",
                "source_query": "q1",
                "observation": (
                    "Meaningful research finding about treatment "
                    "outcomes in a randomized controlled clinical "
                    "trial with statistically significant results"
                ),
                "url": "https://doi.org/10.1234/abc",
            },
            {
                "id": "E2",
                "mcp_server": "s2",
                "source_query": "q2",
                "observation": (
                    "Another substantial evidence item regarding "
                    "pharmacokinetic properties and drug metabolism "
                    "pathways observed in preclinical models tested"
                ),
                "url": "https://doi.org/10.5678/def",
            },
            {
                "id": "E3",
                "mcp_server": "s1",
                "source_query": "q3",
                "observation": "Short",  # ❌ Too short -> noise
                "url": None,
            },
        ]

        mock_agent = _make_mock_agent(
            raw_evidence_records=records,
            servers_used={"s1", "s2"},
        )

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.error is None
        assert len(result.raw_evidence) == 3
        # ✅ Pipeline should remove the short noise item
        assert len(result.cleaned_evidence) == 2
        assert result.cleaning_stats.noise_removed == 1

    @pytest.mark.asyncio
    async def test_run_search_cost_tracking(self) -> None:
        """Cost is estimated from trajectory and recorded ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()

        # 📊 Build a trajectory with usage metadata
        step = MagicMock()
        step.assistant_message = MagicMock()
        step.assistant_message.meta = {
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 1000,
            },
        }
        mock_traj = MagicMock()
        mock_traj.steps = [step]

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=mock_traj,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        # 💰 2000*3/1M + 1000*15/1M = 0.006 + 0.015 = 0.021
        assert result.cost_usd == pytest.approx(0.021)


# ============================================================================
# 🔧 Initialization tests
# ============================================================================


class TestSearchExpInit:
    """Tests for SearchExp initialization 🔧."""

    def test_default_init_creates_infrastructure(self) -> None:
        """Default init creates pipeline, tracker, emitter, token ✅."""
        exp = SearchExp(
            llm=MockLLM(),
            tools=MagicMock(),
        )
        assert isinstance(exp.evidence_pipeline, EvidencePipeline)
        assert exp.cost_tracker is not None
        assert exp.event_emitter is not None
        assert exp.cancellation_token is not None

    def test_custom_infrastructure_used(self) -> None:
        """Custom infrastructure objects are stored correctly ✅."""
        from inquiro.infrastructure.cancellation import CancellationToken
        from inquiro.infrastructure.cost_tracker import CostTracker

        custom_emitter = MagicMock()
        custom_tracker = CostTracker(
            max_per_task=5.0,
            max_total=50.0,
        )
        custom_token = CancellationToken()

        exp = SearchExp(
            llm=MockLLM(),
            tools=MagicMock(),
            event_emitter=custom_emitter,
            cost_tracker=custom_tracker,
            cancellation_token=custom_token,
        )

        assert exp.event_emitter is custom_emitter
        assert exp.cost_tracker is custom_tracker
        assert exp.cancellation_token is custom_token


# ============================================================================
# 📊 Output schema tests
# ============================================================================


class TestSearchOutputSchema:
    """Tests for the search output schema constant 📊."""

    def test_schema_has_required_fields(self) -> None:
        """Schema requires raw_evidence, total_collected, search_gaps ✅."""
        assert "required" in _SEARCH_OUTPUT_SCHEMA
        required = _SEARCH_OUTPUT_SCHEMA["required"]
        assert "raw_evidence" in required
        assert "total_collected" in required
        assert "search_gaps" in required

    def test_schema_is_valid_json_schema(self) -> None:
        """Schema has proper JSON Schema structure ✅."""
        assert _SEARCH_OUTPUT_SCHEMA["type"] == "object"
        assert "properties" in _SEARCH_OUTPUT_SCHEMA
        props = _SEARCH_OUTPUT_SCHEMA["properties"]
        assert "raw_evidence" in props
        assert props["raw_evidence"]["type"] == "array"


# ============================================================================
# 📝 Agent trajectory ref tests
# ============================================================================


class TestAgentTrajectoryRef:
    """Tests for agent_trajectory_ref wiring in run_search() 📝."""

    @pytest.mark.asyncio
    async def test_run_search_sets_agent_trajectory_ref_when_dir_given(
        self,
        tmp_path: Any,
    ) -> None:
        """agent_trajectory_ref is set when agent_trajectory_dir provided ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        records = _sample_raw_evidence_records(2)
        mock_agent = _make_mock_agent(raw_evidence_records=records)

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
                agent_trajectory_dir=str(tmp_path),
            )

        assert result.agent_trajectory_ref is not None
        assert str(tmp_path) in result.agent_trajectory_ref
        assert "search_agent_r1" in result.agent_trajectory_ref
        assert result.agent_trajectory_ref.endswith(".jsonl")

    @pytest.mark.asyncio
    async def test_run_search_no_trajectory_ref_when_dir_not_given(
        self,
    ) -> None:
        """agent_trajectory_ref is None when no agent_trajectory_dir given ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        assert result.agent_trajectory_ref is None

    @pytest.mark.asyncio
    async def test_run_search_sets_trajectory_path_on_agent_instance(
        self,
        tmp_path: Any,
    ) -> None:
        """Agent instance gets _trajectory_file_path set before execution ✅."""
        from pathlib import Path

        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()
        observed_path: list[Any] = []

        async def capture_agent_path(
            agent: Any, timeout: int, **kwargs: Any,
        ) -> None:
            observed_path.append(
                getattr(agent, "_trajectory_file_path", None),
            )
            return None

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                side_effect=capture_agent_path,
            ),
        ):
            await exp.run_search(
                task=task,
                config=config,
                round_number=2,
                agent_trajectory_dir=str(tmp_path),
            )

        assert len(observed_path) == 1
        assert observed_path[0] is not None
        assert isinstance(observed_path[0], Path)
        assert "search_agent_r2" in str(observed_path[0])

    @pytest.mark.asyncio
    async def test_run_search_trajectory_ref_includes_round_number(
        self,
        tmp_path: Any,
    ) -> None:
        """agent_trajectory_ref path encodes the round number ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        config = _build_discovery_config()

        mock_agent = _make_mock_agent()

        with (
            patch.object(
                exp,
                "_create_search_agent",
                return_value=mock_agent,
            ),
            patch.object(
                exp,
                "_execute_agent_with_timeout",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result3 = await exp.run_search(
                task=task,
                config=config,
                round_number=3,
                agent_trajectory_dir=str(tmp_path),
            )
            result7 = await exp.run_search(
                task=task,
                config=config,
                round_number=7,
                agent_trajectory_dir=str(tmp_path),
            )

        assert result3.agent_trajectory_ref is not None
        assert result7.agent_trajectory_ref is not None
        assert "r3" in result3.agent_trajectory_ref
        assert "r7" in result7.agent_trajectory_ref
        assert result3.agent_trajectory_ref != result7.agent_trajectory_ref
