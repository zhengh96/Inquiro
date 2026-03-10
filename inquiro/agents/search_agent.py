"""Inquiro SearchAgent -- Pure search agent for DISCOVERY mode 🔍.

Executes search queries using MCP tools and collects raw evidence.
Does NOT perform any analysis — that is AnalysisAgent's responsibility.

This agent is the first stage of the DISCOVERY pipeline:
    SearchAgent (collect) → EvidencePipeline (clean) → AnalysisAgent (analyze)

Key characteristics:
    - Focused solely on search execution and evidence collection 📥
    - Auto-tags evidence with sequential IDs (E1, E2, ...) 🏷️
    - Tracks which MCP servers and queries were used 📊
    - Extracts URLs from observations when available 🔗
    - Outputs a flat list of RawEvidence items, no analysis
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from evomaster.agent.agent import AgentConfig

from inquiro.agents.base import InquiroAgentBase

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from evomaster.agent.tools import ToolRegistry
    from evomaster.skills import SkillRegistry
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter


class SearchAgent(InquiroAgentBase):
    """Pure search agent for DISCOVERY mode 🔍.

    Executes search queries using MCP tools and collects raw evidence.
    Does NOT perform any analysis -- that is AnalysisAgent's job.

    Inherits: SearchAgent -> InquiroAgentBase -> BaseAgent (EvoMaster)

    SearchAgent-specific features (extends InquiroAgentBase):
        1. Auto-tags evidence with sequential IDs (E1, E2, ...) 🏷️
        2. Records raw observation text without summarization 📄
        3. Tracks MCP server usage for diversity metrics 📊
        4. Extracts URLs from tool observations 🔗

    Attributes:
        _evidence_counter: Auto-incrementing evidence ID counter.
        _raw_evidence_records: Collected raw evidence with provenance.
        _servers_used: Set of MCP server names used during search.
        _queries_executed: List of (server, query) tuples executed.
        _search_gaps: Queries that returned errors or empty results.
    """

    VERSION: str = "1.0"

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str,
        user_prompt: str,
        config: AgentConfig,
        output_schema: dict[str, Any],
        task_id: str = "",
        cost_tracker: CostTracker | None = None,
        cancellation_token: CancellationToken | None = None,
        event_emitter: EventEmitter | None = None,
        skill_registry: SkillRegistry | None = None,
        adaptive_search: bool = False,
    ) -> None:
        """Initialize SearchAgent 🔧.

        Args:
            llm: LLM instance for inference.
            tools: Tool registry (MCP search tools + InquiroFinishTool).
            system_prompt: Pre-rendered system prompt emphasizing pure search.
            user_prompt: Pre-rendered user prompt with topic and checklist.
            config: Agent configuration (max_turns, context, etc.).
            output_schema: JSON Schema for search output validation.
            task_id: Unique task identifier for cost tracking and events.
            cost_tracker: Optional cost tracking instance.
            cancellation_token: Optional cancellation signal.
            event_emitter: Optional event emitter for lifecycle events.
            skill_registry: Optional SkillRegistry for agent skill access.
            adaptive_search: When True, agent operates in SRDR mode --
                performing both search and in-context reasoning within
                the same conversation turn.
        """
        # 🔧 Initialize base infrastructure (prompts, tools, cost, cancellation)
        super().__init__(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            output_schema=output_schema,
            task_id=task_id,
            cost_tracker=cost_tracker,
            cancellation_token=cancellation_token,
            event_emitter=event_emitter,
            skill_registry=skill_registry,
        )

        # 🔄 Adaptive search mode (SRDR: search+reason in one Agent)
        self._adaptive_search = adaptive_search

        # 🏷️ Evidence collection tracking (SearchAgent-specific)
        self._evidence_counter: int = 0
        self._raw_evidence_records: list[dict[str, Any]] = []

        # 📊 Server and query diversity tracking (SearchAgent-specific)
        self._servers_used: set[str] = set()
        self._queries_executed: list[dict[str, str]] = []

        # ⚠️ Search gaps — queries that failed or returned empty
        self._search_gaps: list[dict[str, str]] = []

    def _execute_tool(self, tool_call: Any) -> tuple[str, dict[str, Any]]:
        """Override to collect raw evidence on MCP tool calls 🏷️.

        For MCP-sourced tools (identified by ``_is_mcp_tool`` attribute),
        automatically assigns an evidence_id (E1, E2, ...) and records
        the full raw observation without any filtering or summarization.

        Args:
            tool_call: The tool call to execute.

        Returns:
            (observation, info) tuple with evidence metadata injected.
        """
        # 🔄 Delegate to base class for actual tool execution
        observation, info = super()._execute_tool(tool_call)

        # 🏷️ Auto-tag evidence for MCP tool calls
        tool = self.tools.get_tool(tool_call.function.name)
        if tool and getattr(tool, "_is_mcp_tool", False):
            self._evidence_counter += 1
            evidence_id = f"E{self._evidence_counter}"
            source_server = getattr(tool, "_mcp_server", "unknown")
            query = tool_call.function.arguments

            # 📊 Track server usage and query diversity
            self._servers_used.add(source_server)
            self._queries_executed.append(
                {
                    "server": source_server,
                    "query": query,
                    "evidence_id": evidence_id,
                }
            )

            # 📥 Record raw evidence with full provenance
            self._record_raw_evidence(
                evidence_id,
                source_server,
                query,
                observation,
            )

            # ⚠️ Detect empty or error responses
            if self._is_empty_or_error(observation):
                self._search_gaps.append(
                    {
                        "server": source_server,
                        "query": query,
                        "reason": "empty_or_error",
                    }
                )

            # 🏷️ Prepend evidence tag to observation for agent awareness
            observation = (
                f"[Evidence {evidence_id} | Source: {source_server}] {observation}"
            )

            # ✨ Inject evidence_id into info dict
            info["evidence_id"] = evidence_id
            info["source_server"] = source_server

        return observation, info

    def _get_no_tool_call_prompt(self) -> str:
        """Get the prompt to use when agent does not call a tool 📝.

        Instructs the agent to continue searching or submit results.
        In adaptive mode, allows reasoning alongside collection.

        Returns:
            The prompt string instructing agent to continue searching
            or use the finish tool to submit collected evidence.
        """
        if self._adaptive_search:
            return (
                "You must continue your research. "
                "Use the available MCP search tools to gather evidence, "
                "or use the `think` tool to reason about your findings. "
                "When you have sufficient evidence, use the `finish` tool "
                "to submit your final assessment.\n"
                "IMPORTANT: Do not ask for human help. "
                "You must work autonomously."
            )
        return (
            "You must continue executing search queries. "
            "Use the available MCP search tools to gather more raw "
            "evidence, or if you have completed all searches, use the "
            "`finish` tool to submit your collected evidence.\n"
            "IMPORTANT: Do not ask for human help. "
            "Do not analyze or evaluate the evidence — just collect it. "
            "You must work autonomously."
        )

    def _record_raw_evidence(
        self,
        evidence_id: str,
        source_server: str,
        query: str,
        observation: str,
    ) -> None:
        """Record a piece of raw evidence with full provenance 📥.

        Stores the complete observation text without any summarization
        or filtering. URL extraction is attempted for convenience.

        Args:
            evidence_id: Unique evidence identifier (e.g., "E1", "E2").
            source_server: Name of the MCP server that produced the data.
            query: The original query/arguments sent to the tool.
            observation: The raw observation returned by the tool.
        """
        # 🔗 Attempt URL extraction from observation
        extracted_url = self._extract_url(observation)

        record = {
            "id": evidence_id,
            "mcp_server": source_server,
            "source_query": query,
            "observation": observation,
            "url": extracted_url,
        }
        self._raw_evidence_records.append(record)

        self.logger.debug(
            "🏷️ Collected raw evidence %s from %s",
            evidence_id,
            source_server,
        )

    @staticmethod
    def _extract_url(observation: str) -> str | None:
        """Extract the most relevant URL from a tool observation 🔗.

        Searches for URLs in priority order:
        1. DOI URL (https://doi.org/...)
        2. Raw DOI (doi:10.xxxx/...) -> converted to URL
        3. PMID reference -> PubMed URL
        4. Any HTTPS URL

        Args:
            observation: Raw text returned by an MCP tool call.

        Returns:
            Extracted URL string, or None if no URL found.
        """
        if not observation:
            return None

        # 🥇 Priority 1: DOI URL
        doi_url_match = re.search(
            r"https?://doi\.org/[^\s\])\>,]+",
            observation,
        )
        if doi_url_match:
            return doi_url_match.group(0).rstrip(".")

        # 🥈 Priority 2: Raw DOI -> URL
        raw_doi_match = re.search(
            r"\bdoi:?\s*(10\.\d{4,}/[^\s\])\>,]+)",
            observation,
            re.IGNORECASE,
        )
        if raw_doi_match:
            return f"https://doi.org/{raw_doi_match.group(1).rstrip('.')}"

        # 🥉 Priority 3: PMID -> PubMed URL
        pmid_match = re.search(
            r"\bPMID:\s*(\d+)",
            observation,
            re.IGNORECASE,
        )
        if pmid_match:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"

        # 📎 Priority 4: Generic HTTPS URL
        generic_match = re.search(
            r"https://[^\s\])\>,\"']+",
            observation,
        )
        if generic_match:
            return generic_match.group(0).rstrip(".")

        return None

    @staticmethod
    def _is_empty_or_error(observation: str) -> bool:
        """Check whether a tool observation is empty or an error 🔍.

        Args:
            observation: Raw text returned by an MCP tool call.

        Returns:
            True if the observation is empty, whitespace-only, or
            contains an obvious error indicator.
        """
        if not observation or not observation.strip():
            return True
        # ⚠️ Common error patterns from MCP tools
        lower = observation.strip().lower()
        error_indicators = [
            "error:",
            "tool execution error",
            "no results found",
            "request failed",
            "timeout",
        ]
        return any(lower.startswith(ind) for ind in error_indicators)

    # ================================================================
    # 📊 Public accessors for Exp layer and tests
    # ================================================================

    def get_evidence_count(self) -> int:
        """Return the total number of evidence items collected 🏷️.

        Returns:
            The current evidence counter value.
        """
        return self._evidence_counter

    def get_raw_evidence_records(self) -> list[dict[str, Any]]:
        """Return all collected raw evidence records 📥.

        Each record contains: id, mcp_server, source_query,
        observation, and url fields.

        Returns:
            List of raw evidence record dictionaries.
        """
        return list(self._raw_evidence_records)

    def get_servers_used(self) -> set[str]:
        """Return the set of MCP server names used during search 📊.

        Returns:
            Set of server name strings.
        """
        return set(self._servers_used)

    def get_queries_executed(self) -> list[dict[str, str]]:
        """Return all executed queries with their metadata 📊.

        Returns:
            List of dicts with server, query, and evidence_id fields.
        """
        return list(self._queries_executed)

    def get_search_gaps(self) -> list[dict[str, str]]:
        """Return queries that produced empty or error results ⚠️.

        Returns:
            List of dicts with server, query, and reason fields.
        """
        return list(self._search_gaps)

    def get_server_diversity_ratio(self) -> float:
        """Calculate the server diversity ratio 📊.

        Measures how well the agent distributed searches across
        available MCP servers. A ratio of 1.0 means every query
        went to a unique server.

        Returns:
            Ratio of unique servers used to total queries executed.
            Returns 1.0 if no queries have been executed.
        """
        total_queries = len(self._queries_executed)
        if total_queries == 0:
            return 1.0
        return len(self._servers_used) / total_queries
