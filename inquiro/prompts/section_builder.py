"""Prompt section builders for Inquiro agents 🏗️.

Stateless utility class that extracts prompt section rendering logic
from SearchExp._render_system_prompt(), making each section
independently testable and reusable.

Design principle (review feedback S5):
    - ``augment_*()`` methods MUTATE data structures (side effects).
    - ``format_*()`` methods are PURE — they only produce strings.
    These two concerns are strictly separated.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inquiro.core.types import Checklist


class PromptSectionBuilder:
    """Stateless builder for system prompt sections 📝.

    All methods are static — no instance state required.
    This class serves as a namespace for related section-building
    functions, each independently testable.

    Usage::

        builder = PromptSectionBuilder
        checklist_md = builder.format_checklist(checklist)
        schema = builder.augment_schema_with_coverage(schema, checklist)
        schema_str = builder.format_output_schema(schema)
    """

    # -- Augmentation (side-effect methods) --------------------------------

    @staticmethod
    def augment_schema_with_coverage(
        schema: dict[str, Any],
        checklist: Any,
    ) -> dict[str, Any]:
        """Inject checklist_coverage field into output schema 🔧.

        Creates a deep copy of the schema and adds a ``checklist_coverage``
        object with ``required_covered`` and ``required_missing`` arrays.
        The original schema dict is NOT mutated.

        Args:
            schema: Original JSON Schema dict for output validation.
            checklist: Checklist object with ``required`` attribute.
                If None or has no required items, returns schema unchanged.

        Returns:
            New schema dict with ``checklist_coverage`` field added,
            or a deep copy of the original if no checklist items exist.
        """
        schema = copy.deepcopy(schema)

        has_required = (
            checklist is not None
            and hasattr(checklist, "required")
            and checklist.required
        )
        if not has_required:
            return schema

        schema.setdefault("properties", {})["checklist_coverage"] = {
            "type": "object",
            "properties": {
                "required_covered": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Checklist item IDs (R1, R2, ...) covered "
                        "by at least one evidence item"
                    ),
                },
                "required_missing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": ("Checklist item IDs lacking sufficient evidence"),
                },
            },
            "required": ["required_covered", "required_missing"],
        }
        schema.setdefault("required", []).append("checklist_coverage")

        return schema

    # -- Pure formatting methods -------------------------------------------

    @staticmethod
    def format_rules_summary(
        sub_item_id: str | None,
        global_one_liner: str = "",
    ) -> str:
        """Format a short rules summary plus get_reference guidance 📋.

        Used for Progressive Disclosure: when sub_item_id is set, the
        prompt shows only this summary; full rules are loaded via
        use_skill(sub-item-rules, get_reference, <id>.md).

        Args:
            sub_item_id: Current sub-item identifier (e.g., "market_size").
                When non-empty, output includes get_reference guidance.
            global_one_liner: Optional one-line global evidence requirement
                (e.g., "All factual claims must cite evidence.").

        Returns:
            Markdown string: optional global line, optional Sub-item line,
            and when sub_item_id is set, the get_reference instruction.
        """
        parts: list[str] = []
        if global_one_liner.strip():
            parts.append(global_one_liner.strip())
        if sub_item_id and sub_item_id.strip():
            parts.append(f"Sub-item: **{sub_item_id.strip()}**.")
            parts.append(
                "Full evaluation criteria: use "
                '`use_skill(skill_name="sub-item-rules", '
                'action="get_reference", '
                f'reference_name="{sub_item_id.strip()}.md")` to load.'
            )
        if not parts:
            return (
                "No specific evaluation rules summary. "
                "Use the task context and checklist to guide research."
            )
        return "\n\n".join(parts)

    @staticmethod
    def format_checklist(
        checklist: Checklist,
        sub_item_id: str | None = None,
    ) -> str:
        """Format search checklist as Markdown for prompt injection 📋.

        Converts the structured Checklist into a human-readable
        Markdown list. When sub_item_id is set (Progressive Disclosure),
        only id + description + coverage threshold and get_reference
        guidance are included; keywords and suggested_sources are omitted.

        Args:
            checklist: Search checklist with required and optional items.
            sub_item_id: When set, output is summary-only (id + description
                + threshold + get_reference). When None, full format with
                keywords and suggested_sources is used.

        Returns:
            Formatted Markdown string for the SEARCH CHECKLIST section.
        """
        lines: list[str] = []
        summary_mode = bool(sub_item_id and sub_item_id.strip())

        # ✅ Required items
        if checklist.required:
            lines.append("## Required Items (MUST investigate)\n")
            for item in checklist.required:
                lines.append(f"- **[{item.id}]** {item.description}")
                if not summary_mode:
                    if item.keywords:
                        kw_str = ", ".join(item.keywords)
                        lines.append(f"  - Keywords: {kw_str}")
                    if item.suggested_sources:
                        src_str = ", ".join(item.suggested_sources)
                        lines.append(f"  - Suggested sources: {src_str}")

        # 📌 Optional items
        if checklist.optional:
            lines.append("\n## Optional Items (investigate if time permits)\n")
            for item in checklist.optional:
                lines.append(f"- **[{item.id}]** {item.description}")
                if not summary_mode and item.keywords:
                    kw_str = ", ".join(item.keywords)
                    lines.append(f"  - Keywords: {kw_str}")

        if not lines:
            return "No specific search checklist provided."

        # 📊 Coverage threshold
        lines.append(
            f"\nCoverage threshold: "
            f"{checklist.coverage_threshold:.0%} "
            f"of required items must be addressed."
        )

        if summary_mode:
            lines.append(
                "\nFor detailed keywords and suggested sources, use "
                '`use_skill(skill_name="sub-item-checklists", '
                'action="get_reference", '
                f'reference_name="{sub_item_id.strip()}.yaml")` to load.'
            )

        return "\n".join(lines)

    @staticmethod
    def format_prior_context(prior_context_text: str) -> str:
        """Format prior context section for prompt injection 📝.

        Wraps raw prior context text with the standard section header
        and instructions.

        Args:
            prior_context_text: Raw prior context string. May be empty.

        Returns:
            Formatted PRIOR CONTEXT section, or empty string if no
            context is provided.
        """
        if not prior_context_text:
            return ""

        return (
            "# PRIOR CONTEXT\n\n"
            "The following context is from previous research rounds. "
            "Build upon this knowledge and avoid redundant searches."
            "\n\n"
            f"{prior_context_text}"
        )

    @staticmethod
    def format_output_schema(schema: dict[str, Any]) -> str:
        """Serialize output schema to formatted JSON string 📊.

        Args:
            schema: JSON Schema dict for output validation.

        Returns:
            Pretty-printed JSON string (2-space indent, unicode preserved).
        """
        return json.dumps(schema, indent=2, ensure_ascii=False)

    @staticmethod
    def format_available_tools(tools: Any) -> str:
        """Build available-tools summary from ToolRegistry 🔧.

        Groups MCP tools by their source server and lists tool names
        with descriptions. Produces a Markdown section suitable for
        injection into the system prompt.

        Args:
            tools: ToolRegistry instance. May be None or empty.

        Returns:
            Formatted Markdown section listing tools by server, or a
            fallback message if no MCP tools are registered.
        """
        if tools is None or len(tools) == 0:
            return "No search tools are currently available."

        # 🗂️ Group MCP tools by server
        server_names = tools.get_mcp_server_names()
        if not server_names:
            return "No search tools are currently available."

        lines: list[str] = ["## Available Search Tools\n"]
        for server in server_names:
            server_tools = tools.get_tools_by_server(server)
            if not server_tools:
                continue
            lines.append(f"**{server}** ({len(server_tools)} tools):")
            for tool in server_tools:
                desc = getattr(tool, "tool_description", "") or ""
                # 📐 Truncate long descriptions for prompt compactness
                if len(desc) > 120:
                    desc = desc[:117] + "..."
                lines.append(f"  - `{tool.name}`: {desc}")
            lines.append("")  # blank line between servers

        lines.append(
            "You **SHOULD** use tools from at least 2 different servers "
            "to ensure source diversity."
        )
        return "\n".join(lines)

    @staticmethod
    def format_reasoning_protocol() -> str:
        """Format the explicit reasoning protocol section 🧠.

        Instructs the agent to output structured reasoning before
        and after every tool call, following the ReAct pattern.

        Returns:
            Markdown text for the REASONING PROTOCOL section.
        """
        return (
            "# REASONING PROTOCOL\n\n"
            "Before EVERY tool call, you **MUST** output a "
            "reasoning block:\n"
            "1. **Gap Identified**: What specific information "
            "is currently missing from the checklist?\n"
            "2. **Tool Selection Rationale**: Why this specific "
            "tool? What makes it the best choice for this gap?\n"
            "3. **Expected Evidence**: What type and quality of "
            "evidence should this search yield?\n\n"
            "After EVERY tool result, you **MUST** assess:\n"
            "1. **Evidence Quality**: Rate the result "
            "(high / medium / low / irrelevant)\n"
            "2. **Gap Status**: Was the information gap filled? "
            "Partially? Any new gaps discovered?\n"
            "3. **Next Action**: Continue searching, switch "
            "strategy, reflect on progress, or prepare to "
            "finish?\n\n"
            "You **SHOULD** call the `reflect` tool periodically"
            " (every 3-4 search rounds) to assess overall "
            "progress before deciding to finish.\n"
        )

    @staticmethod
    def format_tool_selection_guide(
        tool_configs: dict[str, dict[str, Any]],
    ) -> str:
        """Format tool selection guidance for the agent 🎯.

        Generates a decision-tree style guide based on MCP server
        configurations, helping the agent choose the right tool
        for each information need.

        Args:
            tool_configs: Mapping of server_name to config dict.
                Expected keys: description, use_when, domains.

        Returns:
            Markdown text for the TOOL SELECTION GUIDE section.
            Empty string if no configs provided.
        """
        if not tool_configs:
            return ""

        lines: list[str] = ["# TOOL SELECTION GUIDE\n"]
        lines.append(
            "Choose the most appropriate search tool based on "
            "your current information need:\n"
        )

        for server_name, config in sorted(tool_configs.items()):
            desc = config.get("description", "No description")
            use_when = config.get("use_when", "")
            differs_from = config.get("differs_from", "")
            domains = config.get("domains", [])

            lines.append(f"- **{server_name}**: {desc}")
            if use_when:
                lines.append(f"  - Use when: {use_when}")
            if differs_from:
                lines.append(f"  - Differs: {differs_from}")
            if domains:
                lines.append(f"  - Best for: {', '.join(domains)} research")
            lines.append("")

        lines.append(
            "When uncertain which tool to use, prefer tools "
            "matching your current research domain. If no "
            "domain-specific tool is available, use a "
            "general-purpose search tool."
        )

        return "\n".join(lines)

    @staticmethod
    def format_query_strategy(
        query_strategy_dict: dict[str, Any] | None,
    ) -> tuple[str, str]:
        """Extract alias expansion and query section guide from strategy 📋.

        Parses the opaque query_strategy dict (produced by TargetMaster's
        QueryTemplateParser) and formats two prompt sections:
        1. alias_expansion — for Phase 1 (Entity Alias Enumeration)
        2. query_section_guide — for guided Query Generation

        Args:
            query_strategy_dict: Opaque strategy dict, or None.

        Returns:
            Tuple of (alias_expansion_text, query_section_guide_text).
            Both are empty strings when strategy is None.
        """
        if query_strategy_dict is None:
            return ("", "")

        # 🔤 Extract alias expansion (always returned when present)
        alias_expansion = query_strategy_dict.get("alias_expansion", "")

        # 🔍 Extract query sections and tool allocations
        query_sections = query_strategy_dict.get("query_sections", [])
        tool_allocations = query_strategy_dict.get("tool_allocations", [])

        # 📋 Build query section guide only when sections are present
        if not query_sections:
            return (alias_expansion, "")

        lines: list[str] = [
            "## Query Section Guide (from template)",
            "",
            "The following pre-defined query sections have been prepared. Use them",
            "as a starting framework — you MAY adjust wording and add additional",
            "queries, but SHOULD execute at least one query per section.",
            "",
        ]

        # 🔢 Render each query section
        for section in query_sections:
            sec_id = section.get("id", "Q?")
            priority = section.get("priority", "?")
            tool_name = section.get("tool_name", "")
            description = section.get("description", "")
            content = section.get("content", "")

            lines.append(f"### {sec_id} [Priority {priority}] → {tool_name}")
            if description:
                lines.append(description)
            if content:
                lines.append(content)
            lines.append("")

        # 📊 Render tool allocation table when present
        if tool_allocations:
            lines.append("### Tool Allocation Guidance")
            lines.append("| Tool | Allocation |")
            lines.append("|------|-----------|")
            for allocation in tool_allocations:
                t_name = allocation.get("tool_name", "")
                pct = allocation.get("percentage", 0)
                lines.append(f"| {t_name} | {pct}% |")

        # 📊 Render evidence tiers when present
        evidence_tiers = query_strategy_dict.get("evidence_tiers", "")
        if evidence_tiers:
            lines.append("")
            lines.append("### Evidence Strength Tiers (from template)")
            lines.append(evidence_tiers)

        query_section_guide = "\n".join(lines)
        return (alias_expansion, query_section_guide)

    # 📐 Max length for one-line skill description in summary (Progressive Disclosure)
    _SKILL_DESC_MAX_LEN = 80

    @staticmethod
    def format_available_skills(skill_registry: Any) -> str:
        """Build short available-skills summary + get_reference guidance 🎯.

        Produces one line per skill (name + truncated description) and a
        unified note to load details via use_skill(..., get_reference).
        Does not inject full SKILL.md content (Progressive Disclosure).

        Args:
            skill_registry: SkillRegistry or SkillService instance. May be None.

        Returns:
            Markdown string with skill list and get_reference instruction,
            or empty string if no skills are available.
        """
        if skill_registry is None:
            return ""

        try:
            skills = getattr(skill_registry, "get_all_skills", lambda: [])()
        except Exception:
            return ""

        if not skills:
            return ""

        lines: list[str] = ["# Available Skills\n"]
        max_len = PromptSectionBuilder._SKILL_DESC_MAX_LEN
        for skill in skills:
            name = getattr(
                getattr(skill, "meta_info", None),
                "name",
                getattr(skill, "name", "unknown"),
            )
            desc_raw = (
                getattr(
                    getattr(skill, "meta_info", None),
                    "description",
                    "",
                )
                or ""
            )
            desc = desc_raw.strip()
            if "." in desc and desc.index(".") < max_len:
                desc = desc.split(".")[0].strip() + "."
            elif len(desc) > max_len:
                desc = desc[: max_len - 3].rstrip()
                if desc and not desc.endswith("."):
                    desc = desc + "..."
            if not desc:
                desc = "No description."
            lines.append(f"- **{name}**: {desc}")

        lines.append(
            "\nFor full content use "
            '`use_skill(skill_name="<name>", action="get_reference", '
            'reference_name="<file>"`. '
            "**Recommended order**: "
            "1) alias-expansion → target/indication/modality_expansion.md "
            "(enumerate synonyms FIRST); "
            "2) query-templates → <sub_item_id>.md; "
            "3) evidence-grader → grading-criteria.md; "
            "4) evidence-type-taxonomy → understand evidence types for gap analysis; "
            "5) sub-item rules/checklists as needed."
        )
        return "\n".join(lines)
