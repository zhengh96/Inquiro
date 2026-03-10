"""Query strategy data models for structured search execution 🗺️.

Defines domain-agnostic models that represent a parsed query strategy:
ordered search queries, tool allocations, and supporting guidance text.
These models live in Inquiro (domain-agnostic layer) because SearchExp
consumes them; TargetMaster's QueryTemplateParser produces them.

Models are designed to be opaque carriers — Inquiro uses the
``query_sections`` ordering and ``tool_name`` routing without
interpreting pharma-specific content.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


__all__ = [
    "QuerySection",
    "ToolAllocation",
    "QueryStrategy",
]


class QuerySection(BaseModel):
    """A single prioritised search query within a strategy 🔍.

    Attributes:
        id: Query identifier extracted from the template header,
            e.g. ``"Q1"``.
        priority: Numeric priority level (lower = higher priority),
            e.g. ``1``, ``2``, ``3``.
        tool_name: Normalised lowercase tool name, e.g.
            ``"bohrium"``, ``"perplexity"``, ``"brave"``.
        description: Optional human-readable focus description
            extracted from the parenthetical annotation in the
            header, e.g. ``"end-to-end feasibility synthesis"``.
        content: Full markdown content of the query section
            (everything between this header and the next).
    """

    id: str = Field(description="Query identifier, e.g. 'Q1'.")
    priority: int = Field(description="Priority level; lower = higher priority.")
    tool_name: str = Field(
        description="Normalised lowercase tool name, e.g. 'bohrium'."
    )
    description: str = Field(
        default="",
        description="Optional parenthetical description from the header.",
    )
    content: str = Field(description="Full markdown body of this query section.")


class ToolAllocation(BaseModel):
    """Percentage allocation for a single search tool 📊.

    Attributes:
        tool_name: Normalised lowercase tool name.
        percentage: Integer allocation percentage (0–100).
    """

    tool_name: str = Field(description="Normalised lowercase tool name.")
    percentage: int = Field(description="Allocation percentage 0-100.", ge=0, le=100)


class QueryStrategy(BaseModel):
    """Fully parsed query strategy for one sub-item 📋.

    Produced by ``QueryTemplateParser.parse()`` and consumed by
    Inquiro's ``SearchExp``.  All text fields carry raw markdown so
    the agent prompt can be built at call-time without further
    parsing.

    Attributes:
        sub_item_id: Identifier of the sub-item this strategy belongs
            to, e.g. ``"genetic_evidence"``.
        alias_expansion: Combined text of the Inputs (Section 0) and
            Alias Expansion (Section 1) blocks.
        query_sections: Ordered list of ``QuerySection`` objects,
            sorted by appearance in the template file.
        tool_allocations: Tool allocation percentages extracted from
            the Tool Allocation Strategy section (may be empty when
            the section is absent).
        follow_up_rules: Raw markdown of the Follow-up Guidance
            section (may be empty string).
        evidence_tiers: Raw markdown of the Evidence Strength Tiers
            section (may be empty string).
    """

    sub_item_id: str = Field(
        description="Sub-item identifier, e.g. 'genetic_evidence'."
    )
    alias_expansion: str = Field(
        description="Combined inputs and alias expansion guidance text."
    )
    query_sections: list[QuerySection] = Field(
        default_factory=list,
        description="Ordered list of query sections.",
    )
    tool_allocations: list[ToolAllocation] = Field(
        default_factory=list,
        description="Tool allocation percentages.",
    )
    follow_up_rules: str = Field(
        default="",
        description="Raw markdown of the follow-up guidance section.",
    )
    evidence_tiers: str = Field(
        default="",
        description="Raw markdown of the evidence strength tiers section.",
    )
