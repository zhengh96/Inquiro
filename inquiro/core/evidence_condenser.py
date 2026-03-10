"""EvidenceCondenser — three-tier evidence condensation to prevent token overflow 🗜️.

When accumulated evidence exceeds LLM context limits (~128K tokens),
this module selects the most relevant subset and generates lightweight
group summaries for evidence that did not make the primary cut.

Tiers:
    Tier 0 (≤150 items): Passthrough — no condensation needed.
    Tier 1 (151–400 items): Multi-signal quality scoring + top-N selection.
    Tier 2 (401+ items): Tier 1 selection + tag-grouped summary text.

Selection algorithm (Tier 1/2):
    1. Score each item with 6 additive normalised signals (keyword relevance,
       source quality, quality-label score, structural completeness, recency,
       journal quality).
    2. Sort descending, take top N items.
    3. Tag safety net: ensure at least one item per evidence_tag type is
       represented (force-insert the highest-scoring item for any absent tag).

Architecture:
    - Stateless: no cross-round state.
    - Domain-agnostic: operates on Evidence fields only; no pharma terms.
    - Zero LLM for Tier 0/1.  Tier 2 generates template fallback summaries;
      DiscoveryLoop enriches them via GroupSummarizer (LLM, Phase 1b).
    - Condenser itself is fully deterministic (no random, no LLM).
      LLM enrichment is done externally via the GroupSummarizer protocol.

Integration point::

    DiscoveryLoop._run_analysis()
        → EvidenceCondenser.condense(all_evidence, checklist_items)
        → condensed.evidence  (filtered list[Evidence])
        → analysis_executor.execute_analysis(evidence=condensed.evidence)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from inquiro.core.types import Evidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CondenserConfig(BaseModel):
    """Thresholds and signal weights for EvidenceCondenser 🔧.

    All scoring signals are normalised to [0, 1] before weighting, so
    the weights are directly comparable.  They must sum to approximately
    1.0 for scores to remain in a consistent range; the condenser does
    NOT enforce this constraint — intentional asymmetric weighting is
    allowed (e.g. keyword-only mode with weight_keyword_relevance=1.0).

    Attributes:
        tier1_threshold: Evidence count at or below which Tier 0 (passthrough)
            is used.  Above this → Tier 1.
        tier2_threshold: Evidence count at or below which Tier 1 is used.
            Above this → Tier 2.
        diversity_fraction: DEPRECATED — retained for backwards compatibility
            only.  Default changed to 0.0; the parameter is no longer used.
            Will be removed in the next major version.
        tag_quality_map: Normalised quality score [0, 1] per evidence_tag
            string.  Keys should match EvidenceTag values.  Items with an
            unknown tag receive 0.3 (below "other" baseline).
        quality_label_map: Normalised quality score [0, 1] per quality_label
            value.  Supports both EvidenceTier keys ("tier_1" … "tier_4")
            and legacy string labels ("high", "medium", "low").
        default_quality_score: Score used when quality_label is None or
            absent from quality_label_map.  Default 0.5 (neutral).
        source_saturation_cap: Maximum evidence items from the same MCP
            source (ev.source) before the greedy pass skips that source.
            Prevents a single search server from dominating the selection.
        enable_tag_safety_net: When True, after the greedy pass any
            evidence_tag with zero selected items has its highest-scoring
            candidate force-inserted (appended beyond the target count).
        doi_prefix_quality_map: DOI prefix to quality score mapping.
            Longest-prefix match is used.  Empty dict disables Signal 6.
        doi_prefix_default_score: Score for DOIs not matching any prefix.
            Defaults to 0.0 (no boost for unrecognised journals).
        weight_keyword_relevance: Weight for the keyword-overlap signal.
        weight_source_quality: Weight for the evidence_tag quality signal.
        weight_quality_label: Weight for the quality_label signal.
        weight_structural_completeness: Weight for the field-presence signal.
        weight_round_recency: Weight for the search-round recency signal.
        weight_journal_quality: Weight for DOI prefix journal quality signal.
    """

    # Tier thresholds
    tier1_threshold: int = Field(
        default=150,
        ge=1,
        description="Max items for Tier 0 passthrough.",
    )
    tier2_threshold: int = Field(
        default=400,
        ge=1,
        description="Max items for Tier 1 filter; above → Tier 2.",
    )
    tier1_target: int = Field(
        default=160,
        ge=1,
        description="Primary evidence budget for Tier 1 selection.",
    )
    tier2_target: int = Field(
        default=150,
        ge=1,
        description="Primary evidence budget for Tier 2 selection.",
    )

    # Deprecated knobs
    diversity_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description=(
            "DEPRECATED — no longer used. Retained for API compatibility. "
            "Will be removed in the next major version."
        ),
    )
    top_k_per_item: int = Field(
        default=15,
        ge=1,
        description=("DEPRECATED — no longer used. Retained for API compatibility."),
    )

    # Source saturation (DEPRECATED)
    source_saturation_cap: int = Field(
        default=20,
        ge=1,
        description=(
            "DEPRECATED — no longer used. Selection is now purely score-based. "
            "Tag safety net provides diversity. Retained for API compatibility."
        ),
    )
    enable_tag_safety_net: bool = Field(
        default=True,
        description=(
            "Ensure at least one item per evidence_tag is included "
            "after the greedy pass (force-inserts if needed)."
        ),
    )

    # Quality signal maps (domain-agnostic: keys are strings, values 0–1)
    tag_quality_map: dict[str, float] = Field(
        default_factory=lambda: {
            "regulatory": 1.0,
            "clinical_trial": 0.8,
            "academic": 0.6,
            "patent": 0.4,
            "other": 0.2,
        },
        description="Normalised source-quality score per evidence_tag value.",
    )
    quality_label_map: dict[str, float] = Field(
        default_factory=lambda: {
            # EvidenceTier keys
            "tier_1": 1.0,
            "tier_2": 0.75,
            "tier_3": 0.50,
            "tier_4": 0.25,
            # Legacy string labels
            "high": 1.0,
            "medium": 0.50,
            "low": 0.25,
        },
        description="Normalised quality score per quality_label value.",
    )
    default_quality_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality score used when quality_label is None.",
    )

    # DOI prefix quality map (domain-agnostic: keys are DOI prefix strings)
    doi_prefix_quality_map: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "DOI prefix to quality score mapping. "
            "Longest-prefix match is used. Empty dict disables this signal."
        ),
    )
    doi_prefix_default_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score for DOIs not matching any prefix. 0.0 = no boost.",
    )

    # Scoring signal weights
    weight_keyword_relevance: float = Field(
        default=0.35,
        ge=0.0,
        description="Weight for keyword-overlap relevance signal.",
    )
    weight_source_quality: float = Field(
        default=0.15,
        ge=0.0,
        description="Weight for evidence_tag quality signal.",
    )
    weight_quality_label: float = Field(
        default=0.15,
        ge=0.0,
        description="Weight for quality_label signal.",
    )
    weight_structural_completeness: float = Field(
        default=0.10,
        ge=0.0,
        description="Weight for field-presence structural completeness signal.",
    )
    weight_round_recency: float = Field(
        default=0.05,
        ge=0.0,
        description="Weight for search-round recency signal.",
    )
    weight_journal_quality: float = Field(
        default=0.10,
        ge=0.0,
        description="Weight for DOI prefix journal quality signal.",
    )


# ---------------------------------------------------------------------------
# Group summarizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GroupSummarizer(Protocol):
    """Protocol for LLM-based group summarisation of excluded evidence 🤖.

    Implementations receive a tag group of excluded evidence items and
    produce a concise synthesis highlighting key findings, contradictions,
    and coverage gaps.  Used in Tier 2 condensation to replace lightweight
    template summaries with richer LLM-generated text.

    The protocol is intentionally simple (one method) to allow swapping
    between mock and real LLM implementations in tests.
    """

    async def summarize(
        self,
        tag: str,
        items: list[Evidence],
        included_count: int,
    ) -> str:
        """Summarize a group of excluded evidence items 📝.

        Args:
            tag: Evidence tag group name (e.g. 'academic', 'patent').
            items: Excluded evidence items in this tag group.
            included_count: Number of items from this tag group that
                are already included in the primary evidence selection.

        Returns:
            Concise text summary (150-400 words) suitable for LLM context.
        """
        ...


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class GroupSummary(BaseModel):
    """Lightweight text summary for a tag group in Tier 2 condensation 📋.

    Attributes:
        tag: EvidenceTag value (e.g. 'academic', 'clinical_trial').
        original_count: Number of items in this group before filtering.
        included_count: Number of items included in primary evidence list.
        excluded_count: Number of items NOT in primary list (summarised here).
        summary_text: Human-readable summary of excluded items.
    """

    tag: str = Field(description="Evidence tag group name.")
    original_count: int = Field(description="Total items in this tag group.")
    included_count: int = Field(description="Items present in primary evidence list.")
    excluded_count: int = Field(description="Items captured only in this summary.")
    summary_text: str = Field(description="Summary of excluded evidence.")


class CondensationMeta(BaseModel):
    """Metadata describing what the condenser did 📊.

    Attributes:
        tier: Which tier was applied (0, 1, or 2).
        original_count: Evidence count before condensation.
        condensed_count: Evidence count after condensation.
        group_summaries: Non-empty only for Tier 2 — one entry per tag.
        transparency_footer: Human-readable accounting string.
    """

    tier: int = Field(ge=0, le=2, description="Condensation tier applied.")
    original_count: int = Field(description="Evidence count before condensation.")
    condensed_count: int = Field(description="Evidence count after condensation.")
    group_summaries: list[GroupSummary] = Field(
        default_factory=list,
        description="Tag-grouped summaries (Tier 2 only).",
    )
    transparency_footer: str = Field(
        default="",
        description="Human-readable accounting sentence for LLM context.",
    )


@dataclass
class CondensedEvidence:
    """Output of EvidenceCondenser.condense() 🗜️.

    Attributes:
        evidence: Selected primary evidence to pass to AnalysisExp.
        meta: Metadata describing condensation tier and statistics.
        excluded_groups: Tag → excluded evidence mapping for Tier 2
            LLM enrichment.  Empty for Tier 0/1.  Keyed by evidence_tag
            string (e.g. 'academic', 'patent', 'other').
    """

    evidence: list[Evidence]
    meta: CondensationMeta
    excluded_groups: dict[str, list[Evidence]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase words (≥3 chars) from text for overlap scoring 🔤."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text)}


def _longest_prefix_match(
    doi: str, prefix_map: dict[str, float], default: float
) -> float:
    """Match DOI against prefix map using longest-prefix-first strategy 🔍.

    Iterates all entries and returns the score of the longest matching prefix.
    This ensures a specific prefix (e.g. "10.1038/s41591") takes precedence
    over a shorter one (e.g. "10.1038/").

    Args:
        doi: DOI string to match.
        prefix_map: Dict of {doi_prefix: score}.
        default: Score returned when no prefix matches.

    Returns:
        Quality score in [0, 1].
    """
    best_score = default
    best_len = 0
    for prefix, score in prefix_map.items():
        if doi.startswith(prefix) and len(prefix) > best_len:
            best_score = score
            best_len = len(prefix)
    return best_score


def _structural_completeness(ev: Evidence) -> float:
    """Compute field-presence completeness score [0, 1] 🏗️.

    Rewards evidence that carries provenance metadata (url, doi,
    clinical_trial_id, evidence_tag, quality_label), each weighted
    by its contribution to verifiability.

    Args:
        ev: Evidence item to evaluate.

    Returns:
        Float in [0, 1].
    """
    return (
        0.20 * (ev.url is not None)
        + 0.30 * (ev.doi is not None)
        + 0.20 * (ev.clinical_trial_id is not None)
        + 0.15 * (ev.evidence_tag is not None)
        + 0.15 * (ev.quality_label is not None)
    )


def _compute_score(
    ev: Evidence,
    checklist_keywords: set[str],
    max_round: int,
    config: CondenserConfig,
) -> float:
    """Compute composite quality-relevance score for one Evidence item 🎯.

    Score = Σ(signal_i × weight_i)

    All signals are normalised to [0, 1] before weighting.

    Signals:
        keyword_relevance: Fraction of checklist keywords present in summary.
        source_quality: Normalised quality tier of the evidence_tag.
        quality_label_score: Normalised quality tier from quality_label field.
        structural_completeness: Fraction of provenance fields populated.
        round_recency: Normalised search-round (later = higher, gap-filling).
        journal_quality: DOI prefix lookup score from caller-supplied map.

    Args:
        ev: Evidence item to score.
        checklist_keywords: Union of keywords from all checklist items.
        max_round: Highest round_number seen in the evidence pool.
        config: CondenserConfig with weights and mapping tables.

    Returns:
        Non-negative float; higher means higher quality + relevance.
    """
    # Signal 1: Keyword relevance
    if checklist_keywords:
        summary_kw = _extract_keywords(ev.summary)
        kw_overlap = len(summary_kw & checklist_keywords) / len(checklist_keywords)
        keyword_relevance = min(kw_overlap, 1.0)
    else:
        keyword_relevance = 0.0

    # Signal 2: Source quality via evidence_tag
    source_quality = config.tag_quality_map.get(
        ev.evidence_tag or "other",
        0.3,  # unknown tag → slightly below "other" baseline
    )

    # Signal 3: Quality label score
    quality_label_score = config.quality_label_map.get(
        ev.quality_label or "",
        config.default_quality_score,
    )

    # Signal 4: Structural completeness
    completeness = _structural_completeness(ev)

    # Signal 5: Round recency (later rounds are gap-filling searches)
    round_num = ev.round_number or 1
    if max_round > 1:
        recency = min((round_num - 1) / (max_round - 1), 1.0)
    else:
        recency = 0.0

    # Signal 6: Journal quality via DOI prefix match
    journal_quality = 0.0
    if config.doi_prefix_quality_map and ev.doi:
        journal_quality = _longest_prefix_match(
            ev.doi,
            config.doi_prefix_quality_map,
            config.doi_prefix_default_score,
        )

    return (
        keyword_relevance * config.weight_keyword_relevance
        + source_quality * config.weight_source_quality
        + quality_label_score * config.weight_quality_label
        + completeness * config.weight_structural_completeness
        + recency * config.weight_round_recency
        + journal_quality * config.weight_journal_quality
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EvidenceCondenser:
    """Three-tier evidence condensation engine 🗜️.

    Prevents LLM token overflow when hundreds of evidence items accumulate
    across search rounds.  Selects the most relevant, highest-quality subset
    via deterministic multi-signal scoring and greedy source-saturation
    filtering, preserving signal from every item via group summaries in Tier 2.

    Design principles:
        - Fully deterministic: same input → same output, always.
        - Zero LLM: all scoring uses Evidence fields only.
        - Domain-agnostic: no hardcoded domain knowledge.
        - Soft diversity: source saturation + tag safety net replace random
          sampling with principled, explainable mechanisms.

    Example::

        condenser = EvidenceCondenser(CondenserConfig())
        condensed = condenser.condense(all_evidence, task.checklist)
        result = await executor.execute_analysis(
            evidence=condensed.evidence, ...
        )
    """

    def __init__(self, config: CondenserConfig | None = None) -> None:
        """Initialise with optional config.

        Args:
            config: CondenserConfig instance.  Defaults to CondenserConfig().
        """
        self.config = config or CondenserConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def condense(
        self,
        evidence: Sequence[Evidence],
        checklist_items: Sequence[str],
    ) -> CondensedEvidence:
        """Select or summarise evidence to fit within analysis context 🗜️.

        Args:
            evidence: Full accumulated evidence list from DiscoveryLoop.
            checklist_items: Checklist item texts used to compute relevance.

        Returns:
            CondensedEvidence with filtered primary evidence and metadata.
        """
        ev_list = list(evidence)
        n = len(ev_list)

        if n <= self.config.tier1_threshold:
            return self._tier0(ev_list)
        if n <= self.config.tier2_threshold:
            return self._tier1(ev_list, checklist_items)
        return self._tier2(ev_list, checklist_items)

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    def _tier0(self, evidence: list[Evidence]) -> CondensedEvidence:
        """Passthrough — no condensation required (Tier 0) ✅."""
        meta = CondensationMeta(
            tier=0,
            original_count=len(evidence),
            condensed_count=len(evidence),
            transparency_footer=(
                f"All {len(evidence)} evidence items passed to analysis."
            ),
        )
        logger.debug("🗜️ Condenser Tier 0: %d items — passthrough.", len(evidence))
        return CondensedEvidence(evidence=evidence, meta=meta)

    def _tier1(
        self,
        evidence: list[Evidence],
        checklist_items: Sequence[str],
    ) -> CondensedEvidence:
        """Multi-signal scoring + greedy saturation selection (Tier 1) 🎯."""
        selected = self._select(evidence, checklist_items, self.config.tier1_target)
        footer = (
            f"{len(evidence)} evidence items total; "
            f"{len(selected)} selected via quality scoring for analysis "
            f"({len(evidence) - len(selected)} de-prioritised)."
        )
        meta = CondensationMeta(
            tier=1,
            original_count=len(evidence),
            condensed_count=len(selected),
            transparency_footer=footer,
        )
        logger.info(
            "🗜️ Condenser Tier 1: %d → %d items (quality score + saturation filter).",
            len(evidence),
            len(selected),
        )
        return CondensedEvidence(evidence=selected, meta=meta)

    def _tier2(
        self,
        evidence: list[Evidence],
        checklist_items: Sequence[str],
    ) -> CondensedEvidence:
        """Tier 1 selection + tag-grouped text summaries (Tier 2) 📋.

        Phase 1a: group summaries are lightweight text templates (fallback).
        Phase 1b: DiscoveryLoop enriches with LLM per-group summarisation
        using excluded_groups data returned here.

        Args:
            evidence: Full evidence list (401+ items).
            checklist_items: Checklist items for scoring.

        Returns:
            CondensedEvidence with primary subset, group summaries, and
            excluded_groups mapping for optional LLM enrichment.
        """
        selected = self._select(evidence, checklist_items, self.config.tier2_target)
        selected_ids = {id(e) for e in selected}

        excluded = [e for e in evidence if id(e) not in selected_ids]
        group_summaries = self._build_group_summaries(excluded, selected)

        # 📦 Build excluded groups for downstream LLM enrichment
        excluded_groups: dict[str, list[Evidence]] = defaultdict(list)
        for ev in excluded:
            excluded_groups[ev.evidence_tag or "other"].append(ev)

        footer = (
            f"{len(evidence)} evidence items total; "
            f"{len(selected)} shown in full below; "
            f"{len(excluded)} captured in tag-group summaries."
        )
        meta = CondensationMeta(
            tier=2,
            original_count=len(evidence),
            condensed_count=len(selected),
            group_summaries=group_summaries,
            transparency_footer=footer,
        )
        logger.info(
            "🗜️ Condenser Tier 2: %d → %d items + %d group summaries.",
            len(evidence),
            len(selected),
            len(group_summaries),
        )
        return CondensedEvidence(
            evidence=selected, meta=meta, excluded_groups=dict(excluded_groups)
        )

    # ------------------------------------------------------------------
    # Core selection algorithm
    # ------------------------------------------------------------------

    def _select(
        self,
        evidence: list[Evidence],
        checklist_items: Sequence[str],
        target: int,
    ) -> list[Evidence]:
        """Select up to `target` items via pure quality scoring 🎯.

        Algorithm:
            1. Build checklist keyword union.
            2. Compute max_round from evidence pool.
            3. Score every item with _compute_score() (6 signals, additive).
            4. Sort descending, take top `target` items.
            5. Tag safety net: force-insert the top-scoring item for any
               evidence_tag type absent from the selected set.

        Args:
            evidence: Full evidence list.
            checklist_items: Checklist texts for keyword extraction.
            target: Desired primary selection count (soft limit).

        Returns:
            List of Evidence items; may slightly exceed target when the
            tag safety net inserts items for under-represented tag types.
        """
        if not evidence:
            return []

        # Build shared checklist keyword set
        checklist_keywords: set[str] = set()
        for item in checklist_items:
            checklist_keywords |= _extract_keywords(item)

        # Determine max round for recency normalisation
        max_round = max((ev.round_number or 1 for ev in evidence), default=1)

        # Score all items once
        scored = sorted(
            [
                (ev, _compute_score(ev, checklist_keywords, max_round, self.config))
                for ev in evidence
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Select top-scored items (pure quality ranking)
        selected: list[Evidence] = [ev for ev, _ in scored[:target]]

        # Tag safety net: ensure every observed tag has ≥ 1 representative
        if self.config.enable_tag_safety_net:
            selected = self._apply_tag_safety_net(selected, scored)

        return selected

    def _apply_tag_safety_net(
        self,
        selected: list[Evidence],
        scored: list[tuple[Evidence, float]],
    ) -> list[Evidence]:
        """Force-insert the top-scoring item for any tag absent from selection 🛡️.

        This guarantees that even a rare evidence_tag type (e.g. regulatory
        documents that score lower on keyword overlap) is not completely
        excluded from the analysis context.

        Args:
            selected: Current selection from the greedy pass.
            scored: All evidence sorted by score descending.

        Returns:
            Updated selection with at most len(unique_tags) extra items.
        """
        selected_tags = {ev.evidence_tag or "other" for ev in selected}
        selected_ids = {id(ev) for ev in selected}

        for ev, _ in scored:
            tag = ev.evidence_tag or "other"
            if tag not in selected_tags and id(ev) not in selected_ids:
                selected.append(ev)
                selected_tags.add(tag)
                selected_ids.add(id(ev))

        return selected

    # ------------------------------------------------------------------
    # Group summaries (Tier 2)
    # ------------------------------------------------------------------

    def _build_group_summaries(
        self,
        excluded: list[Evidence],
        selected: list[Evidence],
    ) -> list[GroupSummary]:
        """Build lightweight tag-group summaries for excluded evidence 📋.

        Phase 1a implementation: generates readable text without LLM.
        Each group summary lists source counts, URL samples, and query hints.

        Args:
            excluded: Evidence items NOT in the primary selection.
            selected: Evidence items in the primary selection.

        Returns:
            List of GroupSummary, one per non-empty tag group.
        """
        selected_by_tag: dict[str, int] = defaultdict(int)
        for ev in selected:
            selected_by_tag[ev.evidence_tag or "other"] += 1

        excluded_by_tag: dict[str, list[Evidence]] = defaultdict(list)
        for ev in excluded:
            excluded_by_tag[ev.evidence_tag or "other"].append(ev)

        all_tags = sorted(set(excluded_by_tag.keys()) | set(selected_by_tag.keys()))

        summaries: list[GroupSummary] = []
        for tag in all_tags:
            tag_excluded = excluded_by_tag.get(tag, [])
            if not tag_excluded:
                continue

            included = selected_by_tag.get(tag, 0)
            excluded_count = len(tag_excluded)
            total = included + excluded_count

            sample_queries = list({ev.query for ev in tag_excluded if ev.query})[:5]
            query_hint = (
                "; ".join(f'"{q}"' for q in sample_queries)
                if sample_queries
                else "various queries"
            )

            summary_text = (
                f"[{tag.upper()} group — {excluded_count} additional items "
                f"(total in group: {total}, {included} shown in full above)]\n"
                f"Search queries producing these items: {query_hint}.\n"
                f"These items were de-prioritised by quality/relevance scoring "
                f"but may contain supporting or contradictory evidence."
            )
            summaries.append(
                GroupSummary(
                    tag=tag,
                    original_count=total,
                    included_count=included,
                    excluded_count=excluded_count,
                    summary_text=summary_text,
                )
            )

        return summaries
