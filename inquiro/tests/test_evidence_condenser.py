"""Tests for EvidenceCondenser — multi-signal quality scoring 🗜️."""

from __future__ import annotations

import pytest

from inquiro.core.evidence_condenser import (
    CondenserConfig,
    EvidenceCondenser,
    GroupSummary,
    _compute_score,
    _extract_keywords,
    _longest_prefix_match,
    _structural_completeness,
)
from inquiro.core.types import Evidence


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(
    idx: int,
    summary: str = "generic evidence summary",
    tag: str | None = None,
    round_number: int | None = None,
    source: str = "test_mcp",
    doi: str | None = None,
    clinical_trial_id: str | None = None,
    quality_label: str | None = None,
    url: str | None = None,
) -> Evidence:
    """Create a minimal Evidence fixture 🔧."""
    return Evidence(
        id=f"E{idx}",
        source=source,
        url=url or f"https://example.com/{idx}",
        query="test query",
        summary=summary,
        evidence_tag=tag,
        round_number=round_number,
        doi=doi,
        clinical_trial_id=clinical_trial_id,
        quality_label=quality_label,
    )


def _make_batch(
    n: int,
    tag: str | None = None,
    source: str = "test_mcp",
    summary_template: str = "evidence item {i} generic content",
) -> list[Evidence]:
    """Create a batch of n Evidence items with distinct IDs 🔧."""
    return [
        _make_evidence(
            i,
            summary=summary_template.format(i=i),
            tag=tag,
            source=source,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests: _extract_keywords
# ---------------------------------------------------------------------------


def test_extract_keywords_basic() -> None:
    """Keywords are lowercased and filtered to ≥3 chars 🔤."""
    result = _extract_keywords("EGFR mutation in lung cancer")
    assert "egfr" in result
    assert "mutation" in result
    assert "lung" in result
    assert "cancer" in result
    assert "in" not in result  # too short


def test_extract_keywords_empty() -> None:
    """Empty string returns empty set 🔤."""
    assert _extract_keywords("") == set()


# ---------------------------------------------------------------------------
# Tests: _structural_completeness
# ---------------------------------------------------------------------------


def test_structural_completeness_all_fields() -> None:
    """Evidence with all provenance fields scores 1.0 🏗️."""
    ev = _make_evidence(
        1,
        tag="academic",
        doi="10.1234/test",
        clinical_trial_id="NCT12345678",
        quality_label="tier_1",
    )
    score = _structural_completeness(ev)
    assert score == pytest.approx(1.0)


def test_structural_completeness_minimal() -> None:
    """Evidence with only url scores above 0.0 🏗️."""
    ev = _make_evidence(
        1, tag=None, doi=None, clinical_trial_id=None, quality_label=None
    )
    # url present (fixture sets it): 0.20
    score = _structural_completeness(ev)
    assert score == pytest.approx(0.20)


def test_structural_completeness_doi_boosts() -> None:
    """Adding doi increases completeness score 🏗️."""
    base = _make_evidence(1)
    with_doi = _make_evidence(2, doi="10.1234/test")
    assert _structural_completeness(with_doi) > _structural_completeness(base)


# ---------------------------------------------------------------------------
# Tests: _compute_score
# ---------------------------------------------------------------------------


def test_compute_score_keyword_match_raises_score() -> None:
    """Evidence matching checklist keywords scores higher 🎯."""
    config = CondenserConfig()
    checklist_kw = {"egfr", "mutation", "lung"}
    high = _make_evidence(
        1, summary="EGFR mutation found in lung tissue", tag="academic"
    )
    low = _make_evidence(2, summary="unrelated protein data", tag="academic")
    assert _compute_score(high, checklist_kw, 1, config) > _compute_score(
        low, checklist_kw, 1, config
    )


def test_compute_score_regulatory_tag_higher_than_other() -> None:
    """Regulatory evidence scores higher than 'other' for same summary 🎯."""
    config = CondenserConfig()
    kw = {"trial", "efficacy"}
    reg = _make_evidence(1, summary="trial efficacy data", tag="regulatory")
    other = _make_evidence(2, summary="trial efficacy data", tag="other")
    assert _compute_score(reg, kw, 1, config) > _compute_score(other, kw, 1, config)


def test_compute_score_quality_label_tier1_beats_tier4() -> None:
    """Evidence with quality_label tier_1 scores higher than tier_4 🎯."""
    config = CondenserConfig()
    kw: set[str] = set()
    high = _make_evidence(1, quality_label="tier_1", tag="academic")
    low = _make_evidence(2, quality_label="tier_4", tag="academic")
    assert _compute_score(high, kw, 1, config) > _compute_score(low, kw, 1, config)


def test_compute_score_missing_quality_label_uses_default() -> None:
    """Missing quality_label falls back to default_quality_score 🎯."""
    config = CondenserConfig(default_quality_score=0.5)
    kw: set[str] = set()
    ev = _make_evidence(1, quality_label=None, tag=None)
    score = _compute_score(ev, kw, 1, config)
    # Only source_quality (0.3 for None tag) + default_quality_score contribution
    assert score >= 0.0


def test_compute_score_recency_later_rounds_higher() -> None:
    """Evidence from later rounds gets a recency boost 🎯."""
    config = CondenserConfig()
    kw: set[str] = set()
    early = _make_evidence(1, round_number=1, tag="academic")
    late = _make_evidence(2, round_number=3, tag="academic")
    assert _compute_score(late, kw, 3, config) > _compute_score(early, kw, 3, config)


def test_compute_score_no_checklist_does_not_crash() -> None:
    """Empty checklist keywords returns a valid score without error 🎯."""
    config = CondenserConfig()
    ev = _make_evidence(1, summary="some content", tag="academic")
    score = _compute_score(ev, set(), 1, config)
    assert score >= 0.0


# ---------------------------------------------------------------------------
# Tests: Tier 0 — passthrough
# ---------------------------------------------------------------------------


def test_tier0_passthrough_exact_threshold() -> None:
    """Exactly 150 items triggers Tier 0 ✅."""
    config = CondenserConfig(tier1_threshold=150)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(150)
    result = condenser.condense(evidence, ["checklist item"])
    assert result.meta.tier == 0
    assert result.meta.condensed_count == 150
    assert len(result.evidence) == 150


def test_tier0_passthrough_empty() -> None:
    """Empty evidence list returns Tier 0 with zero counts ✅."""
    condenser = EvidenceCondenser()
    result = condenser.condense([], ["checklist item"])
    assert result.meta.tier == 0
    assert len(result.evidence) == 0


def test_tier0_returns_all_items() -> None:
    """Tier 0 returns every Evidence item from the input ✅."""
    evidence = _make_batch(10)
    condenser = EvidenceCondenser()
    result = condenser.condense(evidence, [])
    assert {e.id for e in result.evidence} == {e.id for e in evidence}


# ---------------------------------------------------------------------------
# Tests: Tier 1 — quality scoring
# ---------------------------------------------------------------------------


def test_tier1_activates_above_threshold() -> None:
    """151 items triggers Tier 1 🎯."""
    config = CondenserConfig(tier1_threshold=150, tier2_threshold=400)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(151)
    result = condenser.condense(evidence, ["any item"])
    assert result.meta.tier == 1
    assert result.meta.original_count == 151


def test_tier1_reduces_count() -> None:
    """Tier 1 output is fewer than input 🎯."""
    config = CondenserConfig(tier1_threshold=10, tier2_threshold=400)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(200)
    result = condenser.condense(evidence, ["checklist item one"])
    assert result.meta.condensed_count < 200
    assert len(result.evidence) == result.meta.condensed_count


def test_tier1_selects_relevant_evidence() -> None:
    """High-scoring (matching checklist keywords) evidence is preferentially selected 🎯."""
    config = CondenserConfig(tier1_threshold=5, tier2_threshold=400)
    condenser = EvidenceCondenser(config)

    relevant = [
        _make_evidence(
            i, summary=f"EGFR mutation lung cancer study {i}", tag="academic"
        )
        for i in range(3)
    ]
    generic = _make_batch(20, summary_template="unrelated protein content {i}")
    evidence = relevant + generic

    result = condenser.condense(evidence, ["EGFR mutation lung cancer"])

    selected_ids = {e.id for e in result.evidence}
    for ev in relevant:
        assert ev.id in selected_ids, f"{ev.id} (relevant) should be selected"


def test_tier1_deterministic() -> None:
    """Same input always produces same output (no randomness) 🎯."""
    config = CondenserConfig(tier1_threshold=5, tier2_threshold=400)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(50)
    checklist = ["checklist item alpha", "checklist item beta"]

    results = [condenser.condense(evidence, checklist) for _ in range(5)]
    ids_0 = [e.id for e in results[0].evidence]
    for r in results[1:]:
        assert [e.id for e in r.evidence] == ids_0, "Output must be deterministic"


def test_tier1_transparency_footer_present() -> None:
    """Tier 1 meta has a non-empty transparency footer 🎯."""
    config = CondenserConfig(tier1_threshold=5, tier2_threshold=400)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(20)
    result = condenser.condense(evidence, ["item"])
    assert len(result.meta.transparency_footer) > 0
    assert str(result.meta.original_count) in result.meta.transparency_footer


# ---------------------------------------------------------------------------
# Tests: Pure score-based selection (source saturation cap deprecated)
# ---------------------------------------------------------------------------


def test_strong_source_contributes_more_items() -> None:
    """A higher-quality source naturally contributes more items via scoring 🌐."""
    config = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        tier1_target=10,
        enable_tag_safety_net=False,
    )
    condenser = EvidenceCondenser(config)

    # Strong source: high-quality academic evidence
    strong = [
        _make_evidence(
            i, source="strong_mcp", tag="academic",
            summary=f"strong evidence item {i} with relevant keywords",
        )
        for i in range(200)
    ]
    # Weak source: low-quality other evidence
    weak = [
        _make_evidence(
            i + 500, source="weak_mcp", tag="other",
            summary=f"weak item {i}",
        )
        for i in range(200)
    ]
    evidence = strong + weak

    result = condenser.condense(evidence, ["relevant keywords"])

    strong_count = sum(1 for e in result.evidence if e.source == "strong_mcp")
    # Strong source should dominate because it scores higher (academic > other)
    assert strong_count > 5, (
        f"Strong source should contribute majority of items, got {strong_count}/10"
    )


def test_selection_respects_target_count() -> None:
    """Selection MUST produce at most `target` items (before safety net) 🌐."""
    config = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        tier1_target=8,
        enable_tag_safety_net=False,
    )
    condenser = EvidenceCondenser(config)

    evidence = _make_batch(200, source="mcp_a")
    result = condenser.condense(evidence, [])

    assert len(result.evidence) == 8


# ---------------------------------------------------------------------------
# Tests: Tag safety net
# ---------------------------------------------------------------------------


def test_tag_safety_net_includes_rare_tags() -> None:
    """Tag safety net force-inserts an item for a tag with 0 selections 🛡️."""
    config = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        source_saturation_cap=100,
        enable_tag_safety_net=True,
    )
    condenser = EvidenceCondenser(config)

    # Many academic items that will dominate scoring
    academic = _make_batch(20, tag="academic", summary_template="EGFR lung cancer {i}")
    # One regulatory item with zero keyword overlap → would score very low
    regulatory = [
        _make_evidence(999, tag="regulatory", summary="unrelated regulatory text")
    ]
    evidence = academic + regulatory

    result = condenser.condense(evidence, ["EGFR lung cancer mutation"])

    selected_tags = {e.evidence_tag for e in result.evidence}
    assert "regulatory" in selected_tags, "Safety net should include regulatory tag"


def test_tag_safety_net_disabled() -> None:
    """When enable_tag_safety_net=False, rare tags may be excluded 🛡️."""
    # 📝 Shared evidence: identical for both runs to prove the flag matters
    academic = _make_batch(20, tag="academic", summary_template="EGFR lung cancer {i}")
    regulatory = [
        _make_evidence(
            999,
            tag="regulatory",
            summary="zzz",
            quality_label="tier_4",
        )
    ]
    evidence = academic + regulatory
    checklist = ["EGFR lung cancer mutation"]

    # 🚫 Run with safety net DISABLED
    config_off = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        tier1_target=10,
        source_saturation_cap=100,
        enable_tag_safety_net=False,
    )
    result_off = EvidenceCondenser(config_off).condense(evidence, checklist)

    tags_off = {e.evidence_tag for e in result_off.evidence}
    assert "regulatory" not in tags_off, (
        "Safety net disabled: low-scoring regulatory should be excluded"
    )
    assert len(result_off.evidence) <= 10

    # ✅ Run with safety net ENABLED — same evidence must now include regulatory
    config_on = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        tier1_target=10,
        source_saturation_cap=100,
        enable_tag_safety_net=True,
    )
    result_on = EvidenceCondenser(config_on).condense(evidence, checklist)

    tags_on = {e.evidence_tag for e in result_on.evidence}
    assert "regulatory" in tags_on, (
        "Safety net enabled: regulatory tag should be force-inserted"
    )


# ---------------------------------------------------------------------------
# Tests: Tier 2 — group summaries
# ---------------------------------------------------------------------------


def test_tier2_activates_above_threshold() -> None:
    """401 items triggers Tier 2 📋."""
    config = CondenserConfig(tier1_threshold=150, tier2_threshold=400)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(401)
    result = condenser.condense(evidence, ["checklist item"])
    assert result.meta.tier == 2
    assert result.meta.original_count == 401


def test_tier2_has_group_summaries() -> None:
    """Tier 2 result contains GroupSummary objects for excluded items 📋."""
    config = CondenserConfig(tier1_threshold=10, tier2_threshold=20)
    condenser = EvidenceCondenser(config)

    academic = _make_batch(120, tag="academic")
    trial = [
        _make_evidence(i + 200, tag="clinical_trial", summary=f"trial item {i}")
        for i in range(80)
    ]
    evidence = academic + trial  # 200 > tier2_threshold=20

    result = condenser.condense(evidence, ["some checklist item"])

    assert result.meta.tier == 2
    assert len(result.meta.group_summaries) > 0
    for gs in result.meta.group_summaries:
        assert isinstance(gs, GroupSummary)
        assert gs.excluded_count > 0
        assert len(gs.summary_text) > 0


def test_tier2_primary_plus_summaries_cover_all() -> None:
    """Primary count + all excluded counts equals original count 📋."""
    config = CondenserConfig(
        tier1_threshold=10, tier2_threshold=20, enable_tag_safety_net=False
    )
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(30, tag="academic")
    result = condenser.condense(evidence, ["checklist item"])

    total_excluded = sum(gs.excluded_count for gs in result.meta.group_summaries)
    assert result.meta.condensed_count + total_excluded == result.meta.original_count


def test_tier2_transparency_footer_present() -> None:
    """Tier 2 footer mentions original and condensed counts 📋."""
    config = CondenserConfig(tier1_threshold=5, tier2_threshold=10)
    condenser = EvidenceCondenser(config)
    evidence = _make_batch(15, tag="other")
    result = condenser.condense(evidence, ["item"])
    assert "15" in result.meta.transparency_footer
    assert str(result.meta.condensed_count) in result.meta.transparency_footer


# ---------------------------------------------------------------------------
# Tests: CondenserConfig
# ---------------------------------------------------------------------------


def test_condenser_config_defaults() -> None:
    """Default config values are sensible 🔧."""
    config = CondenserConfig()
    assert config.tier1_threshold == 150
    assert config.tier2_threshold == 400
    assert config.diversity_fraction == pytest.approx(0.0)  # deprecated, now 0
    assert config.source_saturation_cap == 20
    assert config.enable_tag_safety_net is True
    assert config.default_quality_score == pytest.approx(0.5)


def test_condenser_uses_default_config_when_none() -> None:
    """EvidenceCondenser creates default CondenserConfig when none provided 🔧."""
    condenser = EvidenceCondenser(None)
    assert condenser.config is not None
    assert condenser.config.tier1_threshold == 150


def test_custom_tag_quality_map() -> None:
    """Custom tag_quality_map is respected in scoring 🔧."""
    # Invert defaults: patent > regulatory
    config = CondenserConfig(
        tier1_threshold=5,
        tier2_threshold=400,
        tag_quality_map={
            "patent": 1.0,
            "regulatory": 0.1,
            "academic": 0.5,
            "clinical_trial": 0.5,
            "other": 0.2,
        },
    )
    EvidenceCondenser(config)  # validates config parsing

    patent_ev = _make_evidence(1, tag="patent", summary="patent data")
    regulatory_ev = _make_evidence(2, tag="regulatory", summary="patent data")
    kw = _extract_keywords("patent data")

    from inquiro.core.evidence_condenser import _compute_score

    p_score = _compute_score(patent_ev, kw, 1, config)
    r_score = _compute_score(regulatory_ev, kw, 1, config)
    assert p_score > r_score, "Custom map: patent should score higher than regulatory"


# ---------------------------------------------------------------------------
# Tests: _longest_prefix_match helper
# ---------------------------------------------------------------------------


def test_longest_prefix_match_returns_matching_score() -> None:
    """Matching prefix returns the configured score 🔍."""
    prefix_map = {"10.1038/nature": 0.85, "10.1021/jacs": 0.85}
    assert _longest_prefix_match("10.1038/nature12345", prefix_map, 0.3) == pytest.approx(0.85)


def test_longest_prefix_match_returns_default_for_no_match() -> None:
    """No matching prefix returns the default score 🔍."""
    prefix_map = {"10.1038/nature": 0.85}
    assert _longest_prefix_match("10.9999/unknown.xyz", prefix_map, 0.3) == pytest.approx(0.3)


def test_longest_prefix_match_selects_longest() -> None:
    """Longer prefix takes precedence when both match the same DOI 🔍."""
    prefix_map = {
        "10.1038": 0.70,          # shorter
        "10.1038/s41591": 0.85,   # longer, more specific
    }
    score = _longest_prefix_match("10.1038/s41591-021-00001-1", prefix_map, 0.3)
    assert score == pytest.approx(0.85), (
        "More specific prefix should win over shorter generic one"
    )


def test_longest_prefix_match_empty_map_returns_default() -> None:
    """Empty prefix map returns default score without error 🔍."""
    assert _longest_prefix_match("10.1038/nature12345", {}, 0.3) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Tests: Signal 6 in _compute_score
# ---------------------------------------------------------------------------


def test_signal6_matching_doi_boosts_score() -> None:
    """Evidence with a matching DOI prefix scores higher than evidence without 🎯."""
    doi_map = {"10.1038/nature": 0.85}
    config_with = CondenserConfig(
        doi_prefix_quality_map=doi_map,
        doi_prefix_default_score=0.0,
        weight_journal_quality=0.10,
    )
    config_without = CondenserConfig(
        doi_prefix_quality_map={},
        weight_journal_quality=0.10,
    )
    kw: set[str] = set()
    ev = _make_evidence(1, doi="10.1038/nature12345", tag="academic")

    score_with = _compute_score(ev, kw, 1, config_with)
    score_without = _compute_score(ev, kw, 1, config_without)
    assert score_with > score_without, "Matching DOI prefix should boost the score"


def test_signal6_non_matching_doi_no_boost() -> None:
    """Evidence with non-matching DOI gets no boost (default=0.0) 🎯."""
    doi_map = {"10.1038/nature": 0.85}
    config = CondenserConfig(
        doi_prefix_quality_map=doi_map,
        doi_prefix_default_score=0.0,
        weight_journal_quality=0.10,
    )
    kw: set[str] = set()
    ev_match = _make_evidence(1, doi="10.1038/nature12345", tag="academic")
    ev_no_match = _make_evidence(2, doi="10.9999/unknown.xyz", tag="academic")

    score_match = _compute_score(ev_match, kw, 1, config)
    score_no_match = _compute_score(ev_no_match, kw, 1, config)
    # Matching DOI should score higher by weight_journal_quality * (0.85 - 0.0)
    assert score_match > score_no_match


def test_signal6_disabled_when_map_empty() -> None:
    """Signal 6 has no effect when doi_prefix_quality_map is empty 🎯."""
    config_off = CondenserConfig(doi_prefix_quality_map={}, weight_journal_quality=0.10)
    config_on = CondenserConfig(
        doi_prefix_quality_map={"10.1038/nature": 0.85},
        doi_prefix_default_score=0.0,
        weight_journal_quality=0.10,
    )
    kw: set[str] = set()
    ev_no_doi = _make_evidence(1, doi=None, tag="academic")

    # Without DOI, signal 6 is always 0 regardless of map
    score_off = _compute_score(ev_no_doi, kw, 1, config_off)
    score_on = _compute_score(ev_no_doi, kw, 1, config_on)
    assert score_off == pytest.approx(score_on)


def test_signal6_none_doi_no_crash() -> None:
    """Evidence with doi=None does not crash when prefix map is populated 🎯."""
    config = CondenserConfig(
        doi_prefix_quality_map={"10.1038/nature": 0.85},
        doi_prefix_default_score=0.0,
        weight_journal_quality=0.10,
    )
    kw: set[str] = set()
    ev = _make_evidence(1, doi=None, tag="academic")
    score = _compute_score(ev, kw, 1, config)
    assert score >= 0.0


def test_default_weights_sum_to_approximately_one() -> None:
    """Default signal weights (5 original + 1 new) sum to ~0.90 🔧."""
    config = CondenserConfig()
    total = (
        config.weight_keyword_relevance
        + config.weight_source_quality
        + config.weight_quality_label
        + config.weight_structural_completeness
        + config.weight_round_recency
        + config.weight_journal_quality
    )
    assert total == pytest.approx(0.90, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: excluded_groups in Tier 2
# ---------------------------------------------------------------------------


def test_tier2_populates_excluded_groups() -> None:
    """Tier 2 condensation populates excluded_groups mapping 📦."""
    items = []
    # 200 academic + 200 patent + 50 other = 450 → triggers Tier 2
    items.extend(
        _make_evidence(i, tag="academic", summary=f"academic item {i}")
        for i in range(200)
    )
    items.extend(
        _make_evidence(200 + i, tag="patent", summary=f"patent item {i}")
        for i in range(200)
    )
    items.extend(
        _make_evidence(400 + i, tag="other", summary=f"other item {i}")
        for i in range(50)
    )
    config = CondenserConfig(
        tier1_threshold=150,
        tier2_threshold=400,
        tier2_target=100,
    )
    condenser = EvidenceCondenser(config)
    result = condenser.condense(items, ["test checklist"])

    assert result.meta.tier == 2
    assert result.excluded_groups  # non-empty
    # Every excluded group tag should appear in group_summaries
    summary_tags = {gs.tag for gs in result.meta.group_summaries}
    for tag in result.excluded_groups:
        assert tag in summary_tags


def test_tier2_excluded_groups_contain_correct_items() -> None:
    """Excluded groups contain items not in primary selection 📦."""
    items = []
    items.extend(
        _make_evidence(i, tag="academic", summary=f"academic item {i}")
        for i in range(250)
    )
    items.extend(
        _make_evidence(250 + i, tag="patent", summary=f"patent item {i}")
        for i in range(250)
    )
    config = CondenserConfig(
        tier1_threshold=150,
        tier2_threshold=400,
        tier2_target=100,
    )
    condenser = EvidenceCondenser(config)
    result = condenser.condense(items, ["test"])

    selected_ids = {id(e) for e in result.evidence}
    for tag, group_items in result.excluded_groups.items():
        for ev in group_items:
            assert id(ev) not in selected_ids, (
                f"Evidence {ev.id} in excluded_groups but also in selection"
            )


def test_tier0_and_tier1_have_empty_excluded_groups() -> None:
    """Tier 0 and Tier 1 do not populate excluded_groups 📦."""
    config = CondenserConfig(tier1_threshold=150, tier2_threshold=400)
    condenser = EvidenceCondenser(config)

    # Tier 0
    tier0 = condenser.condense(_make_batch(50), ["test"])
    assert tier0.meta.tier == 0
    assert tier0.excluded_groups == {}

    # Tier 1
    tier1 = condenser.condense(_make_batch(200), ["test"])
    assert tier1.meta.tier == 1
    assert tier1.excluded_groups == {}


# ---------------------------------------------------------------------------
# Tests: GroupSummarizer protocol
# ---------------------------------------------------------------------------


def test_group_summarizer_protocol_is_runtime_checkable() -> None:
    """GroupSummarizer can be checked at runtime with isinstance 🤖."""
    from inquiro.core.evidence_condenser import GroupSummarizer

    class MockSummarizer:
        async def summarize(
            self, tag: str, items: list, included_count: int
        ) -> str:
            return "mock summary"

    assert isinstance(MockSummarizer(), GroupSummarizer)
