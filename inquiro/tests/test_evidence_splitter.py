"""Tests for EvidenceSplitter multi-item response splitting 🧪.

Covers all splitting formats:
    - Bohrium markdown (multi-paper)
    - Brave JSON (search result arrays)
    - biomcp Record (multi-record)
    - Pass-through (Perplexity, OpenTargets)
    - Pipeline integration (split → dedup interaction)
"""

from __future__ import annotations

import json


from inquiro.core.evidence_splitter import EvidenceSplitter
from inquiro.core.types import Evidence


# ============================================================================
# 🔧 Fixtures
# ============================================================================


def _make_evidence(
    id: str = "E1",
    summary: str = "A meaningful evidence summary with enough content",
    source: str = "unknown",
    url: str | None = None,
    query: str = "test query",
) -> Evidence:
    """Create a test Evidence instance 🔧."""
    return Evidence(
        id=id,
        summary=summary,
        source=source,
        url=url,
        query=query,
    )


# ============================================================================
# 🔬 Bohrium splitting tests
# ============================================================================


BOHRIUM_MULTI_PAPER = """\
# Enhanced Search: 'TL1A obesity adipose tissue'

Found **3** papers

**[1] A TRAIL-TL1A Paracrine Network Involving Adipocytes**
- Authors: Nitzan Maixner, Tal Pecht
- Journal: Diabetes
- Date: 2020-07-30
- DOI: 10.2337/db19-1231
- URL: https://doi.org/10.2337/db19-1231
- Citations: 8
- Abstract: Elevated TL1A in obesity promotes inflammation.

**[2] TL1A Gene Knockout Leads to Ameliorated Arthritis**
- Authors: Xuehai Wang, Yan Hu
- Journal: The Journal of Immunology
- Date: 2013-10-18
- DOI: 10.4049/jimmunol.1301475
- URL: https://doi.org/10.4049/jimmunol.1301475
- Impact Factor: 3.4
- Abstract: TL1A knockout mice show reduced collagen-induced arthritis.

**[3] TNFSF15 Variants and Inflammatory Bowel Disease**
- Authors: John Smith, Jane Doe
- Journal: Nature Genetics
- DOI: 10.1038/ng.1234
- URL: https://doi.org/10.1038/ng.1234
- PMID: 12345678
- Abstract: GWAS meta-analysis identifies TNFSF15 risk variants."""


class TestBohriumSplitting:
    """Test Bohrium multi-paper markdown splitting 🔬."""

    def test_multi_paper_split(self) -> None:
        """Three-paper Bohrium response splits into 3 items ✅."""
        ev = _make_evidence(
            id="E5",
            summary=BOHRIUM_MULTI_PAPER,
            source="bohrium",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 3
        assert stats.expanded == 2  # 3 items - 1 original

    def test_child_ids_use_dot_notation(self) -> None:
        """Child IDs follow E5.1, E5.2, E5.3 pattern ✅."""
        ev = _make_evidence(
            id="E5",
            summary=BOHRIUM_MULTI_PAPER,
            source="bohrium",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].id == "E5.1"
        assert result[1].id == "E5.2"
        assert result[2].id == "E5.3"

    def test_child_preserves_parent_source_and_query(self) -> None:
        """Child items inherit parent's source and query ✅."""
        ev = _make_evidence(
            id="E5",
            summary=BOHRIUM_MULTI_PAPER,
            source="bohrium",
            query="TL1A obesity",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        for child in result:
            assert child.source == "bohrium"
            assert child.query == "TL1A obesity"

    def test_child_extracts_doi_url(self) -> None:
        """Each child extracts its own DOI URL ✅."""
        ev = _make_evidence(
            id="E5",
            summary=BOHRIUM_MULTI_PAPER,
            source="bohrium",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].url == "https://doi.org/10.2337/db19-1231"
        assert result[1].url == "https://doi.org/10.4049/jimmunol.1301475"
        assert result[2].url == "https://doi.org/10.1038/ng.1234"

    def test_single_paper_not_split(self) -> None:
        """Bohrium response with single paper is not split ✅."""
        single_paper = """\
Found **1** papers

**[1] Single Paper Title**
- Authors: Author One
- DOI: 10.1234/test
- URL: https://doi.org/10.1234/test
- Abstract: Single paper abstract."""

        ev = _make_evidence(
            id="E1",
            summary=single_paper,
            source="bohrium",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 1
        assert result[0].id == "E1"  # Original ID preserved
        assert stats.expanded == 0

    def test_child_summary_contains_section_text(self) -> None:
        """Each child's summary contains its paper section ✅."""
        ev = _make_evidence(
            id="E5",
            summary=BOHRIUM_MULTI_PAPER,
            source="bohrium",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert "TRAIL-TL1A" in result[0].summary
        assert "Ameliorated Arthritis" in result[1].summary
        assert "TNFSF15 Variants" in result[2].summary


# ============================================================================
# 🌐 Brave splitting tests
# ============================================================================


BRAVE_JSON_ARRAY = json.dumps(
    [
        {
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6130856/",
            "title": "Reduced monocyte TNFSF15/TL1A expression",
            "description": "TNFSF15 is associated with IBD in multiple populations.",
        },
        {
            "url": "https://www.nature.com/articles/s41586-021-03145-z",
            "title": "Genome-wide association study of IBD",
            "description": "Novel loci identified including TNFSF15 region.",
        },
        {
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
            "title": "TL1A signaling in intestinal inflammation",
            "description": "Review of TL1A-DR3 axis in gut immune regulation.",
        },
    ]
)


class TestBraveSplitting:
    """Test Brave JSON search result splitting 🌐."""

    def test_json_array_split(self) -> None:
        """Three-item Brave JSON array splits into 3 items ✅."""
        ev = _make_evidence(
            id="E3",
            summary=BRAVE_JSON_ARRAY,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 3
        assert stats.expanded == 2

    def test_brave_child_ids(self) -> None:
        """Brave child IDs follow dot notation ✅."""
        ev = _make_evidence(
            id="E3",
            summary=BRAVE_JSON_ARRAY,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].id == "E3.1"
        assert result[1].id == "E3.2"
        assert result[2].id == "E3.3"

    def test_brave_child_extracts_url(self) -> None:
        """Each Brave child gets its own URL ✅."""
        ev = _make_evidence(
            id="E3",
            summary=BRAVE_JSON_ARRAY,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC6130856/"
        assert result[1].url == "https://www.nature.com/articles/s41586-021-03145-z"
        assert result[2].url == "https://pubmed.ncbi.nlm.nih.gov/12345/"

    def test_brave_child_summary_readable(self) -> None:
        """Brave child summaries are human-readable, not raw JSON ✅."""
        ev = _make_evidence(
            id="E3",
            summary=BRAVE_JSON_ARRAY,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        # Summary should contain title and description, not raw JSON
        assert "Reduced monocyte TNFSF15/TL1A expression" in result[0].summary
        assert "associated with IBD" in result[0].summary

    def test_single_json_object_not_split(self) -> None:
        """Single Brave JSON object is not split ✅."""
        single_obj = json.dumps(
            {
                "url": "https://example.com",
                "title": "Single result",
                "description": "Only one result.",
            }
        )
        ev = _make_evidence(
            id="E1",
            summary=single_obj,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 1
        assert result[0].id == "E1"
        assert stats.expanded == 0

    def test_truncated_json_handling(self) -> None:
        """Truncated JSON array is handled gracefully ✅."""
        truncated = (
            '[{"url":"https://a.com","title":"A","description":"desc A"},'
            '{"url":"https://b.com","title":"B","description":"desc B"},'
            '{"url":"https://c.com","title":"C","descr'  # Truncated!
        )
        ev = _make_evidence(
            id="E2",
            summary=truncated,
            source="brave",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        # Should recover at least the 2 complete objects
        assert len(result) >= 2
        assert result[0].id == "E2.1"


# ============================================================================
# 🧬 biomcp splitting tests
# ============================================================================


BIOMCP_MULTI_RECORD = """\
# Record 1
- Title: Clinical Trial of Anti-TL1A in UC
- Doi Url: https://doi.org/10.1056/NEJMoa2301234
- PMID: 38765432
- Journal: NEJM
- Date: 2024-01-15

# Record 2
- Title: Phase 2 Results for Tulisokibart
- PMID: 39012345
- Journal: Lancet
- Date: 2024-03-20

# Record 3
- Title: TL1A Blockade Safety Profile Analysis
- Doi Url: https://doi.org/10.1016/j.jaci.2024.05.001
- Journal: JACI
- Date: 2024-06-01"""


class TestBiomcpSplitting:
    """Test biomcp multi-Record splitting 🧬."""

    def test_multi_record_split(self) -> None:
        """Three biomcp Records split into 3 items ✅."""
        ev = _make_evidence(
            id="E7",
            summary=BIOMCP_MULTI_RECORD,
            source="biomcp",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 3
        assert stats.expanded == 2

    def test_biomcp_child_ids(self) -> None:
        """biomcp child IDs follow dot notation ✅."""
        ev = _make_evidence(
            id="E7",
            summary=BIOMCP_MULTI_RECORD,
            source="biomcp",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].id == "E7.1"
        assert result[1].id == "E7.2"
        assert result[2].id == "E7.3"

    def test_biomcp_doi_url_extraction(self) -> None:
        """biomcp child extracts DOI URL from structured field ✅."""
        ev = _make_evidence(
            id="E7",
            summary=BIOMCP_MULTI_RECORD,
            source="biomcp",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].url == "https://doi.org/10.1056/NEJMoa2301234"
        assert result[2].url == "https://doi.org/10.1016/j.jaci.2024.05.001"

    def test_biomcp_pmid_fallback_url(self) -> None:
        """biomcp child falls back to PMID URL when no DOI ✅."""
        ev = _make_evidence(
            id="E7",
            summary=BIOMCP_MULTI_RECORD,
            source="biomcp",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        # Record 2 has PMID but no DOI URL
        assert result[1].url == "https://pubmed.ncbi.nlm.nih.gov/39012345/"


# ============================================================================
# 🚫 Pass-through tests (no split)
# ============================================================================


class TestPassThrough:
    """Test sources that should never be split 🚫."""

    def test_perplexity_not_split(self) -> None:
        """Perplexity AI-synthesized text is never split ✅."""
        ev = _make_evidence(
            id="E1",
            summary="Based on search results, TL1A is a key cytokine...",
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 1
        assert result[0].id == "E1"
        assert stats.skipped == 1
        assert stats.expanded == 0

    def test_opentargets_not_split(self) -> None:
        """OpenTargets structured result is never split ✅."""
        ev = _make_evidence(
            id="E2",
            summary='{"target":"TNFSF15","score":0.85,"associations":[]}',
            source="opentargets",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 1
        assert result[0].id == "E2"
        assert stats.skipped == 1

    def test_perplexity_without_urls_falls_through(self) -> None:
        """Perplexity without citation URLs falls through to bohrium ✅."""
        ev = _make_evidence(
            id="E1",
            summary=(
                "Found **5** papers related to this topic. "
                "The key findings suggest that TL1A plays "
                "a significant role in inflammatory pathways. "
                "**[1] First study** showed elevated levels."
            ),
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        # 🔍 No citation URLs → perplexity split returns None →
        # falls through to bohrium pattern match (Found **N** papers)
        assert len(result) == 1
        assert stats.bohrium_split == 1
        assert stats.perplexity_split == 0

    def test_plain_text_not_split(self) -> None:
        """Plain text evidence without multi-item markers passes through ✅."""
        ev = _make_evidence(
            id="E4",
            summary="Simple evidence text without any multi-item format markers.",
            source="unknown_mcp",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 1
        assert result[0].id == "E4"
        assert stats.skipped == 1


# ============================================================================
# 🔗 Pipeline integration tests
# ============================================================================


class TestPipelineIntegration:
    """Test splitter behavior in pipeline context 🔗."""

    def test_split_then_dedup_catches_cross_query_duplicates(self) -> None:
        """Split evidence + dedup catches papers found by different queries ✅."""
        # Simulate: two Bohrium searches found the same paper
        bohrium_1 = """\
Found **2** papers

**[1] A TRAIL-TL1A Paracrine Network**
- DOI: 10.2337/db19-1231
- URL: https://doi.org/10.2337/db19-1231
- Abstract: Elevated TL1A in obesity.

**[2] Unique Paper from Query 1**
- DOI: 10.1234/unique1
- URL: https://doi.org/10.1234/unique1
- Abstract: Only found by first query."""

        bohrium_2 = """\
Found **2** papers

**[1] A TRAIL-TL1A Paracrine Network**
- DOI: 10.2337/db19-1231
- URL: https://doi.org/10.2337/db19-1231
- Abstract: Elevated TL1A in obesity.

**[2] Unique Paper from Query 2**
- DOI: 10.5678/unique2
- URL: https://doi.org/10.5678/unique2
- Abstract: Only found by second query."""

        evidence = [
            _make_evidence(id="E1", summary=bohrium_1, source="bohrium"),
            _make_evidence(id="E2", summary=bohrium_2, source="bohrium"),
        ]

        splitter = EvidenceSplitter()
        expanded, stats = splitter.split(evidence)

        # 2 original → 4 children (2+2)
        assert len(expanded) == 4
        assert stats.expanded == 2

        # The duplicated paper (E1.1 and E2.1) will have identical
        # summaries, so downstream dedup should catch it
        _summaries = [e.summary for e in expanded]
        # First paper from each query should be the same text
        assert expanded[0].summary == expanded[2].summary

    def test_empty_input(self) -> None:
        """Empty evidence list returns empty ✅."""
        splitter = EvidenceSplitter()
        result, stats = splitter.split([])

        assert result == []
        assert stats.input_count == 0
        assert stats.output_count == 0

    def test_mixed_sources_processed_correctly(self) -> None:
        """Mix of splittable and non-splittable sources handled ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=BOHRIUM_MULTI_PAPER,
                source="bohrium",
            ),
            _make_evidence(
                id="E2",
                summary="AI synthesis: TL1A is important.",
                source="perplexity",
            ),
            _make_evidence(
                id="E3",
                summary=BRAVE_JSON_ARRAY,
                source="brave",
            ),
        ]

        splitter = EvidenceSplitter()
        result, stats = splitter.split(evidence)

        # E1 → 3 Bohrium children, E2 → pass-through, E3 → 3 Brave children
        assert len(result) == 7
        assert stats.skipped == 1  # Perplexity
        assert stats.expanded == 4  # (3-1) + (3-1)

    def test_stats_counts_correct(self) -> None:
        """SplitStats counts are accurate ✅."""
        evidence = [
            _make_evidence(
                id="E1",
                summary=BOHRIUM_MULTI_PAPER,
                source="bohrium",
            ),
        ]

        splitter = EvidenceSplitter()
        _, stats = splitter.split(evidence)

        assert stats.input_count == 1
        assert stats.output_count == 3
        assert stats.expanded == 2


# ============================================================================
# 🔎 Perplexity splitting tests
# ============================================================================


PERPLEXITY_WITH_CITATION_BLOCK = """\
TL1A (TNF-like ligand 1A) has been extensively studied in inflammatory diseases [1].

DR3 receptor signaling downstream of TL1A promotes Th9 cell differentiation [2].

Both mechanisms contribute to IBD pathogenesis [1][2].

Sources:
[1] https://pubmed.ncbi.nlm.nih.gov/11111111/
[2] https://www.nature.com/articles/s41586-021-test
"""

PERPLEXITY_INLINE_URLS = """\
TL1A promotes intestinal inflammation through DR3 signaling [1](https://pubmed.ncbi.nlm.nih.gov/22222222/).

Genetic variants in TNFSF15 are associated with IBD risk [2](https://www.nejm.org/doi/full/test).
"""

PERPLEXITY_MULTI_CITE = """\
TL1A-DR3 interaction underlies multiple inflammatory pathways [1][2].

Genetic evidence supports TL1A's role in IBD [1].

DR3 signaling controls Treg differentiation [2].

Sources:
[1] https://pubmed.ncbi.nlm.nih.gov/33333333/
[2] https://www.science.org/doi/test
"""


class TestPerplexitySplitting:
    """Test Perplexity AI answer splitting into per-citation items 🔎."""

    def test_perplexity_with_citation_block_splits(self) -> None:
        """Sources block with [N] URL entries splits into N sub-items ✅."""
        ev = _make_evidence(
            id="E1",
            summary=PERPLEXITY_WITH_CITATION_BLOCK,
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 2
        assert stats.perplexity_split == 2
        assert stats.expanded == 1
        assert result[0].url == "https://pubmed.ncbi.nlm.nih.gov/11111111/"
        assert result[1].url == "https://www.nature.com/articles/s41586-021-test"

    def test_perplexity_sub_item_ids_dot_notation(self) -> None:
        """Perplexity child IDs follow E1.1, E1.2 dot notation ✅."""
        ev = _make_evidence(
            id="E1",
            summary=PERPLEXITY_WITH_CITATION_BLOCK,
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert result[0].id == "E1.1"
        assert result[1].id == "E1.2"

    def test_perplexity_inline_url_extraction(self) -> None:
        """Inline [N](https://...) pattern is extracted when no Sources block ✅."""
        ev = _make_evidence(
            id="E2",
            summary=PERPLEXITY_INLINE_URLS,
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        assert len(result) == 2
        assert result[0].id == "E2.1"
        assert result[1].id == "E2.2"
        assert result[0].url == "https://pubmed.ncbi.nlm.nih.gov/22222222/"
        assert result[1].url == "https://www.nejm.org/doi/full/test"

    def test_perplexity_multi_citation_paragraph(self) -> None:
        """Paragraph referencing [1][2] appears in both child items ✅."""
        ev = _make_evidence(
            id="E3",
            summary=PERPLEXITY_MULTI_CITE,
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        result, _ = splitter.split([ev])

        assert len(result) == 2
        # 🔍 The shared paragraph must appear in both children's summaries
        shared_fragment = (
            "TL1A-DR3 interaction underlies multiple inflammatory pathways"
        )
        assert shared_fragment in result[0].summary
        assert shared_fragment in result[1].summary
        # 🔍 Each child also carries its unique paragraph
        assert "Genetic evidence" in result[0].summary
        assert "DR3 signaling controls Treg" in result[1].summary

    def test_perplexity_citation_number_gaps(self) -> None:
        """Non-contiguous citation numbers [1] [3] [5] split correctly ✅."""
        text = (
            "Gene X is involved in immune regulation [1].\n\n"
            "Pathway Y modulates T cell activation [3].\n\n"
            "Compound Z shows therapeutic potential [5].\n\n"
            "Sources:\n"
            "[1] https://pubmed.ncbi.nlm.nih.gov/44444444/\n"
            "[3] https://www.nature.com/articles/gap-test\n"
            "[5] https://www.science.org/doi/gap-test\n"
        )
        ev = _make_evidence(id="E5", summary=text, source="perplexity")
        splitter = EvidenceSplitter()
        result, stats = splitter.split([ev])

        # 🔍 3 citations with gaps → 3 sub-items
        assert len(result) == 3
        assert stats.perplexity_split == 3
        # 🔍 IDs use citation numbers: [1]→.1, [3]→.3, [5]→.5
        assert result[0].id == "E5.1"
        assert result[1].id == "E5.3"
        assert result[2].id == "E5.5"
        # 🔍 Each child references the correct URL
        assert "44444444" in (result[0].url or "")
        assert "gap-test" in (result[1].url or "")

    def test_perplexity_no_urls_returns_none(self) -> None:
        """_split_perplexity returns None when no citation URLs are present ✅."""
        ev = _make_evidence(
            id="E4",
            summary=(
                "TL1A is a critical cytokine [1][2][3] that regulates "
                "intestinal immunity and inflammation in multiple disease "
                "contexts. Its receptor DR3 is broadly expressed on lymphocytes."
            ),
            source="perplexity",
        )
        splitter = EvidenceSplitter()
        # 🔍 Private method must return None — no actionable URLs
        assert splitter._split_perplexity(ev) is None  # type: ignore[attr-defined]
        # 🔍 Via split(), item passes through with no expansion
        result, stats = splitter.split([ev])
        assert len(result) == 1
        assert stats.expanded == 0
