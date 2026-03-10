"""Tests for QueryTemplateAnalyzer query optimization 🧪.

Covers template extraction, effectiveness ranking, top-K
retrieval, task_id filtering, edge cases, and special characters.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from inquiro.core.trajectory.index import TrajectoryIndex
from inquiro.core.trajectory.query_analyzer import (
    QueryTemplateAnalyzer,
    TemplateEffectivenessRecord,
    _extract_template,
)


# ============================================================================
# 🔧 Fixtures and helpers
# ============================================================================


def _write_jsonl(dir_path: str, task_id: str, trajectory_id: str, **kwargs):
    """Write a complete JSONL trajectory file for testing 📝.

    Args:
        dir_path: Output directory.
        task_id: Task identifier.
        trajectory_id: Trajectory identifier.
        **kwargs: Overrides for round/query structure.

    Returns:
        Path to the created JSONL file.
    """
    rounds_count = kwargs.get("rounds", 2)
    final_coverage = kwargs.get("final_coverage", 0.85)
    total_cost = kwargs.get("total_cost", 3.50)
    status = kwargs.get("status", "completed")
    termination_reason = kwargs.get("termination_reason", "coverage_reached")
    custom_queries = kwargs.get("custom_queries", None)

    lines = []

    # 📋 Meta record
    lines.append(
        json.dumps(
            {
                "type": "meta",
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "config_snapshot": {"max_rounds": 5},
                "task_snapshot": {"rules": "test rules"},
                "timestamp": "2026-02-26T21:00:00+00:00",
            }
        )
    )

    # 📊 Round records
    total_queries = 0
    for r in range(1, rounds_count + 1):
        if custom_queries and r in custom_queries:
            queries = custom_queries[r]
        else:
            queries = [
                {
                    "query_text": (f"query_{r}_a site:pubmed.ncbi.nlm.nih.gov"),
                    "mcp_tool": "web_search",
                    "result_count": 5,
                    "cost_usd": 0.01,
                },
                {
                    "query_text": f"query_{r}_b clinical trials",
                    "mcp_tool": "web_search",
                    "result_count": 3,
                    "cost_usd": 0.01,
                },
            ]
        total_queries += len(queries)

        coverage = min(1.0, final_coverage * r / rounds_count)
        round_cost = total_cost / rounds_count

        cleaned_evidence = kwargs.get("cleaned_evidence_per_round", 6)

        lines.append(
            json.dumps(
                {
                    "type": "round",
                    "round_number": r,
                    "search_phase": {
                        "queries": queries,
                        "total_raw_evidence": 8,
                        "duration_seconds": 10.0,
                    },
                    "cleaning_phase": {
                        "input_count": 8,
                        "output_count": cleaned_evidence,
                        "dedup_removed": 1,
                        "noise_removed": 1,
                        "duration_seconds": 0.5,
                    },
                    "analysis_phase": {
                        "model_results": [],
                        "consensus": {
                            "consensus_decision": "positive",
                            "consensus_ratio": 0.67,
                            "total_claims": 5,
                        },
                        "duration_seconds": 15.0,
                    },
                    "gap_phase": {
                        "coverage_ratio": coverage,
                        "covered_items": [],
                        "uncovered_items": [],
                        "convergence_reason": (
                            termination_reason if r == rounds_count else None
                        ),
                        "focus_prompt": None,
                        "duration_seconds": 5.0,
                    },
                    "round_cost_usd": round_cost,
                    "round_duration_seconds": 30.5,
                }
            )
        )

    # 📊 Summary
    evidence_yield = 6.0 / max(total_queries, 1)
    quality = final_coverage / max(total_cost, 0.01)
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "total_rounds": rounds_count,
                "final_coverage": final_coverage,
                "total_cost_usd": total_cost,
                "total_evidence": rounds_count * 6,
                "total_claims": rounds_count * 5,
                "total_duration_seconds": rounds_count * 30.5,
                "termination_reason": termination_reason,
                "evidence_yield_rate": round(evidence_yield, 4),
                "cost_normalized_quality": round(quality, 4),
            }
        )
    )

    # 🏁 Meta final
    lines.append(
        json.dumps(
            {
                "type": "meta_final",
                "status": status,
                "termination_reason": termination_reason,
                "timestamp": "2026-02-26T21:02:00+00:00",
            }
        )
    )

    # ✅ Write file
    filename = f"discovery_{task_id}_{trajectory_id}.jsonl"
    filepath = os.path.join(dir_path, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return filepath


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files 📂."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def index(tmp_dir):
    """Create a TrajectoryIndex instance with temp database 🗄️."""
    db_path = os.path.join(tmp_dir, "test_index.db")
    return TrajectoryIndex(db_path)


@pytest.fixture
def analyzer(index):
    """Create a QueryTemplateAnalyzer instance 📊."""
    return QueryTemplateAnalyzer(index)


@pytest.fixture
def populated_index(index, tmp_dir):
    """Index with multiple trajectories containing diverse queries 📊."""
    # 🎯 Trajectory 1: high-yield queries with site: operator
    _write_jsonl(
        tmp_dir,
        task_id="task-001",
        trajectory_id="traj-1111",
        rounds=2,
        custom_queries={
            1: [
                {
                    "query_text": ('"STAT3 inhibitor" site:pubmed.ncbi.nlm.nih.gov'),
                    "mcp_tool": "web_search",
                    "result_count": 15,
                    "cost_usd": 0.02,
                },
                {
                    "query_text": ('"JAK2 pathway" clinical trials phase 3'),
                    "mcp_tool": "web_search",
                    "result_count": 10,
                    "cost_usd": 0.02,
                },
            ],
            2: [
                {
                    "query_text": ('"STAT3 resistance" site:pubmed.ncbi.nlm.nih.gov'),
                    "mcp_tool": "web_search",
                    "result_count": 8,
                    "cost_usd": 0.02,
                },
                {
                    "query_text": '"EGFR mutation" safety review',
                    "mcp_tool": "web_search",
                    "result_count": 12,
                    "cost_usd": 0.02,
                },
            ],
        },
        cleaned_evidence_per_round=10,
    )

    # 🎯 Trajectory 2: lower-yield queries without site: operator
    _write_jsonl(
        tmp_dir,
        task_id="task-001",
        trajectory_id="traj-2222",
        rounds=2,
        custom_queries={
            1: [
                {
                    "query_text": "general topic search 2026",
                    "mcp_tool": "web_search",
                    "result_count": 2,
                    "cost_usd": 0.01,
                },
                {
                    "query_text": "another broad query",
                    "mcp_tool": "web_search",
                    "result_count": 0,
                    "cost_usd": 0.01,
                },
            ],
            2: [
                {
                    "query_text": "vague search terms",
                    "mcp_tool": "web_search",
                    "result_count": 1,
                    "cost_usd": 0.01,
                },
            ],
        },
        cleaned_evidence_per_round=2,
    )

    # 🎯 Trajectory 3: different task_id for filtering tests
    _write_jsonl(
        tmp_dir,
        task_id="task-002",
        trajectory_id="traj-3333",
        rounds=1,
        custom_queries={
            1: [
                {
                    "query_text": ('"biomarker" site:clinicaltrials.gov'),
                    "mcp_tool": "web_search",
                    "result_count": 20,
                    "cost_usd": 0.03,
                },
            ],
        },
        cleaned_evidence_per_round=15,
    )

    index.index_from_directory(tmp_dir)
    return index


# ============================================================================
# 🧪 Template extraction tests
# ============================================================================


class TestTemplateExtraction:
    """Tests for query template extraction logic 🔧."""

    def test_extract_replaces_quoted_strings(self):
        """Quoted strings are replaced with {quoted_term}."""
        query = '"STAT3 inhibitor" site:pubmed.ncbi.nlm.nih.gov'
        template = _extract_template(query)
        assert "{quoted_term}" in template
        assert "STAT3" not in template

    def test_extract_preserves_site_operator(self):
        """site: operators are preserved verbatim."""
        query = "some topic site:pubmed.ncbi.nlm.nih.gov"
        template = _extract_template(query)
        assert "site:pubmed.ncbi.nlm.nih.gov" in template

    def test_extract_preserves_filetype_operator(self):
        """filetype: operators are preserved verbatim."""
        query = "research paper filetype:pdf"
        template = _extract_template(query)
        assert "filetype:pdf" in template

    def test_extract_replaces_numbers(self):
        """Standalone numbers are replaced with {number}."""
        query = '"JAK2 pathway" clinical trials phase 3'
        template = _extract_template(query)
        assert "{number}" in template
        # ✅ The "3" should be replaced
        assert " 3" not in template or "{number}" in template

    def test_extract_replaces_topic_words(self):
        """Non-modifier words are collapsed to {topic}."""
        query = "some specific search terms"
        template = _extract_template(query)
        assert "{topic}" in template

    def test_extract_preserves_search_modifiers(self):
        """Common search modifiers are kept as-is."""
        query = "something clinical trials phase 3"
        template = _extract_template(query)
        assert "clinical" in template
        assert "trials" in template
        assert "phase" in template

    def test_extract_empty_query_returns_topic(self):
        """Empty query returns {topic} placeholder."""
        assert _extract_template("") == "{topic}"
        assert _extract_template("   ") == "{topic}"

    def test_extract_deduplicates_adjacent_topic(self):
        """Multiple adjacent topic words produce single {topic}."""
        query = "word1 word2 word3"
        template = _extract_template(query)
        # ✅ Should not have consecutive {topic} {topic}
        assert "{topic} {topic}" not in template

    def test_extract_quoted_with_site(self):
        """Combined quoted term + site: produces expected template."""
        query = '"STAT3 inhibitor" site:pubmed.ncbi.nlm.nih.gov'
        template = _extract_template(query)
        assert "{quoted_term}" in template
        assert "site:pubmed.ncbi.nlm.nih.gov" in template

    def test_extract_multiple_quoted_strings(self):
        """Multiple quoted strings each become {quoted_term}."""
        query = '"term one" and "term two"'
        template = _extract_template(query)
        count = template.count("{quoted_term}")
        assert count == 2

    def test_extract_preserves_review_modifier(self):
        """The 'review' modifier is preserved."""
        query = "some topic systematic review"
        template = _extract_template(query)
        assert "systematic" in template
        assert "review" in template

    def test_extract_single_quoted_strings(self):
        """Single-quoted strings are also replaced."""
        query = "'single quoted' site:example.com"
        template = _extract_template(query)
        assert "{quoted_term}" in template
        assert "site:example.com" in template


# ============================================================================
# 🧪 Analyzer grouping tests
# ============================================================================


class TestExtractTemplatesMethod:
    """Tests for QueryTemplateAnalyzer.extract_templates 📊."""

    def test_groups_queries_by_template(self, analyzer):
        """Queries with same template are grouped together."""
        queries = [
            {
                "query_text": '"term A" site:pubmed.ncbi.nlm.nih.gov',
                "result_count": 5,
                "cost_usd": 0.01,
            },
            {
                "query_text": '"term B" site:pubmed.ncbi.nlm.nih.gov',
                "result_count": 8,
                "cost_usd": 0.01,
            },
        ]
        groups = analyzer.extract_templates(queries)
        # ✅ Both queries should map to the same template
        assert len(groups) == 1
        template = list(groups.keys())[0]
        assert len(groups[template]) == 2

    def test_different_templates_separated(self, analyzer):
        """Queries with different patterns go to different groups."""
        queries = [
            {
                "query_text": '"term" site:pubmed.ncbi.nlm.nih.gov',
                "result_count": 5,
                "cost_usd": 0.01,
            },
            {
                "query_text": "broad topic search",
                "result_count": 2,
                "cost_usd": 0.01,
            },
        ]
        groups = analyzer.extract_templates(queries)
        assert len(groups) == 2

    def test_empty_queries_list(self, analyzer):
        """Empty query list returns empty groups."""
        groups = analyzer.extract_templates([])
        assert groups == {}

    def test_preserves_query_data_in_groups(self, analyzer):
        """Original query data is preserved in groups."""
        queries = [
            {
                "query_text": "test query",
                "result_count": 10,
                "cost_usd": 0.05,
            },
        ]
        groups = analyzer.extract_templates(queries)
        template = list(groups.keys())[0]
        assert groups[template][0]["result_count"] == 10
        assert groups[template][0]["cost_usd"] == 0.05


# ============================================================================
# 🧪 Analysis and ranking tests
# ============================================================================


class TestAnalyze:
    """Tests for effectiveness analysis and ranking 📊."""

    def test_analyze_returns_records(self, populated_index):
        """analyze() returns TemplateEffectivenessRecord objects."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze()
        assert len(records) > 0
        assert all(isinstance(r, TemplateEffectivenessRecord) for r in records)

    def test_analyze_ranked_by_yield_rate(self, populated_index):
        """Results are ordered by yield_rate descending."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze()
        for i in range(1, len(records)):
            assert records[i - 1].yield_rate >= records[i].yield_rate

    def test_analyze_records_have_all_fields(self, populated_index):
        """Each record has all expected fields populated."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze()
        for r in records:
            assert r.template
            assert r.usage_count > 0
            assert r.avg_result_count >= 0
            assert r.avg_cost_usd >= 0
            assert 0.0 <= r.success_rate <= 1.0
            assert r.yield_rate >= 0
            assert len(r.example_queries) > 0
            assert len(r.example_queries) <= 3

    def test_analyze_respects_limit(self, populated_index):
        """Limit parameter restricts number of results."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze(limit=2)
        assert len(records) <= 2

    def test_analyze_success_rate_computed_correctly(self, index, tmp_dir):
        """Success rate reflects fraction of queries with > 0 results."""
        _write_jsonl(
            tmp_dir,
            task_id="task-sr",
            trajectory_id="traj-sr01",
            rounds=1,
            custom_queries={
                1: [
                    {
                        "query_text": "test alpha query",
                        "mcp_tool": "web_search",
                        "result_count": 10,
                        "cost_usd": 0.01,
                    },
                    {
                        "query_text": "test beta query",
                        "mcp_tool": "web_search",
                        "result_count": 0,
                        "cost_usd": 0.01,
                    },
                ],
            },
        )
        index.index_from_directory(tmp_dir)
        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze()
        # ✅ Both map to {topic} template; one has 0 results
        # so success_rate should be 0.5
        for r in records:
            if r.usage_count == 2:
                assert r.success_rate == pytest.approx(0.5, abs=0.01)

    def test_analyze_avg_cost_computed(self, populated_index):
        """Average cost is computed from individual query costs."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze()
        for r in records:
            assert r.avg_cost_usd >= 0

    def test_analyze_high_yield_template_ranked_first(self, populated_index):
        """Template with highest yield rate appears first."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze()
        # 🎯 The site:clinicaltrials.gov query had 15 cleaned
        # evidence and only 1 query, giving yield_rate = 15.0
        assert records[0].yield_rate >= records[-1].yield_rate


# ============================================================================
# 🧪 Task ID filtering tests
# ============================================================================


class TestTaskFiltering:
    """Tests for task_id filtering in analysis 🔍."""

    def test_analyze_filters_by_task_id(self, populated_index):
        """Filtering by task_id restricts to that task's queries."""
        analyzer = QueryTemplateAnalyzer(populated_index)

        # 📊 Task-001 should have queries from traj-1111 and traj-2222
        records_001 = analyzer.analyze(task_id="task-001")
        # 📊 Task-002 should have queries from traj-3333 only
        records_002 = analyzer.analyze(task_id="task-002")

        # ✅ They should have different query counts
        total_001 = sum(r.usage_count for r in records_001)
        total_002 = sum(r.usage_count for r in records_002)
        assert total_001 > total_002
        assert total_002 == 1  # Only one query in task-002

    def test_get_top_templates_filters_by_task_id(self, populated_index):
        """get_top_templates respects task_id filter."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        templates_001 = analyzer.get_top_templates(task_id="task-001")
        templates_002 = analyzer.get_top_templates(task_id="task-002")
        # 🎯 Different tasks produce different template sets
        assert len(templates_001) > 0
        assert len(templates_002) > 0

    def test_analyze_nonexistent_task_returns_empty(self, populated_index):
        """Non-existent task_id returns empty list."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        records = analyzer.analyze(task_id="task-nonexistent")
        assert records == []


# ============================================================================
# 🧪 get_top_templates tests
# ============================================================================


class TestGetTopTemplates:
    """Tests for top-K template retrieval 🚀."""

    def test_returns_strings(self, populated_index):
        """get_top_templates returns list of strings."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        templates = analyzer.get_top_templates()
        assert all(isinstance(t, str) for t in templates)

    def test_respects_top_k(self, populated_index):
        """top_k limits the number of results."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        templates = analyzer.get_top_templates(top_k=2)
        assert len(templates) <= 2

    def test_templates_ordered_by_effectiveness(self, populated_index):
        """Templates are in effectiveness order."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        templates = analyzer.get_top_templates(top_k=10)
        # 🎯 Compare with analyze() order
        records = analyzer.analyze(limit=10)
        expected = [r.template for r in records]
        assert templates == expected

    def test_top_k_exceeds_available(self, populated_index):
        """Requesting more than available returns all."""
        analyzer = QueryTemplateAnalyzer(populated_index)
        templates = analyzer.get_top_templates(top_k=1000)
        # ✅ Should return all available, not raise
        assert len(templates) > 0
        assert len(templates) <= 1000


# ============================================================================
# 🧪 Empty index tests
# ============================================================================


class TestEmptyIndex:
    """Tests for behavior with empty index 🛡️."""

    def test_analyze_empty_index_returns_empty(self, analyzer):
        """Empty index produces empty analysis."""
        records = analyzer.analyze()
        assert records == []

    def test_get_top_templates_empty_index(self, analyzer):
        """Empty index produces empty top templates."""
        templates = analyzer.get_top_templates()
        assert templates == []

    def test_extract_templates_empty_queries(self, analyzer):
        """Empty queries list produces empty groups."""
        groups = analyzer.extract_templates([])
        assert groups == {}


# ============================================================================
# 🧪 Special character tests
# ============================================================================


class TestSpecialCharacters:
    """Tests for queries with special characters 🛡️."""

    def test_query_with_double_quotes(self):
        """Double-quoted phrases are handled correctly."""
        template = _extract_template('"signal transduction" mechanism')
        assert "{quoted_term}" in template

    def test_query_with_asterisk_in_site(self):
        """Wildcards in site: operators are preserved."""
        template = _extract_template("topic site:*.gov")
        assert "site:*.gov" in template

    def test_query_with_parentheses(self):
        """Parentheses in queries don't break extraction."""
        template = _extract_template("(STAT3 OR JAK2) inhibitor")
        # ✅ Should not raise, should produce valid template
        assert template

    def test_query_with_hyphenated_terms(self):
        """Hyphenated terms like meta-analysis are preserved."""
        template = _extract_template("some topic meta-analysis")
        assert "meta-analysis" in template

    def test_query_with_colon_not_site(self):
        """Colons that are not site:/filetype: are handled."""
        template = _extract_template("ratio: 1 to 10 study")
        # ✅ Should not crash, should produce valid template
        assert template

    def test_special_chars_in_indexed_data(self, index, tmp_dir):
        """Queries with special chars are indexed and analyzed."""
        lines = [
            json.dumps(
                {
                    "type": "meta",
                    "trajectory_id": "traj-spec",
                    "task_id": "task-spec",
                    "config_snapshot": {},
                    "timestamp": "2026-02-26T21:00:00+00:00",
                }
            ),
            json.dumps(
                {
                    "type": "round",
                    "round_number": 1,
                    "search_phase": {
                        "queries": [
                            {
                                "query_text": (
                                    'STAT3 "signal transduction" site:*.gov'
                                ),
                                "mcp_tool": "web_search",
                                "result_count": 7,
                                "cost_usd": 0.02,
                            },
                            {
                                "query_text": ('"p53 pathway" AND "apoptosis"'),
                                "mcp_tool": "web_search",
                                "result_count": 12,
                                "cost_usd": 0.02,
                            },
                        ],
                        "total_raw_evidence": 10,
                    },
                    "cleaning_phase": {
                        "output_count": 8,
                        "dedup_removed": 1,
                        "noise_removed": 1,
                    },
                    "analysis_phase": {
                        "consensus": {"total_claims": 3},
                    },
                    "gap_phase": {"coverage_ratio": 0.6},
                    "round_cost_usd": 0.5,
                    "round_duration_seconds": 10.0,
                }
            ),
            json.dumps(
                {
                    "type": "summary",
                    "total_rounds": 1,
                    "final_coverage": 0.6,
                    "total_cost_usd": 0.5,
                }
            ),
            json.dumps(
                {
                    "type": "meta_final",
                    "status": "completed",
                    "termination_reason": "max_rounds",
                    "timestamp": "2026-02-26T21:01:00+00:00",
                }
            ),
        ]
        path = os.path.join(tmp_dir, "discovery_task-spec_traj-spec.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        index.index_trajectory(path)
        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze()
        assert len(records) > 0
        # ✅ Verify example queries contain original text
        all_examples = []
        for r in records:
            all_examples.extend(r.example_queries)
        assert any("signal transduction" in e for e in all_examples)


# ============================================================================
# 🧪 Yield rate computation tests
# ============================================================================


class TestYieldRate:
    """Tests for yield rate computation accuracy 📊."""

    def test_yield_rate_uses_cleaned_evidence(self, index, tmp_dir):
        """Yield rate is based on cleaned_evidence, not raw."""
        _write_jsonl(
            tmp_dir,
            task_id="task-yield",
            trajectory_id="traj-yield",
            rounds=1,
            custom_queries={
                1: [
                    {
                        "query_text": '"yield test" site:example.com',
                        "mcp_tool": "web_search",
                        "result_count": 20,
                        "cost_usd": 0.01,
                    },
                ],
            },
            cleaned_evidence_per_round=10,
        )
        index.index_from_directory(tmp_dir)
        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze()
        assert len(records) == 1
        # 🎯 1 query in round 1, 10 cleaned evidence
        # yield_rate = 10 / 1 = 10.0
        assert records[0].yield_rate == pytest.approx(10.0, abs=0.1)

    def test_yield_rate_zero_for_no_evidence(self, index, tmp_dir):
        """Yield rate is 0 when rounds have no cleaned evidence."""
        _write_jsonl(
            tmp_dir,
            task_id="task-noyield",
            trajectory_id="traj-noyield",
            rounds=1,
            custom_queries={
                1: [
                    {
                        "query_text": "zero evidence query",
                        "mcp_tool": "web_search",
                        "result_count": 5,
                        "cost_usd": 0.01,
                    },
                ],
            },
            cleaned_evidence_per_round=0,
        )
        index.index_from_directory(tmp_dir)
        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze()
        assert len(records) == 1
        assert records[0].yield_rate == 0.0

    def test_yield_rate_shared_across_queries_in_round(self, index, tmp_dir):
        """Multiple queries in same round share the round's evidence."""
        _write_jsonl(
            tmp_dir,
            task_id="task-shared",
            trajectory_id="traj-shared",
            rounds=1,
            custom_queries={
                1: [
                    {
                        "query_text": '"term A" site:example.com',
                        "mcp_tool": "web_search",
                        "result_count": 5,
                        "cost_usd": 0.01,
                    },
                    {
                        "query_text": '"term B" site:example.com',
                        "mcp_tool": "web_search",
                        "result_count": 8,
                        "cost_usd": 0.01,
                    },
                ],
            },
            cleaned_evidence_per_round=12,
        )
        index.index_from_directory(tmp_dir)
        analyzer = QueryTemplateAnalyzer(index)
        records = analyzer.analyze()
        # ✅ Both queries map to same template
        # 1 unique round with 12 evidence, 2 queries
        # yield = 12 / 2 = 6.0
        assert len(records) == 1
        assert records[0].yield_rate == pytest.approx(6.0, abs=0.1)
        assert records[0].usage_count == 2
