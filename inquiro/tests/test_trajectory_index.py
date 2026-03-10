"""Tests for TrajectoryIndex SQLite indexing and analytics 🧪.

Covers JSONL parsing, SQLite indexing, querying, trend analysis,
cost breakdown, statistics, and edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from inquiro.core.trajectory.index import (
    CostBreakdown,
    RoundRecord,
    TrajectoryIndex,
    TrajectoryRecord,
    TrendPoint,
    _parse_jsonl,
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
        **kwargs: Overrides for summary fields.

    Returns:
        Path to the created JSONL file.
    """
    rounds = kwargs.get("rounds", 2)
    final_coverage = kwargs.get("final_coverage", 0.85)
    total_cost = kwargs.get("total_cost", 3.50)
    status = kwargs.get("status", "completed")
    termination_reason = kwargs.get("termination_reason", "coverage_reached")
    config_snapshot = kwargs.get("config_snapshot", {"max_rounds": 5})
    extra_events = kwargs.get("extra_events", [])

    lines = []

    # 📋 Meta record
    lines.append(
        json.dumps(
            {
                "type": "meta",
                "trajectory_id": trajectory_id,
                "task_id": task_id,
                "config_snapshot": config_snapshot,
                "task_snapshot": {"rules": "test rules"},
                "timestamp": "2026-02-26T21:00:00+00:00",
            }
        )
    )

    # 📊 Round records
    total_queries = 0
    for r in range(1, rounds + 1):
        queries = [
            {
                "query_text": f"query_{r}_a site:pubmed.ncbi.nlm.nih.gov",
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

        coverage = min(1.0, final_coverage * r / rounds)
        round_cost = total_cost / rounds

        focus_prompt = None
        if r < rounds:
            focus_prompt = {
                "prompt_text": f"Focus on gap_{r}",
                "target_gaps": [f"gap_{r}"],
                "generation_model": "test-model",
                "cost_usd": 0.05,
            }

        lines.append(
            json.dumps(
                {
                    "type": "round",
                    "round_number": r,
                    "search_phase": {
                        "queries": queries,
                        "total_raw_evidence": 8,
                        "agent_trajectory_ref": None,
                        "duration_seconds": 10.0,
                    },
                    "cleaning_phase": {
                        "input_count": 8,
                        "output_count": 6,
                        "dedup_removed": 1,
                        "noise_removed": 1,
                        "tag_distribution": {"academic": 4, "other": 2},
                        "duration_seconds": 0.5,
                    },
                    "analysis_phase": {
                        "model_results": [
                            {
                                "model_name": "model-a",
                                "claims_count": 3,
                                "decision": "positive",
                                "confidence": 0.8,
                                "cost_usd": 0.3,
                            },
                        ],
                        "consensus": {
                            "consensus_decision": "positive",
                            "consensus_ratio": 0.67,
                            "total_claims": 5,
                        },
                        "duration_seconds": 15.0,
                    },
                    "gap_phase": {
                        "coverage_ratio": coverage,
                        "covered_items": [
                            f"item_{i}" for i in range(int(coverage * 5))
                        ],
                        "uncovered_items": ["item_last"],
                        "conflict_signals": [],
                        "convergence_reason": (
                            termination_reason if r == rounds else None
                        ),
                        "focus_prompt": focus_prompt,
                        "duration_seconds": 5.0,
                    },
                    "round_cost_usd": round_cost,
                    "round_duration_seconds": 30.5,
                }
            )
        )

    # ⏱️ Events
    lines.append(
        json.dumps(
            {
                "type": "event",
                "event_type": "discovery_started",
                "timestamp": "2026-02-26T21:00:01+00:00",
                "data": {"task_id": task_id},
            }
        )
    )
    for evt in extra_events:
        lines.append(json.dumps(evt))
    lines.append(
        json.dumps(
            {
                "type": "event",
                "event_type": "convergence_reached",
                "timestamp": "2026-02-26T21:01:00+00:00",
                "data": {"reason": termination_reason},
            }
        )
    )

    # 📝 Synthesis
    if kwargs.get("with_synthesis", True):
        lines.append(
            json.dumps(
                {
                    "type": "synthesis",
                    "model_results": [
                        {
                            "model_name": "model-a",
                            "claims_count": 8,
                            "decision": "positive",
                            "confidence": 0.85,
                            "cost_usd": 0.5,
                        },
                    ],
                    "consensus_decision": "positive",
                    "consensus_ratio": 0.67,
                    "cost_usd": 1.0,
                    "duration_seconds": 20.0,
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
                "total_rounds": rounds,
                "final_coverage": final_coverage,
                "total_cost_usd": total_cost,
                "total_evidence": rounds * 6,
                "total_claims": rounds * 5,
                "total_duration_seconds": rounds * 30.5,
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
def sample_jsonl(tmp_dir):
    """Create a sample JSONL trajectory file 📄."""
    return _write_jsonl(
        tmp_dir,
        task_id="task-001",
        trajectory_id="traj-aaaa-1111",
        rounds=3,
        final_coverage=0.90,
        total_cost=4.50,
    )


# ============================================================================
# 🧪 JSONL parsing tests
# ============================================================================


class TestJsonlParsing:
    """Tests for JSONL trajectory file parsing 📄."""

    def test_parse_complete_jsonl_returns_all_sections(self, sample_jsonl):
        """Parse a complete JSONL file and verify all sections present."""
        data = _parse_jsonl(sample_jsonl)
        assert data["meta"]["type"] == "meta"
        assert data["meta"]["trajectory_id"] == "traj-aaaa-1111"
        assert len(data["rounds"]) == 3
        assert len(data["events"]) >= 2
        assert data["summary"]["total_rounds"] == 3
        assert data["meta_final"]["status"] == "completed"

    def test_parse_extracts_config_snapshot(self, sample_jsonl):
        """Config snapshot should be preserved in meta."""
        data = _parse_jsonl(sample_jsonl)
        assert data["meta"]["config_snapshot"]["max_rounds"] == 5

    def test_parse_round_records_ordered(self, sample_jsonl):
        """Round records should be in insertion order."""
        data = _parse_jsonl(sample_jsonl)
        numbers = [r["round_number"] for r in data["rounds"]]
        assert numbers == [1, 2, 3]

    def test_parse_handles_empty_file(self, tmp_dir):
        """Empty file should return empty sections."""
        path = os.path.join(tmp_dir, "discovery_empty_test.jsonl")
        Path(path).touch()
        data = _parse_jsonl(path)
        assert data["meta"] == {}
        assert data["rounds"] == []

    def test_parse_handles_malformed_lines(self, tmp_dir):
        """Malformed JSON lines should be skipped gracefully."""
        path = os.path.join(tmp_dir, "discovery_bad_test.jsonl")
        with open(path, "w") as f:
            f.write('{"type": "meta", "trajectory_id": "t1", "task_id": "x"}\n')
            f.write("not valid json\n")
            f.write('{"type": "summary", "total_rounds": 1}\n')
        data = _parse_jsonl(path)
        assert data["meta"]["trajectory_id"] == "t1"
        assert data["summary"]["total_rounds"] == 1


# ============================================================================
# 🧪 Indexing tests
# ============================================================================


class TestIndexing:
    """Tests for trajectory indexing operations 📥."""

    def test_index_single_trajectory_returns_id(self, index, sample_jsonl):
        """Indexing a valid JSONL returns trajectory_id."""
        tid = index.index_trajectory(sample_jsonl)
        assert tid == "traj-aaaa-1111"

    def test_index_increments_count(self, index, sample_jsonl):
        """Count increases after indexing."""
        assert index.count() == 0
        index.index_trajectory(sample_jsonl)
        assert index.count() == 1

    def test_index_idempotent_same_file(self, index, sample_jsonl):
        """Re-indexing unchanged file returns None (skipped)."""
        tid1 = index.index_trajectory(sample_jsonl)
        tid2 = index.index_trajectory(sample_jsonl)
        assert tid1 == "traj-aaaa-1111"
        assert tid2 is None
        assert index.count() == 1

    def test_index_reindexes_changed_file(self, index, tmp_dir):
        """Re-indexing a modified file updates the record."""
        path = _write_jsonl(
            tmp_dir,
            "task-002",
            "traj-bbbb-2222",
            final_coverage=0.70,
        )
        index.index_trajectory(path)
        rec = index.get_summary("traj-bbbb-2222")
        assert rec is not None
        assert rec.final_coverage == 0.70

        # ✏️ Modify file
        path2 = _write_jsonl(
            tmp_dir,
            "task-002",
            "traj-bbbb-2222",
            final_coverage=0.95,
        )
        # Overwrite same path
        os.replace(path2, path)
        tid = index.index_trajectory(path)
        assert tid == "traj-bbbb-2222"
        rec = index.get_summary("traj-bbbb-2222")
        assert rec is not None
        assert rec.final_coverage == 0.95

    def test_index_nonexistent_file_returns_none(self, index):
        """Non-existent file returns None."""
        result = index.index_trajectory("/no/such/file.jsonl")
        assert result is None

    def test_index_file_without_meta_returns_none(self, index, tmp_dir):
        """JSONL without meta record returns None."""
        path = os.path.join(tmp_dir, "discovery_nometa_test.jsonl")
        with open(path, "w") as f:
            f.write('{"type": "summary", "total_rounds": 1}\n')
        result = index.index_trajectory(path)
        assert result is None

    def test_index_from_directory_indexes_all(self, index, tmp_dir):
        """Batch index finds and indexes all discovery_*.jsonl files."""
        _write_jsonl(tmp_dir, "task-a", "traj-1111-aaaa")
        _write_jsonl(tmp_dir, "task-b", "traj-2222-bbbb")
        _write_jsonl(tmp_dir, "task-c", "traj-3333-cccc")

        indexed = index.index_from_directory(tmp_dir)
        assert len(indexed) == 3
        assert index.count() == 3

    def test_index_from_directory_skips_non_discovery_files(self, index, tmp_dir):
        """Only files matching discovery_*.jsonl are indexed."""
        _write_jsonl(tmp_dir, "task-a", "traj-1111-aaaa")
        # ❌ Non-matching filename
        other = os.path.join(tmp_dir, "other_data.jsonl")
        with open(other, "w") as f:
            f.write('{"type": "meta", "trajectory_id": "skip"}\n')

        indexed = index.index_from_directory(tmp_dir)
        assert len(indexed) == 1

    def test_index_from_nonexistent_directory_returns_empty(self, index):
        """Non-existent directory returns empty list."""
        result = index.index_from_directory("/no/such/dir")
        assert result == []


# ============================================================================
# 🧪 Query tests
# ============================================================================


class TestQuerying:
    """Tests for trajectory querying operations 🔍."""

    def test_get_summary_returns_record(self, index, sample_jsonl):
        """get_summary returns complete TrajectoryRecord."""
        index.index_trajectory(sample_jsonl)
        rec = index.get_summary("traj-aaaa-1111")
        assert rec is not None
        assert isinstance(rec, TrajectoryRecord)
        assert rec.trajectory_id == "traj-aaaa-1111"
        assert rec.task_id == "task-001"
        assert rec.total_rounds == 3
        assert rec.final_coverage == 0.90
        assert rec.status == "completed"

    def test_get_summary_nonexistent_returns_none(self, index):
        """get_summary returns None for unknown trajectory."""
        assert index.get_summary("unknown") is None

    def test_list_trajectories_returns_all(self, index, tmp_dir):
        """list_trajectories without filters returns all."""
        _write_jsonl(tmp_dir, "t1", "traj-1111")
        _write_jsonl(tmp_dir, "t2", "traj-2222")
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories()
        assert len(results) == 2

    def test_list_filter_by_task_id(self, index, tmp_dir):
        """Filter by task_id returns matching records."""
        _write_jsonl(tmp_dir, "task-alpha", "traj-1111")
        _write_jsonl(tmp_dir, "task-beta", "traj-2222")
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(task_id="task-alpha")
        assert len(results) == 1
        assert results[0].task_id == "task-alpha"

    def test_list_filter_by_status(self, index, tmp_dir):
        """Filter by status returns matching records."""
        _write_jsonl(tmp_dir, "t1", "traj-1111", status="completed")
        _write_jsonl(tmp_dir, "t2", "traj-2222", status="failed")
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(status="completed")
        assert len(results) == 1
        assert results[0].status == "completed"

    def test_list_filter_by_min_coverage(self, index, tmp_dir):
        """Filter by min_coverage returns high-coverage records."""
        _write_jsonl(
            tmp_dir,
            "t1",
            "traj-1111",
            final_coverage=0.50,
        )
        _write_jsonl(
            tmp_dir,
            "t2",
            "traj-2222",
            final_coverage=0.90,
        )
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(min_coverage=0.80)
        assert len(results) == 1
        assert results[0].final_coverage >= 0.80

    def test_list_filter_by_max_cost(self, index, tmp_dir):
        """Filter by max_cost returns affordable records."""
        _write_jsonl(tmp_dir, "t1", "traj-1111", total_cost=2.0)
        _write_jsonl(tmp_dir, "t2", "traj-2222", total_cost=10.0)
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(max_cost=5.0)
        assert len(results) == 1
        assert results[0].total_cost_usd <= 5.0

    def test_list_with_pagination(self, index, tmp_dir):
        """Pagination with limit and offset works."""
        for i in range(5):
            _write_jsonl(tmp_dir, f"t{i}", f"traj-{i:04d}")
        index.index_from_directory(tmp_dir)
        page1 = index.list_trajectories(limit=2, offset=0)
        page2 = index.list_trajectories(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # ✅ No overlap
        ids1 = {r.trajectory_id for r in page1}
        ids2 = {r.trajectory_id for r in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_order_by_coverage(self, index, tmp_dir):
        """Order by final_coverage DESC works."""
        _write_jsonl(tmp_dir, "t1", "traj-low0", final_coverage=0.30)
        _write_jsonl(tmp_dir, "t2", "traj-high", final_coverage=0.99)
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(order_by="final_coverage DESC")
        assert results[0].final_coverage >= results[1].final_coverage

    def test_list_sanitizes_invalid_order_by(self, index, tmp_dir):
        """Invalid order_by falls back to default."""
        _write_jsonl(tmp_dir, "t1", "traj-1111")
        index.index_from_directory(tmp_dir)
        # ✅ Should not raise, falls back to created_at DESC
        results = index.list_trajectories(order_by="DROP TABLE trajectories; --")
        assert len(results) == 1


# ============================================================================
# 🧪 Round detail tests
# ============================================================================


class TestRoundDetails:
    """Tests for per-round querying 📊."""

    def test_get_rounds_returns_ordered_records(self, index, sample_jsonl):
        """Rounds are returned in order by round_number."""
        index.index_trajectory(sample_jsonl)
        rounds = index.get_rounds("traj-aaaa-1111")
        assert len(rounds) == 3
        assert all(isinstance(r, RoundRecord) for r in rounds)
        assert [r.round_number for r in rounds] == [1, 2, 3]

    def test_get_rounds_contains_correct_data(self, index, sample_jsonl):
        """Round records contain expected field values."""
        index.index_trajectory(sample_jsonl)
        rounds = index.get_rounds("traj-aaaa-1111")
        r1 = rounds[0]
        assert r1.search_queries == 2
        assert r1.raw_evidence == 8
        assert r1.cleaned_evidence == 6
        assert r1.dedup_removed == 1
        assert r1.noise_removed == 1
        assert r1.analysis_claims == 5
        assert r1.round_duration_seconds == 30.5

    def test_get_rounds_empty_for_unknown_trajectory(self, index):
        """Unknown trajectory returns empty list."""
        assert index.get_rounds("unknown") == []

    def test_coverage_progression_across_rounds(self, index, sample_jsonl):
        """Coverage should increase across rounds."""
        index.index_trajectory(sample_jsonl)
        rounds = index.get_rounds("traj-aaaa-1111")
        coverages = [r.coverage_ratio for r in rounds]
        # ✅ Monotonically non-decreasing
        for i in range(1, len(coverages)):
            assert coverages[i] >= coverages[i - 1]


# ============================================================================
# 🧪 Cost breakdown tests
# ============================================================================


class TestCostBreakdown:
    """Tests for cost breakdown analysis 💰."""

    def test_get_cost_breakdown_returns_breakdown(self, index, sample_jsonl):
        """Cost breakdown returns populated CostBreakdown."""
        index.index_trajectory(sample_jsonl)
        cb = index.get_cost_breakdown("traj-aaaa-1111")
        assert cb is not None
        assert isinstance(cb, CostBreakdown)
        assert cb.trajectory_id == "traj-aaaa-1111"
        # total = discovery cost (4.50) + synthesis cost (1.0) = 5.50
        assert cb.total_cost_usd == pytest.approx(5.50, abs=0.001)

    def test_cost_breakdown_has_per_round_data(self, index, sample_jsonl):
        """Per-round cost data is populated."""
        index.index_trajectory(sample_jsonl)
        cb = index.get_cost_breakdown("traj-aaaa-1111")
        assert cb is not None
        assert len(cb.per_round) == 3
        assert all("round_number" in r for r in cb.per_round)
        assert all("cost_usd" in r for r in cb.per_round)

    def test_cost_breakdown_search_cost_from_queries(self, index, sample_jsonl):
        """Search cost is aggregated from individual queries."""
        index.index_trajectory(sample_jsonl)
        cb = index.get_cost_breakdown("traj-aaaa-1111")
        assert cb is not None
        # 3 rounds × 2 queries × 0.01 = 0.06
        assert cb.search_cost_usd == pytest.approx(0.06, abs=0.001)

    def test_cost_breakdown_synthesis_cost_populated(self, index, sample_jsonl):
        """Synthesis cost is parsed from JSONL synthesis record."""
        index.index_trajectory(sample_jsonl)
        cb = index.get_cost_breakdown("traj-aaaa-1111")
        assert cb is not None
        # Sample JSONL writes synthesis record with cost_usd=1.0
        assert cb.synthesis_cost_usd == pytest.approx(1.0, abs=0.001)

    def test_cost_breakdown_synthesis_cost_zero_when_no_synthesis(self, index, tmp_dir):
        """synthesis_cost_usd is 0 when no synthesis record in JSONL."""
        path = _write_jsonl(
            tmp_dir,
            task_id="task-nosyn",
            trajectory_id="traj-nosyn-0001",
            with_synthesis=False,
        )
        index.index_trajectory(path)
        cb = index.get_cost_breakdown("traj-nosyn-0001")
        assert cb is not None
        assert cb.synthesis_cost_usd == 0.0
        # total_cost_usd equals discovery cost only
        assert cb.total_cost_usd == pytest.approx(3.50, abs=0.001)

    def test_cost_breakdown_nonexistent_returns_none(self, index):
        """Unknown trajectory returns None."""
        assert index.get_cost_breakdown("unknown") is None


# ============================================================================
# 🧪 find_similar tests
# ============================================================================


class TestFindSimilar:
    """Tests for similar trajectory lookup 🔍."""

    def test_find_similar_returns_matching_task(self, index, tmp_dir):
        """find_similar returns trajectories with same task_id."""
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-1111",
            final_coverage=0.90,
        )
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-2222",
            final_coverage=0.80,
        )
        _write_jsonl(
            tmp_dir,
            "task-y",
            "traj-3333",
            final_coverage=0.95,
        )
        index.index_from_directory(tmp_dir)
        results = index.find_similar("task-x")
        assert len(results) == 2
        assert all(r.task_id == "task-x" for r in results)

    def test_find_similar_ordered_by_quality(self, index, tmp_dir):
        """Results are ordered by cost_normalized_quality DESC."""
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-low0",
            final_coverage=0.50,
            total_cost=5.0,
        )
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-high",
            final_coverage=0.95,
            total_cost=2.0,
        )
        index.index_from_directory(tmp_dir)
        results = index.find_similar("task-x")
        assert len(results) == 2
        assert results[0].cost_normalized_quality >= results[1].cost_normalized_quality

    def test_find_similar_respects_min_coverage(self, index, tmp_dir):
        """min_coverage filter excludes low-coverage trajectories."""
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-low0",
            final_coverage=0.40,
        )
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-high",
            final_coverage=0.90,
        )
        index.index_from_directory(tmp_dir)
        results = index.find_similar("task-x", min_coverage=0.80)
        assert len(results) == 1
        assert results[0].final_coverage >= 0.80

    def test_find_similar_respects_top_k(self, index, tmp_dir):
        """top_k limits the number of results."""
        for i in range(10):
            _write_jsonl(
                tmp_dir,
                "task-x",
                f"traj-{i:04d}",
            )
        index.index_from_directory(tmp_dir)
        results = index.find_similar("task-x", top_k=3)
        assert len(results) == 3

    def test_find_similar_excludes_failed(self, index, tmp_dir):
        """Failed trajectories are excluded."""
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-ok00",
            status="completed",
        )
        _write_jsonl(
            tmp_dir,
            "task-x",
            "traj-fail",
            status="failed",
        )
        index.index_from_directory(tmp_dir)
        results = index.find_similar("task-x")
        assert len(results) == 1
        assert results[0].status == "completed"


# ============================================================================
# 🧪 Trend analysis tests
# ============================================================================


class TestTrendAnalysis:
    """Tests for trend analysis queries 📈."""

    def test_get_trend_returns_points(self, index, tmp_dir):
        """get_trend returns TrendPoint objects."""
        _write_jsonl(tmp_dir, "t1", "traj-1111", final_coverage=0.70)
        _write_jsonl(tmp_dir, "t2", "traj-2222", final_coverage=0.85)
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("final_coverage")
        assert len(trend) == 2
        assert all(isinstance(p, TrendPoint) for p in trend)

    def test_get_trend_values_match_metric(self, index, tmp_dir):
        """Trend values correspond to the requested metric."""
        _write_jsonl(tmp_dir, "t1", "traj-1111", total_cost=2.0)
        _write_jsonl(tmp_dir, "t2", "traj-2222", total_cost=5.0)
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("total_cost_usd")
        values = [p.value for p in trend]
        assert 2.0 in values
        assert 5.0 in values

    def test_get_trend_filters_by_task_id(self, index, tmp_dir):
        """Task_id filter restricts trend data."""
        _write_jsonl(tmp_dir, "task-a", "traj-1111")
        _write_jsonl(tmp_dir, "task-b", "traj-2222")
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("final_coverage", task_id="task-a")
        assert len(trend) == 1
        assert trend[0].task_id == "task-a"

    def test_get_trend_respects_limit(self, index, tmp_dir):
        """Limit controls maximum data points."""
        for i in range(10):
            _write_jsonl(tmp_dir, f"t{i}", f"traj-{i:04d}")
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("final_coverage", limit=3)
        assert len(trend) == 3

    def test_get_trend_invalid_metric_returns_empty(self, index, tmp_dir):
        """Invalid metric name returns empty list."""
        _write_jsonl(tmp_dir, "t1", "traj-1111")
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("invalid_column")
        assert trend == []

    def test_get_trend_excludes_failed(self, index, tmp_dir):
        """Failed trajectories are excluded from trends."""
        _write_jsonl(tmp_dir, "t1", "traj-ok00", status="completed")
        _write_jsonl(tmp_dir, "t2", "traj-fail", status="failed")
        index.index_from_directory(tmp_dir)
        trend = index.get_trend("final_coverage")
        assert len(trend) == 1


# ============================================================================
# 🧪 Statistics tests
# ============================================================================


class TestStatistics:
    """Tests for aggregate statistics 📊."""

    def test_statistics_returns_aggregates(self, index, tmp_dir):
        """Statistics returns all expected aggregate fields."""
        _write_jsonl(tmp_dir, "t1", "traj-1111", final_coverage=0.80)
        _write_jsonl(tmp_dir, "t2", "traj-2222", final_coverage=0.90)
        index.index_from_directory(tmp_dir)
        stats = index.get_statistics()
        assert stats["count"] == 2
        assert "coverage" in stats
        assert "cost_usd" in stats
        assert stats["coverage"]["avg"] == pytest.approx(0.85, abs=0.01)

    def test_statistics_empty_index(self, index):
        """Empty index returns count=0."""
        stats = index.get_statistics()
        assert stats == {"count": 0}

    def test_statistics_filters_by_task_id(self, index, tmp_dir):
        """task_id filter restricts statistics."""
        _write_jsonl(tmp_dir, "task-a", "traj-1111")
        _write_jsonl(tmp_dir, "task-b", "traj-2222")
        index.index_from_directory(tmp_dir)
        stats = index.get_statistics(task_id="task-a")
        assert stats["count"] == 1

    def test_statistics_counts_completed_and_failed(self, index, tmp_dir):
        """Completed and failed counts are tracked."""
        _write_jsonl(tmp_dir, "t1", "traj-ok00", status="completed")
        _write_jsonl(tmp_dir, "t2", "traj-fail", status="failed")
        index.index_from_directory(tmp_dir)
        stats = index.get_statistics()
        assert stats["completed_count"] == 1
        assert stats["failed_count"] == 1


# ============================================================================
# 🧪 Query effectiveness tests
# ============================================================================


class TestQueryEffectiveness:
    """Tests for query effectiveness analysis 📊."""

    def test_query_effectiveness_ranks_by_results(self, index, sample_jsonl):
        """Queries are ranked by average result count."""
        index.index_trajectory(sample_jsonl)
        results = index.get_query_effectiveness()
        assert len(results) > 0
        assert all("query_text" in r for r in results)
        assert all("avg_results" in r for r in results)
        # ✅ First result has highest avg
        if len(results) > 1:
            assert results[0]["avg_results"] >= results[1]["avg_results"]

    def test_query_effectiveness_filters_by_task_id(self, index, tmp_dir):
        """task_id filter restricts to specific task's queries."""
        _write_jsonl(tmp_dir, "task-a", "traj-1111")
        _write_jsonl(tmp_dir, "task-b", "traj-2222")
        index.index_from_directory(tmp_dir)
        results = index.get_query_effectiveness(task_id="task-a")
        assert len(results) > 0

    def test_query_effectiveness_respects_limit(self, index, sample_jsonl):
        """Limit controls maximum results."""
        index.index_trajectory(sample_jsonl)
        results = index.get_query_effectiveness(limit=1)
        assert len(results) == 1


# ============================================================================
# 🧪 Deletion tests
# ============================================================================


class TestDeletion:
    """Tests for trajectory deletion 🗑️."""

    def test_delete_removes_trajectory(self, index, sample_jsonl):
        """Deleting a trajectory removes it from index."""
        index.index_trajectory(sample_jsonl)
        assert index.count() == 1
        result = index.delete_trajectory("traj-aaaa-1111")
        assert result is True
        assert index.count() == 0

    def test_delete_removes_related_records(self, index, sample_jsonl):
        """Deleting also removes rounds, queries, events."""
        index.index_trajectory(sample_jsonl)
        assert len(index.get_rounds("traj-aaaa-1111")) == 3
        index.delete_trajectory("traj-aaaa-1111")
        assert index.get_rounds("traj-aaaa-1111") == []

    def test_delete_nonexistent_returns_false(self, index):
        """Deleting unknown trajectory returns False."""
        assert index.delete_trajectory("unknown") is False


# ============================================================================
# 🧪 Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness 🛡️."""

    def test_database_created_on_init(self, tmp_dir):
        """Database file is created on initialization."""
        db_path = os.path.join(tmp_dir, "subdir", "test.db")
        TrajectoryIndex(db_path)
        assert os.path.isfile(db_path)

    def test_concurrent_index_operations(self, tmp_dir):
        """Multiple index instances on same DB don't conflict."""
        db_path = os.path.join(tmp_dir, "shared.db")
        idx1 = TrajectoryIndex(db_path)
        idx2 = TrajectoryIndex(db_path)

        path = _write_jsonl(tmp_dir, "t1", "traj-1111")
        idx1.index_trajectory(path)
        # ✅ idx2 sees the same data
        assert idx2.count() == 1

    def test_jsonl_with_no_rounds(self, index, tmp_dir):
        """Trajectory with zero rounds is indexed correctly."""
        path = _write_jsonl(
            tmp_dir,
            "task-empty",
            "traj-empty",
            rounds=0,
            final_coverage=0.0,
            total_cost=0.0,
        )
        tid = index.index_trajectory(path)
        assert tid == "traj-empty"
        rec = index.get_summary("traj-empty")
        assert rec is not None
        assert rec.total_rounds == 0
        assert index.get_rounds("traj-empty") == []

    def test_special_characters_in_query_text(self, index, tmp_dir):
        """Query text with special characters is handled."""
        # 📝 Build custom JSONL with special characters
        lines = [
            json.dumps(
                {
                    "type": "meta",
                    "trajectory_id": "traj-special",
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
                                "query_text": 'STAT3 "signal transduction" site:*.gov',
                                "mcp_tool": "web_search",
                                "result_count": 3,
                                "cost_usd": 0.01,
                            }
                        ],
                        "total_raw_evidence": 3,
                    },
                    "cleaning_phase": {},
                    "analysis_phase": {},
                    "gap_phase": {"coverage_ratio": 0.5},
                    "round_cost_usd": 0.5,
                    "round_duration_seconds": 10.0,
                }
            ),
            json.dumps(
                {
                    "type": "summary",
                    "total_rounds": 1,
                    "final_coverage": 0.5,
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
        path = os.path.join(tmp_dir, "discovery_task-spec_special.jsonl")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        tid = index.index_trajectory(path)
        assert tid == "traj-special"
        results = index.get_query_effectiveness()
        assert any('STAT3 "signal transduction"' in r["query_text"] for r in results)

    def test_filter_by_termination_reason(self, index, tmp_dir):
        """Filter by termination_reason works."""
        _write_jsonl(
            tmp_dir,
            "t1",
            "traj-cov0",
            termination_reason="coverage_reached",
        )
        _write_jsonl(
            tmp_dir,
            "t2",
            "traj-budg",
            termination_reason="budget_exhausted",
        )
        index.index_from_directory(tmp_dir)
        results = index.list_trajectories(termination_reason="budget_exhausted")
        assert len(results) == 1
        assert results[0].termination_reason == "budget_exhausted"
