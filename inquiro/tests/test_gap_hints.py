"""Tests for GapSearchHintAccumulator (Phase 3 O3) 🧪.

Covers hint extraction from trajectory data, accumulation across
multiple trajectories, YAML export format, and relevance matching.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import yaml

from inquiro.core.trajectory.gap_hints import (
    GapSearchHint,
    GapSearchHintAccumulator,
    _generalize_gap,
    _generalize_query,
    _keyword_overlap,
)
from inquiro.core.trajectory.index import TrajectoryIndex


# ============================================================================
# 🔧 Fixtures and helpers
# ============================================================================


def _write_gap_trajectory(
    dir_path: str,
    task_id: str,
    trajectory_id: str,
    *,
    rounds_data: list[dict] | None = None,
    status: str = "completed",
) -> str:
    """Write a JSONL trajectory file with explicit round-level control 📝.

    Allows specifying per-round coverage, focus prompts, target gaps,
    queries, and MCP tools for precise testing of hint extraction.

    Args:
        dir_path: Output directory.
        task_id: Task identifier.
        trajectory_id: Trajectory identifier.
        rounds_data: List of dicts with per-round overrides.
            Each dict may contain:
                - coverage_ratio (float)
                - focus_prompt_text (str)
                - target_gaps (list[str])
                - queries (list[dict]) with query_text, mcp_tool
        status: Run status.

    Returns:
        Path to the created JSONL file.
    """
    if rounds_data is None:
        rounds_data = []

    lines: list[str] = []

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
    final_coverage = 0.0
    total_cost = 0.0
    for i, rd in enumerate(rounds_data, start=1):
        coverage = rd.get("coverage_ratio", 0.0)
        final_coverage = coverage

        queries = rd.get(
            "queries",
            [
                {
                    "query_text": f"default query {i}",
                    "mcp_tool": "web_search",
                    "result_count": 3,
                    "cost_usd": 0.01,
                },
            ],
        )

        focus_prompt = None
        focus_text = rd.get("focus_prompt_text", "")
        target_gaps = rd.get("target_gaps", [])
        if focus_text:
            focus_prompt = {
                "prompt_text": focus_text,
                "target_gaps": target_gaps,
                "generation_model": "test-model",
                "cost_usd": 0.05,
            }

        round_cost = 0.5
        total_cost += round_cost

        lines.append(
            json.dumps(
                {
                    "type": "round",
                    "round_number": i,
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
                            }
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
                        "covered_items": [],
                        "uncovered_items": [],
                        "conflict_signals": [],
                        "convergence_reason": None,
                        "focus_prompt": focus_prompt,
                        "duration_seconds": 5.0,
                    },
                    "round_cost_usd": round_cost,
                    "round_duration_seconds": 30.0,
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

    # 📝 Summary
    num_rounds = len(rounds_data)
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "total_rounds": num_rounds,
                "final_coverage": final_coverage,
                "total_cost_usd": total_cost,
                "total_evidence": num_rounds * 6,
                "total_claims": num_rounds * 5,
                "total_duration_seconds": num_rounds * 30.0,
                "termination_reason": "coverage_reached",
                "evidence_yield_rate": 0.5,
                "cost_normalized_quality": (final_coverage / max(total_cost, 0.01)),
            }
        )
    )

    # 🏁 Meta final
    lines.append(
        json.dumps(
            {
                "type": "meta_final",
                "status": status,
                "termination_reason": "coverage_reached",
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
def accumulator(index):
    """Create a GapSearchHintAccumulator instance 🎯."""
    return GapSearchHintAccumulator(index)


# ============================================================================
# 🧪 Helper function tests
# ============================================================================


class TestHelperFunctions:
    """Tests for gap pattern generalization helpers 🔧."""

    def test_generalize_gap_strips_numbering(self):
        """Numbered prefixes are removed from gap text."""
        assert _generalize_gap("1. regulatory approval status") == (
            "regulatory approval status"
        )
        assert _generalize_gap("2) safety profile data") == ("safety profile data")

    def test_generalize_gap_normalizes_whitespace(self):
        """Extra whitespace is collapsed to single spaces."""
        assert _generalize_gap("  multiple   spaces  ") == ("multiple spaces")

    def test_generalize_gap_lowercases(self):
        """Gap patterns are lowercased for canonical comparison."""
        assert _generalize_gap("Regulatory Approval") == ("regulatory approval")

    def test_generalize_query_preserves_structure(self):
        """Query template retains search structure."""
        result = _generalize_query("topic FDA approval status")
        assert "topic FDA approval status" == result

    def test_generalize_query_replaces_site_filter(self):
        """site: filters are replaced with {domain} placeholder."""
        result = _generalize_query("topic site:pubmed.ncbi.nlm.nih.gov")
        assert "site:{domain}" in result
        assert "pubmed" not in result

    def test_keyword_overlap_identical_texts(self):
        """Identical texts should have overlap of 1.0."""
        assert _keyword_overlap("foo bar", "foo bar") == 1.0

    def test_keyword_overlap_disjoint_texts(self):
        """Completely different texts should have 0.0 overlap."""
        assert _keyword_overlap("alpha beta", "gamma delta") == 0.0

    def test_keyword_overlap_partial(self):
        """Partially overlapping texts produce a score between 0-1."""
        score = _keyword_overlap(
            "regulatory approval status",
            "regulatory filing status",
        )
        assert 0.0 < score < 1.0

    def test_keyword_overlap_empty_text(self):
        """Empty text returns 0.0."""
        assert _keyword_overlap("", "something") == 0.0
        assert _keyword_overlap("something", "") == 0.0


# ============================================================================
# 🧪 Hint extraction from single trajectory
# ============================================================================


class TestAnalyzeTrajectory:
    """Tests for hint extraction from a single trajectory 🔍."""

    def test_extract_hints_with_coverage_improvement(self, index, accumulator, tmp_dir):
        """Hints are extracted when coverage improves after focus prompt."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-001",
            "traj-improve",
            rounds_data=[
                {
                    "coverage_ratio": 0.40,
                    "focus_prompt_text": "Focus on regulatory approval status",
                    "target_gaps": ["regulatory approval status"],
                    "queries": [
                        {
                            "query_text": "initial search query",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        },
                    ],
                },
                {
                    "coverage_ratio": 0.65,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                    "queries": [
                        {
                            "query_text": "topic FDA approval status",
                            "mcp_tool": "web_search",
                            "result_count": 8,
                            "cost_usd": 0.01,
                        },
                        {
                            "query_text": "topic regulatory filing",
                            "mcp_tool": "scholarly_search",
                            "result_count": 4,
                            "cost_usd": 0.02,
                        },
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-improve")

        assert len(hints) == 1
        hint = hints[0]
        assert hint.gap_pattern == "regulatory approval status"
        assert hint.success_count == 1
        assert hint.avg_coverage_delta == pytest.approx(0.25, abs=0.01)
        assert len(hint.effective_queries) == 2
        assert len(hint.recommended_tools) >= 1
        assert "web_search" in hint.recommended_tools

    def test_no_hints_when_no_coverage_improvement(self, index, accumulator, tmp_dir):
        """No hints when coverage does not improve after focus prompt."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-002",
            "traj-noimprove",
            rounds_data=[
                {
                    "coverage_ratio": 0.50,
                    "focus_prompt_text": "Focus on safety data",
                    "target_gaps": ["safety data"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.50,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                    "queries": [
                        {
                            "query_text": "q2",
                            "mcp_tool": "web_search",
                            "result_count": 2,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-noimprove")
        assert hints == []

    def test_no_hints_when_coverage_decreases(self, index, accumulator, tmp_dir):
        """No hints when coverage decreases after focus prompt."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-003",
            "traj-decrease",
            rounds_data=[
                {
                    "coverage_ratio": 0.60,
                    "focus_prompt_text": "Focus on mechanism data",
                    "target_gaps": ["mechanism data"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.55,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                    "queries": [
                        {
                            "query_text": "q2",
                            "mcp_tool": "web_search",
                            "result_count": 2,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-decrease")
        assert hints == []

    def test_no_hints_with_single_round(self, index, accumulator, tmp_dir):
        """Single-round trajectory has no consecutive pairs to analyze."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-004",
            "traj-single",
            rounds_data=[
                {
                    "coverage_ratio": 0.80,
                    "focus_prompt_text": "Focus on something",
                    "target_gaps": ["something"],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-single")
        assert hints == []

    def test_no_hints_without_focus_prompt(self, index, accumulator, tmp_dir):
        """No hints when there is no focus prompt in any round."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-005",
            "traj-nofocus",
            rounds_data=[
                {
                    "coverage_ratio": 0.40,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                },
                {
                    "coverage_ratio": 0.70,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-nofocus")
        assert hints == []

    def test_multiple_effective_rounds(self, index, accumulator, tmp_dir):
        """Multiple consecutive improvements yield multiple hints."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-006",
            "traj-multi",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on efficacy data",
                    "target_gaps": ["efficacy data"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.50,
                    "focus_prompt_text": "Focus on safety profile",
                    "target_gaps": ["safety profile"],
                    "queries": [
                        {
                            "query_text": "efficacy trial results",
                            "mcp_tool": "web_search",
                            "result_count": 8,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.75,
                    "focus_prompt_text": "",
                    "target_gaps": [],
                    "queries": [
                        {
                            "query_text": "safety profile data",
                            "mcp_tool": "scholarly_search",
                            "result_count": 6,
                            "cost_usd": 0.02,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-multi")

        assert len(hints) == 2
        patterns = {h.gap_pattern for h in hints}
        assert "efficacy data" in patterns
        assert "safety profile" in patterns

    def test_nonexistent_trajectory_returns_empty(self, index, accumulator):
        """Analyzing non-existent trajectory returns empty list."""
        hints = accumulator.analyze_trajectory("nonexistent")
        assert hints == []


# ============================================================================
# 🧪 Accumulation across multiple trajectories
# ============================================================================


class TestAccumulate:
    """Tests for accumulation across multiple trajectories 📊."""

    def test_accumulate_merges_same_pattern(self, index, accumulator, tmp_dir):
        """Same gap pattern across trajectories gets merged."""
        # 🔧 Trajectory 1: "regulatory approval status" effective
        path1 = _write_gap_trajectory(
            tmp_dir,
            "task-a",
            "traj-acc-01",
            rounds_data=[
                {
                    "coverage_ratio": 0.40,
                    "focus_prompt_text": "Focus on regulatory approval status",
                    "target_gaps": ["regulatory approval status"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "regulatory filing status",
                            "mcp_tool": "web_search",
                            "result_count": 7,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        # 🔧 Trajectory 2: same pattern, different queries
        path2 = _write_gap_trajectory(
            tmp_dir,
            "task-a",
            "traj-acc-02",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on regulatory approval status",
                    "target_gaps": ["regulatory approval status"],
                    "queries": [
                        {
                            "query_text": "q2",
                            "mcp_tool": "web_search",
                            "result_count": 4,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.55,
                    "queries": [
                        {
                            "query_text": "approval pathway analysis",
                            "mcp_tool": "scholarly_search",
                            "result_count": 6,
                            "cost_usd": 0.02,
                        }
                    ],
                },
            ],
        )

        index.index_trajectory(path1)
        index.index_trajectory(path2)

        hints = accumulator.accumulate(task_id="task-a")

        assert len(hints) == 1
        hint = hints[0]
        assert hint.gap_pattern == "regulatory approval status"
        assert hint.success_count == 2
        # ✅ Both trajectories' queries should be present
        assert len(hint.effective_queries) >= 2

    def test_accumulate_ordered_by_success_count(self, index, accumulator, tmp_dir):
        """Hints are ordered by success_count descending."""
        # 🔧 Pattern A: appears in 2 trajectories
        for tid in ["traj-ord-01", "traj-ord-02"]:
            path = _write_gap_trajectory(
                tmp_dir,
                "task-b",
                tid,
                rounds_data=[
                    {
                        "coverage_ratio": 0.30,
                        "focus_prompt_text": "Focus on pattern alpha",
                        "target_gaps": ["pattern alpha"],
                        "queries": [
                            {
                                "query_text": "q",
                                "mcp_tool": "web_search",
                                "result_count": 3,
                                "cost_usd": 0.01,
                            }
                        ],
                    },
                    {
                        "coverage_ratio": 0.50,
                        "queries": [
                            {
                                "query_text": "alpha result",
                                "mcp_tool": "web_search",
                                "result_count": 5,
                                "cost_usd": 0.01,
                            }
                        ],
                    },
                ],
            )
            index.index_trajectory(path)

        # 🔧 Pattern B: appears in 1 trajectory
        path = _write_gap_trajectory(
            tmp_dir,
            "task-b",
            "traj-ord-03",
            rounds_data=[
                {
                    "coverage_ratio": 0.20,
                    "focus_prompt_text": "Focus on pattern beta",
                    "target_gaps": ["pattern beta"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.50,
                    "queries": [
                        {
                            "query_text": "beta result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)

        hints = accumulator.accumulate(task_id="task-b")
        assert len(hints) == 2
        assert hints[0].success_count >= hints[1].success_count
        assert hints[0].gap_pattern == "pattern alpha"

    def test_accumulate_empty_index_returns_empty(self, index, accumulator):
        """Empty index returns empty results."""
        hints = accumulator.accumulate()
        assert hints == []

    def test_accumulate_no_effective_prompts_returns_empty(
        self, index, accumulator, tmp_dir
    ):
        """Trajectories with no coverage improvement yield no hints."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-c",
            "traj-flat",
            rounds_data=[
                {
                    "coverage_ratio": 0.50,
                    "focus_prompt_text": "Focus on something",
                    "target_gaps": ["something"],
                },
                {
                    "coverage_ratio": 0.50,
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.accumulate(task_id="task-c")
        assert hints == []

    def test_accumulate_filters_by_task_id(self, index, accumulator, tmp_dir):
        """task_id filter restricts which trajectories are analyzed."""
        # 🔧 Task X trajectory with hint
        path_x = _write_gap_trajectory(
            tmp_dir,
            "task-x",
            "traj-filtx",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on X pattern",
                    "target_gaps": ["x pattern"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "x result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        # 🔧 Task Y trajectory with hint
        path_y = _write_gap_trajectory(
            tmp_dir,
            "task-y",
            "traj-filty",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on Y pattern",
                    "target_gaps": ["y pattern"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "y result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )

        index.index_trajectory(path_x)
        index.index_trajectory(path_y)

        hints_x = accumulator.accumulate(task_id="task-x")
        assert len(hints_x) == 1
        assert hints_x[0].gap_pattern == "x pattern"

        hints_y = accumulator.accumulate(task_id="task-y")
        assert len(hints_y) == 1
        assert hints_y[0].gap_pattern == "y pattern"

    def test_accumulate_excludes_failed_trajectories(self, index, accumulator, tmp_dir):
        """Failed trajectories are excluded from accumulation."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-d",
            "traj-failed",
            status="failed",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on data",
                    "target_gaps": ["data"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.accumulate(task_id="task-d")
        assert hints == []


# ============================================================================
# 🧪 YAML export tests
# ============================================================================


class TestYamlExport:
    """Tests for YAML export format 📄."""

    def test_export_yaml_creates_valid_file(self, accumulator, tmp_dir):
        """Exported YAML file is valid and parseable."""
        hints = [
            GapSearchHint(
                gap_pattern="regulatory approval status",
                effective_queries=[
                    "{topic} FDA approval status",
                    "{topic} regulatory filing",
                ],
                success_count=5,
                avg_coverage_delta=0.15,
                recommended_tools=["web_search"],
            ),
        ]
        output_path = os.path.join(tmp_dir, "gap_search_hints.yaml")
        accumulator.export_yaml(hints, output_path)

        assert os.path.isfile(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "hints" in data
        assert len(data["hints"]) == 1

    def test_export_yaml_preserves_all_fields(self, accumulator, tmp_dir):
        """All hint fields are present in the exported YAML."""
        hints = [
            GapSearchHint(
                gap_pattern="safety profile data",
                effective_queries=["query_a", "query_b"],
                success_count=3,
                avg_coverage_delta=0.12,
                recommended_tools=["web_search", "scholarly_search"],
            ),
        ]
        output_path = os.path.join(tmp_dir, "hints.yaml")
        accumulator.export_yaml(hints, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        hint = data["hints"][0]
        assert hint["gap_pattern"] == "safety profile data"
        assert hint["effective_queries"] == ["query_a", "query_b"]
        assert hint["success_count"] == 3
        assert hint["avg_coverage_delta"] == pytest.approx(0.12, abs=0.001)
        assert hint["recommended_tools"] == ["web_search", "scholarly_search"]

    def test_export_yaml_multiple_hints(self, accumulator, tmp_dir):
        """Multiple hints are exported correctly."""
        hints = [
            GapSearchHint(
                gap_pattern="pattern_a",
                effective_queries=["qa"],
                success_count=5,
                avg_coverage_delta=0.2,
                recommended_tools=["web_search"],
            ),
            GapSearchHint(
                gap_pattern="pattern_b",
                effective_queries=["qb"],
                success_count=2,
                avg_coverage_delta=0.1,
                recommended_tools=["scholarly_search"],
            ),
        ]
        output_path = os.path.join(tmp_dir, "multi_hints.yaml")
        accumulator.export_yaml(hints, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert len(data["hints"]) == 2
        assert data["hints"][0]["gap_pattern"] == "pattern_a"
        assert data["hints"][1]["gap_pattern"] == "pattern_b"

    def test_export_yaml_empty_hints(self, accumulator, tmp_dir):
        """Empty hints list exports valid YAML with empty array."""
        output_path = os.path.join(tmp_dir, "empty_hints.yaml")
        accumulator.export_yaml([], output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["hints"] == []


# ============================================================================
# 🧪 Relevance matching tests
# ============================================================================


class TestGetHintsForGaps:
    """Tests for get_hints_for_gaps relevance matching 🔍."""

    def test_relevant_hints_ranked_higher(self, index, accumulator, tmp_dir):
        """Hints matching the gap description are ranked first."""
        # 🔧 Create trajectory with two different effective patterns
        path = _write_gap_trajectory(
            tmp_dir,
            "task-rel",
            "traj-rel01",
            rounds_data=[
                {
                    "coverage_ratio": 0.20,
                    "focus_prompt_text": "Focus on safety profile data",
                    "target_gaps": ["safety profile data"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.40,
                    "focus_prompt_text": "Focus on market competition analysis",
                    "target_gaps": ["market competition analysis"],
                    "queries": [
                        {
                            "query_text": "safety data review",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.65,
                    "queries": [
                        {
                            "query_text": "market analysis result",
                            "mcp_tool": "web_search",
                            "result_count": 4,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)

        # 🔍 Query for safety-related gaps
        results = accumulator.get_hints_for_gaps(["safety profile evaluation"], top_k=2)
        assert len(results) >= 1
        # ✅ Safety hint should be ranked higher than market
        assert results[0].gap_pattern == "safety profile data"

    def test_empty_gap_descriptions_returns_top_hints(
        self, index, accumulator, tmp_dir
    ):
        """Empty gap list returns top hints by success count."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-empty-gap",
            "traj-emptygap",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on any pattern",
                    "target_gaps": ["any pattern"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "any result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        results = accumulator.get_hints_for_gaps([], top_k=5)
        assert len(results) >= 1

    def test_no_hints_available_returns_empty(self, index, accumulator):
        """Empty index with no hints returns empty list."""
        results = accumulator.get_hints_for_gaps(["some gap description"])
        assert results == []

    def test_top_k_limits_results(self, index, accumulator, tmp_dir):
        """top_k parameter limits the number of returned hints."""
        # 🔧 Create trajectory with 3 different effective patterns
        path = _write_gap_trajectory(
            tmp_dir,
            "task-topk",
            "traj-topk01",
            rounds_data=[
                {
                    "coverage_ratio": 0.10,
                    "focus_prompt_text": "Focus on alpha data",
                    "target_gaps": ["alpha data"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on beta data",
                    "target_gaps": ["beta data"],
                    "queries": [
                        {
                            "query_text": "alpha result",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.55,
                    "focus_prompt_text": "Focus on gamma data",
                    "target_gaps": ["gamma data"],
                    "queries": [
                        {
                            "query_text": "beta result",
                            "mcp_tool": "web_search",
                            "result_count": 4,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.80,
                    "queries": [
                        {
                            "query_text": "gamma result",
                            "mcp_tool": "web_search",
                            "result_count": 6,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        results = accumulator.get_hints_for_gaps(["alpha data search"], top_k=1)
        assert len(results) == 1


# ============================================================================
# 🧪 Edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and robustness 🛡️."""

    def test_pydantic_model_serialization(self):
        """GapSearchHint model supports round-trip serialization."""
        hint = GapSearchHint(
            gap_pattern="test pattern",
            effective_queries=["query_a"],
            success_count=3,
            avg_coverage_delta=0.15,
            recommended_tools=["web_search"],
        )
        data = hint.model_dump()
        restored = GapSearchHint.model_validate(data)
        assert restored == hint

    def test_coverage_delta_calculation(self, index, accumulator, tmp_dir):
        """Average coverage delta is correctly computed across merges."""
        # 🔧 Two trajectories with same pattern, different deltas
        path1 = _write_gap_trajectory(
            tmp_dir,
            "task-delta",
            "traj-delta-01",
            rounds_data=[
                {
                    "coverage_ratio": 0.40,
                    "focus_prompt_text": "Focus on common pattern",
                    "target_gaps": ["common pattern"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "result1",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        path2 = _write_gap_trajectory(
            tmp_dir,
            "task-delta",
            "traj-delta-02",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on common pattern",
                    "target_gaps": ["common pattern"],
                    "queries": [
                        {
                            "query_text": "q",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.60,
                    "queries": [
                        {
                            "query_text": "result2",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )

        index.index_trajectory(path1)
        index.index_trajectory(path2)

        hints = accumulator.accumulate(task_id="task-delta")
        assert len(hints) == 1
        # ✅ Average of 0.20 and 0.30 = 0.25
        assert hints[0].avg_coverage_delta == pytest.approx(0.25, abs=0.01)

    def test_multiple_target_gaps_per_prompt(self, index, accumulator, tmp_dir):
        """Focus prompt with multiple target gaps produces one hint per gap."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-multi-gap",
            "traj-multigap",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on X and Y",
                    "target_gaps": ["gap alpha", "gap beta"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.55,
                    "queries": [
                        {
                            "query_text": "result for alpha and beta",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        }
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-multigap")
        assert len(hints) == 2
        patterns = {h.gap_pattern for h in hints}
        assert "gap alpha" in patterns
        assert "gap beta" in patterns

    def test_tools_are_deduplicated(self, index, accumulator, tmp_dir):
        """Recommended tools are deduplicated in merged hints."""
        path = _write_gap_trajectory(
            tmp_dir,
            "task-dedup",
            "traj-dedup",
            rounds_data=[
                {
                    "coverage_ratio": 0.30,
                    "focus_prompt_text": "Focus on dedup test",
                    "target_gaps": ["dedup test"],
                    "queries": [
                        {
                            "query_text": "q1",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        }
                    ],
                },
                {
                    "coverage_ratio": 0.55,
                    "queries": [
                        {
                            "query_text": "result a",
                            "mcp_tool": "web_search",
                            "result_count": 5,
                            "cost_usd": 0.01,
                        },
                        {
                            "query_text": "result b",
                            "mcp_tool": "web_search",
                            "result_count": 3,
                            "cost_usd": 0.01,
                        },
                    ],
                },
            ],
        )
        index.index_trajectory(path)
        hints = accumulator.analyze_trajectory("traj-dedup")
        assert len(hints) == 1
        # ✅ "web_search" should appear only once
        assert hints[0].recommended_tools.count("web_search") == 1
