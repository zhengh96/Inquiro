"""Tests for Discovery Trajectory recording system 🧪.

Covers models, JSONL writer, and event handling.
"""

from __future__ import annotations

import json
import os
import tempfile


from inquiro.core.trajectory.models import (
    AnalysisPhaseRecord,
    CleaningPhaseRecord,
    ConsensusRecord,
    DiscoveryRoundRecord,
    DiscoverySummary,
    DiscoveryTrajectory,
    FocusPromptRecord,
    GapPhaseRecord,
    ModelAnalysisRecord,
    QueryRecord,
    SearchPhaseRecord,
    SynthesisRecord,
    TrajectoryEvent,
    TrajectoryEventType,
)
from inquiro.core.trajectory.writer import TrajectoryWriter


# ============================================================================
# 📊 Model tests
# ============================================================================


class TestTrajectoryModels:
    """Test Trajectory data model construction and serialization 📊."""

    def test_query_record_default_values(self) -> None:
        """QueryRecord has sane defaults ✅."""
        qr = QueryRecord(query_text="STAT3 inhibitor clinical trials")
        assert qr.mcp_tool == ""
        assert qr.result_count == 0
        assert qr.cost_usd == 0.0

    def test_search_phase_record_with_queries(self) -> None:
        """SearchPhaseRecord holds multiple query records ✅."""
        spr = SearchPhaseRecord(
            queries=[
                QueryRecord(query_text="q1", result_count=5),
                QueryRecord(query_text="q2", result_count=3),
            ],
            total_raw_evidence=8,
            duration_seconds=12.5,
        )
        assert len(spr.queries) == 2
        assert spr.total_raw_evidence == 8

    def test_cleaning_phase_record(self) -> None:
        """CleaningPhaseRecord captures dedup and noise stats ✅."""
        cpr = CleaningPhaseRecord(
            input_count=25,
            output_count=18,
            dedup_removed=4,
            noise_removed=3,
            tag_distribution={"academic": 10, "patent": 5, "other": 3},
        )
        assert cpr.input_count == 25
        assert cpr.output_count == 18
        assert cpr.tag_distribution["academic"] == 10

    def test_model_analysis_record(self) -> None:
        """ModelAnalysisRecord captures per-model results ✅."""
        mar = ModelAnalysisRecord(
            model_name="gpt-5.2",
            claims_count=7,
            decision="positive",
            confidence=0.85,
            cost_usd=0.52,
        )
        assert mar.model_name == "gpt-5.2"
        assert mar.confidence == 0.85

    def test_analysis_phase_record_with_consensus(self) -> None:
        """AnalysisPhaseRecord combines models and consensus ✅."""
        apr = AnalysisPhaseRecord(
            model_results=[
                ModelAnalysisRecord(model_name="m1", claims_count=5),
                ModelAnalysisRecord(model_name="m2", claims_count=4),
            ],
            consensus=ConsensusRecord(
                consensus_decision="positive",
                consensus_ratio=0.67,
                total_claims=6,
            ),
        )
        assert len(apr.model_results) == 2
        assert apr.consensus.consensus_ratio == 0.67

    def test_gap_phase_record_with_focus_prompt(self) -> None:
        """GapPhaseRecord includes optional focus prompt ✅."""
        gpr = GapPhaseRecord(
            coverage_ratio=0.6,
            covered_items=["R1", "R2", "R3"],
            uncovered_items=["R4", "R5"],
            focus_prompt=FocusPromptRecord(
                prompt_text="Focus on R4 and R5",
                target_gaps=["R4", "R5"],
            ),
        )
        assert gpr.coverage_ratio == 0.6
        assert gpr.focus_prompt is not None
        assert gpr.focus_prompt.target_gaps == ["R4", "R5"]

    def test_discovery_round_record_complete(self) -> None:
        """DiscoveryRoundRecord contains all four phases ✅."""
        drr = DiscoveryRoundRecord(
            round_number=1,
            search_phase=SearchPhaseRecord(total_raw_evidence=20),
            cleaning_phase=CleaningPhaseRecord(input_count=20, output_count=15),
            analysis_phase=AnalysisPhaseRecord(),
            gap_phase=GapPhaseRecord(coverage_ratio=0.7),
            round_cost_usd=2.15,
        )
        assert drr.round_number == 1
        assert drr.search_phase.total_raw_evidence == 20
        assert drr.gap_phase.coverage_ratio == 0.7

    def test_discovery_trajectory_full_structure(self) -> None:
        """DiscoveryTrajectory composes all sub-records ✅."""
        traj = DiscoveryTrajectory(
            task_id="task-001",
            config_snapshot={"max_rounds": 3},
            rounds=[
                DiscoveryRoundRecord(round_number=1),
                DiscoveryRoundRecord(round_number=2),
            ],
            summary=DiscoverySummary(
                total_rounds=2,
                final_coverage=0.85,
                total_cost_usd=4.30,
                termination_reason="coverage_reached",
            ),
        )
        assert traj.task_id == "task-001"
        assert len(traj.rounds) == 2
        assert traj.summary.final_coverage == 0.85
        assert traj.trajectory_id  # Auto-generated UUID

    def test_trajectory_event_creation(self) -> None:
        """TrajectoryEvent captures event type and data ✅."""
        event = TrajectoryEvent(
            event_type=TrajectoryEventType.DISCOVERY_STARTED,
            data={"task_id": "t1"},
        )
        assert event.event_type == TrajectoryEventType.DISCOVERY_STARTED
        assert event.data["task_id"] == "t1"
        assert event.timestamp is not None

    def test_discovery_summary_metrics(self) -> None:
        """DiscoverySummary computes quality metrics ✅."""
        summary = DiscoverySummary(
            total_rounds=3,
            final_coverage=0.90,
            total_cost_usd=6.21,
            total_evidence=21,
            total_claims=8,
            total_duration_seconds=180.0,
            termination_reason="coverage_reached",
            evidence_yield_rate=0.84,
            cost_normalized_quality=0.145,
        )
        assert summary.cost_normalized_quality > 0
        assert summary.evidence_yield_rate > 0

    def test_model_dump_round_trip(self) -> None:
        """Models serialize and deserialize correctly ✅."""
        traj = DiscoveryTrajectory(
            task_id="test",
            rounds=[DiscoveryRoundRecord(round_number=1)],
        )
        dumped = traj.model_dump()
        restored = DiscoveryTrajectory.model_validate(dumped)
        assert restored.task_id == "test"
        assert len(restored.rounds) == 1


# ============================================================================
# 💾 Writer tests
# ============================================================================


class TestTrajectoryWriter:
    """Test JSONL trajectory writer 💾."""

    def test_write_meta_creates_file(self) -> None:
        """write_meta creates the JSONL file with meta record ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-001")
            writer.write_meta(
                trajectory_id="traj-1",
                config_snapshot={"max_rounds": 3},
            )
            assert os.path.exists(writer.file_path)
            with open(writer.file_path) as f:
                line = json.loads(f.readline())
            assert line["type"] == "meta"
            assert line["trajectory_id"] == "traj-1"
            assert line["config_snapshot"]["max_rounds"] == 3

    def test_write_round_appends_to_file(self) -> None:
        """write_round appends round record to existing file ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-002")
            writer.write_meta(trajectory_id="traj-2", config_snapshot={})
            writer.write_round(
                DiscoveryRoundRecord(
                    round_number=1,
                    round_cost_usd=2.07,
                )
            )
            with open(writer.file_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            round_record = json.loads(lines[1])
            assert round_record["type"] == "round"
            assert round_record["round_number"] == 1

    def test_write_summary(self) -> None:
        """write_summary writes summary record ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-003")
            writer.write_summary(
                DiscoverySummary(
                    total_rounds=2,
                    final_coverage=0.85,
                )
            )
            with open(writer.file_path) as f:
                line = json.loads(f.readline())
            assert line["type"] == "summary"
            assert line["final_coverage"] == 0.85

    def test_write_event(self) -> None:
        """write_event writes event record ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-004")
            event = TrajectoryWriter.emit_event(
                TrajectoryEventType.DISCOVERY_STARTED,
                data={"round": 1},
            )
            writer.write_event(event)
            with open(writer.file_path) as f:
                line = json.loads(f.readline())
            assert line["type"] == "event"
            assert line["event_type"] == "discovery_started"

    def test_finalize_writes_meta_final(self) -> None:
        """finalize writes the closing meta record ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-005")
            writer.finalize(
                status="completed",
                termination_reason="coverage_reached",
            )
            with open(writer.file_path) as f:
                line = json.loads(f.readline())
            assert line["type"] == "meta_final"
            assert line["status"] == "completed"
            assert line["termination_reason"] == "coverage_reached"

    def test_full_trajectory_write_sequence(self) -> None:
        """Full write sequence produces valid JSONL ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = TrajectoryWriter(tmpdir, "task-full")
            writer.write_meta("traj-full", {"max_rounds": 3})
            writer.write_round(DiscoveryRoundRecord(round_number=1))
            writer.write_round(DiscoveryRoundRecord(round_number=2))
            writer.write_synthesis(SynthesisRecord(cost_usd=1.50))
            writer.write_summary(DiscoverySummary(total_rounds=2, final_coverage=0.90))
            writer.finalize("completed", "coverage_reached")

            with open(writer.file_path) as f:
                lines = f.readlines()
            assert len(lines) == 6

            # ✅ Verify each line is valid JSON
            types = [json.loads(line)["type"] for line in lines]
            assert types == [
                "meta",
                "round",
                "round",
                "synthesis",
                "summary",
                "meta_final",
            ]

    def test_write_to_nonexistent_dir_creates_it(self) -> None:
        """Writer creates output directory if needed ✅."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "deep", "nested", "dir")
            writer = TrajectoryWriter(nested, "task-nested")
            writer.write_meta("traj-n", {})
            assert os.path.exists(writer.file_path)

    def test_write_failure_does_not_raise(self) -> None:
        """Write failures are logged but never propagated ✅."""
        # 🔧 Use an invalid path that will fail
        writer = TrajectoryWriter("/proc/nonexistent", "task-fail")
        # ✅ Should not raise
        writer.write_meta("traj-fail", {})
        writer.write_round(DiscoveryRoundRecord(round_number=1))
        writer.finalize("failed", "error")

    def test_emit_event_helper(self) -> None:
        """emit_event creates properly typed events ✅."""
        event = TrajectoryWriter.emit_event(
            TrajectoryEventType.CONVERGENCE_REACHED,
            data={"reason": "coverage_threshold"},
        )
        assert event.event_type == TrajectoryEventType.CONVERGENCE_REACHED
        assert event.data["reason"] == "coverage_threshold"
        assert event.timestamp is not None
