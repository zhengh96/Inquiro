"""Tests for Inquiro TrajectoryCollector 🧪.

Tests the trajectory data collection system:
- Tool call extraction from EvoMaster trajectories
- Result metrics extraction from finish tool calls
- Timing information extraction
- Edge case handling (empty trajectory, missing finish tool, malformed args)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from inquiro.evolution.collector import TrajectoryCollector
from inquiro.evolution.types import TrajectorySnapshot


# ============================================================
# 🏗️ Fixtures
# ============================================================


@pytest.fixture
def collector() -> TrajectoryCollector:
    """Create a TrajectoryCollector instance 🔧."""
    return TrajectoryCollector()


@pytest.fixture
def mock_task() -> Mock:
    """Create a mock task object 📋."""
    task = Mock()
    task.evaluation_id = "eval_123"
    task.task_id = "task_456"
    task.topic = "Test Topic"
    return task


@pytest.fixture
def mock_trajectory_with_finish() -> Mock:
    """Create a mock trajectory with tool calls and finish tool 🔧."""
    # 🔍 Create mock tool calls
    search_tool = Mock()
    search_tool.function = Mock()
    search_tool.function.name = "search_pubmed"
    search_tool.function.arguments = json.dumps({"query": "test query"})

    finish_tool = Mock()
    finish_tool.function = Mock()
    finish_tool.function.name = "finish"
    finish_result = {
        "decision": "positive",
        "confidence": 0.85,
        "evidence_index": [
            {"id": "E1", "source": "PubMed:123"},
            {"id": "E2", "source": "PubMed:456"},
        ],
        "search_rounds": 2,
        "checklist_coverage": {
            "required_covered": ["item1", "item2"],
            "required_missing": ["item3"],
        },
    }
    finish_tool.function.arguments = json.dumps(
        {"result_json": json.dumps(finish_result)}
    )

    # 📝 Create mock steps
    step1 = Mock()
    step1.assistant_message = Mock()
    step1.assistant_message.tool_calls = [search_tool]
    step1.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

    step2 = Mock()
    step2.assistant_message = Mock()
    step2.assistant_message.tool_calls = [finish_tool]
    step2.timestamp = datetime(2026, 2, 19, 10, 5, 0, tzinfo=timezone.utc)

    # 🧬 Create trajectory
    trajectory = Mock()
    trajectory.steps = [step1, step2]

    return trajectory


@pytest.fixture
def mock_empty_trajectory() -> Mock:
    """Create a mock empty trajectory 📋."""
    trajectory = Mock()
    trajectory.steps = []
    return trajectory


@pytest.fixture
def mock_trajectory_no_finish() -> Mock:
    """Create a mock trajectory without finish tool 🔧."""
    # 🔍 Create mock tool calls (no finish)
    search_tool = Mock()
    search_tool.function = Mock()
    search_tool.function.name = "search_pubmed"
    search_tool.function.arguments = json.dumps({"query": "test query"})

    # 📝 Create mock steps
    step1 = Mock()
    step1.assistant_message = Mock()
    step1.assistant_message.tool_calls = [search_tool]
    step1.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

    # 🧬 Create trajectory
    trajectory = Mock()
    trajectory.steps = [step1]

    return trajectory


@pytest.fixture
def mock_trajectory_malformed_finish() -> Mock:
    """Create a mock trajectory with malformed finish tool 🔧."""
    # 🔍 Create mock finish tool with malformed arguments
    finish_tool = Mock()
    finish_tool.function = Mock()
    finish_tool.function.name = "finish"
    finish_tool.function.arguments = "not valid json{"

    # 📝 Create mock steps
    step1 = Mock()
    step1.assistant_message = Mock()
    step1.assistant_message.tool_calls = [finish_tool]
    step1.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

    # 🧬 Create trajectory
    trajectory = Mock()
    trajectory.steps = [step1]

    return trajectory


# ============================================================
# ✅ Basic Collection Tests
# ============================================================


class TestBasicCollection:
    """Tests for basic trajectory collection functionality ✅."""

    def test_collect_returns_snapshot(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should return a TrajectorySnapshot."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
            context_tags=["test_tag"],
            sub_item_id="sub_123",
        )

        assert isinstance(snapshot, TrajectorySnapshot)
        assert snapshot.evaluation_id == "eval_123"
        assert snapshot.task_id == "task_456"
        assert snapshot.topic == "Test Topic"
        assert snapshot.context_tags == ["test_tag"]
        assert snapshot.sub_item_id == "sub_123"

    def test_collect_extracts_tool_calls(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should extract all tool calls from trajectory."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
        )

        assert len(snapshot.tool_calls) == 2
        assert snapshot.tool_calls[0].tool_name == "search_pubmed"
        assert snapshot.tool_calls[0].round_number == 1
        assert "test query" in snapshot.tool_calls[0].arguments_summary
        assert snapshot.tool_calls[1].tool_name == "finish"
        assert snapshot.tool_calls[1].round_number == 2

    def test_collect_extracts_metrics(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should extract result metrics from finish tool."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
        )

        assert snapshot.metrics.evidence_count == 2
        assert snapshot.metrics.confidence == 0.85
        assert snapshot.metrics.decision == "positive"
        assert snapshot.metrics.search_rounds == 2
        # 📋 Coverage: 2 covered / (2 covered + 1 missing) = 2/3 = 0.667
        assert abs(snapshot.metrics.checklist_coverage - 0.667) < 0.01

    def test_collect_extracts_timing(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should extract timing information from trajectory."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
        )

        assert snapshot.started_at == datetime(
            2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc
        )
        assert snapshot.completed_at == datetime(
            2026, 2, 19, 10, 5, 0, tzinfo=timezone.utc
        )
        # 🕒 Wall time: 5 minutes = 300 seconds
        assert snapshot.wall_time_seconds == 300.0


# ============================================================
# 🛡️ Edge Case Tests
# ============================================================


class TestEdgeCases:
    """Tests for edge case handling 🛡️."""

    def test_collect_raises_on_none_trajectory(
        self,
        collector: TrajectoryCollector,
        mock_task: Mock,
    ) -> None:
        """Collector should raise ValueError if trajectory is None."""
        with pytest.raises(ValueError, match="Trajectory cannot be None"):
            collector.collect(None, mock_task)

    def test_collect_raises_on_none_task(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
    ) -> None:
        """Collector should raise ValueError if task is None."""
        with pytest.raises(ValueError, match="Task cannot be None"):
            collector.collect(mock_trajectory_with_finish, None)

    def test_collect_empty_trajectory(
        self,
        collector: TrajectoryCollector,
        mock_empty_trajectory: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should handle empty trajectory gracefully."""
        snapshot = collector.collect(
            mock_empty_trajectory,
            mock_task,
        )

        assert len(snapshot.tool_calls) == 0
        # 📊 Metrics should be defaults
        assert snapshot.metrics.evidence_count == 0
        assert snapshot.metrics.confidence == 0.0
        assert snapshot.metrics.decision == ""
        assert snapshot.metrics.search_rounds == 0

    def test_collect_no_finish_tool(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_no_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should handle trajectory without finish tool."""
        snapshot = collector.collect(
            mock_trajectory_no_finish,
            mock_task,
        )

        # 🔧 Tool calls should still be extracted
        assert len(snapshot.tool_calls) == 1
        assert snapshot.tool_calls[0].tool_name == "search_pubmed"

        # 📊 Metrics should be defaults
        assert snapshot.metrics.evidence_count == 0
        assert snapshot.metrics.confidence == 0.0
        assert snapshot.metrics.decision == ""

    def test_collect_malformed_finish_tool(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_malformed_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should handle malformed finish tool arguments."""
        snapshot = collector.collect(
            mock_trajectory_malformed_finish,
            mock_task,
        )

        # 🔧 Tool calls should still be extracted
        assert len(snapshot.tool_calls) == 1
        assert snapshot.tool_calls[0].tool_name == "finish"

        # 📊 Metrics should be defaults (extraction failed)
        assert snapshot.metrics.evidence_count == 0
        assert snapshot.metrics.confidence == 0.0
        assert snapshot.metrics.decision == ""

    def test_collect_default_context_tags(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should use empty list if context_tags not provided."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
        )

        assert snapshot.context_tags == []

    def test_collect_default_sub_item_id(
        self,
        collector: TrajectoryCollector,
        mock_trajectory_with_finish: Mock,
        mock_task: Mock,
    ) -> None:
        """Collector should use empty string if sub_item_id not provided."""
        snapshot = collector.collect(
            mock_trajectory_with_finish,
            mock_task,
        )

        assert snapshot.sub_item_id == ""


# ============================================================
# 🔧 Tool Call Extraction Tests
# ============================================================


class TestToolCallExtraction:
    """Tests for tool call record extraction 🔧."""

    def test_extract_multiple_tool_calls_same_step(
        self,
        collector: TrajectoryCollector,
        mock_task: Mock,
    ) -> None:
        """Collector should extract multiple tool calls from same step."""
        # 🔍 Create mock tool calls in same step
        tool1 = Mock()
        tool1.function = Mock()
        tool1.function.name = "search_pubmed"
        tool1.function.arguments = json.dumps({"query": "query1"})

        tool2 = Mock()
        tool2.function = Mock()
        tool2.function.name = "search_clinicaltrials"
        tool2.function.arguments = json.dumps({"query": "query2"})

        # 📝 Create mock step with multiple tools
        step = Mock()
        step.assistant_message = Mock()
        step.assistant_message.tool_calls = [tool1, tool2]
        step.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

        # 🧬 Create trajectory
        trajectory = Mock()
        trajectory.steps = [step]

        snapshot = collector.collect(trajectory, mock_task)

        assert len(snapshot.tool_calls) == 2
        assert snapshot.tool_calls[0].tool_name == "search_pubmed"
        assert snapshot.tool_calls[0].round_number == 1
        assert snapshot.tool_calls[1].tool_name == "search_clinicaltrials"
        assert snapshot.tool_calls[1].round_number == 1  # Same round

    def test_extract_tool_calls_across_rounds(
        self,
        collector: TrajectoryCollector,
        mock_task: Mock,
    ) -> None:
        """Collector should track round numbers across steps."""
        # 🔍 Create mock tool calls in different steps
        tool1 = Mock()
        tool1.function = Mock()
        tool1.function.name = "search_pubmed"
        tool1.function.arguments = json.dumps({"query": "query1"})

        tool2 = Mock()
        tool2.function = Mock()
        tool2.function.name = "search_clinicaltrials"
        tool2.function.arguments = json.dumps({"query": "query2"})

        # 📝 Create mock steps
        step1 = Mock()
        step1.assistant_message = Mock()
        step1.assistant_message.tool_calls = [tool1]
        step1.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

        step2 = Mock()
        step2.assistant_message = Mock()
        step2.assistant_message.tool_calls = [tool2]
        step2.timestamp = datetime(2026, 2, 19, 10, 1, 0, tzinfo=timezone.utc)

        # 🧬 Create trajectory
        trajectory = Mock()
        trajectory.steps = [step1, step2]

        snapshot = collector.collect(trajectory, mock_task)

        assert len(snapshot.tool_calls) == 2
        assert snapshot.tool_calls[0].round_number == 1
        assert snapshot.tool_calls[1].round_number == 2  # Different round

    def test_extract_truncates_long_arguments(
        self,
        collector: TrajectoryCollector,
        mock_task: Mock,
    ) -> None:
        """Collector should truncate long arguments to 200 chars."""
        # 🔍 Create mock tool call with long arguments
        long_query = "a" * 300  # 300 chars
        tool = Mock()
        tool.function = Mock()
        tool.function.name = "search_pubmed"
        tool.function.arguments = json.dumps({"query": long_query})

        # 📝 Create mock step
        step = Mock()
        step.assistant_message = Mock()
        step.assistant_message.tool_calls = [tool]
        step.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

        # 🧬 Create trajectory
        trajectory = Mock()
        trajectory.steps = [step]

        snapshot = collector.collect(trajectory, mock_task)

        assert len(snapshot.tool_calls) == 1
        # 📏 Arguments should be truncated to 200 chars
        assert len(snapshot.tool_calls[0].arguments_summary) == 200

    def test_extract_handles_dict_arguments(
        self,
        collector: TrajectoryCollector,
        mock_task: Mock,
    ) -> None:
        """Collector should handle dict arguments (not string)."""
        # 🔍 Create mock tool call with dict arguments
        tool = Mock()
        tool.function = Mock()
        tool.function.name = "search_pubmed"
        tool.function.arguments = {"query": "test query"}  # Dict, not string

        # 📝 Create mock step
        step = Mock()
        step.assistant_message = Mock()
        step.assistant_message.tool_calls = [tool]
        step.timestamp = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)

        # 🧬 Create trajectory
        trajectory = Mock()
        trajectory.steps = [step]

        snapshot = collector.collect(trajectory, mock_task)

        assert len(snapshot.tool_calls) == 1
        # 📝 Arguments should be JSON-serialized
        assert "test query" in snapshot.tool_calls[0].arguments_summary
