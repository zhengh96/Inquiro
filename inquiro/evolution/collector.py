"""Inquiro TrajectoryCollector — generic trajectory data collection 🧬.

Collects structured execution data from EvoMaster trajectories without
interpreting domain meaning. Extracts raw tool calls, result metrics,
and timing information for later analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from inquiro.core.trajectory_utils import extract_finish_result
from inquiro.evolution.types import (
    ResultMetrics,
    ToolCallRecord,
    TrajectorySnapshot,
)


class TrajectoryCollector:
    """Collects trajectory data for evolution analysis 📸.

    Domain-agnostic collector that extracts raw execution data from
    EvoMaster trajectories. Does not interpret the semantic meaning
    of tools, metrics, or outcomes.

    All context classification (context_tags, sub_item_id) is provided
    by the caller (upper layer), not derived by the collector.
    """

    def __init__(self):
        """Initialize the collector 🔧."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def collect(
        self,
        trajectory: Any,
        task: Any,
        context_tags: list[str] | None = None,
        sub_item_id: str = "",
    ) -> TrajectorySnapshot:
        """Collect trajectory data into a structured snapshot 📸.

        Args:
            trajectory: EvoMaster trajectory object with steps.
            task: Task object with task_id, evaluation_id, topic.
            context_tags: Context tags from caller (opaque to collector).
            sub_item_id: Sub-item identifier from caller (opaque).

        Returns:
            TrajectorySnapshot with tool calls, metrics, and timing.

        Raises:
            ValueError: If trajectory or task is None.
        """
        if trajectory is None:
            raise ValueError("Trajectory cannot be None")
        if task is None:
            raise ValueError("Task cannot be None")

        # 🆔 Extract identity fields
        evaluation_id = getattr(task, "evaluation_id", "")
        task_id = getattr(task, "task_id", "")
        topic = getattr(task, "topic", "")

        # 🔧 Extract tool call records
        tool_calls = self._extract_tool_calls(trajectory)

        # 📊 Extract result metrics from finish tool
        metrics = self._extract_metrics(trajectory)

        # ⏱️ Extract timing information
        started_at, completed_at, wall_time = self._extract_timing(trajectory)

        return TrajectorySnapshot(
            evaluation_id=evaluation_id,
            task_id=task_id,
            topic=topic,
            context_tags=context_tags or [],
            sub_item_id=sub_item_id,
            tool_calls=tool_calls,
            metrics=metrics,
            started_at=started_at,
            completed_at=completed_at,
            wall_time_seconds=wall_time,
        )

    def _extract_tool_calls(self, trajectory: Any) -> list[ToolCallRecord]:
        """Extract all tool call records from trajectory 🔧.

        Iterates through trajectory steps and collects information about
        each MCP tool invocation.

        Args:
            trajectory: EvoMaster trajectory object.

        Returns:
            List of ToolCallRecord objects.
        """
        tool_calls: list[ToolCallRecord] = []

        if not trajectory or not hasattr(trajectory, "steps"):
            return tool_calls

        round_number = 0
        for step in trajectory.steps:
            # 📝 Increment round for each step with tool calls
            if (
                hasattr(step, "assistant_message")
                and step.assistant_message
                and hasattr(step.assistant_message, "tool_calls")
                and step.assistant_message.tool_calls
            ):
                round_number += 1

                for tool_call in step.assistant_message.tool_calls:
                    if not hasattr(tool_call, "function"):
                        continue

                    function = tool_call.function
                    tool_name = getattr(function, "name", "")
                    arguments = getattr(function, "arguments", "")

                    # 📏 Summarize arguments (first 200 chars)
                    arguments_summary = ""
                    try:
                        if isinstance(arguments, str):
                            arguments_summary = arguments[:200]
                        elif isinstance(arguments, dict):
                            arguments_summary = json.dumps(arguments)[:200]
                    except Exception:
                        arguments_summary = str(arguments)[:200]

                    # 📊 Result size and success are not directly available
                    # from trajectory structure — left as defaults
                    result_size = 0
                    success = True

                    tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            arguments_summary=arguments_summary,
                            result_size=result_size,
                            success=success,
                            round_number=round_number,
                        )
                    )

        return tool_calls

    def _extract_metrics(self, trajectory: Any) -> ResultMetrics:
        """Extract result metrics from finish tool call 📊.

        Searches for the finish tool call in the trajectory and extracts
        outcome metrics. Uses the same logic as InquiroBaseExp._extract_result.

        Args:
            trajectory: EvoMaster trajectory object.

        Returns:
            ResultMetrics object (defaults if no finish tool found).
        """
        raw_result = self._extract_finish_result(trajectory)

        if not raw_result:
            return ResultMetrics()

        # 📊 Extract metrics from raw result
        evidence_count = len(raw_result.get("evidence_index", []))
        confidence = float(raw_result.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        decision = raw_result.get("decision", "")
        search_rounds = raw_result.get("search_rounds", 0)

        # 📋 Calculate checklist coverage
        coverage_raw = raw_result.get("checklist_coverage", {})
        required_covered = coverage_raw.get("required_covered", [])
        required_total = len(required_covered) + len(
            coverage_raw.get("required_missing", [])
        )
        checklist_coverage = (
            len(required_covered) / required_total if required_total > 0 else 0.0
        )

        return ResultMetrics(
            evidence_count=evidence_count,
            confidence=confidence,
            decision=decision,
            search_rounds=search_rounds,
            checklist_coverage=checklist_coverage,
            cost_usd=0.0,  # 💰 Cost tracking is handled separately
        )

    def _extract_finish_result(self, trajectory: Any) -> dict[str, Any]:
        """Extract result from finish tool call 📝.

        Delegates to the shared ``extract_finish_result`` utility in
        ``inquiro.core.trajectory_utils``, which handles both object-style
        and dict-style trajectories, applies the two-level JSON parse
        (outer args → ``result_json``), and returns an empty dict on any
        failure.

        Args:
            trajectory: EvoMaster trajectory object.

        Returns:
            Raw result dictionary, or empty dict if not found.
        """
        return extract_finish_result(trajectory)

    def _extract_timing(self, trajectory: Any) -> tuple[datetime, datetime, float]:
        """Extract timing information from trajectory ⏱️.

        Args:
            trajectory: EvoMaster trajectory object.

        Returns:
            Tuple of (started_at, completed_at, wall_time_seconds).
        """
        now = datetime.now(timezone.utc)
        started_at = now
        completed_at = now
        wall_time = 0.0

        if trajectory and hasattr(trajectory, "steps") and trajectory.steps:
            # 🏁 Extract from first and last step if available
            first_step = trajectory.steps[0]
            last_step = trajectory.steps[-1]

            # Try to extract timestamps from steps
            if hasattr(first_step, "timestamp"):
                started_at = first_step.timestamp
            if hasattr(last_step, "timestamp"):
                completed_at = last_step.timestamp

            # Calculate wall time
            if isinstance(started_at, datetime) and isinstance(completed_at, datetime):
                wall_time = (completed_at - started_at).total_seconds()

        return started_at, completed_at, wall_time
