"""Discovery Trajectory JSONL writer 💾.

Provides streaming-safe writing of trajectory records to JSONL files.
Each discovery run produces one JSONL file with line-delimited records.

Write semantics:
    - Non-blocking: write failures are logged but never propagate
    - Crash-safe: each line is flushed independently
    - Append-only: supports streaming mid-flight and post-hoc writes

JSONL format:
    {"type": "meta", "trajectory_id": "...", "config_snapshot": {...}}
    {"type": "round", "round_number": 1, "search_phase": {...}, ...}
    {"type": "synthesis", ...}
    {"type": "summary", "total_rounds": 3, "final_coverage": 0.90, ...}
    {"type": "meta_final", "status": "completed", "termination_reason": "..."}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from inquiro.core.trajectory.models import (
    DiscoveryRoundRecord,
    DiscoverySummary,
    SynthesisRecord,
    TrajectoryEvent,
    TrajectoryEventType,
)

_UTC = ZoneInfo("UTC")

logger = logging.getLogger(__name__)


class TrajectoryWriter:
    """Streaming JSONL writer for Discovery trajectories 💾.

    Non-blocking, crash-safe writer that appends trajectory records
    to a JSONL file.  All write methods catch exceptions internally
    and log warnings — they never raise.

    Example::

        writer = TrajectoryWriter("/path/to/trajectories", "task-123")
        writer.write_meta(trajectory_id="abc", config={...})
        writer.write_round(round_record)
        writer.write_summary(summary)
        writer.finalize("completed", "coverage_reached")

    Attributes:
        output_dir: Directory for JSONL files.
        task_id: Associated task identifier.
        file_path: Full path to the JSONL file.
    """

    def __init__(self, output_dir: str, task_id: str) -> None:
        """Initialize the trajectory writer 🔧.

        Args:
            output_dir: Directory where JSONL files are created.
            task_id: Task identifier used in the filename.
        """
        self.output_dir = output_dir
        self.task_id = task_id
        timestamp = datetime.now(tz=_UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"discovery_{task_id}_{timestamp}.jsonl"
        self.file_path = os.path.join(output_dir, filename)
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create output directory if it doesn't exist 📁."""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "💾 Could not create trajectory dir %s: %s",
                self.output_dir,
                e,
            )

    def _append_line(self, record: dict[str, Any]) -> None:
        """Append a single JSON line to the file 📝.

        Non-blocking: catches all exceptions and logs warnings.

        Args:
            record: Dictionary to serialize as one JSON line.
        """
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception as e:
            logger.warning(
                "💾 Trajectory write failed for %s: %s",
                self.task_id,
                e,
            )

    def write_meta(
        self,
        trajectory_id: str,
        config_snapshot: dict[str, Any],
        task_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Write the initial meta record 📋.

        Args:
            trajectory_id: Unique trajectory identifier.
            config_snapshot: DiscoveryConfig parameters.
            task_snapshot: Task rules and checklist snapshot.
        """
        self._append_line(
            {
                "type": "meta",
                "trajectory_id": trajectory_id,
                "task_id": self.task_id,
                "config_snapshot": config_snapshot,
                "task_snapshot": task_snapshot or {},
                "timestamp": datetime.now(tz=_UTC).isoformat(),
            }
        )

    def write_round(self, round_record: DiscoveryRoundRecord) -> None:
        """Write a round record after completion 📊.

        Args:
            round_record: Complete record for one discovery round.
        """
        self._append_line(
            {
                "type": "round",
                **round_record.model_dump(),
            }
        )

    def write_event(self, event: TrajectoryEvent) -> None:
        """Write a timeline event ⏱️.

        Args:
            event: Timestamped trajectory event.
        """
        self._append_line(
            {
                "type": "event",
                **event.model_dump(),
            }
        )

    def write_synthesis(self, synthesis_record: SynthesisRecord) -> None:
        """Write the synthesis phase record 📝.

        Args:
            synthesis_record: Synthesis phase record.
        """
        self._append_line(
            {
                "type": "synthesis",
                **synthesis_record.model_dump(),
            }
        )

    def write_summary(self, summary: DiscoverySummary) -> None:
        """Write the aggregate summary record 📊.

        Args:
            summary: Aggregate summary of the discovery run.
        """
        self._append_line(
            {
                "type": "summary",
                **summary.model_dump(),
            }
        )

    def finalize(
        self,
        status: str = "completed",
        termination_reason: str = "",
    ) -> None:
        """Write the final meta record and close 🏁.

        Args:
            status: Final status (completed/failed/cancelled).
            termination_reason: Why the run terminated.
        """
        self._append_line(
            {
                "type": "meta_final",
                "status": status,
                "termination_reason": termination_reason,
                "timestamp": datetime.now(tz=_UTC).isoformat(),
            }
        )
        logger.info(
            "💾 Trajectory finalized: %s (status=%s, reason=%s)",
            self.file_path,
            status,
            termination_reason,
        )

    @staticmethod
    def emit_event(
        event_type: TrajectoryEventType,
        data: dict[str, Any] | None = None,
    ) -> TrajectoryEvent:
        """Create a trajectory event 🔧.

        Helper to create TrajectoryEvent instances with current timestamp.

        Args:
            event_type: Type of event.
            data: Optional additional data.

        Returns:
            New TrajectoryEvent instance.
        """
        return TrajectoryEvent(
            event_type=event_type,
            timestamp=datetime.now(tz=_UTC),
            data=data or {},
        )
