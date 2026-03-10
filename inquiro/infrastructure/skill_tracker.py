"""Skill execution tracking for post-hoc analysis 📊.

Records each ``use_skill`` invocation with input parameters,
output, timing, and error information. Writes a JSONL log to
the task's run directory for full traceability.

Usage::

    tracker = SkillExecutionTracker()
    tracker.record(
        skill_name="evidence-grader",
        action="get_info",
        output="...",
        duration_ms=42,
    )
    tracker.flush(Path("/runs/abc/task_123"))
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillExecutionRecord:
    """Single skill invocation record 📝.

    Attributes:
        timestamp: ISO 8601 timestamp of the invocation.
        skill_name: Name of the skill invoked.
        action: Action type (get_info, get_reference, run_script).
        reference_name: Reference document name (if applicable).
        script_name: Script name (if applicable).
        script_args: Script arguments (if applicable).
        output_preview: First 500 chars of output.
        duration_ms: Execution time in milliseconds.
        success: Whether the invocation succeeded.
        error: Error message (if failed).
    """

    timestamp: str = ""
    skill_name: str = ""
    action: str = ""
    reference_name: str | None = None
    script_name: str | None = None
    script_args: str | None = None
    output_preview: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


class SkillExecutionTracker:
    """Thread-safe tracker for skill invocations 📊.

    Accumulates records in memory and flushes to a JSONL file
    on demand or at finalization.

    Attributes:
        records: List of accumulated execution records.
    """

    def __init__(self) -> None:
        """Initialize tracker (empty) 🔧."""
        self._records: list[SkillExecutionRecord] = []
        self._lock = threading.Lock()

    @property
    def records(self) -> list[SkillExecutionRecord]:
        """Return a copy of accumulated records 📋.

        Returns:
            List of SkillExecutionRecord.
        """
        with self._lock:
            return list(self._records)

    def record(
        self,
        skill_name: str,
        action: str,
        output: str = "",
        duration_ms: float = 0.0,
        success: bool = True,
        error: str | None = None,
        reference_name: str | None = None,
        script_name: str | None = None,
        script_args: str | None = None,
    ) -> None:
        """Add a skill execution record 📝.

        Args:
            skill_name: Name of the invoked skill.
            action: Action type.
            output: Full output text (truncated for preview).
            duration_ms: Execution duration in milliseconds.
            success: Whether the call succeeded.
            error: Error message if failed.
            reference_name: Reference name (for get_reference).
            script_name: Script name (for run_script).
            script_args: Script arguments (for run_script).
        """
        entry = SkillExecutionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            skill_name=skill_name,
            action=action,
            reference_name=reference_name,
            script_name=script_name,
            script_args=script_args,
            output_preview=output[:500] if output else "",
            duration_ms=round(duration_ms, 2),
            success=success,
            error=error,
        )
        with self._lock:
            self._records.append(entry)

    def flush(self, task_dir: Path) -> Path | None:
        """Write all records to a JSONL file 💾.

        Args:
            task_dir: Task directory to write the log into.

        Returns:
            Path to the skill execution log, or None on failure.
        """
        with self._lock:
            if not self._records:
                return None
            snapshot = list(self._records)

        try:
            task_dir.mkdir(parents=True, exist_ok=True)
            path = task_dir / "skill_executions.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in snapshot:
                    f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            logger.info(
                "📊 Flushed %d skill execution record(s) to %s",
                len(snapshot),
                path,
            )
            return path
        except Exception as exc:
            logger.warning("⚠️ Failed to flush skill executions: %s", exc)
            return None

    def summary(self) -> dict:
        """Generate summary statistics 📊.

        Returns:
            Dict with total calls, unique skills, error count, etc.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "total_calls": 0,
                "unique_skills": 0,
                "error_count": 0,
                "total_duration_ms": 0.0,
            }

        skills = set(r.skill_name for r in records)
        errors = sum(1 for r in records if not r.success)
        total_ms = sum(r.duration_ms for r in records)

        return {
            "total_calls": len(records),
            "unique_skills": len(skills),
            "skills_used": sorted(skills),
            "error_count": errors,
            "total_duration_ms": round(total_ms, 2),
        }


@dataclass
class _TimingContext:
    """Helper for timing a skill execution ⏱️."""

    start: float = field(default_factory=time.monotonic)

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds ⏱️."""
        return (time.monotonic() - self.start) * 1000


def create_timing_context() -> _TimingContext:
    """Create a new timing context ⏱️.

    Returns:
        _TimingContext with start time set to now.
    """
    return _TimingContext()
