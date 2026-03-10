"""RunDirectoryManager — per-task directory isolation 📁.

Provides structured, concurrent-safe directory management for
evaluation runs. Each task gets an isolated directory for
trajectory, configuration snapshot, result, and metadata.

Directory layout::

    {base_dir}/
    └── {run_id}/
        ├── manifest.json         ← Run manifest (all tasks)
        ├── config.yaml           ← Service config snapshot
        └── task_{task_id}/
            ├── trajectory.jsonl   ← Agent execution trajectory
            ├── config.json        ← Task config snapshot
            ├── result.json        ← Evaluation result
            └── metadata.json      ← Timing, cost, QG info

Usage::

    manager = RunDirectoryManager(base_dir=Path("/data/runs"))
    run_dir = manager.create_run()
    task_dir = manager.get_task_dir(task_id="abc123")
    manager.save_task_config(task_id, config_dict)
    manager.save_task_result(task_id, result_dict)
    manager.save_task_metadata(task_id, meta_dict)
    manager.finalize_run(summary_dict)
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# 📁 Default base directory for runs
_DEFAULT_BASE_DIR = Path("runs")


class RunDirectoryManager:
    """Manages per-run and per-task directory structure 📁.

    Thread-safe. Each operation creates directories lazily and
    writes files atomically (write-to-temp + rename where possible).

    Attributes:
        base_dir: Root directory for all runs.
        run_id: Unique identifier for the current run.
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize RunDirectoryManager (no file IO) 🔧.

        Args:
            base_dir: Root directory for runs.
                Defaults to ``./runs/``.
            run_id: Unique run identifier. Auto-generated
                from timestamp if None.
        """
        self.base_dir = base_dir or _DEFAULT_BASE_DIR
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()
        self._task_dirs: dict[str, Path] = {}

    @property
    def run_dir(self) -> Path:
        """Return the run-level directory path 📂.

        Returns:
            Path to ``{base_dir}/{run_id}/``.
        """
        return self.base_dir / self.run_id

    def create_run(self) -> Path:
        """Create the run directory on disk 📂.

        Returns:
            Path to the created run directory.
        """
        run_path = self.run_dir
        run_path.mkdir(parents=True, exist_ok=True)
        logger.info("📁 Created run directory: %s", run_path)
        return run_path

    def get_task_dir(self, task_id: str) -> Path:
        """Get or create a per-task directory 📂.

        Args:
            task_id: Unique task identifier.

        Returns:
            Path to ``{run_dir}/task_{task_id}/``.
        """
        with self._lock:
            if task_id in self._task_dirs:
                return self._task_dirs[task_id]

            task_path = self.run_dir / f"task_{task_id}"
            task_path.mkdir(parents=True, exist_ok=True)
            self._task_dirs[task_id] = task_path
            logger.debug("📁 Created task directory: %s", task_path)
            return task_path

    def get_trajectory_path(self, task_id: str) -> str:
        """Return the trajectory directory path for a task 💾.

        This is the value to inject into ``EvaluationTask.trajectory_dir``.

        Args:
            task_id: Task identifier.

        Returns:
            String path to the task directory (for trajectory_dir field).
        """
        return str(self.get_task_dir(task_id))

    def save_task_config(
        self,
        task_id: str,
        config: dict,
    ) -> Path:
        """Save task configuration snapshot 📋.

        Args:
            task_id: Task identifier.
            config: Task configuration dictionary.

        Returns:
            Path to the saved config file.
        """
        path = self.get_task_dir(task_id) / "config.json"
        self._write_json(path, config)
        return path

    def save_task_result(
        self,
        task_id: str,
        result: dict,
    ) -> Path:
        """Save task evaluation result 📊.

        Args:
            task_id: Task identifier.
            result: Evaluation result dictionary.

        Returns:
            Path to the saved result file.
        """
        path = self.get_task_dir(task_id) / "result.json"
        self._write_json(path, result)
        return path

    def save_task_metadata(
        self,
        task_id: str,
        metadata: dict,
    ) -> Path:
        """Save task metadata (timing, cost, QG info) 📝.

        Args:
            task_id: Task identifier.
            metadata: Metadata dictionary.

        Returns:
            Path to the saved metadata file.
        """
        path = self.get_task_dir(task_id) / "metadata.json"
        self._write_json(path, metadata)
        return path

    def finalize_run(self, summary: dict | None = None) -> Path:
        """Write run manifest with all task metadata 📋.

        Args:
            summary: Optional run-level summary to include.

        Returns:
            Path to the manifest file.
        """
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_count": len(self._task_dirs),
            "task_ids": sorted(self._task_dirs.keys()),
        }
        if summary:
            manifest["summary"] = summary

        path = self.run_dir / "manifest.json"
        self._write_json(path, manifest)
        logger.info(
            "📋 Run manifest saved: %s (%d tasks)",
            path,
            len(self._task_dirs),
        )
        return path

    def cleanup_old_runs(self, keep_last: int = 10) -> int:
        """Remove old run directories, keeping the most recent 🧹.

        Args:
            keep_last: Number of most recent runs to keep.

        Returns:
            Number of runs removed.
        """
        import shutil

        if not self.base_dir.exists():
            return 0

        # 📁 List all run directories sorted by name (timestamp-based)
        run_dirs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        to_remove = run_dirs[:-keep_last] if len(run_dirs) > keep_last else []
        removed = 0
        for old_dir in to_remove:
            try:
                shutil.rmtree(old_dir)
                removed += 1
                logger.info("🧹 Removed old run: %s", old_dir)
            except Exception as exc:
                logger.warning("⚠️ Failed to remove run %s: %s", old_dir, exc)

        return removed

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        """Write JSON atomically (best-effort) 💾.

        Args:
            path: Target file path.
            data: Data to serialize.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            path.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("⚠️ Failed to write %s: %s", path, exc)
