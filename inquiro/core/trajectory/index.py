"""Trajectory SQLite index for analytics and optimization 📊.

Provides persistent indexing of Discovery trajectory JSONL files
into a SQLite database for efficient querying, trend analysis,
and data-driven optimization (Phase 3 O1).

Design principles:
    - Lightweight: indexes DiscoverySummary fields (< 2KB per record)
    - Incremental: re-indexing skips already-indexed files
    - Read-heavy: optimized for analytical queries
    - Self-contained: single .db file, no external dependencies
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

_UTC = ZoneInfo("UTC")

logger = logging.getLogger(__name__)

# ============================================================================
# 📊 Query result models
# ============================================================================


class TrajectoryRecord(BaseModel):
    """Summary record from the trajectory index 📋.

    Attributes:
        trajectory_id: Unique trajectory identifier.
        task_id: Associated task identifier.
        jsonl_path: Path to the source JSONL file.
        status: Run status (completed/failed/cancelled).
        total_rounds: Number of rounds executed.
        final_coverage: Final checklist coverage ratio.
        total_cost_usd: Total cost across all phases.
        total_evidence: Total cleaned evidence items.
        total_claims: Total consensus claims.
        total_duration_seconds: Total wall-clock time.
        termination_reason: Why the loop stopped.
        evidence_yield_rate: Evidence yield ratio.
        cost_normalized_quality: Quality metric (coverage / cost).
        created_at: When the trajectory was created.
        completed_at: When the trajectory was finalized.
    """

    trajectory_id: str = Field(description="Unique trajectory identifier")
    task_id: str = Field(description="Associated task identifier")
    jsonl_path: str = Field(description="Source JSONL file path")
    status: str = Field(default="unknown", description="Run status")
    total_rounds: int = Field(default=0, description="Rounds executed")
    final_coverage: float = Field(default=0.0, description="Final coverage")
    total_cost_usd: float = Field(default=0.0, description="Total cost")
    total_evidence: int = Field(default=0, description="Total evidence")
    total_claims: int = Field(default=0, description="Total claims")
    total_duration_seconds: float = Field(default=0.0, description="Duration")
    termination_reason: str = Field(default="", description="Termination reason")
    evidence_yield_rate: float = Field(default=0.0, description="Yield rate")
    cost_normalized_quality: float = Field(default=0.0, description="Quality metric")
    created_at: str = Field(default="", description="Creation timestamp")
    completed_at: str = Field(default="", description="Completion timestamp")


class RoundRecord(BaseModel):
    """Per-round detail from the trajectory index 📊.

    Attributes:
        trajectory_id: Parent trajectory identifier.
        round_number: Round index (1-based).
        search_queries: Number of search queries executed.
        raw_evidence: Raw evidence items collected.
        cleaned_evidence: Evidence after cleaning.
        dedup_removed: Items removed by dedup.
        noise_removed: Items removed by noise filter.
        analysis_claims: Claims from analysis consensus.
        coverage_ratio: Gap analysis coverage after this round.
        round_cost_usd: Total cost for this round.
        round_duration_seconds: Total round duration.
    """

    trajectory_id: str = Field(description="Parent trajectory")
    round_number: int = Field(description="Round index")
    search_queries: int = Field(default=0, description="Queries executed")
    raw_evidence: int = Field(default=0, description="Raw evidence count")
    cleaned_evidence: int = Field(default=0, description="Cleaned evidence")
    dedup_removed: int = Field(default=0, description="Dedup removed")
    noise_removed: int = Field(default=0, description="Noise removed")
    analysis_claims: int = Field(default=0, description="Analysis claims")
    coverage_ratio: float = Field(default=0.0, description="Coverage")
    round_cost_usd: float = Field(default=0.0, description="Round cost")
    round_duration_seconds: float = Field(default=0.0, description="Duration")


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for a trajectory 💰.

    Attributes:
        trajectory_id: Trajectory identifier.
        total_cost_usd: Total cost.
        search_cost_usd: Total search phase cost.
        analysis_cost_usd: Total analysis phase cost.
        gap_cost_usd: Total gap analysis cost.
        synthesis_cost_usd: Synthesis phase cost.
        focus_prompt_cost_usd: Focus prompt generation cost.
        per_round: Cost per round.
    """

    trajectory_id: str = Field(description="Trajectory identifier")
    total_cost_usd: float = Field(default=0.0, description="Total cost")
    search_cost_usd: float = Field(default=0.0, description="Search cost")
    analysis_cost_usd: float = Field(default=0.0, description="Analysis cost")
    gap_cost_usd: float = Field(default=0.0, description="Gap analysis cost")
    synthesis_cost_usd: float = Field(default=0.0, description="Synthesis cost")
    focus_prompt_cost_usd: float = Field(default=0.0, description="Focus prompt cost")
    per_round: list[dict[str, Any]] = Field(
        default_factory=list, description="Cost per round"
    )


class TrendPoint(BaseModel):
    """Single data point in a trend series 📈.

    Attributes:
        trajectory_id: Trajectory identifier.
        task_id: Task identifier.
        created_at: Timestamp.
        value: Metric value.
    """

    trajectory_id: str = Field(description="Trajectory identifier")
    task_id: str = Field(description="Task identifier")
    created_at: str = Field(description="Timestamp")
    value: float = Field(description="Metric value")


# ============================================================================
# 🗄️ SQLite schema
# ============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    jsonl_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    status TEXT DEFAULT 'unknown',
    total_rounds INTEGER DEFAULT 0,
    final_coverage REAL DEFAULT 0.0,
    total_cost_usd REAL DEFAULT 0.0,
    synthesis_cost_usd REAL DEFAULT 0.0,
    total_evidence INTEGER DEFAULT 0,
    total_claims INTEGER DEFAULT 0,
    total_duration_seconds REAL DEFAULT 0.0,
    termination_reason TEXT DEFAULT '',
    evidence_yield_rate REAL DEFAULT 0.0,
    cost_normalized_quality REAL DEFAULT 0.0,
    config_snapshot TEXT DEFAULT '{}',
    created_at TEXT DEFAULT '',
    completed_at TEXT DEFAULT '',
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    search_queries INTEGER DEFAULT 0,
    raw_evidence INTEGER DEFAULT 0,
    cleaned_evidence INTEGER DEFAULT 0,
    dedup_removed INTEGER DEFAULT 0,
    noise_removed INTEGER DEFAULT 0,
    analysis_claims INTEGER DEFAULT 0,
    coverage_ratio REAL DEFAULT 0.0,
    convergence_reason TEXT DEFAULT '',
    focus_prompt_text TEXT DEFAULT '',
    round_cost_usd REAL DEFAULT 0.0,
    round_duration_seconds REAL DEFAULT 0.0,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id),
    UNIQUE(trajectory_id, round_number)
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    query_text TEXT NOT NULL,
    mcp_tool TEXT DEFAULT '',
    result_count INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data TEXT DEFAULT '{}',
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
);

CREATE INDEX IF NOT EXISTS idx_trajectories_task_id
    ON trajectories(task_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_status
    ON trajectories(status);
CREATE INDEX IF NOT EXISTS idx_trajectories_created_at
    ON trajectories(created_at);
CREATE INDEX IF NOT EXISTS idx_trajectories_final_coverage
    ON trajectories(final_coverage);
CREATE INDEX IF NOT EXISTS idx_trajectories_total_cost_usd
    ON trajectories(total_cost_usd);
CREATE INDEX IF NOT EXISTS idx_rounds_trajectory_id
    ON rounds(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_queries_trajectory_id
    ON queries(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_events_trajectory_id
    ON events(trajectory_id);
"""


# ============================================================================
# 🗄️ TrajectoryIndex
# ============================================================================


def _file_hash(path: str) -> str:
    """Compute MD5 hash of a file for change detection 🔑.

    Args:
        path: File path.

    Returns:
        Hex digest of the file content.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_jsonl(path: str) -> dict[str, Any]:
    """Parse a trajectory JSONL file into structured data 📄.

    Args:
        path: Path to a JSONL file.

    Returns:
        Dict with keys: meta, rounds, events, synthesis, summary, meta_final.
    """
    result: dict[str, Any] = {
        "meta": {},
        "rounds": [],
        "events": [],
        "synthesis": None,
        "summary": {},
        "meta_final": {},
    }
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("📄 Skipping malformed JSON at %s:%d", path, line_num)
                continue

            record_type = record.get("type", "")
            if record_type == "meta":
                result["meta"] = record
            elif record_type == "round":
                result["rounds"].append(record)
            elif record_type == "event":
                result["events"].append(record)
            elif record_type == "synthesis":
                result["synthesis"] = record
            elif record_type == "summary":
                result["summary"] = record
            elif record_type == "meta_final":
                result["meta_final"] = record

    return result


class TrajectoryIndex:
    """SQLite-backed index for Discovery trajectory analytics 🗄️.

    Indexes JSONL trajectory files into a SQLite database for
    efficient querying, trend analysis, and strategy optimization.

    Example::

        index = TrajectoryIndex("/path/to/index.db")
        index.index_from_directory("/path/to/trajectories")
        records = index.list_trajectories(status="completed")
        trend = index.get_trend("coverage", limit=20)

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the trajectory index 🔧.

        Creates the database file and schema if they don't exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # 🔄 Migration: add synthesis_cost_usd for existing databases
            try:
                conn.execute(
                    "ALTER TABLE trajectories "
                    "ADD COLUMN synthesis_cost_usd REAL DEFAULT 0.0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists — no-op

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections 🔌.

        Yields:
            SQLite connection with row_factory set.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ========================================================================
    # 📥 Indexing
    # ========================================================================

    def index_trajectory(self, jsonl_path: str) -> str | None:
        """Index a single JSONL trajectory file 📥.

        Parses the file, extracts summary + round data, and inserts
        into the SQLite database.  Skips if file already indexed with
        same hash (idempotent).

        Args:
            jsonl_path: Path to the JSONL trajectory file.

        Returns:
            The trajectory_id if indexed, None if skipped.
        """
        abs_path = os.path.abspath(jsonl_path)
        if not os.path.isfile(abs_path):
            logger.warning("📥 File not found: %s", abs_path)
            return None

        fhash = _file_hash(abs_path)

        with self._connect() as conn:
            # ✅ Check if already indexed with same hash
            existing = conn.execute(
                "SELECT trajectory_id, file_hash FROM trajectories "
                "WHERE jsonl_path = ?",
                (abs_path,),
            ).fetchone()
            if existing and existing["file_hash"] == fhash:
                return None  # 🔁 Already up-to-date
            if existing:
                # 🔄 File changed, re-index: delete old records
                tid = existing["trajectory_id"]
                conn.execute("DELETE FROM queries WHERE trajectory_id = ?", (tid,))
                conn.execute("DELETE FROM events WHERE trajectory_id = ?", (tid,))
                conn.execute("DELETE FROM rounds WHERE trajectory_id = ?", (tid,))
                conn.execute("DELETE FROM trajectories WHERE trajectory_id = ?", (tid,))

            # 📄 Parse JSONL
            data = _parse_jsonl(abs_path)
            meta = data["meta"]
            summary = data["summary"]
            meta_final = data["meta_final"]

            trajectory_id = meta.get("trajectory_id", "")
            if not trajectory_id:
                logger.warning("📥 No trajectory_id in meta record: %s", abs_path)
                return None

            task_id = meta.get("task_id", "")
            now = datetime.now(tz=_UTC).isoformat()

            # 💰 Extract synthesis cost from the synthesis record (if present)
            synthesis_record = data.get("synthesis") or {}
            synthesis_cost_usd = float(synthesis_record.get("cost_usd", 0.0) or 0.0)

            # 📊 Insert trajectory summary
            conn.execute(
                """INSERT INTO trajectories (
                    trajectory_id, task_id, jsonl_path, file_hash,
                    status, total_rounds, final_coverage, total_cost_usd,
                    synthesis_cost_usd, total_evidence, total_claims,
                    total_duration_seconds, termination_reason,
                    evidence_yield_rate, cost_normalized_quality,
                    config_snapshot, created_at, completed_at, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory_id,
                    task_id,
                    abs_path,
                    fhash,
                    meta_final.get("status", "unknown"),
                    summary.get("total_rounds", 0),
                    summary.get("final_coverage", 0.0),
                    summary.get("total_cost_usd", 0.0),
                    synthesis_cost_usd,
                    summary.get("total_evidence", 0),
                    summary.get("total_claims", 0),
                    summary.get("total_duration_seconds", 0.0),
                    summary.get(
                        "termination_reason",
                        meta_final.get("termination_reason", ""),
                    ),
                    summary.get("evidence_yield_rate", 0.0),
                    summary.get("cost_normalized_quality", 0.0),
                    json.dumps(meta.get("config_snapshot", {}), ensure_ascii=False),
                    meta.get("timestamp", ""),
                    meta_final.get("timestamp", ""),
                    now,
                ),
            )

            # 📊 Insert round records
            for rnd in data["rounds"]:
                search = rnd.get("search_phase", {})
                cleaning = rnd.get("cleaning_phase", {})
                analysis = rnd.get("analysis_phase", {})
                gap = rnd.get("gap_phase", {})

                round_number = rnd.get("round_number", 0)
                queries_list = search.get("queries", [])

                conn.execute(
                    """INSERT INTO rounds (
                        trajectory_id, round_number, search_queries,
                        raw_evidence, cleaned_evidence, dedup_removed,
                        noise_removed, analysis_claims, coverage_ratio,
                        convergence_reason, focus_prompt_text,
                        round_cost_usd, round_duration_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trajectory_id,
                        round_number,
                        len(queries_list),
                        search.get("total_raw_evidence", 0),
                        cleaning.get("output_count", 0),
                        cleaning.get("dedup_removed", 0),
                        cleaning.get("noise_removed", 0),
                        analysis.get("consensus", {}).get("total_claims", 0),
                        gap.get("coverage_ratio", 0.0),
                        gap.get("convergence_reason", ""),
                        (gap.get("focus_prompt") or {}).get("prompt_text", ""),
                        rnd.get("round_cost_usd", 0.0),
                        rnd.get("round_duration_seconds", 0.0),
                    ),
                )

                # 📊 Insert individual query records
                for query in queries_list:
                    conn.execute(
                        """INSERT INTO queries (
                            trajectory_id, round_number, query_text,
                            mcp_tool, result_count, cost_usd
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trajectory_id,
                            round_number,
                            query.get("query_text", ""),
                            query.get("mcp_tool", ""),
                            query.get("result_count", 0),
                            query.get("cost_usd", 0.0),
                        ),
                    )

            # ⏱️ Insert events
            for evt in data["events"]:
                conn.execute(
                    """INSERT INTO events (
                        trajectory_id, event_type, timestamp, data
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        trajectory_id,
                        evt.get("event_type", ""),
                        evt.get("timestamp", ""),
                        json.dumps(evt.get("data", {}), ensure_ascii=False),
                    ),
                )

        logger.info("📥 Indexed trajectory %s from %s", trajectory_id, abs_path)
        return trajectory_id

    def index_from_directory(self, dir_path: str) -> list[str]:
        """Batch-index all JSONL trajectory files in a directory 📂.

        Scans for ``discovery_*.jsonl`` files and indexes each one.
        Already-indexed files with unchanged content are skipped.

        Args:
            dir_path: Directory containing JSONL trajectory files.

        Returns:
            List of newly indexed trajectory_ids.
        """
        indexed: list[str] = []
        dirp = Path(dir_path)
        if not dirp.is_dir():
            logger.warning("📂 Directory not found: %s", dir_path)
            return indexed

        for jsonl_file in sorted(dirp.glob("discovery_*.jsonl")):
            tid = self.index_trajectory(str(jsonl_file))
            if tid:
                indexed.append(tid)

        logger.info("📂 Indexed %d new trajectories from %s", len(indexed), dir_path)
        return indexed

    # ========================================================================
    # 🔍 Querying
    # ========================================================================

    def get_summary(self, trajectory_id: str) -> TrajectoryRecord | None:
        """Get summary record for a specific trajectory 📋.

        Args:
            trajectory_id: Unique trajectory identifier.

        Returns:
            TrajectoryRecord if found, None otherwise.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def list_trajectories(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
        min_coverage: float | None = None,
        max_cost: float | None = None,
        termination_reason: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC",
    ) -> list[TrajectoryRecord]:
        """List trajectories with optional filters 📋.

        Args:
            task_id: Filter by task_id (exact match).
            status: Filter by status (exact match).
            min_coverage: Filter by minimum final_coverage.
            max_cost: Filter by maximum total_cost_usd.
            termination_reason: Filter by termination_reason.
            limit: Maximum number of results.
            offset: Result offset for pagination.
            order_by: SQL ORDER BY clause.

        Returns:
            List of matching TrajectoryRecord objects.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if task_id is not None:
            conditions.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if min_coverage is not None:
            conditions.append("final_coverage >= ?")
            params.append(min_coverage)
        if max_cost is not None:
            conditions.append("total_cost_usd <= ?")
            params.append(max_cost)
        if termination_reason is not None:
            conditions.append("termination_reason = ?")
            params.append(termination_reason)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        # ✅ Sanitize order_by to prevent injection
        allowed_columns = {
            "created_at",
            "completed_at",
            "final_coverage",
            "total_cost_usd",
            "total_rounds",
            "total_evidence",
            "evidence_yield_rate",
            "cost_normalized_quality",
            "total_duration_seconds",
        }
        allowed_dirs = {"ASC", "DESC"}
        parts = order_by.strip().split()
        if len(parts) == 2:
            col, direction = parts
            if col not in allowed_columns or direction.upper() not in allowed_dirs:
                order_by = "created_at DESC"
        elif len(parts) == 1:
            if parts[0] not in allowed_columns:
                order_by = "created_at DESC"
        else:
            order_by = "created_at DESC"

        sql = f"SELECT * FROM trajectories {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_rounds(self, trajectory_id: str) -> list[RoundRecord]:
        """Get per-round details for a trajectory 📊.

        Args:
            trajectory_id: Trajectory identifier.

        Returns:
            List of RoundRecord objects ordered by round_number.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rounds WHERE trajectory_id = ? ORDER BY round_number",
                (trajectory_id,),
            ).fetchall()
            return [
                RoundRecord(
                    trajectory_id=r["trajectory_id"],
                    round_number=r["round_number"],
                    search_queries=r["search_queries"],
                    raw_evidence=r["raw_evidence"],
                    cleaned_evidence=r["cleaned_evidence"],
                    dedup_removed=r["dedup_removed"],
                    noise_removed=r["noise_removed"],
                    analysis_claims=r["analysis_claims"],
                    coverage_ratio=r["coverage_ratio"],
                    round_cost_usd=r["round_cost_usd"],
                    round_duration_seconds=r["round_duration_seconds"],
                )
                for r in rows
            ]

    def get_cost_breakdown(self, trajectory_id: str) -> CostBreakdown | None:
        """Get detailed cost breakdown for a trajectory 💰.

        Computes per-phase costs by aggregating round and query data.

        Args:
            trajectory_id: Trajectory identifier.

        Returns:
            CostBreakdown if trajectory exists, None otherwise.
        """
        with self._connect() as conn:
            traj = conn.execute(
                "SELECT total_cost_usd, synthesis_cost_usd FROM trajectories "
                "WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if not traj:
                return None

            rounds = conn.execute(
                "SELECT * FROM rounds WHERE trajectory_id = ? ORDER BY round_number",
                (trajectory_id,),
            ).fetchall()

            # 💰 Aggregate search cost from queries
            search_cost_row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) as total "
                "FROM queries WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            search_cost = search_cost_row["total"] if search_cost_row else 0.0

            # 💰 Synthesis cost — stored during indexing from the JSONL
            # synthesis record written by runner.py after the loop completes
            synthesis_cost = float(traj["synthesis_cost_usd"] or 0.0)

            focus_cost = 0.0
            per_round: list[dict[str, Any]] = []

            for rnd in rounds:
                round_cost = rnd["round_cost_usd"]
                per_round.append(
                    {
                        "round_number": rnd["round_number"],
                        "cost_usd": round_cost,
                        "search_queries": rnd["search_queries"],
                        "cleaned_evidence": rnd["cleaned_evidence"],
                        "coverage_ratio": rnd["coverage_ratio"],
                    }
                )

            # 💰 Total cost = discovery loop cost + synthesis cost
            discovery_cost = float(traj["total_cost_usd"] or 0.0)
            total_cost = discovery_cost + synthesis_cost

            # ✅ Estimate analysis cost as discovery remainder after search
            analysis_cost = max(0.0, discovery_cost - search_cost - focus_cost)

            return CostBreakdown(
                trajectory_id=trajectory_id,
                total_cost_usd=total_cost,
                search_cost_usd=search_cost,
                analysis_cost_usd=analysis_cost,
                gap_cost_usd=0.0,
                synthesis_cost_usd=synthesis_cost,
                focus_prompt_cost_usd=focus_cost,
                per_round=per_round,
            )

    def find_similar(
        self,
        task_id: str,
        *,
        top_k: int = 5,
        min_coverage: float = 0.0,
    ) -> list[TrajectoryRecord]:
        """Find trajectories for the same or similar tasks 🔍.

        Queries by exact task_id match, ordered by quality metric.
        Useful for replaying successful strategies.

        Args:
            task_id: Task identifier to match.
            top_k: Maximum results to return.
            min_coverage: Minimum coverage filter.

        Returns:
            List of TrajectoryRecord sorted by cost_normalized_quality.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trajectories "
                "WHERE task_id = ? AND final_coverage >= ? "
                "AND status = 'completed' "
                "ORDER BY cost_normalized_quality DESC "
                "LIMIT ?",
                (task_id, min_coverage, top_k),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_trend(
        self,
        metric: str = "final_coverage",
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[TrendPoint]:
        """Get time-series trend for a metric 📈.

        Returns data points ordered by creation time (oldest first).

        Args:
            metric: Column name to track. Must be a valid numeric column.
            task_id: Optional filter by task_id.
            limit: Maximum data points.

        Returns:
            List of TrendPoint objects.
        """
        # ✅ Validate metric name
        allowed_metrics = {
            "final_coverage",
            "total_cost_usd",
            "total_evidence",
            "total_claims",
            "total_duration_seconds",
            "evidence_yield_rate",
            "cost_normalized_quality",
            "total_rounds",
        }
        if metric not in allowed_metrics:
            logger.warning("📈 Invalid metric: %s", metric)
            return []

        conditions = ["status = 'completed'"]
        params: list[Any] = []
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = "WHERE " + " AND ".join(conditions)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT trajectory_id, task_id, created_at, "
                f"{metric} as value FROM trajectories "
                f"{where} ORDER BY created_at ASC LIMIT ?",
                params,
            ).fetchall()
            return [
                TrendPoint(
                    trajectory_id=r["trajectory_id"],
                    task_id=r["task_id"],
                    created_at=r["created_at"],
                    value=r["value"],
                )
                for r in rows
            ]

    def get_statistics(
        self,
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregate statistics across trajectories 📊.

        Args:
            task_id: Optional filter by task_id.

        Returns:
            Dict with count, averages, min/max for key metrics.
        """
        conditions = []
        params: list[Any] = []
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT "
                f"COUNT(*) as count, "
                f"AVG(final_coverage) as avg_coverage, "
                f"MIN(final_coverage) as min_coverage, "
                f"MAX(final_coverage) as max_coverage, "
                f"AVG(total_cost_usd) as avg_cost, "
                f"MIN(total_cost_usd) as min_cost, "
                f"MAX(total_cost_usd) as max_cost, "
                f"AVG(total_rounds) as avg_rounds, "
                f"AVG(total_evidence) as avg_evidence, "
                f"AVG(evidence_yield_rate) as avg_yield_rate, "
                f"AVG(cost_normalized_quality) as avg_quality, "
                f"SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) "
                f"as completed_count, "
                f"SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) "
                f"as failed_count "
                f"FROM trajectories {where}",
                params,
            ).fetchone()
            if not row or row["count"] == 0:
                return {"count": 0}
            return {
                "count": row["count"],
                "completed_count": row["completed_count"],
                "failed_count": row["failed_count"],
                "coverage": {
                    "avg": round(row["avg_coverage"], 4),
                    "min": round(row["min_coverage"], 4),
                    "max": round(row["max_coverage"], 4),
                },
                "cost_usd": {
                    "avg": round(row["avg_cost"], 4),
                    "min": round(row["min_cost"], 4),
                    "max": round(row["max_cost"], 4),
                },
                "avg_rounds": round(row["avg_rounds"], 2),
                "avg_evidence": round(row["avg_evidence"], 2),
                "avg_yield_rate": round(row["avg_yield_rate"], 4),
                "avg_quality": round(row["avg_quality"], 4),
            }

    def get_query_effectiveness(
        self,
        *,
        task_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Rank query templates by effectiveness 📊.

        Groups queries by text, ranks by average result count.
        Foundation for O2 (Query Template Optimization).

        Args:
            task_id: Optional filter by task_id.
            limit: Maximum results.

        Returns:
            List of dicts with query_text, count, avg_results, total_cost.
        """
        join_clause = ""
        conditions = []
        params: list[Any] = []
        if task_id:
            join_clause = "JOIN trajectories t ON q.trajectory_id = t.trajectory_id"
            conditions.append("t.task_id = ?")
            params.append(task_id)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT q.query_text, "
                f"COUNT(*) as usage_count, "
                f"AVG(q.result_count) as avg_results, "
                f"SUM(q.cost_usd) as total_cost "
                f"FROM queries q {join_clause} {where} "
                f"GROUP BY q.query_text "
                f"ORDER BY avg_results DESC "
                f"LIMIT ?",
                params,
            ).fetchall()
            return [
                {
                    "query_text": r["query_text"],
                    "usage_count": r["usage_count"],
                    "avg_results": round(r["avg_results"], 2),
                    "total_cost": round(r["total_cost"], 4),
                }
                for r in rows
            ]

    def delete_trajectory(self, trajectory_id: str) -> bool:
        """Delete a trajectory and all related records 🗑️.

        Args:
            trajectory_id: Trajectory identifier to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
            if not existing:
                return False

            conn.execute(
                "DELETE FROM queries WHERE trajectory_id = ?",
                (trajectory_id,),
            )
            conn.execute(
                "DELETE FROM events WHERE trajectory_id = ?",
                (trajectory_id,),
            )
            conn.execute(
                "DELETE FROM rounds WHERE trajectory_id = ?",
                (trajectory_id,),
            )
            conn.execute(
                "DELETE FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            )
            return True

    def count(self) -> int:
        """Get total number of indexed trajectories 📊.

        Returns:
            Total trajectory count.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM trajectories").fetchone()
            return row["cnt"] if row else 0

    # ========================================================================
    # 🔧 Internal helpers
    # ========================================================================

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TrajectoryRecord:
        """Convert a SQLite Row to a TrajectoryRecord 🔧.

        Args:
            row: SQLite Row from trajectories table.

        Returns:
            TrajectoryRecord instance.
        """
        return TrajectoryRecord(
            trajectory_id=row["trajectory_id"],
            task_id=row["task_id"],
            jsonl_path=row["jsonl_path"],
            status=row["status"],
            total_rounds=row["total_rounds"],
            final_coverage=row["final_coverage"],
            total_cost_usd=row["total_cost_usd"],
            total_evidence=row["total_evidence"],
            total_claims=row["total_claims"],
            total_duration_seconds=row["total_duration_seconds"],
            termination_reason=row["termination_reason"],
            evidence_yield_rate=row["evidence_yield_rate"],
            cost_normalized_quality=row["cost_normalized_quality"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )
