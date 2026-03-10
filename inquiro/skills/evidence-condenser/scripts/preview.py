#!/usr/bin/env python3
"""Preview what EvidenceCondenser would keep or drop — without modifying state 🔍.

Reads evidence from file or stdin, runs condensation in dry-run mode, and
prints a human-readable summary table showing:
- Which tier would be applied
- Score breakdown for kept vs. dropped items
- Tag distribution before and after

Usage:
    python preview.py --input /tmp/evidence.json --checklist /tmp/checklist.json
    cat evidence.json | python preview.py

Exit codes:
    0 — Success
    2 — Input error (invalid JSON, missing file, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments 📋."""
    parser = argparse.ArgumentParser(
        description=(
            "Preview EvidenceCondenser output: show what would be kept vs. dropped. "
            "No files are modified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preview.py --input /tmp/evidence.json
  python preview.py --input /tmp/evidence.json --checklist /tmp/checklist.json
  python preview.py --input /tmp/evidence.json --top-n 20
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Path to JSON evidence list. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--checklist",
        "-c",
        metavar="FILE",
        help="Path to JSON checklist items list (for keyword scoring).",
    )
    parser.add_argument(
        "--top-n",
        metavar="N",
        type=int,
        default=10,
        help="Show top-N and bottom-N items by score (default: 10).",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output machine-readable JSON instead of human-readable text.",
    )
    return parser.parse_args()


def _load_json(source: str | None) -> list:
    """Load JSON from file path or stdin 📂."""
    try:
        if source:
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = sys.stdin.read()
        return json.loads(text)
    except FileNotFoundError:
        logger.error("File not found: %s", source)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON: %s", exc)
        sys.exit(2)


def _format_table(rows: list[dict], columns: list[str]) -> str:
    """Format a list of dicts as a plain-text table 📊.

    Args:
        rows: Data rows.
        columns: Column keys to include.

    Returns:
        Formatted table string.
    """
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    sep = "  ".join("-" * widths[col] for col in columns)
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    lines = [header, sep]
    for row in rows:
        lines.append(
            "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        )
    return "\n".join(lines)


def main() -> None:
    """Entry point 🚀."""
    args = _parse_args()

    raw_evidence = _load_json(args.input)
    if not isinstance(raw_evidence, list):
        logger.error("Evidence input must be a JSON array.")
        sys.exit(2)

    checklist_items: list[str] = []
    if args.checklist:
        raw_checklist = _load_json(args.checklist)
        if isinstance(raw_checklist, list):
            checklist_items = [str(item) for item in raw_checklist]

    # --- Import project modules ---
    try:
        project_root = Path(__file__).resolve().parents[4]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from inquiro.core.evidence_condenser import (
            CondenserConfig,
            EvidenceCondenser,
            _compute_score,
            _extract_keywords,
        )
        from inquiro.core.types import Evidence
    except ImportError as exc:
        logger.error("Cannot import project modules: %s", exc)
        sys.exit(1)

    # --- Convert to Evidence objects ---
    try:
        evidence_objects: list[Evidence] = [
            Evidence.model_validate(ev) if isinstance(ev, dict) else ev
            for ev in raw_evidence
        ]
    except Exception as exc:
        logger.error("Failed to parse evidence items: %s", exc)
        sys.exit(2)

    config = CondenserConfig()
    condenser = EvidenceCondenser(config)

    # Determine which tier would be applied
    n = len(evidence_objects)
    if n <= config.tier1_threshold:
        target = n
    elif n <= config.tier2_threshold:
        target = config.tier1_target
    else:
        target = config.tier2_target

    # Run condensation
    result = condenser.condense(evidence_objects, checklist_items)

    # Compute per-item scores for preview
    checklist_keywords: set[str] = set()
    for item in checklist_items:
        checklist_keywords |= _extract_keywords(item)
    max_round = max((ev.round_number or 1 for ev in evidence_objects), default=1)

    scored_items = sorted(
        [
            {
                "id": ev.evidence_id or ev.url or ev.summary[:40],
                "score": round(
                    _compute_score(ev, checklist_keywords, max_round, config), 4
                ),
                "tag": ev.evidence_tag or "other",
                "source": (ev.source or "")[:20],
                "quality_label": ev.quality_label or "",
            }
            for ev in evidence_objects
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    kept_ids = {id(ev) for ev in result.evidence}
    for item in scored_items:
        item["status"] = "KEPT" if id(item) not in kept_ids else "KEPT"

    if args.json_output:
        output = {
            "tier_applied": result.meta.tier,
            "original_count": n,
            "would_keep": result.meta.condensed_count,
            "would_drop": n - result.meta.condensed_count,
            "target": target,
            "transparency_footer": result.meta.transparency_footer,
            "top_scored": scored_items[: args.top_n],
            "bottom_scored": scored_items[-args.top_n :] if n > args.top_n else [],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print("  Evidence Condenser Preview 🔍")
        print(f"{'=' * 60}")
        print(f"  Input:  {n} evidence items")
        tier_thresholds = "threshold: Tier0≤150, Tier1≤400, Tier2>400"
        print(f"  Tier:   {result.meta.tier} ({tier_thresholds})")
        print(f"  Target: {target} items")
        dropped = n - result.meta.condensed_count
        print(f"  Result: {result.meta.condensed_count} kept, {dropped} de-prioritised")
        print(f"\n  {result.meta.transparency_footer}")

        print(f"\n--- Top {args.top_n} items (KEPT) ---")
        columns = ["id", "score", "tag", "source", "quality_label"]
        print(_format_table(scored_items[: args.top_n], columns))

        if n > args.top_n:
            print(f"\n--- Bottom {args.top_n} items (likely DROPPED) ---")
            print(_format_table(scored_items[-args.top_n :], columns))

        # Tag distribution
        from collections import Counter
        tag_dist = Counter(item["tag"] for item in scored_items)
        print("\n--- Tag distribution ---")
        for tag, count in tag_dist.most_common():
            print(f"  {tag:<20} {count}")
        print()


if __name__ == "__main__":
    main()
