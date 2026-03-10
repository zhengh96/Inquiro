#!/usr/bin/env python3
"""CLI wrapper for EvidenceCondenser — condense evidence to prevent token overflow 🗜️.

Reads a JSON evidence list from file or stdin, runs EvidenceCondenser, and writes
the condensed result to stdout as JSON.

Usage:
    python condense.py --input /tmp/evidence.json --checklist /tmp/checklist.json
    cat evidence.json | python condense.py --checklist /tmp/checklist.json

Exit codes:
    0 — Success
    1 — Condensation error
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
            "Condense an evidence JSON list using multi-signal scoring. "
            "Reads evidence from --input or stdin. Writes condensed evidence to stdout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python condense.py --input /tmp/evidence.json --checklist /tmp/checklist.json
  cat evidence.json | python condense.py --tier1-target 120
  python condense.py --input ev.json --force-tier 1 --quiet
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Path to JSON file containing evidence list. Reads from stdin if omitted.",
    )
    parser.add_argument(
        "--checklist",
        "-c",
        metavar="FILE",
        help=(
            "Path to JSON file containing checklist items (list of strings). "
            "Used for keyword relevance scoring. Defaults to empty list."
        ),
    )
    parser.add_argument(
        "--force-tier",
        metavar="N",
        type=int,
        choices=[0, 1, 2],
        help="Force a specific tier (0/1/2), ignoring count-based selection.",
    )
    parser.add_argument(
        "--tier1-target",
        metavar="N",
        type=int,
        default=160,
        help="Target count for Tier 1 selection (default: 160).",
    )
    parser.add_argument(
        "--tier2-target",
        metavar="N",
        type=int,
        default=150,
        help="Target count for Tier 2 selection (default: 150).",
    )
    parser.add_argument(
        "--saturation-cap",
        metavar="N",
        type=int,
        default=20,
        help="Max evidence items per MCP source (default: 20).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress INFO logging.",
    )
    return parser.parse_args()


def _load_json(source: str | None) -> list:
    """Load JSON from file path or stdin 📂.

    Args:
        source: File path string, or None to read from stdin.

    Returns:
        Parsed JSON object (expected to be a list).

    Raises:
        SystemExit: On file-not-found or JSON parse error.
    """
    try:
        if source:
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = sys.stdin.read()
        return json.loads(text)
    except FileNotFoundError:
        logger.error("Input file not found: %s", source)
        sys.exit(2)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON input: %s", exc)
        sys.exit(2)


def main() -> None:
    """Entry point 🚀."""
    args = _parse_args()

    if args.quiet:
        logging.disable(logging.INFO)

    # --- Load evidence ---
    raw_evidence = _load_json(args.input)
    if not isinstance(raw_evidence, list):
        logger.error("Evidence input must be a JSON array.")
        sys.exit(2)

    # --- Load checklist ---
    checklist_items: list[str] = []
    if args.checklist:
        raw_checklist = _load_json(args.checklist)
        if isinstance(raw_checklist, list):
            checklist_items = [str(item) for item in raw_checklist]
        else:
            logger.warning("Checklist is not a list; using empty checklist.")

    # --- Import project modules ---
    try:
        # Add project root to path if running outside package context
        project_root = Path(__file__).resolve().parents[4]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from inquiro.core.evidence_condenser import CondenserConfig, EvidenceCondenser
        from inquiro.core.types import Evidence
    except ImportError as exc:
        logger.error(
            "Cannot import project modules. "
            "Run from project root or activate the virtual environment. "
            "Error: %s",
            exc,
        )
        sys.exit(1)

    # --- Build config ---
    config = CondenserConfig(
        tier1_target=args.tier1_target,
        tier2_target=args.tier2_target,
        source_saturation_cap=args.saturation_cap,
    )

    # --- Convert dicts to Evidence objects ---
    try:
        evidence_objects: list[Evidence] = [
            Evidence.model_validate(ev) if isinstance(ev, dict) else ev
            for ev in raw_evidence
        ]
    except Exception as exc:
        logger.error("Failed to parse evidence items: %s", exc)
        sys.exit(2)

    # --- Run condenser ---
    condenser = EvidenceCondenser(config)

    if args.force_tier == 0:
        result = condenser._tier0(evidence_objects)
    elif args.force_tier == 1:
        result = condenser._tier1(evidence_objects, checklist_items)
    elif args.force_tier == 2:
        result = condenser._tier2(evidence_objects, checklist_items)
    else:
        result = condenser.condense(evidence_objects, checklist_items)

    # --- Emit output ---
    output = {
        "tier": result.meta.tier,
        "original_count": result.meta.original_count,
        "condensed_count": result.meta.condensed_count,
        "transparency_footer": result.meta.transparency_footer,
        "group_summaries": [gs.model_dump() for gs in result.meta.group_summaries],
        "evidence": [ev.model_dump() for ev in result.evidence],
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    logger.info(
        "🗜️ Condensed %d → %d items (Tier %d)",
        result.meta.original_count,
        result.meta.condensed_count,
        result.meta.tier,
    )


if __name__ == "__main__":
    main()
