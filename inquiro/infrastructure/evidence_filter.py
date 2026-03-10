"""Inquiro EvidenceFilter — filter invalid evidence records 🧹.

Detects and filters out invalid evidence items such as HTTP errors,
tool execution failures, empty results, and blank summaries. Also
performs deduplication (title similarity, content prefix) and cleans
LLM response residuals from evidence text. Invalid records are
preserved with an ``_invalid`` marker for audit purposes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# 🔍 Patterns that indicate an invalid evidence summary
_INVALID_PATTERNS: list[re.Pattern[str]] = [
    # HTTP error format: "401, message='Unauthorized'"
    re.compile(r"^\d{3},\s*message="),
    # Tool execution errors
    re.compile(r"^Error executing tool\b"),
    # Generic error prefix
    re.compile(r"^Error:\s"),
    # ✨ API parameter validation errors
    re.compile(r"^Input validation error:\s"),
    # ✨ MCP tool call validation errors (e.g., "call[search_entities] ...")
    re.compile(r"^call\[.+\]\s"),
    # ✨ Pydantic / schema validation errors (e.g., "1 validation error for ...")
    re.compile(r"^\d+\s+validation\s+error", re.IGNORECASE),
    # ✨ Entity not found in database (e.g., "Entity 'X' not found in MyChem.info")
    re.compile(r"^[A-Za-z]+\s+'[^']+'\s+not found in \w+"),
    # ✨ Entity not found — Chinese locale MCP responses
    re.compile(r"^在\s+\S+\s+中未找到"),
    # ✅ Emoji status messages (e.g., "✅ AI Search Session Created")
    re.compile(r"^✅\s"),
    # 🤖 AI-generated summary status (e.g., "🤖 AI-Generated Summary")
    re.compile(r"^🤖\s"),
    # ⚠️ Unknown tool errors (e.g., "Unknown tool: 'get_target_safety'")
    re.compile(r"^Unknown tool:\s"),
    # 🔍 Search count summaries — not actual evidence
    re.compile(r"^🔍\s*Found\s+\d+\s+papers"),
    # 📚 Search count variant (e.g., "📚 Found N papers (sorted by ...)")
    re.compile(r"^📚\s*Found\s+\d+\s+papers"),
    # 🔧 Markdown bold count summaries (e.g., "Found **13** papers.")
    re.compile(r"^Found\s+\*\*\d+\*\*\s+papers", re.IGNORECASE),
    # 🔧 Bohrium / generic API errors
    re.compile(r"Bohrium API error:", re.IGNORECASE),
    re.compile(r"API error:.*(?:Unknown|code=)", re.IGNORECASE),
    # 🔧 MCP protocol errors (e.g., "MCP error -32602: Invalid arguments")
    re.compile(r"^MCP error\b"),
    # 🔧 Empty web search results
    re.compile(r"^No web results found", re.IGNORECASE),
    # 🔁 Bohrium processing status leaked as evidence (JSON fragment)
    re.compile(r'"status":\s*"processing"'),
    # 🔁 Bohrium processing status leaked as evidence (plain text)
    re.compile(r"is still processing\.\s*Wait", re.IGNORECASE),
    # 🔁 AI processing note appended by tool fallback
    re.compile(r"The AI may still be processing", re.IGNORECASE),
]

# 🗑️ Exact-match invalid summaries (normalized to stripped lowercase)
_INVALID_EXACT: set[str] = {
    '{"results": []}',
    '{"results":[]}',
    "[]",
    "{}",
}

# 🗑️ Compacted versions for pretty-printed JSON matching
_INVALID_EXACT_COMPACTED: set[str] = {
    '{"results":[]}',
    "[]",
    "{}",
}


# 🧼 LLM preamble patterns to strip from evidence text
_LLM_PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:Here is|Here are|Here's)\s.{0,80}?[:\.]?\s*\n?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:Based on|According to)\s.{0,120}?[:\.]?\s*\n?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^The following\s.{0,80}?[:\.]?\s*\n?",
        re.IGNORECASE,
    ),
]

# 🧼 LLM boilerplate patterns to strip from evidence text tail
_LLM_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\s*(?:I hope this helps|Hope this helps)[.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:Let me know if)[^.]*[.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:Feel free to)[^.]*[.!]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:If you (?:need|have|want) (?:any |more )?)[^.]*[.!]?\s*$",
        re.IGNORECASE,
    ),
]

# 🧼 LLM citation bracket pattern (e.g., 【1†source】) but preserve [E1] refs
_LLM_CITATION_BRACKET = re.compile(r"【[^】]*†[^】]*】")
_GENERIC_SOURCE_BRACKET = re.compile(r"\[source\]", re.IGNORECASE)

# 🧼 Markdown formatting patterns
_MARKDOWN_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MARKDOWN_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_MARKDOWN_HEADER_LINE = re.compile(r"^#{1,4}\s+", re.MULTILINE)

# Dedup threshold for Jaccard similarity on titles
_TITLE_DEDUP_THRESHOLD = 0.8

# Content prefix length for prefix-based dedup
_CONTENT_PREFIX_LEN = 100


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings tokenized by whitespace 📏.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Jaccard similarity coefficient in [0.0, 1.0].
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


@dataclass
class FilteredEvidence:
    """Result of evidence filtering 📊.

    Attributes:
        valid: Evidence records that passed all validity checks.
        invalid: Evidence records that failed checks (marked ``_invalid=True``).
        error_rate: Fraction of invalid records over total.
    """

    valid: list[dict[str, Any]] = field(default_factory=list)
    invalid: list[dict[str, Any]] = field(default_factory=list)
    error_rate: float = 0.0


class EvidenceFilter:
    """Filter invalid evidence records from agent output 🧹.

    Detects common invalid patterns:
    - Empty result sets: ``{"results": []}``, ``[]``, ``{}``
    - HTTP errors: ``401, message='Unauthorized'``
    - Tool errors: ``Error executing tool ...``, ``Error: ...``
    - Blank or empty summaries

    Invalid records are preserved in ``FilteredEvidence.invalid`` with
    an ``_invalid=True`` marker for audit trail.

    Example::

        filtered = EvidenceFilter.filter(agent_evidence_records)
        print(f"Valid: {len(filtered.valid)}, Invalid: {len(filtered.invalid)}")
    """

    @staticmethod
    def _clean_evidence_text(text: str) -> str:
        """Clean LLM response residuals from evidence text 🧼.

        Strips preamble phrases, trailing boilerplate, markdown
        formatting, and LLM-style citation brackets while preserving
        evidence reference tags like ``[E1]``.

        Args:
            text: Raw evidence text (summary or content).

        Returns:
            Cleaned text with residuals removed.
        """
        if not text:
            return text

        cleaned = text.strip()

        # Strip leading LLM preamble patterns
        for pattern in _LLM_PREAMBLE_PATTERNS:
            cleaned = pattern.sub("", cleaned, count=1).lstrip()

        # Strip trailing LLM boilerplate patterns
        for pattern in _LLM_BOILERPLATE_PATTERNS:
            cleaned = pattern.sub("", cleaned).rstrip()

        # Remove LLM citation brackets (【1†source】) but keep [E1] refs
        cleaned = _LLM_CITATION_BRACKET.sub("", cleaned)
        cleaned = _GENERIC_SOURCE_BRACKET.sub("", cleaned)

        # Remove markdown bold/italic formatting
        cleaned = _MARKDOWN_BOLD.sub(r"\1", cleaned)
        cleaned = _MARKDOWN_ITALIC.sub(r"\1", cleaned)

        # Remove excessive markdown headers at line starts
        cleaned = _MARKDOWN_HEADER_LINE.sub("", cleaned)

        # Normalize excessive whitespace/newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        return cleaned.strip()

    @classmethod
    def _dedup_by_title(
        cls,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate evidence based on title Jaccard similarity 🔗.

        When two records have title similarity > threshold, the one with
        more content (longer summary) is kept.

        Args:
            records: List of valid evidence record dicts.

        Returns:
            Deduplicated list of records.
        """
        if len(records) <= 1:
            return records

        keep: list[dict[str, Any]] = []
        for record in records:
            title = (record.get("title") or "").strip()
            if not title:
                keep.append(record)
                continue

            is_dup = False
            for i, kept in enumerate(keep):
                kept_title = (kept.get("title") or "").strip()
                if not kept_title:
                    continue
                sim = _jaccard_similarity(title, kept_title)
                if sim > _TITLE_DEDUP_THRESHOLD:
                    is_dup = True
                    # Keep the one with more content
                    rec_len = len(record.get("summary") or "")
                    kept_len = len(kept.get("summary") or "")
                    if rec_len > kept_len:
                        keep[i] = record
                        logger.debug(
                            "🔗 Title dedup: replaced shorter duplicate "
                            "(sim=%.2f) title=%r",
                            sim,
                            kept_title[:60],
                        )
                    else:
                        logger.debug(
                            "🔗 Title dedup: dropped duplicate "
                            "(sim=%.2f) title=%r",
                            sim,
                            title[:60],
                        )
                    break
            if not is_dup:
                keep.append(record)

        return keep

    @classmethod
    def _dedup_by_content_prefix(
        cls,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicates sharing the same content prefix 📝.

        If multiple evidence items start with the same first N characters
        in their summary, they are likely duplicates — keep the longest.

        Args:
            records: List of evidence record dicts.

        Returns:
            Deduplicated list of records.
        """
        if len(records) <= 1:
            return records

        # Group by content prefix
        prefix_groups: dict[str, list[dict[str, Any]]] = {}
        no_prefix: list[dict[str, Any]] = []

        for record in records:
            summary = (record.get("summary") or "").strip()
            if len(summary) < _CONTENT_PREFIX_LEN:
                no_prefix.append(record)
                continue

            prefix = summary[:_CONTENT_PREFIX_LEN].lower()
            prefix_groups.setdefault(prefix, []).append(record)

        result = list(no_prefix)
        for prefix, group in prefix_groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Keep the longest summary
                best = max(group, key=lambda r: len(r.get("summary") or ""))
                result.append(best)
                logger.debug(
                    "📝 Prefix dedup: kept 1 of %d records with prefix=%r",
                    len(group),
                    prefix[:40],
                )

        return result

    @staticmethod
    def _is_think_tool_response(summary: str) -> bool:
        """Detect ThinkTool JSON responses misrecorded as evidence 🧠.

        Args:
            summary: Raw evidence summary text.

        Returns:
            True if the summary is a ThinkTool response, not real evidence.
        """
        return '"domain": "thinking"' in summary and '"thoughtNumber"' in summary

    @staticmethod
    def is_valid_evidence(summary: str | None) -> bool:
        """Check whether an evidence summary represents valid content 🔍.

        Args:
            summary: Raw evidence summary text.

        Returns:
            True if the summary contains meaningful evidence content.
        """
        if not summary or not summary.strip():
            return False

        stripped = summary.strip()

        # 🗑️ Check exact-match invalid content (collapse whitespace for
        # pretty-printed JSON variants like "{\n  \"results\": []\n}")
        compacted = re.sub(r"\s+", "", stripped)
        if compacted in _INVALID_EXACT_COMPACTED:
            return False
        if stripped.lower() in _INVALID_EXACT or stripped in _INVALID_EXACT:
            return False

        # 🧠 ThinkTool response detection
        if EvidenceFilter._is_think_tool_response(stripped):
            return False

        # 🔍 Check regex patterns against first line
        first_line = stripped.split("\n")[0].strip()
        for pattern in _INVALID_PATTERNS:
            if pattern.search(first_line):
                return False

        return True

    @classmethod
    def filter(
        cls,
        records: list[dict[str, Any]],
    ) -> FilteredEvidence:
        """Filter evidence records into valid and invalid sets 🧹.

        Pipeline: validity check → text cleaning → title dedup →
        content prefix dedup → hallucination flagging.

        Args:
            records: List of evidence record dicts with at least a
                ``summary`` key.

        Returns:
            FilteredEvidence with valid/invalid splits and error_rate.
        """
        valid: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []

        for record in records:
            summary = record.get("summary", "")
            if cls.is_valid_evidence(summary):
                valid.append(record)
            else:
                # 🏷️ Mark as invalid for audit trail
                marked = dict(record)
                marked["_invalid"] = True
                invalid.append(marked)

        # 🧼 Clean LLM response residuals from valid evidence
        for record in valid:
            if record.get("summary"):
                record["summary"] = cls._clean_evidence_text(
                    record["summary"],
                )
            if record.get("content"):
                record["content"] = cls._clean_evidence_text(
                    record["content"],
                )

        # 🔗 Deduplicate by title similarity
        valid = cls._dedup_by_title(valid)

        # 📝 Deduplicate by content prefix
        valid = cls._dedup_by_content_prefix(valid)

        # 🚩 Flag potentially hallucinated entities
        # TODO: Implement full named-entity uniqueness check across all
        # evidence for more robust hallucination detection.
        cls._flag_suspicious_synergizes(valid)

        total = len(valid) + len(invalid)
        error_rate = len(invalid) / total if total > 0 else 0.0

        return FilteredEvidence(
            valid=valid,
            invalid=invalid,
            error_rate=error_rate,
        )

    @classmethod
    def _flag_suspicious_synergizes(
        cls,
        records: list[dict[str, Any]],
    ) -> None:
        """Flag evidence containing 'synergizes' with unique company names 🚩.

        If evidence text contains "synergizes" and mentions a capitalized
        entity (likely a company name) not found in any other evidence
        record, it may be hallucinated. Sets ``_suspicious_hallucination``
        flag on the record.

        Args:
            records: List of valid evidence record dicts (modified in place).
        """
        if len(records) <= 1:
            return

        # Collect all capitalized multi-word entities across records
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        all_entities: dict[str, int] = {}
        record_entities: list[set[str]] = []

        for record in records:
            text = (record.get("summary") or "") + " " + (
                record.get("content") or ""
            )
            entities = set(entity_pattern.findall(text))
            record_entities.append(entities)
            for entity in entities:
                all_entities[entity] = all_entities.get(entity, 0) + 1

        for i, record in enumerate(records):
            text = (record.get("summary") or "") + " " + (
                record.get("content") or ""
            )
            if "synergizes" not in text.lower():
                continue
            # Check if any entity in this record is unique
            for entity in record_entities[i]:
                if all_entities.get(entity, 0) == 1:
                    record["_suspicious_hallucination"] = True
                    logger.warning(
                        "🚩 Suspicious hallucination: 'synergizes' with "
                        "unique entity %r in evidence %s",
                        entity,
                        record.get("evidence_id", "?"),
                    )
                    break
