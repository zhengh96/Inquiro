"""Evidence metadata extraction — zero-LLM field enrichment 🏷️.

Stateless utilities for extracting structured metadata from evidence
URLs and summaries.  Used by EvidencePipeline to populate the
``clinical_trial_id`` and ``doi`` fields on Evidence objects.

All extraction is regex-based with zero LLM cost.
"""

from __future__ import annotations

import re


# 📋 NCT ID pattern: "NCT" followed by 8+ digits
_NCT_PATTERN = re.compile(r"NCT\d{8,}", re.IGNORECASE)

# 📄 DOI pattern: "10.<registrant>/<suffix>"
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,;)]+")


def extract_clinical_trial_id(
    url: str | None,
    summary: str | None,
) -> str | None:
    """Extract an NCT clinical trial identifier from URL or summary 📋.

    Checks URL first (higher confidence), then falls back to summary.
    Returns the normalised (uppercased) NCT ID or None.

    Args:
        url: Evidence source URL (may be None).
        summary: Evidence summary text (may be None).

    Returns:
        Uppercased NCT ID (e.g. ``"NCT12345678"``), or None if not found.
    """
    # 🔍 Check URL first (higher confidence)
    if url:
        match = _NCT_PATTERN.search(url)
        if match:
            return match.group(0).upper()

    # 🔍 Fallback to summary
    if summary:
        match = _NCT_PATTERN.search(summary)
        if match:
            return match.group(0).upper()

    return None


def extract_doi(
    url: str | None,
    summary: str | None,
) -> str | None:
    """Extract a DOI (Digital Object Identifier) from URL or summary 📄.

    Checks URL first, then falls back to summary.  Strips trailing
    punctuation that is commonly appended in citations.

    Args:
        url: Evidence source URL (may be None).
        summary: Evidence summary text (may be None).

    Returns:
        DOI string (e.g. ``"10.1234/example.2024"``), or None if not found.
    """
    # 🔍 Check URL first
    if url:
        match = _DOI_PATTERN.search(url)
        if match:
            return _strip_trailing_punctuation(match.group(0))

    # 🔍 Fallback to summary
    if summary:
        match = _DOI_PATTERN.search(summary)
        if match:
            return _strip_trailing_punctuation(match.group(0))

    return None


def _strip_trailing_punctuation(doi: str) -> str:
    """Remove trailing punctuation commonly found in citations 🔧.

    Args:
        doi: Raw DOI string potentially ending with ``.,;)``.

    Returns:
        DOI with trailing punctuation stripped.
    """
    return doi.rstrip(".,;)")
