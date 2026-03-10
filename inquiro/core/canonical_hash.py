"""Canonical evidence hashing — unified dedup key for Evidence KB 🔑.

Provides a single deterministic hash function for evidence deduplication
across the entire codebase.  Replaces the previous inconsistent use of
MD5 (EvidencePipeline) and inline SHA-256 (SharedEvidencePool) with a
unified SHA-256 algorithm that considers both URL and summary.

Architecture position::

    EvidencePipeline._dedup()  ──→  canonical_evidence_hash()
    SharedEvidencePool._compute_hash()  ──→  canonical_evidence_hash()
    EvidenceKnowledgeBase (future)  ──→  canonical_evidence_hash()

Key design decisions:
    - SHA-256 for collision resistance (64-char hex digest).
    - Normalisation: URL lowered + stripped; summary lowered + stripped
      + truncated at 500 chars.
    - ``None`` and empty-string URL treated identically.
    - ``hash_version`` parameter for future algorithm migration.
"""

from __future__ import annotations

import hashlib


def canonical_evidence_hash(
    url: str | None,
    summary: str,
    *,
    hash_version: int = 1,
) -> str:
    """Compute a deterministic content hash for evidence deduplication 🔑.

    Normalises both URL and summary text before hashing, ensuring that
    minor formatting differences (case, whitespace) do not produce
    different hashes for semantically identical evidence.

    Args:
        url: Evidence source URL. ``None`` and ``""`` are treated the same.
        summary: Evidence summary text. Truncated to first 500 characters
            after normalisation.
        hash_version: Hash algorithm version (currently only ``1`` is
            supported). Reserved for future migration.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If ``hash_version`` is not a supported value.
    """
    if hash_version != 1:
        raise ValueError(
            f"Unsupported hash_version={hash_version}. "
            "Only version 1 is currently supported."
        )

    # 🔧 Normalise URL: None → "", then strip + lower
    normalized_url = (url or "").strip().lower()

    # 🔧 Normalise summary: strip + lower + truncate to 500 chars
    normalized_summary = summary.strip().lower()[:500]

    # 🔑 Combine with pipe separator and hash
    hash_input = f"{normalized_url}|{normalized_summary}"
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
