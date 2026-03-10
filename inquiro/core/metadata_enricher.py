"""MetadataEnricher — LLM-based structured metadata extraction 🏷️.

Extracts rich metadata (title, authors, journal, year, trial phase, etc.)
from evidence text using Haiku batch calls.  Runs as an optional async
step in DiscoveryLoop between cleaning and analysis.

Architecture position::

    DiscoveryLoop
        ├── EvidencePipeline.clean()   (deterministic, zero LLM)
        ├── MetadataEnricher.enrich()  (THIS MODULE — optional LLM)
        └── AnalysisExp                (multi-model analysis)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from inquiro.core.types import Evidence, EvidenceMetadata

if TYPE_CHECKING:
    from inquiro.infrastructure.llm_pool import LLMProviderPool

logger = logging.getLogger(__name__)

# 📦 Batch size for LLM extraction calls
_BATCH_SIZE = 30

_SYSTEM_PROMPT = """\
You are a metadata extraction specialist. Given a list of evidence items \
(each with id, source, url, and summary), extract structured metadata \
from the text.

For each item, return a JSON object with the item's "id" and any fields \
you can confidently extract:
- title: Title of the source document
- authors: Array of author names (last name first, e.g. ["Smith J", "Lee K"])
- publication_year: Integer year (e.g. 2024)
- publication_date: Date string (e.g. "2024-03-15" or "2024")
- journal: Journal or conference name
- pmid: PubMed ID (digits only)
- trial_phase: Clinical trial phase (e.g. "Phase 1", "Phase 2/3")
- trial_status: Trial status (e.g. "Recruiting", "Completed", "Not yet recruiting")
- sponsor: Trial sponsor organization
- enrollment: Enrollment count (integer)
- patent_number: Patent number (e.g. "US11234567B2")
- patent_assignee: Patent assignee organization
- news_agency: News agency or publisher

Rules:
- Only include fields you can CONFIDENTLY extract from the text.
- Omit fields that are uncertain or not present.
- Output ONLY a JSON array, no markdown, no explanation.
"""


def _build_user_prompt(batch: list[Evidence]) -> str:
    """Build user prompt for a batch of evidence items 📝.

    Args:
        batch: Evidence items to extract metadata from.

    Returns:
        Formatted user prompt string.
    """
    lines: list[str] = []
    for ev in batch:
        summary_truncated = (ev.summary or "")[:400]
        lines.append(
            f"[{ev.id}] source={ev.source} "
            f"url={ev.url or 'N/A'} "
            f"summary={summary_truncated}"
        )
    return "\n".join(lines)


def _parse_response(
    response_text: str,
    valid_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Parse LLM response into per-evidence metadata dicts 🔍.

    Args:
        response_text: Raw LLM response text.
        valid_ids: Set of valid evidence IDs to filter against.

    Returns:
        Mapping from evidence ID to extracted metadata dict.
    """
    cleaned = response_text.strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        logger.warning(
            "⚠️ MetadataEnricher: failed to extract JSON array from response"
        )
        return {}

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("⚠️ MetadataEnricher: failed to parse JSON response")
        return {}

    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        eid = item.get("id", "")
        if eid not in valid_ids:
            continue
        # Remove the id key, keep only metadata fields
        metadata = {k: v for k, v in item.items() if k != "id"}
        if metadata:
            result[eid] = metadata

    return result


def _apply_metadata(evidence: Evidence, raw: dict[str, Any]) -> None:
    """Apply extracted metadata dict to an Evidence object 🏷️.

    Args:
        evidence: Evidence object to enrich.
        raw: Raw metadata dict from LLM response.
    """
    # Build EvidenceMetadata from the raw dict, ignoring unknown fields
    known_fields = set(EvidenceMetadata.model_fields.keys())
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    if not filtered:
        return

    try:
        metadata = EvidenceMetadata.model_validate(filtered)
        evidence.metadata = metadata
    except Exception:
        logger.debug(
            "⚠️ MetadataEnricher: failed to validate metadata for %s",
            evidence.id,
        )


class MetadataEnricher:
    """LLM-based evidence metadata extraction using batch calls 🏷️.

    Extracts structured metadata (title, authors, journal, year, etc.)
    from evidence summary text.  Processes evidence in batches of 30.
    Uses EvoMaster's BaseLLM via LLMProviderPool (synchronous query
    wrapped in asyncio.to_thread).

    Example::

        enricher = MetadataEnricher(llm_pool=pool, model="haiku")
        enriched = await enricher.enrich(evidence_list)

    Attributes:
        _llm_pool: LLM provider pool for model access.
        _model: Model name for extraction calls.
    """

    def __init__(
        self,
        llm_pool: LLMProviderPool | None = None,
        model: str = "haiku",
    ) -> None:
        """Initialize MetadataEnricher 🏗️.

        Args:
            llm_pool: LLM provider pool.  When None,
                enrich() is a no-op.
            model: Model name to request from the pool.
        """
        self._llm_pool = llm_pool
        self._model = model

    async def enrich(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """Extract metadata for all evidence items via LLM 🏷️.

        Skips items that already have metadata populated.
        Processes remaining items in batches.

        Args:
            evidence: Evidence items to enrich.

        Returns:
            The same list with metadata fields populated.
        """
        if not self._llm_pool or not evidence:
            return evidence

        # Filter to items needing enrichment
        to_enrich = [ev for ev in evidence if ev.metadata is None]
        if not to_enrich:
            return evidence

        logger.info(
            "🏷️ MetadataEnricher: enriching %d/%d evidence items",
            len(to_enrich),
            len(evidence),
        )

        batches = [
            to_enrich[i : i + _BATCH_SIZE]
            for i in range(0, len(to_enrich), _BATCH_SIZE)
        ]

        total_enriched = 0
        for batch in batches:
            try:
                enriched_count = await self._enrich_batch(batch)
                total_enriched += enriched_count
            except Exception:
                logger.warning(
                    "⚠️ MetadataEnricher: batch failed for %d items, skipping",
                    len(batch),
                    exc_info=True,
                )

        logger.info(
            "🏷️ MetadataEnricher: enriched %d/%d items",
            total_enriched,
            len(to_enrich),
        )

        return evidence

    async def _enrich_batch(self, batch: list[Evidence]) -> int:
        """Extract metadata for a single batch via LLM 📦.

        Args:
            batch: Batch of evidence items (max 30).

        Returns:
            Number of items successfully enriched.
        """
        from evomaster.utils.llm import Dialog, UserMessage

        user_prompt = _build_user_prompt(batch)
        valid_ids = {ev.id for ev in batch}

        dialog = Dialog(
            system=_SYSTEM_PROMPT,
            messages=[UserMessage(content=user_prompt)],
        )

        llm = self._llm_pool.get_llm(self._model)
        response = await asyncio.to_thread(llm.query, dialog)
        response_text = getattr(response, "content", "") or ""

        parsed = _parse_response(response_text, valid_ids)

        # Apply to evidence objects
        id_to_ev = {ev.id: ev for ev in batch}
        enriched = 0
        for eid, raw_meta in parsed.items():
            ev = id_to_ev.get(eid)
            if ev:
                _apply_metadata(ev, raw_meta)
                enriched += 1

        return enriched
