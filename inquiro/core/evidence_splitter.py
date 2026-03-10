"""Inquiro EvidenceSplitter -- multi-item MCP response splitting 🔀.

Deterministic, zero-LLM splitting of multi-item MCP tool responses
into individual evidence entries. Runs as Stage 0 of EvidencePipeline,
before dedup and noise filtering.

Supported formats:
    - Perplexity citations: AI answer with [N] markers + URL block → per-source items
    - Bohrium markdown: ``Found **N** papers`` + ``**[N] Title**`` blocks
    - Brave JSON: JSON array or multi-line JSON objects with ``url`` keys
    - biomcp Record: ``# Record N`` delimited sections

Explicitly NOT split:
    - OpenTargets (single structured result)

Architecture position:
    EvidencePipeline.clean()
        -> Stage 0: EvidenceSplitter.split()   <-- this module
        -> Stage 1: dedup
        -> Stage 2: noise filter
        -> Stage 3: source tagging
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from inquiro.core.types import Evidence

logger = logging.getLogger(__name__)


# ============================================================================
# 📊 Split statistics
# ============================================================================


@dataclass
class SplitStats:
    """Statistics from an evidence split operation 📊.

    Attributes:
        input_count: Number of evidence items before splitting.
        output_count: Number of evidence items after splitting.
        expanded: Number of new items created by splitting.
        bohrium_split: Items split from Bohrium multi-paper responses.
        brave_split: Items split from Brave JSON responses.
        biomcp_split: Items split from biomcp Record responses.
        skipped: Items that were not split (pass-through).
    """

    input_count: int = 0
    output_count: int = 0
    expanded: int = 0
    bohrium_split: int = 0
    brave_split: int = 0
    biomcp_split: int = 0
    perplexity_split: int = 0
    skipped: int = 0


# ============================================================================
# 🔀 EvidenceSplitter
# ============================================================================


class EvidenceSplitter:
    """Deterministic multi-item evidence splitter — zero LLM cost 🔀.

    Detects multi-item MCP responses and splits them into individual
    evidence entries. Each child entry inherits the parent's source
    and query, with an independent URL extracted from its section.

    Child IDs use dot notation: E3.1, E3.2, E3.3, etc.

    Example::

        splitter = EvidenceSplitter()
        expanded, stats = splitter.split(raw_evidence)
    """

    # 🔍 Bohrium header detection pattern
    _BOHRIUM_HEADER_RE = re.compile(
        r"Found\s+\*\*(\d+)\*\*\s+papers?",
        re.IGNORECASE,
    )
    # 🔍 Bohrium alternate header: "# Enhanced Search:" format
    _BOHRIUM_HEADER_ALT_RE = re.compile(
        r"^#\s+Enhanced\s+Search:",
        re.MULTILINE | re.IGNORECASE,
    )
    # 🔍 Bohrium paper boundary pattern: **[N] Title**
    _BOHRIUM_PAPER_RE = re.compile(r"\*\*\[(\d+)\]\s+(.+?)\*\*")

    # 🔍 biomcp Record boundary pattern: # Record N
    _BIOMCP_RECORD_RE = re.compile(r"^#\s+Record\s+(\d+)", re.MULTILINE)

    # 🚫 Sources that should never be split
    _NO_SPLIT_SOURCES: frozenset[str] = frozenset({"opentargets"})

    # 🔍 Perplexity citation inline markers: [1], [2], [1][2], etc.
    _PERPLEXITY_CITE_RE = re.compile(r"\[(\d+)\]")

    # 🔍 Perplexity citation URL block patterns
    _PERPLEXITY_URL_BLOCK_RE = re.compile(
        r"(?:^|\n)\s*(?:Citations?|Sources?|References?)\s*:?\s*\n"
        r"((?:\s*\[?\d+\]?\s*(?:https?://\S+).*\n?)+)",
        re.IGNORECASE | re.MULTILINE,
    )

    # 🔍 Individual citation-URL pair in the block
    _PERPLEXITY_CITE_URL_RE = re.compile(r"\[?(\d+)\]?\s*(https?://\S+)")

    def split(
        self,
        evidence: list[Evidence],
    ) -> tuple[list[Evidence], SplitStats]:
        """Split multi-item evidence into individual entries 🔀.

        Args:
            evidence: Raw evidence items, potentially containing
                multi-item MCP responses.

        Returns:
            Tuple of (expanded_evidence, split_stats).
        """
        stats = SplitStats(input_count=len(evidence))
        result: list[Evidence] = []

        for ev in evidence:
            source_lower = (ev.source or "").strip().lower()

            # 🚫 Skip sources that should never be split
            if any(ns in source_lower for ns in self._NO_SPLIT_SOURCES):
                result.append(ev)
                stats.skipped += 1
                continue

            # 🔀 Try splitting in priority order
            split_result = self._try_split(ev, source_lower)

            if split_result is None:
                # ✅ Not a multi-item response, pass through
                result.append(ev)
                stats.skipped += 1
            else:
                children, fmt = split_result
                result.extend(children)
                n_new = len(children) - 1  # -1 because parent is replaced
                stats.expanded += n_new
                # 📊 Per-format tracking
                if fmt == "bohrium":
                    stats.bohrium_split += len(children)
                elif fmt == "brave":
                    stats.brave_split += len(children)
                elif fmt == "biomcp":
                    stats.biomcp_split += len(children)
                elif fmt == "perplexity":
                    stats.perplexity_split += len(children)
                logger.info(
                    "🔀 Split evidence %s into %d items (source: %s, format: %s)",
                    ev.id,
                    len(children),
                    ev.source,
                    fmt,
                )

        stats.output_count = len(result)

        if stats.expanded > 0:
            logger.info(
                "🔀 EvidenceSplitter: %d → %d items (+%d expanded, "
                "bohrium=%d, brave=%d, biomcp=%d, perplexity=%d)",
                stats.input_count,
                stats.output_count,
                stats.expanded,
                stats.bohrium_split,
                stats.brave_split,
                stats.biomcp_split,
                stats.perplexity_split,
            )

        return result, stats

    def _try_split(
        self,
        ev: Evidence,
        source_lower: str,
    ) -> tuple[list[Evidence], str] | None:
        """Attempt to split an evidence item by detected format 🔍.

        Args:
            ev: Evidence item to potentially split.
            source_lower: Lowered source string for pattern matching.

        Returns:
            Tuple of (children, format_tag) if split succeeded, or None
            if the item is not a multi-item response. Format tag is one
            of "bohrium", "brave", "biomcp", "perplexity".
        """
        summary = ev.summary or ""

        # 🔍 Priority 0: Perplexity AI answer with citations
        if "perplexity" in source_lower:
            children = self._split_perplexity(ev)
            if children is not None:
                return children, "perplexity"

        # 🔍 Priority 1: Bohrium markdown format (standard or alt header)
        if self._BOHRIUM_HEADER_RE.search(
            summary
        ) or self._BOHRIUM_HEADER_ALT_RE.search(summary):
            children = self._split_bohrium(ev)
            if children is not None:
                return children, "bohrium"

        # 🔍 Priority 2: Brave JSON format
        stripped = summary.strip()
        if self._looks_like_brave_json(stripped):
            children = self._split_brave(ev)
            if children is not None:
                return children, "brave"

        # 🔍 Priority 3: biomcp Record format
        if self._BIOMCP_RECORD_RE.search(summary):
            children = self._split_biomcp(ev)
            if children is not None:
                return children, "biomcp"

        return None

    # ------------------------------------------------------------------
    # 🔬 Bohrium splitting
    # ------------------------------------------------------------------

    def _split_bohrium(self, ev: Evidence) -> list[Evidence] | None:
        """Split Bohrium multi-paper markdown response 🔬.

        Bohrium format::

            # Enhanced Search: 'query'
            Found **N** papers
            **[1] Paper Title**
            - Authors: ...
            - DOI: ...
            **[2] Paper Title**
            ...

        Args:
            ev: Evidence item with Bohrium markdown content.

        Returns:
            List of child Evidence items, or None if only one paper.
        """
        summary = ev.summary or ""

        # 📍 Find all paper boundaries
        boundaries = list(self._BOHRIUM_PAPER_RE.finditer(summary))
        if not boundaries:
            return None
        # ✨ Single-paper: strip preamble ("Found **N** papers") to avoid
        # noise filter false-positive on the "Found **" pattern.
        if len(boundaries) == 1:
            section = summary[boundaries[0].start() :].strip()
            if section == summary.strip():
                return None  # No preamble to strip
            url = self._extract_url_from_section(section)
            return [
                Evidence(
                    id=ev.id,
                    source=ev.source,
                    url=url or ev.url,
                    query=ev.query,
                    summary=section,
                    quality_label=ev.quality_label,
                    round_number=ev.round_number,
                    timestamp=ev.timestamp,
                )
            ]

        children: list[Evidence] = []
        for i, match in enumerate(boundaries):
            # 📄 Extract section text between this paper and next
            start = match.start()
            end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(summary)
            section = summary[start:end].strip()

            child_id = f"{ev.id}.{i + 1}"
            url = self._extract_url_from_section(section)

            children.append(
                Evidence(
                    id=child_id,
                    source=ev.source,
                    url=url or ev.url,
                    query=ev.query,
                    summary=section,
                    quality_label=ev.quality_label,
                    round_number=ev.round_number,
                    timestamp=ev.timestamp,
                )
            )

        return children

    # ------------------------------------------------------------------
    # 🌐 Brave splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_brave_json(stripped: str) -> bool:
        """Check if text looks like Brave JSON response 🔍.

        Args:
            stripped: Stripped summary text.

        Returns:
            True if the text appears to be Brave JSON format.
        """
        # 📋 JSON array of objects with url keys
        if stripped.startswith("[") and '"url"' in stripped:
            return True
        # 📋 Multiple JSON objects (one per line or concatenated)
        if stripped.startswith('{"') and '"url"' in stripped:
            # Check for multiple objects
            return stripped.count('"url"') > 1
        return False

    def _split_brave(self, ev: Evidence) -> list[Evidence] | None:
        """Split Brave JSON search results into individual entries 🌐.

        Brave format — JSON array::

            [{"url": "...", "title": "...", "description": "..."},
             {"url": "...", "title": "...", "description": "..."}]

        Or single object (no split needed if only one result).

        Args:
            ev: Evidence item with Brave JSON content.

        Returns:
            List of child Evidence items, or None if only one result.
        """
        summary = (ev.summary or "").strip()

        items = self._parse_brave_json(summary)
        if items is None or len(items) < 2:
            return None

        children: list[Evidence] = []
        for i, item in enumerate(items):
            child_id = f"{ev.id}.{i + 1}"
            url = item.get("url", "")
            title = item.get("title", "")
            description = item.get("description", "")

            # 📝 Build human-readable summary from JSON fields
            parts: list[str] = []
            if title:
                parts.append(f"**{title}**")
            if description:
                parts.append(description)
            if url:
                parts.append(f"URL: {url}")

            child_summary = "\n".join(parts) if parts else json.dumps(item)

            children.append(
                Evidence(
                    id=child_id,
                    source=ev.source,
                    url=url or ev.url,
                    query=ev.query,
                    summary=child_summary,
                    quality_label=ev.quality_label,
                    round_number=ev.round_number,
                    timestamp=ev.timestamp,
                )
            )

        return children

    @staticmethod
    def _parse_brave_json(text: str) -> list[dict] | None:
        """Parse Brave JSON response, handling truncated JSON 🔧.

        Args:
            text: Raw JSON text from Brave MCP response.

        Returns:
            List of parsed JSON objects, or None if parsing fails.
        """
        # 🔍 Try parsing as JSON array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

        # 🔧 Try fixing truncated JSON array (missing closing bracket)
        if text.startswith("["):
            # Find last complete object
            last_brace = text.rfind("}")
            if last_brace > 0:
                try:
                    fixed = text[: last_brace + 1] + "]"
                    parsed = json.loads(fixed)
                    if isinstance(parsed, list):
                        return [item for item in parsed if isinstance(item, dict)]
                except json.JSONDecodeError:
                    pass

        # 🔧 Try parsing as newline-separated JSON objects
        objects: list[dict] = []
        for line in text.split("\n"):
            line = line.strip().rstrip(",")
            if not line or line in ("[]", "[", "]"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        return objects if len(objects) >= 2 else None

    # ------------------------------------------------------------------
    # 🧬 biomcp Record splitting
    # ------------------------------------------------------------------

    def _split_biomcp(self, ev: Evidence) -> list[Evidence] | None:
        """Split biomcp multi-Record response into individual entries 🧬.

        biomcp format::

            # Record 1
            - Title: ...
            - Doi Url: https://doi.org/...
            - PMID: 12345

            # Record 2
            - Title: ...
            ...

        Args:
            ev: Evidence item with biomcp Record content.

        Returns:
            List of child Evidence items, or None if only one record.
        """
        summary = ev.summary or ""

        # 📍 Find all Record boundaries
        boundaries = list(self._BIOMCP_RECORD_RE.finditer(summary))
        if len(boundaries) < 2:
            return None

        children: list[Evidence] = []
        for i, match in enumerate(boundaries):
            start = match.start()
            end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(summary)
            section = summary[start:end].strip()

            child_id = f"{ev.id}.{i + 1}"
            url = self._extract_url_from_section(section)

            children.append(
                Evidence(
                    id=child_id,
                    source=ev.source,
                    url=url or ev.url,
                    query=ev.query,
                    summary=section,
                    quality_label=ev.quality_label,
                    round_number=ev.round_number,
                    timestamp=ev.timestamp,
                )
            )

        return children

    # ------------------------------------------------------------------
    # 🔎 Perplexity splitting
    # ------------------------------------------------------------------

    def _split_perplexity(self, ev: Evidence) -> list[Evidence] | None:
        """Split Perplexity AI answer into per-citation evidence items 🔎.

        Perplexity returns an AI-synthesized answer with inline citation
        markers [1], [2], etc. and a citation URL block.  This method
        extracts the citation-URL mapping and creates individual evidence
        items, one per unique cited source, with the relevant text
        fragments that reference each citation.

        If no citation block is found, the item is returned as-is
        (single-element list) to avoid data loss.

        Args:
            ev: Evidence item from a Perplexity MCP tool call.

        Returns:
            List of child Evidence items (one per citation), or None
            if no actionable citations were found.
        """
        summary = ev.summary or ""
        if not summary.strip():
            return None

        # 📝 Step 1: Extract citation-URL mapping
        cite_urls = self._extract_perplexity_citations(summary)
        if not cite_urls:
            return None

        # 📝 Step 2: Split text into paragraphs/sentences
        # and map each to its referenced citations
        paragraphs = re.split(r"\n\n+", summary)

        # 📝 Step 3: Build per-citation text fragments
        cite_texts: dict[int, list[str]] = {n: [] for n in cite_urls}

        for para in paragraphs:
            # 🔍 Skip citation block itself
            if re.match(
                r"\s*(?:Citations?|Sources?|References?)\s*:?",
                para.strip(),
                re.IGNORECASE,
            ):
                continue

            # 🎯 Find which citations this paragraph references
            refs = set(int(m.group(1)) for m in self._PERPLEXITY_CITE_RE.finditer(para))

            for ref_num in refs:
                if ref_num in cite_texts:
                    # 🧹 Strip citation markers for cleaner text
                    cleaned = re.sub(
                        r"\[(\d+)\]",
                        "",
                        para,
                    ).strip()
                    if cleaned:
                        cite_texts[ref_num].append(cleaned)

        # 📝 Step 4: Create child Evidence items
        children: list[Evidence] = []
        for cite_num in sorted(cite_urls.keys()):
            url = cite_urls[cite_num]
            fragments = cite_texts.get(cite_num, [])

            if not fragments:
                continue

            child_summary = "\n\n".join(fragments)
            child_id = f"{ev.id}.{cite_num}"

            children.append(
                Evidence(
                    id=child_id,
                    source=ev.source,
                    url=url,
                    query=ev.query,
                    summary=child_summary,
                    quality_label=ev.quality_label,
                    round_number=ev.round_number,
                    timestamp=ev.timestamp,
                )
            )

        if not children:
            return None

        return children

    def _extract_perplexity_citations(
        self,
        summary: str,
    ) -> dict[int, str]:
        """Extract citation number → URL mapping from Perplexity text 🔗.

        Searches for a citation/sources/references block at the end of
        the text, then extracts [N] URL pairs.  Falls back to scanning
        the entire text for inline URL references with citation markers.

        Args:
            summary: Full Perplexity AI answer text.

        Returns:
            Dict mapping citation number to URL string.
        """
        cite_urls: dict[int, str] = {}

        # 🔍 Strategy 1: Look for explicit citation block
        block_match = self._PERPLEXITY_URL_BLOCK_RE.search(summary)
        if block_match:
            block = block_match.group(1)
            for m in self._PERPLEXITY_CITE_URL_RE.finditer(block):
                num = int(m.group(1))
                url = m.group(2).rstrip(".,;)")
                cite_urls[num] = url

        if cite_urls:
            return cite_urls

        # 🔍 Strategy 2: Scan for inline patterns like
        #   [1](https://...) or [1] https://...
        inline_re = re.compile(r"\[(\d+)\]\s*\(?(https?://[^\s\)\]]+)\)?")
        for m in inline_re.finditer(summary):
            num = int(m.group(1))
            url = m.group(2).rstrip(".,;)")
            if num not in cite_urls:
                cite_urls[num] = url

        return cite_urls

    # ------------------------------------------------------------------
    # 🔗 URL extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_url_from_section(section: str) -> str:
        """Extract the best URL from a text section 🔗.

        Priority order:
        1. DOI URL (https://doi.org/...)
        2. Structured field (``- URL:``, ``- Doi Url:``, ``- Study Url:``)
        3. Raw DOI → converted URL (doi:10.xxxx/...)
        4. PMID → PubMed URL
        5. Generic HTTPS URL

        Args:
            section: Text section to extract URL from.

        Returns:
            Extracted URL string, or empty string if none found.
        """
        if not section:
            return ""

        # 🥇 Priority 1: DOI URL
        doi_url_match = re.search(
            r"https?://doi\.org/[^\s\])\>,\"']+",
            section,
        )
        if doi_url_match:
            return doi_url_match.group(0).rstrip(".")

        # 🥈 Priority 2: Structured field URL
        field_match = re.search(
            r"-\s*(?:URL|Doi Url|Study Url|Link):\s*(https?://[^\s\])\>,\"']+)",
            section,
            re.IGNORECASE,
        )
        if field_match:
            return field_match.group(1).rstrip(".")

        # 🥉 Priority 3: Raw DOI → URL
        raw_doi_match = re.search(
            r"\bDOI:\s*(10\.\d{4,}/[^\s\])\>,\"']+)",
            section,
            re.IGNORECASE,
        )
        if raw_doi_match:
            return f"https://doi.org/{raw_doi_match.group(1).rstrip('.')}"

        # 🏅 Priority 4: PMID → PubMed URL
        pmid_match = re.search(
            r"\bPMID:\s*(\d+)",
            section,
            re.IGNORECASE,
        )
        if pmid_match:
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"

        # 📎 Priority 5: Generic HTTPS URL
        generic_match = re.search(
            r"https://[^\s\])\>,\"']+",
            section,
        )
        if generic_match:
            return generic_match.group(0).rstrip(".")

        return ""
