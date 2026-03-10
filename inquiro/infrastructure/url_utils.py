"""URL extraction and normalisation utilities for evidence enrichment 🔗.

Stateless helpers that convert raw text (DOI citations, PMID references,
plain URLs) into canonical clickable HTTPS URLs.  Used by EvidencePipeline
to backfill missing ``url`` fields on Evidence objects.

All logic is regex-based with zero LLM cost.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 🔍 Compiled regex patterns (module-level singletons for performance)
# ---------------------------------------------------------------------------

# Priority 1 — DOI URL already in https://doi.org/… form
_DOI_URL_PATTERN = re.compile(
    r"https?://doi\.org/[^\s\]\)>,\"']+",
    re.IGNORECASE,
)

# Priority 2 — Raw DOI reference: "doi:10.x", "DOI: 10.x", or bare "10.xxxx/…"
_RAW_DOI_PATTERN = re.compile(
    r"\bdoi:?\s*(10\.\d{4,9}/[^\s\]\)>,\"']+)",
    re.IGNORECASE,
)

# Priority 3 — PMID reference: "PMID: 12345678" or "PMID:12345678"
_PMID_PATTERN = re.compile(r"\bPMID:?\s*(\d{1,8})\b", re.IGNORECASE)

# Priority 4 — Generic HTTPS URL
_HTTPS_URL_PATTERN = re.compile(r"https://[^\s\]\)>,\"']+")

# Characters commonly appended as trailing punctuation in citations
_TRAILING_PUNCT = ".,;)]>"


def extract_and_normalize_url(text: str) -> str | None:
    """Extract the most relevant URL from free text and normalise it 🔗.

    Searches for URLs in priority order:

    1. **DOI URL** — ``https://doi.org/10.xxxx/…`` already in canonical form.
    2. **Raw DOI** — ``doi:10.xxxx/…`` or ``DOI: 10.xxxx/…``
       → converted to ``https://doi.org/…``.
    3. **PMID** — ``PMID: 12345678``
       → converted to ``https://pubmed.ncbi.nlm.nih.gov/12345678/``.
    4. **Generic HTTPS URL** — any ``https://…`` link in the text.

    Trailing punctuation (``.,;)]>``) is stripped from all matches to
    handle citations embedded in sentences.

    Args:
        text: Raw text that may contain a URL, DOI, or PMID identifier.
            Typically an evidence summary or search observation.

    Returns:
        Canonical HTTPS URL string, or ``None`` if no extractable URL
        is found.

    Examples::

        >>> extract_and_normalize_url("See https://doi.org/10.1038/s41586 for details.")
        'https://doi.org/10.1038/s41586'

        >>> extract_and_normalize_url("doi: 10.1038/nature.2024.001")
        'https://doi.org/10.1038/nature.2024.001'

        >>> extract_and_normalize_url("PMID: 12345678")
        'https://pubmed.ncbi.nlm.nih.gov/12345678/'

        >>> extract_and_normalize_url("Read more at https://example.com/paper.")
        'https://example.com/paper'

        >>> extract_and_normalize_url("No links here.")
        None
    """
    if not text:
        return None

    # 🥇 Priority 1: DOI URL already in canonical https://doi.org/… form
    doi_url_match = _DOI_URL_PATTERN.search(text)
    if doi_url_match:
        return doi_url_match.group(0).rstrip(_TRAILING_PUNCT)

    # 🥈 Priority 2: Raw DOI reference → convert to canonical URL
    raw_doi_match = _RAW_DOI_PATTERN.search(text)
    if raw_doi_match:
        doi = raw_doi_match.group(1).rstrip(_TRAILING_PUNCT)
        return f"https://doi.org/{doi}"

    # 🥉 Priority 3: PMID → PubMed URL
    pmid_match = _PMID_PATTERN.search(text)
    if pmid_match:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"

    # 📎 Priority 4: Generic HTTPS URL
    generic_match = _HTTPS_URL_PATTERN.search(text)
    if generic_match:
        return generic_match.group(0).rstrip(_TRAILING_PUNCT)

    return None
