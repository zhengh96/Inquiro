"""Inquiro EvidencePipeline -- deterministic evidence cleaning 🧹.

Four-step cleaning pipeline with zero LLM cost:
    0. Multi-item splitting (EvidenceSplitter — deterministic format parsing)
    1. Content dedup (hash-based exact + near-duplicate detection)
    2. Noise filter (pattern matching known garbage strings)
    3. Source tagging (URL-based evidence classification)

Architecture position:
    DiscoveryLoop / Runner
        -> EvidencePipeline.clean()   <-- this module
        -> (cleaned_evidence, removal_log)

The pipeline is stateless, deterministic, and domain-agnostic.
All classification rules are based on URL patterns, not domain knowledge.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from inquiro.core.canonical_hash import canonical_evidence_hash
from inquiro.core.evidence_metadata import extract_clinical_trial_id, extract_doi
from inquiro.core.evidence_splitter import EvidenceSplitter
from inquiro.core.types import Evidence, EvidenceTag
from inquiro.infrastructure.url_utils import extract_and_normalize_url

logger = logging.getLogger(__name__)


# ============================================================================
# 📋 Removal log entry
# ============================================================================


class RemovalRecord(BaseModel):
    """Record of a removed evidence item during cleaning 📋.

    Attributes:
        evidence_id: ID of the removed evidence item.
        reason: Human-readable reason for removal.
        stage: Pipeline stage where removal occurred.
    """

    evidence_id: str = Field(description="ID of the removed evidence item")
    reason: str = Field(description="Reason for removal")
    stage: str = Field(description="Pipeline stage: 'dedup' | 'noise_filter'")


# ============================================================================
# 📊 Cleaning statistics
# ============================================================================


class CleaningStats(BaseModel):
    """Summary statistics from a cleaning run 📊.

    Attributes:
        input_count: Number of evidence items before cleaning.
        output_count: Number of evidence items after cleaning.
        dedup_removed: Items removed by deduplication.
        noise_removed: Items removed by noise filtering.
        tag_distribution: Count of each EvidenceTag assigned.
    """

    input_count: int = Field(description="Evidence count before cleaning")
    output_count: int = Field(description="Evidence count after cleaning")
    dedup_removed: int = Field(default=0, description="Items removed by dedup")
    noise_removed: int = Field(default=0, description="Items removed by noise filter")
    split_expanded: int = Field(
        default=0, description="Items expanded by multi-item splitting"
    )
    tag_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each EvidenceTag assigned",
    )


# ============================================================================
# 📁 Skill reference: patterns loaded from evidence-source-classification
# ============================================================================

# 📍 Path to the Skill's patterns.yaml, relative to this file.
_SKILL_PATTERNS_PATH: Path = (
    Path(__file__).parent.parent
    / "skills"
    / "evidence-source-classification"
    / "references"
    / "patterns.yaml"
)

# 🔒 Hardcoded fallback defaults (used when patterns.yaml cannot be loaded).
# Keep in sync with the YAML file; these are the safety net only.
_FALLBACK_NOISE_PATTERNS: list[str] = [
    "AI Search Session Created",
    "No papers found",
    "Follow-up Question Submitted",
    "No relevant results",
    "Search session expired",
    "Connection timed out",
    "Error fetching results",
    "Rate limit exceeded",
    "Please try again",
    "No results found for",
    "Unable to process",
    "Session ID:",
    "Bohrium API error:",
    "API error:",
    "Found **",
    "🔍 Found",
    "📚 Found",
    "Error executing tool",
    "Error:",
    "Input validation error:",
    "validation error",
    "not found in ",
    "Unknown tool:",
    "✅ ",
    "🤖 ",
    "MCP error",
    "No web results found",
]

_FALLBACK_ACADEMIC_PATTERNS: list[str] = [
    r"pubmed\.ncbi\.nlm\.nih\.gov",
    r"pmc\.ncbi\.nlm\.nih\.gov",
    r"doi\.org",
    r"frontiersin\.org",
    r"nature\.com",
    r"sciencedirect\.com",
    r"springer\.com",
    r"wiley\.com",
    r"cell\.com",
    r"acs\.org",
    r"biorxiv\.org",
    r"medrxiv\.org",
    r"scholar\.google\.com",
    r"ncbi\.nlm\.nih\.gov/books",
    r"academic\.oup\.com",
    r"pnas\.org",
    r"science\.org",
    r"thelancet\.com",
    r"bmj\.com",
    r"nejm\.org",
    r"jama\.jamanetwork\.com",
]

_FALLBACK_PATENT_PATTERNS: list[str] = [
    r"patents\.google\.com",
    r"uspto\.gov",
    r"epo\.org",
    r"patentscope\.wipo\.int",
    r"lens\.org/lens/patent",
]

_FALLBACK_CLINICAL_TRIAL_PATTERNS: list[str] = [
    r"clinicaltrials\.gov",
    r"centerwatch\.com",
    r"who\.int/clinical-trials",
    r"isrctn\.com",
    r"anzctr\.org",
    r"eu-clinical-trials",
]

_FALLBACK_REGULATORY_PATTERNS: list[str] = [
    r"fda\.gov",
    r"ema\.europa\.eu",
    r"pmda\.go\.jp",
    r"accessdata\.fda\.gov",
    r"drugs\.com/.*fda",
]


def _load_classification_patterns() -> dict[str, Any]:
    """Load noise and URL classification patterns from the Skill reference YAML 📂.

    Reads ``references/patterns.yaml`` from the ``evidence-source-classification``
    Skill directory.  Returns a dict with keys ``noise_patterns``,
    ``academic``, ``patent``, ``clinical_trial``, and ``regulatory``.

    Falls back to hardcoded defaults if the file is missing or malformed so
    that the pipeline never breaks due to a missing Skill asset.

    Returns:
        Dict containing ``noise_patterns`` (list[str]) and per-category
        URL pattern lists under their respective keys.
    """
    fallback: dict[str, Any] = {
        "noise_patterns": _FALLBACK_NOISE_PATTERNS,
        "academic": _FALLBACK_ACADEMIC_PATTERNS,
        "patent": _FALLBACK_PATENT_PATTERNS,
        "clinical_trial": _FALLBACK_CLINICAL_TRIAL_PATTERNS,
        "regulatory": _FALLBACK_REGULATORY_PATTERNS,
    }

    try:
        raw = _SKILL_PATTERNS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)

        if not isinstance(data, dict):
            logger.warning(
                "⚠️ patterns.yaml is not a dict — using fallback classification patterns"
            )
            return fallback

        source_cls = data.get("source_classification", {})

        loaded: dict[str, Any] = {
            "noise_patterns": data.get("noise_patterns", _FALLBACK_NOISE_PATTERNS),
            "academic": source_cls.get("academic", {}).get(
                "url_patterns", _FALLBACK_ACADEMIC_PATTERNS
            ),
            "patent": source_cls.get("patent", {}).get(
                "url_patterns", _FALLBACK_PATENT_PATTERNS
            ),
            "clinical_trial": source_cls.get("clinical_trial", {}).get(
                "url_patterns", _FALLBACK_CLINICAL_TRIAL_PATTERNS
            ),
            "regulatory": source_cls.get("regulatory", {}).get(
                "url_patterns", _FALLBACK_REGULATORY_PATTERNS
            ),
        }

        logger.debug(
            "📂 Loaded classification patterns from Skill reference: %s",
            _SKILL_PATTERNS_PATH,
        )
        return loaded

    except FileNotFoundError:
        logger.warning(
            "⚠️ Skill patterns.yaml not found at %s — using fallback patterns",
            _SKILL_PATTERNS_PATH,
        )
        return fallback
    except yaml.YAMLError as exc:
        logger.warning(
            "⚠️ Failed to parse patterns.yaml (%s) — using fallback patterns",
            exc,
        )
        return fallback
    except OSError as exc:
        logger.warning(
            "⚠️ Could not read patterns.yaml (%s) — using fallback patterns",
            exc,
        )
        return fallback


# 🔧 Load patterns once at module import time (cached in module globals).
_LOADED_PATTERNS: dict[str, Any] = _load_classification_patterns()


# ============================================================================
# 🧹 Noise patterns (known garbage from MCP responses)
# ============================================================================

# ⚠️ Patterns that indicate garbage or non-evidence content.
# Loaded from the evidence-source-classification Skill reference YAML.
NOISE_PATTERNS: list[str] = _LOADED_PATTERNS["noise_patterns"]

# 📏 Minimum meaningful evidence length (characters)
MIN_EVIDENCE_LENGTH = 50


# ============================================================================
# 🌐 Source classification rules (URL pattern → EvidenceTag)
# ============================================================================

# 🎓 Academic source URL patterns
_ACADEMIC_PATTERNS: list[str] = _LOADED_PATTERNS["academic"]

# 📄 Patent source URL patterns
_PATENT_PATTERNS: list[str] = _LOADED_PATTERNS["patent"]

# 📋 Study registry source URL patterns (domain-agnostic: matches any
# publicly registered study database, not pharma-specific)
_CLINICAL_TRIAL_PATTERNS: list[str] = _LOADED_PATTERNS["clinical_trial"]

# 🏛️ Regulatory authority source URL patterns (government agencies
# and their affiliated content aggregators)
_REGULATORY_PATTERNS: list[str] = _LOADED_PATTERNS["regulatory"]


def _compile_tag_rules() -> list[tuple[re.Pattern[str], EvidenceTag]]:
    """Compile URL pattern rules into regex patterns 🔧.

    Returns:
        List of (compiled_regex, EvidenceTag) tuples, checked in order.
    """
    rules: list[tuple[re.Pattern[str], EvidenceTag]] = []
    for pattern in _ACADEMIC_PATTERNS:
        rules.append((re.compile(pattern, re.IGNORECASE), EvidenceTag.ACADEMIC))
    for pattern in _PATENT_PATTERNS:
        rules.append((re.compile(pattern, re.IGNORECASE), EvidenceTag.PATENT))
    for pattern in _CLINICAL_TRIAL_PATTERNS:
        rules.append((re.compile(pattern, re.IGNORECASE), EvidenceTag.CLINICAL_TRIAL))
    for pattern in _REGULATORY_PATTERNS:
        rules.append((re.compile(pattern, re.IGNORECASE), EvidenceTag.REGULATORY))
    return rules


# 🔧 Pre-compiled tag rules (module-level singleton)
_TAG_RULES = _compile_tag_rules()


# ============================================================================
# 🧹 EvidencePipeline
# ============================================================================


class EvidencePipeline:
    """Deterministic evidence cleaning pipeline — zero LLM cost 🧹.

    Six-step pipeline:
        0. **Multi-item splitting**: Deterministic format parsing via
           EvidenceSplitter.  Splits Bohrium multi-paper, Brave JSON,
           and biomcp Record responses into individual evidence items.
        1. **Content dedup**: Hash-based exact duplicate removal.  Uses
           ``canonical_evidence_hash`` (SHA-256 of URL + summary).
        2. **Noise filter**: Pattern matching against known garbage
           strings from MCP responses.  Also filters evidence below
           the minimum length threshold.
        3. **Source tagging**: URL-based classification into EvidenceTag
           categories (academic, patent, study_registry, regulatory,
           other).  Tags are backfilled onto Evidence objects.
        4. **Metadata extraction**: Regex-based extraction of
           ``clinical_trial_id`` (NCT) and ``doi`` fields.
        5. **URL backfill**: Derives a clickable URL from DOI or PMID
           identifiers in the summary when no URL is present.

    The pipeline is stateless and can be called multiple times.  Each call
    returns a new list of cleaned evidence and a removal log.

    Example::

        pipeline = EvidencePipeline()
        cleaned, stats = pipeline.clean(raw_evidence)

    Args:
        min_evidence_length: Minimum character length for evidence summaries.
            Items shorter than this are filtered out in the noise step.
    """

    def __init__(self, *, min_evidence_length: int = MIN_EVIDENCE_LENGTH) -> None:
        """Initialize the pipeline with configurable thresholds 🔧.

        Args:
            min_evidence_length: Minimum character length for evidence
                summaries. Defaults to ``MIN_EVIDENCE_LENGTH`` (50).
        """
        self._min_evidence_length = min_evidence_length

    def clean(
        self,
        evidence: list[Evidence],
    ) -> tuple[list[Evidence], CleaningStats]:
        """Run the full six-step cleaning pipeline 🧹.

        Args:
            evidence: Raw evidence items to clean.

        Returns:
            Tuple of (cleaned_evidence, cleaning_stats).
        """
        input_count = len(evidence)

        if not evidence:
            return [], CleaningStats(input_count=0, output_count=0)

        # 🔀 Step 0: Multi-item splitting
        splitter = EvidenceSplitter()
        evidence, split_stats = splitter.split(evidence)

        # 🔧 Step 1: Content dedup
        deduped, dedup_removed = self._dedup(evidence)

        # 🔧 Step 2: Noise filter
        filtered, noise_removed = self._filter_noise(deduped)

        # 🔧 Step 3: Source tagging
        tag_map = self._tag_sources(filtered)

        # 🏷️ Step 3b: Backfill tags onto Evidence objects
        for ev in filtered:
            tag = tag_map.get(ev.id, EvidenceTag.OTHER)
            ev.evidence_tag = tag.value

        # 🏷️ Step 4: Metadata extraction (clinical_trial_id + doi)
        for ev in filtered:
            if ev.clinical_trial_id is None:
                ev.clinical_trial_id = extract_clinical_trial_id(ev.url, ev.summary)
            if ev.doi is None:
                ev.doi = extract_doi(ev.url, ev.summary)

        # 🔗 Step 5: URL backfill — derive clickable URL from DOI/PMID in summary
        for ev in filtered:
            if not ev.url:
                ev.url = extract_and_normalize_url(ev.summary or "")
                if ev.url:
                    logger.debug(
                        "🔗 URL backfilled for evidence %s: %s", ev.id, ev.url
                    )

        # 📊 Compute tag distribution from returned mapping
        tag_dist: dict[str, int] = {}
        for ev in filtered:
            tag = tag_map.get(ev.id, EvidenceTag.OTHER)
            tag_dist[tag.value] = tag_dist.get(tag.value, 0) + 1

        stats = CleaningStats(
            input_count=input_count,
            output_count=len(filtered),
            dedup_removed=dedup_removed,
            noise_removed=noise_removed,
            split_expanded=split_stats.expanded,
            tag_distribution=tag_dist,
        )

        logger.info(
            "🧹 EvidencePipeline: %d → %d items (split +%d, dedup -%d, noise -%d)",
            input_count,
            len(filtered),
            split_stats.expanded,
            dedup_removed,
            noise_removed,
        )

        return filtered, stats

    # -- Step 1: Content dedup ------------------------------------------------

    @staticmethod
    def _dedup(evidence: list[Evidence]) -> tuple[list[Evidence], int]:
        """Remove duplicate evidence by canonical content hash 🔑.

        Uses ``canonical_evidence_hash`` (SHA-256 of URL + summary)
        as the dedup key.  First occurrence wins.

        Args:
            evidence: Evidence items to deduplicate.

        Returns:
            Tuple of (deduplicated_list, removed_count).
        """
        seen_hashes: set[str] = set()
        unique: list[Evidence] = []
        removed = 0

        for ev in evidence:
            # 🔑 Canonical hash: URL + summary (SHA-256)
            content_hash = canonical_evidence_hash(ev.url, ev.summary or "")

            if content_hash in seen_hashes:
                removed += 1
                logger.debug("🔑 Dedup removed evidence %s (hash collision)", ev.id)
                continue

            seen_hashes.add(content_hash)
            unique.append(ev)

        return unique, removed

    # -- Step 2: Noise filter -------------------------------------------------

    def _filter_noise(self, evidence: list[Evidence]) -> tuple[list[Evidence], int]:
        """Remove noise and garbage evidence items 🚫.

        Filters out:
        - Evidence matching known noise patterns (MCP session markers, etc.)
        - Evidence below minimum length threshold

        Args:
            evidence: Evidence items to filter.

        Returns:
            Tuple of (filtered_list, removed_count).
        """
        cleaned: list[Evidence] = []
        removed = 0

        for ev in evidence:
            summary = ev.summary or ""

            # 📏 Check minimum length
            if len(summary.strip()) < self._min_evidence_length:
                removed += 1
                logger.debug(
                    "🚫 Noise filter removed evidence %s (too short: %d chars)",
                    ev.id,
                    len(summary.strip()),
                )
                continue

            # 🔍 Check noise patterns
            is_noise = False
            for pattern in NOISE_PATTERNS:
                if pattern.lower() in summary.lower():
                    is_noise = True
                    logger.debug(
                        "🚫 Noise filter removed evidence %s (matched: '%s')",
                        ev.id,
                        pattern,
                    )
                    break

            if is_noise:
                removed += 1
                continue

            cleaned.append(ev)

        return cleaned, removed

    # -- Step 3: Source tagging -----------------------------------------------

    @staticmethod
    def _tag_sources(evidence: list[Evidence]) -> dict[str, EvidenceTag]:
        """Classify evidence by source URL pattern 🏷️.

        Matches the evidence URL against pre-compiled regex patterns
        to assign an EvidenceTag. Returns a mapping from evidence ID
        to the assigned tag.

        Args:
            evidence: Evidence items to classify.

        Returns:
            Mapping of evidence ID to assigned EvidenceTag.
        """
        tag_map: dict[str, EvidenceTag] = {}
        for ev in evidence:
            tag = EvidenceTag.OTHER
            url = ev.url or ""

            for regex, evidence_tag in _TAG_RULES:
                if regex.search(url):
                    tag = evidence_tag
                    break

            tag_map[ev.id] = tag

        return tag_map

    # -- Utility: Classify a single URL ----------------------------------------

    @staticmethod
    def classify_url(url: str) -> EvidenceTag:
        """Classify a single URL into an EvidenceTag 🏷️.

        Utility method for external callers.

        Args:
            url: URL string to classify.

        Returns:
            Matching EvidenceTag based on URL patterns.
        """
        for regex, evidence_tag in _TAG_RULES:
            if regex.search(url):
                return evidence_tag
        return EvidenceTag.OTHER
