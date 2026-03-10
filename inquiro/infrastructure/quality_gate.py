"""Inquiro QualityGate — deterministic output validation 🔍.

Performs schema validation, required-field checks, checklist coverage
analysis, and evidence reference integrity checks on agent output.
Shared by both SearchAgent and SynthesisAgent.
"""

from __future__ import annotations

import enum
import logging
from typing import Any

import jsonschema
from pydantic import BaseModel, Field

from inquiro.core.types import (
    ConfidenceBreakdown,
    QualityChecks,
    QualityGateConfig as CoreQualityGateConfig,
)


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class QualityGateCheck(str, enum.Enum):
    """Available quality gate check types 🎯."""

    SCHEMA_VALIDATION = "schema_validation"
    COVERAGE_CHECK = "coverage_check"
    EVIDENCE_REFERENCE_CHECK = "evidence_reference_check"
    CROSS_REFERENCE_CHECK = "cross_reference_check"
    SOURCE_DIVERSITY_CHECK = "source_diversity_check"
    EVIDENCE_URL_CHECK = "evidence_url_check"


class QualityGateResult(BaseModel):
    """Outcome of running the QualityGate validation pipeline 📋.

    Attributes:
        passed: ``True`` when no hard failures exist.
        hard_failures: Errors that require a retry (e.g. invalid schema).
        soft_failures: Warnings that cap confidence / flag risks.
        confidence_cap: If set, agent confidence is capped to this value.
        risk_flags: Human-readable risk flag strings.
    """

    passed: bool = Field(description="True if no hard failures were detected")
    hard_failures: list[str] = Field(
        default_factory=list,
        description="Errors requiring retry (schema invalid, missing fields)",
    )
    soft_failures: list[str] = Field(
        default_factory=list, description="Warnings that cap confidence or flag risks"
    )
    confidence_cap: float | None = Field(
        default=None, description="Maximum confidence score after soft failures"
    )
    risk_flags: list[str] = Field(
        default_factory=list, description="Human-readable risk flags"
    )


# ✅ Backward-compatible alias — QualityChecks from core.types is the
# single source of truth for check toggles (100% field-identical).
QualityGateChecksConfig = QualityChecks


class QualityGateConfig(CoreQualityGateConfig):
    """Runtime configuration for the QualityGate validator ⚙️.

    Extends core QualityGateConfig with runtime-specific settings
    for coverage analysis and source diversity checking.

    Inherited from CoreQualityGateConfig:
        enabled: Whether quality gate validation is active.
        max_retries: Maximum retry attempts on hard quality failures.
        checks: Individual check toggles (QualityChecks).

    Attributes:
        coverage_threshold: Minimum required coverage ratio (0.0-1.0).
        min_source_count: Minimum unique evidence sources for diversity.
        min_evidence_per_source: Minimum evidence items per source.
        source_diversity_confidence_caps: Graduated caps by diversity level.
    """

    coverage_threshold: float = Field(
        default=0.80, description="Min required checklist coverage (0.0-1.0)"
    )
    min_source_count: int = Field(
        default=2,
        description="Minimum unique evidence sources required for diversity check",
    )
    min_evidence_per_source: int = Field(
        default=1,
        description=(
            "Minimum evidence items from a source to count "
            "it as a meaningful contributor"
        ),
    )
    source_diversity_confidence_caps: dict[str, float] = Field(
        default_factory=lambda: {
            "no_diversity": 0.49,
            "low_diversity": 0.59,
            "moderate_diversity": 0.69,
        },
        description=(
            "Graduated confidence caps based on diversity level. "
            "Keys: no_diversity, low_diversity, moderate_diversity."
        ),
    )
    url_missing_penalty: float = Field(
        default=0.05,
        description=("Confidence penalty per evidence item with missing URL"),
    )
    url_missing_penalty_max: float = Field(
        default=0.30,
        description=("Maximum total confidence penalty for missing URLs"),
    )


# 🔒 Confidence cap applied on soft failures
_SOFT_FAILURE_CONFIDENCE_CAP = 0.69


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------


class QualityGate:
    """Configurable output quality validation 🔍.

    Performs deterministic checks on agent output.
    Works for both EvaluationResult and SynthesisResult.

    Checks:
        - Schema validation (hard fail) 🔒
        - Required field completeness (hard fail) 📋
        - Search checklist coverage (soft fail) 📊
        - Evidence reference integrity (soft fail) 🔗
        - Cross-reference consistency (soft fail, optional) 🔄

    Example::

        qg = QualityGate(config, output_schema)
        result = qg.validate(raw_agent_output)
        if not result.passed:
            # retry ...
    """

    def __init__(
        self,
        config: QualityGateConfig,
        output_schema: dict[str, Any],
    ) -> None:
        """Initialize QualityGate 🔧.

        Args:
            config: Quality gate configuration.
            output_schema: JSON Schema dict for output validation.
        """
        self.config = config
        self.output_schema = output_schema
        self.max_retries = config.max_retries
        self._logger = logging.getLogger(self.__class__.__name__)

    # -- Public API ----------------------------------------------------------

    def validate(
        self,
        result: dict[str, Any],
        checklist: Any | None = None,
        checks_config: QualityGateChecksConfig | None = None,
    ) -> QualityGateResult:
        """Run all configured checks on an agent result ✅.

        Args:
            result: Raw result dictionary produced by the agent.
            checklist: Optional search checklist for coverage analysis.
            checks_config: Override per-call check toggles. Falls back
                to ``self.config.checks`` when ``None``.

        Returns:
            QualityGateResult with pass/fail status and details.
        """
        checks = checks_config or self.config.checks

        hard_failures: list[str] = []
        soft_failures: list[str] = []
        risk_flags: list[str] = []
        confidence_caps: list[float] = []

        # 🚨 Silent failure detection: zero search rounds (hard fail)
        search_rounds = result.get("search_rounds", None)
        if search_rounds is not None and search_rounds == 0:
            hard_failures.append("silent_failure: No search rounds completed")
            self._logger.warning("🚨 Silent failure detected: search_rounds=0")

        # 🚨 Silent failure detection: zero evidence without error
        # (hard fail)
        evidence_index = result.get("evidence_index", [])
        has_error_indicator = bool(result.get("error"))
        if len(evidence_index) == 0 and not has_error_indicator:
            gaps = result.get("gaps_remaining", [])
            gap_context = ""
            if gaps:
                gap_context = f" (gaps_remaining: {', '.join(gaps[:3])})"
            hard_failures.append(
                "silent_failure: Zero evidence collected "
                "without error explanation" + gap_context
            )
            self._logger.warning(
                "🚨 Silent failure detected: evidence_index is empty with no error"
            )

        # 🔒 Schema validation (hard fail)
        if checks.schema_validation:
            schema_errors = self._validate_schema(result, self.output_schema)
            if schema_errors:
                hard_failures.extend(f"schema_invalid: {e}" for e in schema_errors)

        # 📊 Coverage check (soft fail)
        if checks.coverage_check:
            coverage = self._check_coverage(result, checklist)
            if coverage < self.config.coverage_threshold:
                soft_failures.append(
                    f"low_coverage: {coverage:.2f} "
                    f"< {self.config.coverage_threshold:.2f}"
                )
                confidence_caps.append(_SOFT_FAILURE_CONFIDENCE_CAP)
                risk_flags.append(
                    f"Checklist coverage {coverage:.0%} below "
                    f"threshold {self.config.coverage_threshold:.0%}"
                )

        # 🔗 Evidence reference integrity (soft fail)
        if checks.evidence_reference_check:
            orphans = self._check_evidence_references(result)
            if orphans:
                soft_failures.append(
                    f"orphan_claims: {len(orphans)} references to non-existent evidence"
                )
                confidence_caps.append(_SOFT_FAILURE_CONFIDENCE_CAP)
                risk_flags.extend(orphans)

        # 🔄 Cross-reference consistency (soft fail, optional)
        if checks.cross_reference_check:
            inconsistencies = self._check_cross_references(result)
            if inconsistencies:
                soft_failures.append(
                    f"cross_ref_inconsistencies: {len(inconsistencies)} found"
                )
                confidence_caps.append(_SOFT_FAILURE_CONFIDENCE_CAP)
                risk_flags.extend(inconsistencies)

        # 🌐 Source diversity check (hard + soft fail, graduated caps)
        if checks.source_diversity_check:
            div_hard, div_soft, diversity_cap = self._check_source_diversity(result)
            if div_hard:
                hard_failures.extend(div_hard)
            if div_soft:
                soft_failures.extend(div_soft)
                risk_flags.extend(div_soft)
                if diversity_cap is not None:
                    confidence_caps.append(diversity_cap)

        # 🔗 Evidence URL presence check (soft fail)
        if checks.evidence_url_check:
            url_warnings, url_penalty = self._check_evidence_urls(result)
            if url_warnings:
                soft_failures.extend(url_warnings)
                risk_flags.extend(url_warnings)
            if url_penalty > 0.0:
                # 📉 Convert penalty to cap: 1.0 - penalty
                confidence_caps.append(1.0 - url_penalty)

        # 🎯 Determine final confidence cap
        confidence_cap = min(confidence_caps) if confidence_caps else None

        passed = len(hard_failures) == 0

        return QualityGateResult(
            passed=passed,
            hard_failures=hard_failures,
            soft_failures=soft_failures,
            confidence_cap=confidence_cap,
            risk_flags=risk_flags,
        )

    def generate_reflection(
        self,
        qg_result: QualityGateResult,
        checklist_items: list[str] | None = None,
        coverage_map: dict[str, bool] | None = None,
    ) -> str:
        """Generate structured reflection text for retry injection 🪞.

        Implements the Reflexion pattern (Shinn et al., 2023):
        produces a markdown feedback message that explains what
        went wrong, what is missing, and suggests improvement
        strategies for the next agent attempt.

        Args:
            qg_result: The QualityGateResult from the failed attempt.
            checklist_items: Optional list of all required checklist
                item IDs for missing-item analysis.
            coverage_map: Optional mapping of checklist item ID to
                boolean coverage status.

        Returns:
            Structured markdown reflection string. Empty string when
            there are no failures (no reflection needed).
        """
        has_hard = bool(qg_result.hard_failures)
        has_soft = bool(qg_result.soft_failures)

        if not has_hard and not has_soft:
            return ""

        sections: list[str] = []
        sections.append(
            "# QUALITY GATE REFLECTION\n\n"
            "Your previous attempt did not fully pass quality "
            "validation. Review the feedback below and improve "
            "your next attempt accordingly."
        )

        # 🔴 Section 1: Hard Failures
        if has_hard:
            lines = ["## Hard Failures (must fix)\n"]
            for failure in qg_result.hard_failures:
                lines.append(f"- {failure}")
            sections.append("\n".join(lines))

        # 🟡 Section 2: Quality Issues (soft failures)
        if has_soft:
            lines = ["## Quality Issues (should improve)\n"]
            for failure in qg_result.soft_failures:
                lines.append(f"- {failure}")
            sections.append("\n".join(lines))

        # 📋 Section 3: Missing Checklist Items
        if checklist_items and coverage_map is not None:
            missing = [
                item for item in checklist_items if not coverage_map.get(item, False)
            ]
            if missing:
                lines = ["## Missing Checklist Items\n"]
                for item in missing:
                    lines.append(f"- {item}")
                sections.append("\n".join(lines))

        # 🎯 Section 4: Suggested Strategy
        all_failures = qg_result.hard_failures + qg_result.soft_failures
        strategies = self._suggest_retry_strategies(all_failures)
        if strategies:
            lines = ["## Suggested Strategy\n"]
            for strategy in strategies:
                lines.append(f"- {strategy}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _suggest_retry_strategies(
        failures: list[str],
    ) -> list[str]:
        """Derive retry strategies from failure messages 🎯.

        Scans failure strings for known keywords and returns
        actionable strategy suggestions tailored to each
        failure category.

        Args:
            failures: Combined list of hard and soft failure
                description strings.

        Returns:
            List of strategy suggestion strings.
        """
        strategies: list[str] = []
        joined = " ".join(failures).lower()

        if "schema" in joined or "required" in joined:
            strategies.append(
                "Ensure your output strictly conforms to the "
                "required JSON schema. Include ALL required "
                "fields (decision, confidence, reasoning, "
                "evidence_index, checklist_coverage)."
            )

        if "coverage" in joined:
            strategies.append(
                "Expand your search to cover more checklist "
                "items. Use different search queries and tools "
                "to address the missing items identified above."
            )

        if "evidence" in joined or "orphan" in joined:
            strategies.append(
                "Verify that every evidence_id referenced in "
                "reasoning exists in the evidence_index. Remove "
                "or correct any orphan references."
            )

        if "divers" in joined or "source" in joined:
            strategies.append(
                "Use multiple different search tools to gather "
                "evidence from diverse sources. Avoid relying "
                "on a single tool or database."
            )

        if "missing url" in joined:
            strategies.append(
                "Ensure every evidence item includes a valid "
                "URL linking to the original source. Prefer "
                "primary source URLs over aggregator pages."
            )

        if "cross_ref" in joined or "inconsisten" in joined:
            strategies.append(
                "Review your reasoning for internal consistency. "
                "Ensure claims do not contradict each other."
            )

        # 🔄 Fallback: generic strategy if no keyword matched
        if not strategies:
            strategies.append(
                "Carefully review the failures listed above and "
                "address each one in your next attempt."
            )

        return strategies

    # -- Individual Checks ---------------------------------------------------

    def _validate_schema(
        self,
        result: dict[str, Any],
        schema: dict[str, Any],
    ) -> list[str]:
        """Validate *result* against a JSON Schema (hard fail) 🔒.

        Uses ``jsonschema.Draft7Validator`` internally.

        Args:
            result: Result dictionary to validate.
            schema: JSON Schema dict.

        Returns:
            List of human-readable schema error messages.
            Empty list means the result is valid.
        """
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(
            validator.iter_errors(result),
            key=lambda e: list(e.absolute_path),
        )
        return [e.message for e in errors]

    def _check_required_fields(
        self,
        result: dict[str, Any],
    ) -> list[str]:
        """Check for missing required fields defined in the schema 📋.

        Args:
            result: Result dictionary to inspect.

        Returns:
            List of missing required field names.
        """
        required = self.output_schema.get("required", [])
        missing = [field for field in required if field not in result]
        return missing

    def _check_coverage(
        self,
        result: dict[str, Any],
        checklist: Any | None = None,
    ) -> float:
        """Calculate search checklist coverage ratio 📊.

        Coverage = ``len(covered) / (len(covered) + len(missing))``.
        When no checklist information is present, returns ``1.0``.

        Args:
            result: Result dictionary containing ``checklist_coverage``.
            checklist: Optional checklist definition for cross-check.

        Returns:
            Coverage ratio between 0.0 and 1.0.
        """
        coverage_data = result.get("checklist_coverage", {})

        covered = coverage_data.get(
            "required_covered", coverage_data.get("covered", [])
        )
        missing = coverage_data.get(
            "required_missing", coverage_data.get("missing", [])
        )

        # ✨ If checklist is provided, use its required_items
        if checklist is not None:
            required_items = []
            if isinstance(checklist, dict):
                required_items = checklist.get("required_items", [])
            elif isinstance(checklist, list):
                required_items = checklist

            if required_items:
                effective_covered = [item for item in required_items if item in covered]
                effective_missing = [
                    item for item in required_items if item not in covered
                ]
                covered = effective_covered
                missing = effective_missing

        total = len(covered) + len(missing)
        if total == 0:
            return 1.0

        return len(covered) / total

    def _check_evidence_references(
        self,
        result: dict[str, Any],
    ) -> list[str]:
        """Verify evidence reference integrity 🔗.

        Every ``evidence_id`` referenced in ``reasoning`` or ``claims``
        must exist in the ``evidence_index`` list.  Both fields are
        scanned to catch hallucinated IDs regardless of which output
        format the LLM used.

        Args:
            result: Result dictionary with ``reasoning``, ``claims``
                and ``evidence_index`` keys.

        Returns:
            List of orphan-claim descriptions (claims referencing
            non-existent evidence IDs).
        """
        evidence_index = result.get("evidence_index", [])

        # 🗂️ Build set of valid evidence IDs
        valid_ids: set[str] = set()
        for entry in evidence_index:
            if isinstance(entry, dict):
                eid = entry.get("id") or entry.get("evidence_id")
                if eid:
                    valid_ids.add(str(eid))
            elif isinstance(entry, str):
                valid_ids.add(entry)

        # 📋 Collect all candidate entries from both ``reasoning`` and
        # ``claims`` so we cover every output format an LLM may emit.
        candidate_entries: list[tuple[str, dict[str, Any]]] = []
        for field in ("reasoning", "claims"):
            for i, entry in enumerate(result.get(field, [])):
                if isinstance(entry, dict):
                    candidate_entries.append((f"{field}[{i}]", entry))

        # 🔍 Detect orphan evidence_id references
        orphans: list[str] = []
        seen_orphan_ids: set[str] = set()
        for label, entry in candidate_entries:
            claim_text = entry.get("claim", label)
            for eid in entry.get("evidence_ids", []):
                eid_str = str(eid)
                if eid_str not in valid_ids:
                    orphans.append(
                        f"Claim '{claim_text}' references non-existent evidence"
                        f" '{eid_str}'"
                    )
                    seen_orphan_ids.add(eid_str)

        # 📢 Prepend a summary line listing the distinct orphan IDs so
        # callers can quickly identify which IDs are hallucinated.
        if seen_orphan_ids:
            id_list = ", ".join(sorted(seen_orphan_ids))
            orphans.insert(0, f"orphan_evidence_ids: [{id_list}]")

        # 🔍 Detect unreferenced evidence items (collected but never cited)
        if valid_ids:
            referenced_ids: set[str] = set()
            for _, entry in candidate_entries:
                for eid in entry.get("evidence_ids", []):
                    referenced_ids.add(str(eid))

            unreferenced = valid_ids - referenced_ids
            if unreferenced:
                pct = len(unreferenced) / len(valid_ids) * 100
                orphans.append(
                    f"unreferenced_evidence: {len(unreferenced)}/{len(valid_ids)} "
                    f"items ({pct:.0f}%) collected but never cited in claims"
                )

        return orphans

    def _check_cross_references(
        self,
        result: dict[str, Any],
    ) -> list[str]:
        """Detect cross-reference inconsistencies 🔄.

        Optional deeper analysis that compares claims across
        reasoning entries for internal consistency.

        Args:
            result: Result dictionary to analyse.

        Returns:
            List of inconsistency descriptions.
        """
        # 🔄 Basic cross-reference check:
        # look for contradictory claims in reasoning
        reasoning = result.get("reasoning", [])
        inconsistencies: list[str] = []

        claims: list[str] = []
        for entry in reasoning:
            if isinstance(entry, dict):
                claim = entry.get("claim", "")
                if claim:
                    claims.append(claim)

        # ⚠️ Simple duplicate detection as a baseline
        seen: set[str] = set()
        for claim in claims:
            normalized = claim.strip().lower()
            if normalized in seen:
                inconsistencies.append(f"Duplicate claim detected: '{claim}'")
            seen.add(normalized)

        return inconsistencies

    def _check_source_diversity(
        self,
        result: dict[str, Any],
    ) -> tuple[list[str], list[str], float | None]:
        """Check evidence source diversity with graduated assessment 🌐.

        Performs a multi-tier diversity assessment:
        1. Hard fail when unique source count < min_source_count.
        2. Apply graduated confidence caps by source count.
        3. Assess distribution balance (no single source > 80%).

        Args:
            result: Result dictionary with ``evidence_index`` key.

        Returns:
            Tuple of (hard_failures, soft_warnings, confidence_cap).
            Empty lists and None cap when diversity is sufficient.
        """
        evidence_index = result.get("evidence_index", [])
        source_counts: dict[str, int] = {}

        # 📊 Count evidence items per source
        for entry in evidence_index:
            if isinstance(entry, dict):
                source = entry.get("source", "")
                if source:
                    source_counts[source] = source_counts.get(source, 0) + 1

        total_evidence = sum(source_counts.values())
        min_required = self.config.min_source_count
        caps = self.config.source_diversity_confidence_caps
        hard_failures: list[str] = []
        warnings: list[str] = []
        confidence_cap: float | None = None
        unique_source_count = len(source_counts)

        # 🔴 Hard failure: insufficient source diversity
        if unique_source_count < min_required:
            hard_failures.append(
                f"Insufficient source diversity: "
                f"{unique_source_count} source(s), "
                f"minimum {min_required} required"
            )

        # 📉 Graduated confidence caps by source count
        if unique_source_count <= 1:
            confidence_cap = caps.get("no_diversity", 0.49)
            warnings.append(
                f"no_source_diversity: {unique_source_count} "
                f"source(s), minimum {min_required} required"
            )
            return hard_failures, warnings, confidence_cap

        # 🟠 Tier 2: Below minimum meaningful source count
        meaningful_sources = {
            s: c
            for s, c in source_counts.items()
            if c >= self.config.min_evidence_per_source
        }
        if len(meaningful_sources) < min_required:
            warnings.append(
                f"low_source_diversity: "
                f"{len(meaningful_sources)} meaningful "
                f"source(s) < {min_required} required"
            )
            confidence_cap = caps.get("low_diversity", 0.59)

        # 🟡 Tier 3: Unbalanced distribution (single source > 80%)
        if total_evidence > 0:
            for source, count in source_counts.items():
                ratio = count / total_evidence
                if ratio > 0.80:
                    warnings.append(
                        f"source_imbalance: '{source}' provides "
                        f"{ratio:.0%} of all evidence"
                    )
                    tier3_cap = caps.get("moderate_diversity", 0.69)
                    if confidence_cap is None or tier3_cap < confidence_cap:
                        confidence_cap = tier3_cap

        return hard_failures, warnings, confidence_cap

    def _check_evidence_urls(
        self,
        result: dict[str, Any],
    ) -> tuple[list[str], float]:
        """Check that evidence items include valid URLs 🔗.

        Each evidence item missing a URL incurs a confidence penalty.
        The total penalty is capped at ``url_missing_penalty_max``.

        Args:
            result: Result dictionary with ``evidence_index`` key.

        Returns:
            Tuple of (warning_messages, total_penalty).
            Empty warnings and 0.0 penalty when all URLs are present.
        """
        evidence_index = result.get("evidence_index", [])
        per_penalty = self.config.url_missing_penalty
        max_penalty = self.config.url_missing_penalty_max
        warnings: list[str] = []
        missing_count = 0

        for entry in evidence_index:
            if not isinstance(entry, dict):
                continue
            eid = entry.get("id") or entry.get("evidence_id") or "unknown"
            url = entry.get("url")
            if not url or (isinstance(url, str) and not url.strip()):
                missing_count += 1
                warnings.append(
                    f"Evidence {eid} missing URL — confidence penalty applied"
                )

        total_penalty = min(missing_count * per_penalty, max_penalty)
        return warnings, total_penalty

    # -- Confidence Breakdown ------------------------------------------------

    # 📊 Dimension weights for overall composite score
    _DIMENSION_WEIGHTS: dict[str, float] = {
        "strength": 0.3,
        "coverage": 0.3,
        "consistency": 0.25,
        "recency": 0.15,
    }

    # 🏷️ Quality label → numeric score mapping
    _QUALITY_SCORES: dict[str, float] = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.4,
    }

    _DEFAULT_QUALITY_SCORE: float = 0.5

    def compute_confidence_breakdown(
        self,
        result_data: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
    ) -> ConfidenceBreakdown:
        """Compute multi-dimensional confidence breakdown 📊.

        Analyzes the result data across 4 quality dimensions and
        produces a weighted overall score. Each dimension yields
        a value in [0.0, 1.0].

        Args:
            result_data: The agent's output data dictionary.
            output_schema: JSON Schema for reference (unused
                for now, reserved for future schema-aware
                dimension weighting).

        Returns:
            ConfidenceBreakdown with all dimensions computed.
        """
        strength = self._compute_evidence_strength(result_data)
        coverage = self._compute_evidence_coverage(result_data)
        consistency = self._compute_evidence_consistency(result_data)
        recency = self._compute_information_recency(result_data)

        w = self._DIMENSION_WEIGHTS
        overall = (
            w["strength"] * strength
            + w["coverage"] * coverage
            + w["consistency"] * consistency
            + w["recency"] * recency
        )
        # 🔒 Clamp to [0.0, 1.0] for safety
        overall = max(0.0, min(1.0, overall))

        self._logger.debug(
            "📊 Confidence breakdown: strength=%.2f "
            "coverage=%.2f consistency=%.2f recency=%.2f "
            "overall=%.2f",
            strength,
            coverage,
            consistency,
            recency,
            overall,
        )

        return ConfidenceBreakdown(
            evidence_strength=strength,
            evidence_coverage=coverage,
            evidence_consistency=consistency,
            information_recency=recency,
            overall=overall,
        )

    def _compute_evidence_strength(
        self,
        result_data: dict[str, Any],
    ) -> float:
        """Compute evidence strength from quality labels 💪.

        Averages the numeric score mapped from each evidence
        item's ``quality_label``. Returns 0.0 when no evidence
        is present.

        Args:
            result_data: Agent output with ``evidence_index``.

        Returns:
            Average quality score in [0.0, 1.0].
        """
        evidence_index = result_data.get("evidence_index", [])
        if not evidence_index:
            return 0.0

        total = 0.0
        count = 0
        for entry in evidence_index:
            if not isinstance(entry, dict):
                continue
            label = (entry.get("quality_label") or "").lower()
            score = self._QUALITY_SCORES.get(label, self._DEFAULT_QUALITY_SCORE)
            total += score
            count += 1

        if count == 0:
            return 0.0
        return total / count

    def _compute_evidence_coverage(
        self,
        result_data: dict[str, Any],
    ) -> float:
        """Compute coverage dimension from checklist data 📋.

        Uses ``checklist_coverage`` if available (ratio of
        covered vs. total required items). Falls back to an
        evidence count heuristic: min(evidence_count / 5, 1.0).

        Args:
            result_data: Agent output with
                ``checklist_coverage`` and/or
                ``evidence_index``.

        Returns:
            Coverage ratio in [0.0, 1.0].
        """
        coverage_data = result_data.get("checklist_coverage", {})

        if isinstance(coverage_data, dict):
            covered = coverage_data.get(
                "required_covered",
                coverage_data.get("covered", []),
            )
            missing = coverage_data.get(
                "required_missing",
                coverage_data.get("missing", []),
            )
            total = len(covered) + len(missing)
            if total > 0:
                return len(covered) / total

        # 📐 Fallback heuristic: scale by evidence count
        evidence_count = len(result_data.get("evidence_index", []))
        return min(evidence_count / 5.0, 1.0)

    def _compute_evidence_consistency(
        self,
        result_data: dict[str, Any],
    ) -> float:
        """Compute consistency from gaps and doubts ⚖️.

        Starts at 1.0 (fully consistent). Each doubt reduces
        the score by 0.1, and each gap reduces it by 0.05.
        The result is clamped to [0.0, 1.0].

        Args:
            result_data: Agent output with
                ``doubts_remaining`` and ``gaps_remaining``.

        Returns:
            Consistency score in [0.0, 1.0].
        """
        doubts = result_data.get("doubts_remaining", [])
        gaps = result_data.get("gaps_remaining", [])

        score = 1.0
        score -= len(doubts) * 0.1
        score -= len(gaps) * 0.05
        return max(0.0, min(1.0, score))

    def _compute_information_recency(
        self,
        result_data: dict[str, Any],
    ) -> float:
        """Compute information recency from evidence URLs 🕐.

        Default baseline is 0.8. Evidence items that include
        a non-empty ``url`` field each add a 0.1 bonus (once
        applied, the score caps at 1.0).

        Args:
            result_data: Agent output with ``evidence_index``.

        Returns:
            Recency score in [0.0, 1.0].
        """
        evidence_index = result_data.get("evidence_index", [])
        if not evidence_index:
            return 0.8

        url_count = 0
        for entry in evidence_index:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            if url and isinstance(url, str) and url.strip():
                url_count += 1

        bonus = url_count * 0.1
        return min(0.8 + bonus, 1.0)
