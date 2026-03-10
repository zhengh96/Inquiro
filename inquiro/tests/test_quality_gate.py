"""Tests for Inquiro QualityGate 🧪.

Tests the deterministic output quality validation system:
- Schema validation (hard fail → retry)
- Required field completeness (hard fail → retry)
- Search checklist coverage (soft fail → confidence cap)
- Evidence reference integrity (soft fail → confidence cap)
- Combined check interactions
- Retry behavior on hard failures
"""

from __future__ import annotations

from typing import Any

import pytest

from inquiro.infrastructure.quality_gate import (
    QualityGate,
    QualityGateChecksConfig,
    QualityGateConfig,
)


# ============================================================
# 🏗️ Fixtures
# ============================================================


@pytest.fixture
def sample_output_schema() -> dict[str, Any]:
    """A JSON Schema representing a typical evaluation result 📝."""
    return {
        "type": "object",
        "required": ["decision", "confidence", "reasoning", "evidence_index"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve", "reject", "escalate"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reasoning": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "evidence_index": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "source": {"type": "string"},
                    },
                },
            },
            "checklist_coverage": {
                "type": "object",
                "properties": {
                    "covered": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "missing": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    }


@pytest.fixture
def sample_evaluation_result() -> dict[str, Any]:
    """A valid evaluation result that passes all checks ✅."""
    return {
        "decision": "approve",
        "confidence": 0.85,
        "reasoning": [
            {
                "claim": "Drug shows efficacy in Phase II trials",
                "evidence_ids": ["E1", "E2"],
            },
            {
                "claim": "Safety profile is acceptable",
                "evidence_ids": ["E3"],
            },
        ],
        "evidence_index": [
            {"id": "E1", "source": "PubMed:12345"},
            {"id": "E2", "source": "ClinicalTrials:NCT001"},
            {"id": "E3", "source": "FDA:Safety-Report-001"},
        ],
        "checklist_coverage": {
            "covered": ["efficacy", "safety", "mechanism"],
            "missing": [],
        },
    }


def _make_qg(
    schema: dict[str, Any],
    checks: QualityGateChecksConfig | None = None,
    coverage_threshold: float = 0.80,
) -> QualityGate:
    """Create QualityGate with sensible test defaults 🔧."""
    config = QualityGateConfig(
        checks=checks or QualityGateChecksConfig(),
        coverage_threshold=coverage_threshold,
    )
    return QualityGate(config, schema)


# ============================================================
# ✅ Schema Validation Tests (Hard Fail)
# ============================================================


class TestSchemaValidation:
    """Tests for QualityGate schema validation checks ✅."""

    def test_valid_result_passes_schema(
        self,
        sample_output_schema: dict[str, Any],
        sample_evaluation_result: dict[str, Any],
    ) -> None:
        """Result conforming to schema should pass validation."""
        qg = _make_qg(sample_output_schema)
        result = qg.validate(sample_evaluation_result)
        assert result.passed is True
        assert len(result.hard_failures) == 0

    def test_missing_required_field_hard_fails(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Missing required field should trigger hard failure."""
        # ❌ Result missing "decision" field
        incomplete = {
            "confidence": 0.5,
            "reasoning": [],
            "evidence_index": [],
        }
        qg = _make_qg(sample_output_schema)
        result = qg.validate(incomplete)
        assert result.passed is False
        assert any("schema_invalid" in f for f in result.hard_failures)

    def test_wrong_type_hard_fails(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Wrong field type should trigger hard failure."""
        # ❌ confidence should be number, not string
        bad_type = {
            "decision": "approve",
            "confidence": "high",
            "reasoning": [],
            "evidence_index": [],
        }
        qg = _make_qg(sample_output_schema)
        result = qg.validate(bad_type)
        assert result.passed is False

    def test_invalid_enum_value_hard_fails(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Invalid enum value should trigger hard failure."""
        # ❌ "maybe" is not in ["approve", "reject", "escalate"]
        bad_enum = {
            "decision": "maybe",
            "confidence": 0.5,
            "reasoning": [],
            "evidence_index": [],
        }
        qg = _make_qg(sample_output_schema)
        result = qg.validate(bad_enum)
        assert result.passed is False

    def test_empty_result_hard_fails(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Empty result dict should trigger hard failure."""
        qg = _make_qg(sample_output_schema)
        result = qg.validate({})
        assert result.passed is False

    def test_schema_validation_disabled(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """When schema_validation=False, invalid schema should not fail."""
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(sample_output_schema, checks=checks)
        # ✨ Result with minimal evidence passes when schema check disabled
        result = qg.validate(
            {
                "evidence_index": [{"id": "E1", "source": "test"}],
                "search_rounds": 1,
            }
        )
        assert result.passed is True


# ============================================================
# 📊 Coverage Check Tests (Soft Fail)
# ============================================================


class TestCoverageCheck:
    """Tests for QualityGate coverage check (soft fail) 📊."""

    def test_full_coverage_passes(self) -> None:
        """Result covering all required checklist items should pass."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "checklist_coverage": {
                "covered": ["a", "b", "c", "d", "e"],
                "missing": [],
            },
        }
        result = qg.validate(result_data)
        assert not any("low_coverage" in f for f in result.soft_failures)

    def test_low_coverage_soft_fails(self) -> None:
        """Coverage below threshold should trigger soft failure."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks, coverage_threshold=0.8)
        # 📊 2 covered, 2 missing = 50% < 80%
        result_data = {
            "checklist_coverage": {
                "covered": ["a", "b"],
                "missing": ["c", "d"],
            },
        }
        result = qg.validate(result_data)
        assert any("low_coverage" in f for f in result.soft_failures)

    def test_low_coverage_caps_confidence(self) -> None:
        """Low coverage soft failure should cap confidence at 0.69."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks, coverage_threshold=0.8)
        result_data = {
            "checklist_coverage": {
                "covered": ["a"],
                "missing": ["b", "c", "d"],
            },
        }
        result = qg.validate(result_data)
        assert result.confidence_cap == 0.69

    def test_coverage_check_disabled(self) -> None:
        """When coverage_check=False, low coverage should not fail."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "checklist_coverage": {
                "covered": [],
                "missing": ["a", "b", "c"],
            },
        }
        result = qg.validate(result_data)
        assert not any("low_coverage" in f for f in result.soft_failures)

    def test_empty_checklist_full_coverage(self) -> None:
        """Empty required checklist should count as 100% coverage."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "checklist_coverage": {
                "covered": [],
                "missing": [],
            },
        }
        result = qg.validate(result_data)
        assert not any("low_coverage" in f for f in result.soft_failures)


# ============================================================
# 🔗 Evidence Reference Integrity Tests (Soft Fail)
# ============================================================


class TestEvidenceReferenceCheck:
    """Tests for QualityGate evidence reference integrity 🔗."""

    def test_all_references_valid(
        self,
        sample_evaluation_result: dict[str, Any],
    ) -> None:
        """Result with all valid evidence references should pass."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result = qg.validate(sample_evaluation_result)
        assert not any("orphan_claims" in f for f in result.soft_failures)

    def test_orphan_claim_soft_fails(self) -> None:
        """Claim referencing non-existent evidence should soft fail."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "reasoning": [
                {"claim": "Test claim", "evidence_ids": ["E99"]},
            ],
            "evidence_index": [
                {"id": "E1", "source": "PubMed:12345"},
            ],
        }
        result = qg.validate(result_data)
        assert any("orphan_claims" in f for f in result.soft_failures)

    def test_orphan_claims_cap_confidence(self) -> None:
        """Orphan claims should cap confidence at 0.69."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "reasoning": [
                {"claim": "Orphan claim", "evidence_ids": ["E99"]},
            ],
            "evidence_index": [],
        }
        result = qg.validate(result_data)
        assert result.confidence_cap == 0.69

    def test_evidence_check_disabled(self) -> None:
        """When evidence_reference_check=False, orphans should not fail."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "reasoning": [
                {"claim": "Orphan claim", "evidence_ids": ["E99"]},
            ],
            "evidence_index": [],
        }
        result = qg.validate(result_data)
        assert not any("orphan_claims" in f for f in result.soft_failures)

    def test_orphan_ids_listed_in_risk_flags(self) -> None:
        """Orphan IDs should appear in risk_flags for easy identification."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "reasoning": [
                {"claim": "Some claim", "evidence_ids": ["E1", "E99"]},
            ],
            "evidence_index": [{"id": "E1", "source": "PubMed"}],
        }
        result = qg.validate(result_data)
        # 🔍 The summary line must mention the orphan ID
        assert any("E99" in flag for flag in result.risk_flags)

    def test_orphan_in_claims_field_detected(self) -> None:
        """Orphan evidence_ids in a 'claims' field should be caught."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            # 🧠 AnalysisExp consensus output uses 'claims' not 'reasoning'
            "claims": [
                {"claim": "Hallucinated cite", "evidence_ids": ["GHOST1"]},
            ],
            "evidence_index": [{"id": "E1", "source": "ArXiv"}],
        }
        result = qg.validate(result_data)
        assert any("orphan_claims" in f for f in result.soft_failures)
        assert any("GHOST1" in flag for flag in result.risk_flags)

    def test_orphan_in_both_reasoning_and_claims(self) -> None:
        """Orphans across both 'reasoning' and 'claims' are all reported."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "reasoning": [{"claim": "R1", "evidence_ids": ["MISS_A"]}],
            "claims": [{"claim": "C1", "evidence_ids": ["MISS_B"]}],
            "evidence_index": [{"id": "E1", "source": "Nature"}],
        }
        result = qg.validate(result_data)
        # Both orphan IDs must surface somewhere in risk_flags
        all_flags = " ".join(result.risk_flags)
        assert "MISS_A" in all_flags
        assert "MISS_B" in all_flags

    def test_no_orphans_when_all_ids_valid_in_claims(self) -> None:
        """No soft failure when all claims' evidence_ids exist."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "claims": [{"claim": "Valid claim", "evidence_ids": ["E1", "E2"]}],
            "evidence_index": [
                {"id": "E1", "source": "PubMed"},
                {"id": "E2", "source": "Nature"},
            ],
        }
        result = qg.validate(result_data)
        assert not any("orphan_claims" in f for f in result.soft_failures)


# ============================================================
# 🔄 Combined Check & Retry Tests
# ============================================================


class TestCombinedChecks:
    """Tests for QualityGate combined check behavior 🔄."""

    def test_hard_failure_overrides_soft_pass(self) -> None:
        """Hard failure should cause overall failure even if soft checks pass."""
        schema: dict[str, Any] = {
            "type": "object",
            "required": ["decision"],
            "properties": {
                "decision": {"type": "string"},
            },
        }
        checks = QualityGateChecksConfig(
            schema_validation=True,
            coverage_check=False,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        # ❌ Missing required "decision" field
        result = qg.validate({})
        assert result.passed is False

    def test_soft_failures_only_still_passes(self) -> None:
        """Only soft failures should still result in overall pass."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=True,
            evidence_reference_check=False,
        )
        qg = _make_qg(schema, checks=checks, coverage_threshold=0.8)
        result_data = {
            "checklist_coverage": {
                "covered": ["a"],
                "missing": ["b", "c", "d"],
            },
            "evidence_index": [{"id": "E1", "source": "test"}],
            "search_rounds": 1,
        }
        result = qg.validate(result_data)
        # ✅ Passed overall (no hard failures)
        assert result.passed is True
        # ⚠️ But has soft failures
        assert len(result.soft_failures) > 0

    def test_multiple_soft_failures_minimum_cap(self) -> None:
        """Multiple soft failures should use the minimum confidence cap."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=True,
            evidence_reference_check=True,
        )
        qg = _make_qg(schema, checks=checks, coverage_threshold=0.8)
        # 📊 Both low coverage AND orphan claims
        result_data = {
            "checklist_coverage": {
                "covered": ["a"],
                "missing": ["b", "c", "d"],
            },
            "reasoning": [
                {"claim": "Test", "evidence_ids": ["E99"]},
            ],
            "evidence_index": [],
        }
        result = qg.validate(result_data)
        # 🎯 Both soft failures present
        assert len(result.soft_failures) == 2
        # Confidence cap should be min(0.69, 0.69) = 0.69
        assert result.confidence_cap == 0.69

    def test_hard_failure_triggers_retry_path(self) -> None:
        """Hard failure should indicate retry is needed."""
        schema: dict[str, Any] = {
            "type": "object",
            "required": ["decision"],
            "properties": {
                "decision": {"type": "string", "enum": ["yes", "no"]},
            },
        }
        qg = _make_qg(schema)
        # ❌ Schema error: decision="maybe" not in enum
        result = qg.validate({"decision": "maybe"})
        assert result.passed is False
        assert len(result.hard_failures) > 0

    def test_no_checks_enabled_always_passes(self) -> None:
        """When all checks disabled, any result should pass (with evidence)."""
        schema: dict[str, Any] = {
            "type": "object",
            "required": ["foo"],
            "properties": {"foo": {"type": "string"}},
        }
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            cross_reference_check=False,
            evidence_url_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        # ✨ Include minimal evidence to avoid silent failure detection
        result = qg.validate(
            {
                "totally": "invalid",
                "evidence_index": [{"id": "E1", "source": "test"}],
                "search_rounds": 1,
            }
        )
        assert result.passed is True
        assert len(result.hard_failures) == 0
        assert len(result.soft_failures) == 0


# ============================================================
# 🌐 Source Diversity Check Tests (Soft Fail)
# ============================================================


class TestSourceDiversityCheck:
    """Tests for QualityGate source diversity check 🌐."""

    def test_diverse_sources_passes(self) -> None:
        """Multiple different sources should pass diversity check ✅."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
                {"id": "E2", "source": "perplexity"},
                {"id": "E3", "source": "opentargets"},
            ],
        }
        result = qg.validate(result_data)
        assert not any(
            "source_diversity" in f or "source_imbalance" in f
            for f in result.soft_failures
        )

    def test_single_source_soft_fails(self) -> None:
        """Single source should trigger no_source_diversity ❌."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
                {"id": "E2", "source": "biomcp"},
            ],
        }
        result = qg.validate(result_data)
        assert any("no_source_diversity" in f for f in result.soft_failures)

    def test_source_diversity_disabled(self) -> None:
        """When source_diversity_check=False, single source should not fail ✅."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=False,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
            ],
        }
        result = qg.validate(result_data)
        assert not any(
            "source_diversity" in f or "source_imbalance" in f
            for f in result.soft_failures
        )

    def test_source_diversity_caps_confidence(self) -> None:
        """Single source should cap confidence at 0.49 (no_diversity) 🔴."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        qg = _make_qg(schema, checks=checks)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
            ],
        }
        result = qg.validate(result_data)
        assert result.confidence_cap == 0.49

    def test_custom_min_source_count(self) -> None:
        """Custom min_source_count=3 with 2 sources should soft fail ✅."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=3,
        )
        qg = QualityGate(config, schema)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
                {"id": "E2", "source": "perplexity"},
            ],
        }
        result = qg.validate(result_data)
        assert any("low_source_diversity" in f for f in result.soft_failures)

    def test_empty_evidence_fails_diversity(self) -> None:
        """Empty evidence index should fail diversity check ❌."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        qg = _make_qg(schema, checks=checks)
        result_data: dict[str, Any] = {"evidence_index": []}
        result = qg.validate(result_data)
        assert any("no_source_diversity" in f for f in result.soft_failures)

    # ================================================================
    # 🧪 New: Graduated source diversity tests
    # ================================================================

    def test_no_diversity_caps_at_049(self) -> None:
        """Single source should trigger no_diversity cap at 0.49 🔴."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=2,
        )
        qg = QualityGate(config, schema)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
                {"id": "E2", "source": "biomcp"},
                {"id": "E3", "source": "biomcp"},
            ],
        }
        result = qg.validate(result_data)
        assert result.confidence_cap == 0.49
        assert any("no_source_diversity" in f for f in result.soft_failures)

    def test_low_diversity_caps_at_059(self) -> None:
        """Below min_source_count should trigger low_diversity cap 🟠."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=3,
        )
        qg = QualityGate(config, schema)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "biomcp"},
                {"id": "E2", "source": "biomcp"},
                {"id": "E3", "source": "perplexity"},
                {"id": "E4", "source": "perplexity"},
            ],
        }
        result = qg.validate(result_data)
        assert result.confidence_cap == 0.59
        assert any("low_source_diversity" in f for f in result.soft_failures)

    def test_imbalanced_source_flags_warning(self) -> None:
        """Single source providing >80% evidence should flag imbalance 🟡."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=2,
        )
        qg = QualityGate(config, schema)
        # 📊 9 from perplexity, 1 from biomcp = 90% imbalance
        result_data = {
            "evidence_index": [
                {"id": f"E{i}", "source": "perplexity"} for i in range(9)
            ]
            + [{"id": "E10", "source": "biomcp"}],
        }
        result = qg.validate(result_data)
        assert any("source_imbalance" in f for f in result.soft_failures)
        assert result.confidence_cap == 0.69

    def test_meaningful_source_threshold(self) -> None:
        """Source below min_evidence_per_source should not count 📊."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=3,
            min_evidence_per_source=2,
        )
        qg = QualityGate(config, schema)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "perplexity"},
                {"id": "E2", "source": "perplexity"},
                {"id": "E3", "source": "biomcp"},
                {"id": "E4", "source": "biomcp"},
                # ⚠️ opentargets has only 1 item, below threshold
                {"id": "E5", "source": "opentargets"},
            ],
        }
        result = qg.validate(result_data)
        # Only 2 meaningful sources (perplexity, biomcp) < 3 required
        assert any("low_source_diversity" in f for f in result.soft_failures)

    def test_balanced_sources_passes_fully(self) -> None:
        """Evenly distributed sources should pass with no cap ✅."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        checks = QualityGateChecksConfig(
            schema_validation=False,
            coverage_check=False,
            evidence_reference_check=False,
            source_diversity_check=True,
            evidence_url_check=False,
        )
        config = QualityGateConfig(
            checks=checks,
            min_source_count=3,
        )
        qg = QualityGate(config, schema)
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "perplexity"},
                {"id": "E2", "source": "perplexity"},
                {"id": "E3", "source": "biomcp"},
                {"id": "E4", "source": "biomcp"},
                {"id": "E5", "source": "opentargets"},
                {"id": "E6", "source": "opentargets"},
            ],
        }
        result = qg.validate(result_data)
        assert result.confidence_cap is None
        assert not any(
            "source_diversity" in f or "source_imbalance" in f
            for f in result.soft_failures
        )


# ============================================================
# 🔍 Unreferenced Evidence Detection Tests
# ============================================================


class TestUnreferencedEvidenceDetection:
    """Tests for detecting evidence items that are collected but never cited 🔍."""

    def test_unreferenced_evidence_detected(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Unreferenced evidence items produce a warning 📢."""
        qg = QualityGate(
            config=QualityGateConfig(
                checks=QualityGateChecksConfig(
                    schema_validation=False,
                    evidence_reference_check=True,
                    source_diversity_check=False,
                    evidence_url_check=False,
                ),
            ),
            output_schema=sample_output_schema,
        )
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "test"},
                {"id": "E2", "source": "test"},
                {"id": "E3", "source": "test"},
            ],
            "reasoning": [
                {"claim": "Only uses E1", "evidence_ids": ["E1"]},
            ],
            "search_rounds": 1,
        }
        result = qg.validate(result_data)
        unreferenced = [
            f for f in result.risk_flags if "unreferenced_evidence" in f
        ]
        assert len(unreferenced) == 1
        assert "2/3" in unreferenced[0]
        assert "67%" in unreferenced[0]

    def test_no_unreferenced_when_all_cited(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """No warning when all evidence items are cited 📢."""
        qg = QualityGate(
            config=QualityGateConfig(
                checks=QualityGateChecksConfig(
                    schema_validation=False,
                    evidence_reference_check=True,
                    source_diversity_check=False,
                    evidence_url_check=False,
                ),
            ),
            output_schema=sample_output_schema,
        )
        result_data = {
            "evidence_index": [
                {"id": "E1", "source": "test"},
                {"id": "E2", "source": "test"},
            ],
            "reasoning": [
                {"claim": "Claim A", "evidence_ids": ["E1"]},
                {"claim": "Claim B", "evidence_ids": ["E2"]},
            ],
            "search_rounds": 1,
        }
        result = qg.validate(result_data)
        unreferenced = [
            f for f in result.risk_flags if "unreferenced_evidence" in f
        ]
        assert len(unreferenced) == 0

    def test_no_unreferenced_warning_with_empty_evidence(
        self,
        sample_output_schema: dict[str, Any],
    ) -> None:
        """No unreferenced warning when evidence_index is empty 📢."""
        qg = QualityGate(
            config=QualityGateConfig(
                checks=QualityGateChecksConfig(
                    schema_validation=False,
                    evidence_reference_check=True,
                    source_diversity_check=False,
                    evidence_url_check=False,
                ),
            ),
            output_schema=sample_output_schema,
        )
        result_data: dict[str, Any] = {
            "evidence_index": [],
            "reasoning": [],
            "search_rounds": 1,
            "error": "no data",
        }
        result = qg.validate(result_data)
        unreferenced = [
            f for f in result.risk_flags if "unreferenced_evidence" in f
        ]
        assert len(unreferenced) == 0
