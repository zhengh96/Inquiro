"""Tests for checklist coverage tracking in QualityGate 🧪.

Covers QualityGate._check_coverage: reads both new and legacy
field names correctly.

Each test is independent and follows the Arrange-Act-Assert pattern.
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
# 🏗️ Helpers
# ============================================================


def _make_coverage_qg(
    coverage_threshold: float = 0.80,
) -> QualityGate:
    """Create QualityGate with only coverage check enabled 🔧.

    Args:
        coverage_threshold: Minimum coverage ratio (0.0-1.0).

    Returns:
        QualityGate configured for coverage-only testing.
    """
    checks = QualityGateChecksConfig(
        schema_validation=False,
        coverage_check=True,
        evidence_reference_check=False,
    )
    config = QualityGateConfig(
        checks=checks,
        coverage_threshold=coverage_threshold,
    )
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    return QualityGate(config, schema)


# ============================================================
# 📊 Tests: QualityGate Coverage Check
# ============================================================


class TestQGCoverageCheck:
    """Tests for QualityGate._check_coverage with new and legacy field names 📊."""

    def test_coverage_check_passes_with_sufficient_coverage(
        self,
    ) -> None:
        """All required items covered -> coverage=1.0, no soft failure ✅.

        _check_coverage with required_covered=["R1","R2","R3"] and
        checklist required_items=["R1","R2","R3"] should yield 1.0.
        """
        # Arrange
        qg = _make_coverage_qg(coverage_threshold=0.80)
        result_data: dict[str, Any] = {
            "checklist_coverage": {
                "required_covered": ["R1", "R2", "R3"],
                "required_missing": [],
            },
        }
        checklist = {"required_items": ["R1", "R2", "R3"]}

        # Act
        coverage = qg._check_coverage(
            result_data,
            checklist=checklist,
        )

        # Assert
        assert coverage == 1.0

    def test_coverage_check_fails_with_low_coverage(self) -> None:
        """1 of 4 items covered -> coverage=0.25, soft failure cap 0.69 ⚠️.

        _check_coverage with required_covered=["R1"] and
        checklist required_items=["R1","R2","R3","R4"] yields 0.25,
        which is below the 0.80 threshold and triggers a soft failure.
        """
        # Arrange
        qg = _make_coverage_qg(coverage_threshold=0.80)
        result_data: dict[str, Any] = {
            "checklist_coverage": {
                "required_covered": ["R1"],
                "required_missing": ["R2", "R3", "R4"],
            },
        }
        checklist = {"required_items": ["R1", "R2", "R3", "R4"]}

        # Act
        coverage = qg._check_coverage(
            result_data,
            checklist=checklist,
        )
        qg_result = qg.validate(result_data, checklist=checklist)

        # Assert
        assert coverage == pytest.approx(0.25)
        assert any("low_coverage" in f for f in qg_result.soft_failures)
        assert qg_result.confidence_cap == 0.69

    def test_coverage_check_at_threshold(self) -> None:
        """4 of 5 items -> coverage=0.80 -> PASSES (NOT < 0.80) ✅.

        Boundary test: coverage exactly at threshold should NOT trigger
        a soft failure. The comparison uses strict less-than (<).
        """
        # Arrange
        qg = _make_coverage_qg(coverage_threshold=0.80)
        result_data: dict[str, Any] = {
            "checklist_coverage": {
                "required_covered": [
                    "R1",
                    "R2",
                    "R3",
                    "R4",
                ],
                "required_missing": ["R5"],
            },
        }
        checklist = {
            "required_items": [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
            ],
        }

        # Act
        coverage = qg._check_coverage(
            result_data,
            checklist=checklist,
        )
        qg_result = qg.validate(result_data, checklist=checklist)

        # Assert
        assert coverage == pytest.approx(0.80)
        assert not any("low_coverage" in f for f in qg_result.soft_failures)
        assert qg_result.confidence_cap is None

    def test_field_name_compatibility(self) -> None:
        """_check_coverage reads both new and legacy field name styles 🔄.

        Both required_covered/required_missing and legacy covered/missing
        conventions should produce the same coverage when the underlying
        data is equivalent.
        """
        # Arrange
        qg = _make_coverage_qg(coverage_threshold=0.80)

        # ✅ New-style: required_covered / required_missing
        new_style: dict[str, Any] = {
            "checklist_coverage": {
                "required_covered": ["R1", "R2", "R3"],
                "required_missing": [],
            },
        }
        # ✅ Legacy-style: covered / missing
        legacy_style: dict[str, Any] = {
            "checklist_coverage": {
                "covered": ["R1", "R2", "R3"],
                "missing": [],
            },
        }

        # Act
        coverage_new = qg._check_coverage(new_style)
        coverage_legacy = qg._check_coverage(legacy_style)

        # Assert — both styles should yield full coverage
        assert coverage_new == 1.0
        assert coverage_legacy == 1.0

    def test_coverage_check_with_legacy_field_names(self) -> None:
        """Legacy covered/missing keys still work via fallback 📊.

        Old-style results using 'covered'/'missing' should produce
        correct coverage calculations and trigger soft failures when
        coverage is below threshold.
        """
        # Arrange
        qg = _make_coverage_qg(coverage_threshold=0.80)
        result_data: dict[str, Any] = {
            "checklist_coverage": {
                "covered": ["R1"],
                "missing": ["R2", "R3", "R4"],
            },
        }

        # Act
        coverage = qg._check_coverage(result_data)
        qg_result = qg.validate(result_data)

        # Assert — 1/4 = 0.25 < 0.80 -> soft failure
        assert coverage == pytest.approx(0.25)
        assert any("low_coverage" in f for f in qg_result.soft_failures)
        assert qg_result.confidence_cap == 0.69


# 📊 Note: TestBuildResult (ResearchExp._build_result) was removed along
# with ResearchExp. Checklist coverage parsing in EvaluationResult is
# validated through the AnalysisExp and DiscoveryLoop test suites.
