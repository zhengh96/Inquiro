"""Tests for LLMCoverageJudge — LLM-based coverage assessment 🧪.

Covers:
- Successful LLM judgment with valid JSON response
- Fallback to MockCoverageJudge on LLM error
- Evidence truncation (claims-referenced only, top-20 cap)
- Reconciliation of LLM output with original checklist
- Empty inputs (empty checklist, empty claims)
- Cost tracking in CoverageResult
- GapAnalysis integration with coverage_judge_mode

Each test is independent and follows Arrange → Act → Assert.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inquiro.core.gap_analysis import CoverageResult, GapAnalysis
from inquiro.core.llm_coverage_judge import LLMCoverageJudge


# ============================================================================
# 🏗️ Helpers
# ============================================================================


def _make_mock_llm_pool(response_content: str):
    """Build a mock LLMProviderPool that returns fixed content 🔧."""
    mock_response = MagicMock()
    mock_response.content = response_content
    mock_response.meta = {"usage": {"prompt_tokens": 500, "completion_tokens": 100}}

    mock_llm = MagicMock()
    mock_llm.query.return_value = mock_response

    mock_pool = MagicMock()
    mock_pool.get_llm.return_value = mock_llm
    return mock_pool


def _make_evidence(
    ev_id: str,
    summary: str = "Test evidence summary",
    tag: str = "academic",
    quality: str = "high",
) -> MagicMock:
    """Build a mock Evidence object 🔧."""
    ev = MagicMock()
    ev.id = ev_id
    ev.summary = summary
    ev.evidence_tag = tag
    ev.quality_label = quality
    return ev


def _make_claims(*items: tuple[str, list[str]]) -> list[dict[str, Any]]:
    """Build claim dicts from (text, evidence_ids) tuples 📝."""
    return [
        {"claim": text, "evidence_ids": ids, "strength": "moderate"}
        for text, ids in items
    ]


# ============================================================================
# 🧪 LLMCoverageJudge Tests
# ============================================================================


class TestLLMCoverageJudge:
    """Tests for LLMCoverageJudge core functionality 🧪."""

    @pytest.mark.asyncio
    async def test_successful_judgment(self):
        """LLM returns valid JSON → correct CoverageResult 🎯."""
        response_json = json.dumps({
            "covered": ["Assess target druggability"],
            "uncovered": ["Evaluate safety profile"],
            "conflict_signals": [],
            "reasoning": "Druggability has strong evidence support.",
        })
        pool = _make_mock_llm_pool(response_json)
        judge = LLMCoverageJudge(llm_pool=pool, model="haiku")

        checklist = ["Assess target druggability", "Evaluate safety profile"]
        claims = _make_claims(
            ("Target shows good druggability based on structure", ["E1", "E2"]),
        )
        evidence = [_make_evidence("E1"), _make_evidence("E2")]

        result = await judge.judge_coverage(checklist, claims, evidence)

        assert "Assess target druggability" in result.covered
        assert "Evaluate safety profile" in result.uncovered
        assert result.judge_cost_usd > 0

    @pytest.mark.asyncio
    async def test_empty_checklist(self):
        """Empty checklist → empty result without LLM call 📋."""
        pool = _make_mock_llm_pool("")
        judge = LLMCoverageJudge(llm_pool=pool)

        result = await judge.judge_coverage([], [], [])

        assert result.covered == []
        assert result.uncovered == []
        pool.get_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self):
        """LLM error → fallback to MockCoverageJudge 🔄."""
        pool = MagicMock()
        pool.get_llm.side_effect = RuntimeError("LLM unavailable")
        judge = LLMCoverageJudge(llm_pool=pool)

        checklist = ["Assess efficacy data"]
        claims = _make_claims(
            ("Strong efficacy demonstrated in trials", ["E1"]),
        )
        evidence = [_make_evidence("E1")]

        result = await judge.judge_coverage(checklist, claims, evidence)

        # ✅ Should not raise, should return MockCoverageJudge result
        assert isinstance(result, CoverageResult)

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        """LLM returns invalid JSON → fallback to mock 📝."""
        pool = _make_mock_llm_pool("This is not JSON at all")
        judge = LLMCoverageJudge(llm_pool=pool)

        checklist = ["Item A"]
        claims = _make_claims(("Claim about A", ["E1"]))
        evidence = [_make_evidence("E1")]

        result = await judge.judge_coverage(checklist, claims, evidence)

        # ✅ Fallback should still produce a result
        assert isinstance(result, CoverageResult)

    @pytest.mark.asyncio
    async def test_reconciliation_missing_items(self):
        """Items missing from LLM output → marked uncovered 🔧."""
        response_json = json.dumps({
            "covered": ["Item A"],
            "uncovered": [],
            "conflict_signals": [],
        })
        pool = _make_mock_llm_pool(response_json)
        judge = LLMCoverageJudge(llm_pool=pool)

        checklist = ["Item A", "Item B", "Item C"]
        claims = _make_claims(("Claim about A", ["E1"]))
        evidence = [_make_evidence("E1")]

        result = await judge.judge_coverage(checklist, claims, evidence)

        assert "Item A" in result.covered
        assert "Item B" in result.uncovered
        assert "Item C" in result.uncovered

    @pytest.mark.asyncio
    async def test_evidence_filtering_only_referenced(self):
        """Only claims-referenced evidence appears in prompt 📝."""
        response_json = json.dumps({
            "covered": ["Item A"],
            "uncovered": [],
            "conflict_signals": [],
        })
        pool = _make_mock_llm_pool(response_json)
        judge = LLMCoverageJudge(llm_pool=pool)

        claims = _make_claims(("Claim", ["E1"]))
        evidence = [
            _make_evidence("E1"),
            _make_evidence("E2"),  # Not referenced
            _make_evidence("E3"),  # Not referenced
        ]

        prompt = judge._build_user_prompt(["Item A"], claims, evidence)

        assert "[E1]" in prompt
        assert "[E2]" not in prompt
        assert "[E3]" not in prompt

    @pytest.mark.asyncio
    async def test_conflict_signals_detected(self):
        """LLM reports conflicts → included in result ⚡."""
        response_json = json.dumps({
            "covered": ["Item A", "Item B"],
            "uncovered": [],
            "conflict_signals": ["Item A"],
        })
        pool = _make_mock_llm_pool(response_json)
        judge = LLMCoverageJudge(llm_pool=pool)

        checklist = ["Item A", "Item B"]
        claims = _make_claims(("Claim A", ["E1"]), ("Claim B", ["E2"]))
        evidence = [_make_evidence("E1"), _make_evidence("E2")]

        result = await judge.judge_coverage(checklist, claims, evidence)

        assert "Item A" in result.conflict_signals
        assert "Item B" not in result.conflict_signals


# ============================================================================
# 🧪 GapAnalysis Mode Tests
# ============================================================================


class TestGapAnalysisCoverageMode:
    """Tests for GapAnalysis coverage_judge_mode behavior 🧪."""

    @pytest.mark.asyncio
    async def test_always_mode_ignores_precomputed(self):
        """'always' mode runs judge even with pre_computed_coverage 🔄."""
        from inquiro.core.types import DiscoveryConfig

        mock_judge = AsyncMock()
        mock_judge.judge_coverage.return_value = CoverageResult(
            covered=["Item A"],
            uncovered=["Item B"],
        )

        gap = GapAnalysis(
            coverage_judge=mock_judge,
            coverage_judge_mode="always",
        )

        pre_computed = CoverageResult(
            covered=["Item A", "Item B"],
            uncovered=[],
        )

        report = await gap.analyze(
            checklist=["Item A", "Item B"],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=DiscoveryConfig(),
            cost_spent=0.0,
            pre_computed_coverage=pre_computed,
        )

        # ✅ Judge should have been called (not using pre-computed)
        mock_judge.judge_coverage.assert_called_once()
        assert report.coverage_ratio == 0.5  # 1/2 covered

    @pytest.mark.asyncio
    async def test_fallback_mode_uses_precomputed(self):
        """'fallback' mode uses pre_computed when available 📊."""
        from inquiro.core.types import DiscoveryConfig

        mock_judge = AsyncMock()
        gap = GapAnalysis(
            coverage_judge=mock_judge,
            coverage_judge_mode="fallback",
        )

        pre_computed = CoverageResult(
            covered=["Item A", "Item B"],
            uncovered=[],
        )

        report = await gap.analyze(
            checklist=["Item A", "Item B"],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=DiscoveryConfig(),
            cost_spent=0.0,
            pre_computed_coverage=pre_computed,
        )

        # ✅ Judge should NOT have been called
        mock_judge.judge_coverage.assert_not_called()
        assert report.coverage_ratio == 1.0  # 2/2 covered

    @pytest.mark.asyncio
    async def test_fallback_mode_calls_judge_without_precomputed(self):
        """'fallback' mode calls judge when no pre_computed 🤖."""
        from inquiro.core.types import DiscoveryConfig

        mock_judge = AsyncMock()
        mock_judge.judge_coverage.return_value = CoverageResult(
            covered=["Item A"],
            uncovered=["Item B"],
        )

        gap = GapAnalysis(
            coverage_judge=mock_judge,
            coverage_judge_mode="fallback",
        )

        report = await gap.analyze(
            checklist=["Item A", "Item B"],
            claims=[],
            evidence=[],
            previous_coverage=0.0,
            round_number=1,
            config=DiscoveryConfig(),
            cost_spent=0.0,
            pre_computed_coverage=None,
        )

        mock_judge.judge_coverage.assert_called_once()
        assert report.coverage_ratio == 0.5
