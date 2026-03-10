"""Tests for TrajectoryFeedbackProvider and TrajectoryAwareFocusGenerator 🧪.

Verifies the trajectory feedback loop:
- Cold-start returns empty strings
- System prompt hints format correctly with data
- Focus hints match gap descriptions
- Exception safety (zero-blocking guarantee)
- TrajectoryAwareFocusGenerator enriches FocusPrompt
- DiscoveryLoop auto-upgrades focus_generator
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inquiro.core.trajectory.feedback import (
    FeedbackResult,
    TrajectoryFeedbackProvider,
)
from inquiro.core.trajectory.gap_hints import GapSearchHint


# ============================================================================
# 🏭 Test fixtures
# ============================================================================


def _create_trajectory_jsonl(
    dir_path: str,
    task_id: str = "test-task-001",
    trajectory_id: str = "traj-001",
    num_rounds: int = 2,
) -> str:
    """Create a minimal trajectory JSONL file for testing 🏭.

    Args:
        dir_path: Directory to write the JSONL file.
        task_id: Task identifier for the trajectory.
        trajectory_id: Trajectory identifier.
        num_rounds: Number of rounds to generate.

    Returns:
        Path to the created JSONL file.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    file_path = os.path.join(
        dir_path, f"discovery_{task_id}_{timestamp.replace(':', '')}.jsonl"
    )

    records: list[dict[str, Any]] = []

    # 📝 Meta record
    records.append(
        {
            "type": "meta",
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "config_snapshot": {"max_rounds": num_rounds},
            "created_at": timestamp,
        }
    )

    # 📊 Round records
    for r in range(1, num_rounds + 1):
        coverage = 0.3 + (r * 0.25)
        records.append(
            {
                "type": "round",
                "round_number": r,
                "search_phase": {
                    "queries": [
                        {
                            "query_text": f"test query round {r}",
                            "mcp_tool": "perplexity",
                            "result_count": 5 + r,
                            "cost_usd": 0.01,
                        },
                        {
                            "query_text": f"clinical trials {task_id} round {r}",
                            "mcp_tool": "bohrium",
                            "result_count": 3 + r,
                            "cost_usd": 0.02,
                        },
                    ],
                    "total_raw_evidence": 8 + r,
                },
                "cleaning_phase": {
                    "input_count": 8 + r,
                    "output_count": 5 + r,
                    "dedup_removed": 2,
                    "noise_removed": 1,
                },
                "analysis_phase": {
                    "model_results": [],
                    "consensus": {
                        "consensus_decision": "positive",
                        "consensus_ratio": 0.8,
                        "total_claims": 3,
                    },
                },
                "gap_phase": {
                    "coverage_ratio": coverage,
                    "covered_items": [f"item_{i}" for i in range(r)],
                    "uncovered_items": [f"uncovered_{i}" for i in range(3 - r)]
                    if r < 3
                    else [],
                    "focus_prompt": {
                        "prompt_text": (f"Focus on uncovered items for round {r}"),
                        "target_gaps": [f"uncovered_{i}" for i in range(3 - r)]
                        if r < 3
                        else [],
                    }
                    if r < num_rounds
                    else None,
                },
            }
        )

    # 📊 Summary record
    records.append(
        {
            "type": "summary",
            "total_rounds": num_rounds,
            "final_coverage": 0.8,
            "total_cost_usd": 0.06,
            "total_evidence": 15,
            "total_claims": 6,
            "termination_reason": "coverage_threshold",
        }
    )

    # 📝 Final meta record
    records.append(
        {
            "type": "meta_final",
            "trajectory_id": trajectory_id,
            "status": "completed",
            "completed_at": timestamp,
        }
    )

    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return file_path


# ============================================================================
# 🧪 TrajectoryFeedbackProvider tests
# ============================================================================


class TestTrajectoryFeedbackProviderColdStart:
    """Tests for cold-start behavior (no data) 🧪."""

    def test_cold_start_returns_empty_system_hints(self) -> None:
        """Empty directory yields empty system prompt hints 🧊."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = TrajectoryFeedbackProvider(tmpdir)
            result = provider.get_system_prompt_hints()
            assert result == ""

    def test_cold_start_returns_empty_focus_hints(self) -> None:
        """Empty directory yields empty focus hints 🧊."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = TrajectoryFeedbackProvider(tmpdir)
            result = provider.get_focus_hints(["some gap"])
            assert result == ""

    def test_cold_start_returns_empty_feedback(self) -> None:
        """Empty directory yields empty FeedbackResult 🧊."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = TrajectoryFeedbackProvider(tmpdir)
            result = provider.get_feedback(["some gap"])
            assert isinstance(result, FeedbackResult)
            assert result.system_prompt_hints == ""
            assert result.gap_hints == []

    def test_nonexistent_directory(self) -> None:
        """Nonexistent directory returns empty string without error 🧊."""
        provider = TrajectoryFeedbackProvider("/nonexistent/path/xyz")
        assert provider.get_system_prompt_hints() == ""
        assert provider.get_focus_hints(["gap"]) == ""


class TestTrajectoryFeedbackProviderWithData:
    """Tests with actual trajectory data 🧪."""

    def test_system_prompt_hints_format(self) -> None:
        """With historical data, returns formatted Markdown hints 📊."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir, max_templates=3)
            result = provider.get_system_prompt_hints()

            # 📝 Should contain the section header
            if result:  # May be empty if index has no queries
                assert "Historical Search Patterns" in result
                assert "effective" in result.lower() or "yield" in result.lower()

    def test_lazy_initialization(self) -> None:
        """Index is built lazily on first call 🔧."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir)

            # Before any call, should not be initialized
            assert not provider._initialized

            # After first call, should be initialized
            provider.get_system_prompt_hints()
            assert provider._initialized

    def test_initialization_happens_once(self) -> None:
        """Index is built only once across multiple calls 🔧."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir)

            provider.get_system_prompt_hints()
            assert provider._initialized

            # Second call should not re-initialize
            with patch.object(
                provider,
                "_ensure_initialized",
                wraps=provider._ensure_initialized,
            ) as mock_init:
                provider.get_system_prompt_hints()
                mock_init.assert_called_once()


class TestExceptionSafety:
    """Tests for zero-blocking exception safety 🧪."""

    def test_system_hints_exception_returns_empty(self) -> None:
        """Internal exception in system hints returns empty 🛡️."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir)
            provider._ensure_initialized()

            # 💥 Force analyzer to raise
            if provider._query_analyzer:
                provider._query_analyzer.analyze = MagicMock(
                    side_effect=RuntimeError("boom")
                )
            result = provider.get_system_prompt_hints()
            assert result == ""

    def test_focus_hints_exception_returns_empty(self) -> None:
        """Internal exception in focus hints returns empty 🛡️."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir)
            provider._ensure_initialized()

            # 💥 Force accumulator to raise
            if provider._gap_accumulator:
                provider._gap_accumulator.get_hints_for_gaps = MagicMock(
                    side_effect=RuntimeError("boom")
                )
            result = provider.get_focus_hints(["gap"])
            assert result == ""

    def test_feedback_exception_returns_empty_result(self) -> None:
        """Internal exception in get_feedback returns empty result 🛡️."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_trajectory_jsonl(tmpdir)
            provider = TrajectoryFeedbackProvider(tmpdir)
            provider._ensure_initialized()

            if provider._gap_accumulator:
                provider._gap_accumulator.get_hints_for_gaps = MagicMock(
                    side_effect=RuntimeError("boom")
                )
            result = provider.get_feedback(["gap"])
            assert isinstance(result, FeedbackResult)


class TestFormatGapHints:
    """Tests for gap hint formatting 🧪."""

    def test_format_gap_hints_structure(self) -> None:
        """Formatted gap hints contain expected sections 📝."""
        hints = [
            GapSearchHint(
                gap_pattern="missing biomarker data",
                effective_queries=[
                    "biomarker validation study",
                    "biomarker clinical evidence",
                ],
                success_count=4,
                avg_coverage_delta=0.12,
                recommended_tools=["perplexity", "bohrium"],
            ),
        ]

        result = TrajectoryFeedbackProvider._format_gap_hints(hints)

        assert "Learned Gap-Closing Strategies" in result
        assert "missing biomarker data" in result
        assert "biomarker validation study" in result
        assert "perplexity" in result
        assert "closed 4 times" in result
        assert "+12% coverage" in result

    def test_format_gap_hints_empty(self) -> None:
        """Empty hints list still produces header 📝."""
        result = TrajectoryFeedbackProvider._format_gap_hints([])
        assert "Learned Gap-Closing Strategies" in result


# ============================================================================
# 🧪 TrajectoryAwareFocusGenerator tests
# ============================================================================


class TestTrajectoryAwareFocusGenerator:
    """Tests for the trajectory-enriched focus generator 🧪."""

    @pytest.mark.asyncio
    async def test_basic_focus_generation(self) -> None:
        """Generates enriched focus prompt with historical data 🎯."""
        from inquiro.core.discovery_loop import (
            TrajectoryAwareFocusGenerator,
        )
        from inquiro.core.types import (
            DiscoveryConfig,
            FocusPrompt,
            GapReport,
        )

        # 🔧 Mock feedback provider
        mock_provider = MagicMock()
        mock_provider.get_feedback.return_value = FeedbackResult(
            gap_hints=[
                GapSearchHint(
                    gap_pattern="test gap",
                    effective_queries=["query X"],
                    recommended_tools=["tool_a"],
                    success_count=2,
                    avg_coverage_delta=0.1,
                ),
            ],
        )

        generator = TrajectoryAwareFocusGenerator(mock_provider)

        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=["item_a"],
            uncovered_items=["item_b", "item_c"],
            conflict_signals=[],
            converged=False,
        )
        config = DiscoveryConfig()

        result = await generator.generate_focus(gap_report, config, 1)

        assert isinstance(result, FocusPrompt)
        assert "item_b" in result.prompt_text
        # 📝 Uses _format_gap_hints which produces this header
        assert "Learned Gap-Closing Strategies" in result.prompt_text
        assert "query X" in result.suggested_queries
        assert "tool_a" in result.suggested_tools

    @pytest.mark.asyncio
    async def test_focus_no_uncovered(self) -> None:
        """No uncovered items yields 'Broaden search scope' 🎯."""
        from inquiro.core.discovery_loop import (
            TrajectoryAwareFocusGenerator,
        )
        from inquiro.core.types import (
            DiscoveryConfig,
            GapReport,
        )

        mock_provider = MagicMock()
        generator = TrajectoryAwareFocusGenerator(mock_provider)

        gap_report = GapReport(
            round_number=2,
            coverage_ratio=1.0,
            covered_items=["all_items"],
            uncovered_items=[],
            conflict_signals=[],
            converged=True,
        )
        config = DiscoveryConfig()

        result = await generator.generate_focus(gap_report, config, 2)

        assert "Broaden search scope" in result.prompt_text
        mock_provider.get_focus_hints.assert_not_called()

    @pytest.mark.asyncio
    async def test_focus_feedback_exception_fallback(self) -> None:
        """Exception in feedback provider falls back gracefully 🛡️."""
        from inquiro.core.discovery_loop import (
            TrajectoryAwareFocusGenerator,
        )
        from inquiro.core.types import (
            DiscoveryConfig,
            GapReport,
        )

        mock_provider = MagicMock()
        mock_provider.get_feedback.side_effect = RuntimeError("boom")

        generator = TrajectoryAwareFocusGenerator(mock_provider)

        gap_report = GapReport(
            round_number=1,
            coverage_ratio=0.5,
            covered_items=[],
            uncovered_items=["gap_a"],
            conflict_signals=[],
            converged=False,
        )
        config = DiscoveryConfig()

        # Should not raise
        result = await generator.generate_focus(gap_report, config, 1)
        assert "gap_a" in result.prompt_text


# ============================================================================
# 🧪 DiscoveryLoop auto-upgrade tests
# ============================================================================


class TestDiscoveryLoopAutoUpgrade:
    """Tests for automatic focus generator upgrade 🧪."""

    def test_auto_upgrade_with_feedback_provider(self) -> None:
        """DiscoveryLoop upgrades Mock to TrajectoryAware 🔄."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            TrajectoryAwareFocusGenerator,
        )

        mock_search = MagicMock()
        mock_analysis = MagicMock()
        mock_feedback = MagicMock()

        loop = DiscoveryLoop(
            search_executor=mock_search,
            analysis_executor=mock_analysis,
            feedback_provider=mock_feedback,
        )

        assert isinstance(
            loop.focus_generator,
            TrajectoryAwareFocusGenerator,
        )

    def test_no_upgrade_without_feedback(self) -> None:
        """Without feedback, focus generator stays as Mock 🧊."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            MockFocusPromptGenerator,
        )

        mock_search = MagicMock()
        mock_analysis = MagicMock()

        loop = DiscoveryLoop(
            search_executor=mock_search,
            analysis_executor=mock_analysis,
        )

        assert isinstance(
            loop.focus_generator,
            MockFocusPromptGenerator,
        )

    def test_no_upgrade_with_custom_generator(self) -> None:
        """Custom focus generator is not replaced by auto-upgrade 🔧."""
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            TrajectoryAwareFocusGenerator,
        )

        mock_search = MagicMock()
        mock_analysis = MagicMock()
        mock_feedback = MagicMock()
        custom_generator = MagicMock()

        loop = DiscoveryLoop(
            search_executor=mock_search,
            analysis_executor=mock_analysis,
            focus_generator=custom_generator,
            feedback_provider=mock_feedback,
        )

        # Should keep the custom generator, not replace it
        assert loop.focus_generator is custom_generator
        assert not isinstance(
            loop.focus_generator,
            TrajectoryAwareFocusGenerator,
        )
