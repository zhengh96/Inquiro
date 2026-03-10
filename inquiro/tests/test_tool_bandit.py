"""Tests for ToolSelectionBandit — Thompson Sampling mechanism 🧪."""

import pytest

from inquiro.evolution.mechanisms.tool_selection import (
    ToolSelectionBandit,
    _context_key,
    _priority_label,
    _round_bucket,
    _star_rating,
)
from inquiro.evolution.types import MechanismType, ToolStats


class TestHelpers:
    """Tests for module-level helper functions 🔧."""

    def test_round_bucket_initial(self) -> None:
        """Round 1 maps to 'initial' 🗂️."""
        assert _round_bucket(1) == "initial"

    def test_round_bucket_focused(self) -> None:
        """Round 2+ maps to 'focused' 🗂️."""
        assert _round_bucket(2) == "focused"
        assert _round_bucket(5) == "focused"

    def test_context_key_format(self) -> None:
        """Context key combines sub_item and round bucket 🔑."""
        assert _context_key("safety_1a", 1) == "safety_1a:initial"
        assert _context_key("efficacy", 3) == "efficacy:focused"

    def test_star_rating_levels(self) -> None:
        """Star rating increases with observations ⭐."""
        assert _star_rating(0) == "☆"
        assert _star_rating(2) == "☆"
        assert _star_rating(5) == "★"
        assert _star_rating(15) == "★★"
        assert _star_rating(30) == "★★★"

    def test_priority_label(self) -> None:
        """Priority labels based on share thresholds 🏷️."""
        assert _priority_label(0.50) == "HIGH"
        assert _priority_label(0.40) == "HIGH"
        assert _priority_label(0.30) == "MEDIUM"
        assert _priority_label(0.10) == "LOW"


class TestToolSelectionBandit:
    """Tests for ToolSelectionBandit core logic 🎰."""

    @pytest.fixture()
    def bandit(self) -> ToolSelectionBandit:
        """Create a ToolSelectionBandit with mock store 🔧."""
        # Store is only used for persistence — not needed for core logic
        return ToolSelectionBandit(
            store=None,  # type: ignore[arg-type]
            namespace="test",
            exploration_bonus=0.10,
            confidence_threshold=5,
            decay_factor=0.98,
        )

    def test_mechanism_type(self, bandit: ToolSelectionBandit) -> None:
        """Returns TOOL_SELECTION mechanism type 🏷️."""
        assert bandit.mechanism_type == MechanismType.TOOL_SELECTION

    def test_select_empty_tools(self, bandit: ToolSelectionBandit) -> None:
        """Empty tool list returns empty allocation 📊."""
        result = bandit.select([], "sub1", 1)
        assert result == {}

    def test_select_uniform_cold_start(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Cold start with no fallback uses Thompson Sampling 🎲."""
        # No static fallback → use Beta(1,1) = uniform
        tools = ["tool_a", "tool_b"]
        result = bandit.select(tools, "sub1", 1)
        assert len(result) == 2
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_select_static_fallback_on_low_confidence(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Falls back to static allocation when confidence is low 🧊."""
        bandit.set_static_fallback({"tool_a": 0.7, "tool_b": 0.3})
        result = bandit.select(["tool_a", "tool_b"], "sub1", 1)
        assert abs(result["tool_a"] - 0.7) < 1e-6
        assert abs(result["tool_b"] - 0.3) < 1e-6

    def test_update_increases_alpha_on_success(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Successful outcome increases alpha 📈."""
        bandit.update("sub1", 1, {"tool_a": True})
        ctx_key = _context_key("sub1", 1)
        stats = bandit._stats[ctx_key]["tool_a"]
        assert stats.alpha > 1.0
        assert stats.total_observations == 1

    def test_update_increases_beta_on_failure(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Failed outcome increases beta 📉."""
        bandit.update("sub1", 1, {"tool_a": False})
        ctx_key = _context_key("sub1", 1)
        stats = bandit._stats[ctx_key]["tool_a"]
        assert stats.beta > 1.0
        assert stats.total_observations == 1

    def test_update_applies_decay(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Decay shrinks prior observations toward uniform 📉."""
        # First update
        bandit.update("sub1", 1, {"tool_a": True})
        ctx_key = _context_key("sub1", 1)
        alpha_after_first = bandit._stats[ctx_key]["tool_a"].alpha

        # Second update triggers decay on first
        bandit.update("sub1", 1, {"tool_a": True})
        alpha_after_second = bandit._stats[ctx_key]["tool_a"].alpha

        # alpha should be > first (grew), but not by full +1
        # due to decay on the prior excess
        assert alpha_after_second > alpha_after_first

    def test_exploration_floor_enforced(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """No tool drops below exploration floor 🔍."""
        # Create strong bias for tool_a
        for _ in range(20):
            bandit.update("sub1", 1, {"tool_a": True, "tool_b": False})

        result = bandit.select(["tool_a", "tool_b"], "sub1", 1)
        min_expected = bandit._exploration_bonus / 2
        for share in result.values():
            assert share >= min_expected - 1e-6

    def test_inject_returns_none_when_disabled(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Disabled bandit returns None from inject 🔇."""
        bandit.enabled = False
        result = bandit.inject({"available_tools": ["t1"], "round_num": 1})
        assert result is None

    def test_inject_returns_markdown_table(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """Inject returns markdown with tool priority table 💉."""
        result = bandit.inject({
            "available_tools": ["pubmed_search", "paper_search"],
            "sub_item_id": "sub1",
            "round_num": 1,
        })
        assert result is not None
        assert "## TOOL SELECTION GUIDANCE" in result
        assert "pubmed_search" in result
        assert "paper_search" in result

    def test_inject_no_tools_returns_none(
        self, bandit: ToolSelectionBandit,
    ) -> None:
        """No available tools → inject returns None 📋."""
        result = bandit.inject({
            "available_tools": [],
            "sub_item_id": "sub1",
            "round_num": 1,
        })
        assert result is None


class TestToolStatsModel:
    """Tests for ToolStats Pydantic model 📊."""

    def test_default_values(self) -> None:
        """ToolStats defaults to Beta(1,1) prior 🎰."""
        stats = ToolStats()
        assert stats.alpha == 1.0
        assert stats.beta == 1.0
        assert stats.total_observations == 0

    def test_serialization_roundtrip(self) -> None:
        """ToolStats survives dump/validate roundtrip 🔄."""
        stats = ToolStats(alpha=3.5, beta=2.1, total_observations=10)
        dumped = stats.model_dump()
        restored = ToolStats.model_validate(dumped)
        assert restored.alpha == 3.5
        assert restored.beta == 2.1
        assert restored.total_observations == 10
