"""Tests for ActionPrincipleDistiller 🧪."""

import json

import pytest

from inquiro.evolution.mechanisms.action_principles import (
    ActionPrincipleDistiller,
)
from inquiro.evolution.types import ActionPrinciple, MechanismType


class TestActionPrincipleDistiller:
    """Tests for ActionPrincipleDistiller core logic 📜."""

    @pytest.fixture()
    def mock_llm_fn(self):
        """Create an async mock LLM function 🤖."""
        async def _llm_fn(prompt: str) -> str:
            return json.dumps([
                {
                    "text": "Always verify clinical trial data with primary sources",
                    "source_insight_ids": ["id1", "id2"],
                },
                {
                    "text": "Prioritize systematic reviews over individual studies",
                    "source_insight_ids": ["id3"],
                },
                {
                    "text": "Cross-reference findings across multiple databases",
                    "source_insight_ids": ["id4", "id5"],
                },
            ])
        return _llm_fn

    @pytest.fixture()
    def distiller(self, mock_llm_fn) -> ActionPrincipleDistiller:
        """Create a distiller with mock store and LLM 🔧."""
        return ActionPrincipleDistiller(
            store=None,  # type: ignore[arg-type]
            llm_fn=mock_llm_fn,
            namespace="test",
            enabled=True,
        )

    def test_mechanism_type(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """Returns ACTION_PRINCIPLES mechanism type 🏷️."""
        assert distiller.mechanism_type == MechanismType.ACTION_PRINCIPLES

    def test_inject_no_active_principles(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """inject() returns None when no active principles 💉."""
        result = distiller.inject({"round_num": 1})
        assert result is None

    def test_inject_with_active_principles(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """inject() returns formatted principles 💉."""
        distiller._principles = [
            ActionPrinciple(
                text="Always check multiple sources",
                status="active",
            ),
            ActionPrinciple(
                text="Prioritize recent publications",
                status="active",
            ),
            ActionPrinciple(
                text="Old retired principle",
                status="retired",
            ),
        ]
        result = distiller.inject({"round_num": 1})
        assert result is not None
        assert "## OPERATING PRINCIPLES" in result
        assert "1. Always check multiple sources" in result
        assert "2. Prioritize recent publications" in result
        assert "Old retired" not in result

    def test_ab_test_update_treatment(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """A/B test updates treatment coverage 🔬."""
        p = ActionPrinciple(text="Test principle", status="candidate")
        distiller._principles = [p]

        distiller.update_ab_test(p.id, is_treatment=True, coverage=0.80)
        assert p.evaluation_count == 1
        assert p.treatment_coverage == 0.80

    def test_ab_test_update_control(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """A/B test updates control coverage 🔬."""
        p = ActionPrinciple(text="Test principle", status="candidate")
        distiller._principles = [p]

        distiller.update_ab_test(p.id, is_treatment=False, coverage=0.60)
        assert p.evaluation_count == 1
        assert p.control_coverage == 0.60

    def test_promotion_on_sufficient_lift(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """Principle promoted when lift >= threshold 🎓."""
        p = ActionPrinciple(
            text="Good principle",
            status="candidate",
            treatment_coverage=0.80,
            control_coverage=0.70,
            evaluation_count=9,
        )
        distiller._principles = [p]

        # 10th evaluation triggers promotion check
        distiller.update_ab_test(p.id, is_treatment=True, coverage=0.85)
        assert p.status == "active"

    def test_retirement_on_insufficient_lift(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """Principle retired when lift < threshold 🗑️."""
        p = ActionPrinciple(
            text="Weak principle",
            status="candidate",
            treatment_coverage=0.72,
            control_coverage=0.70,
            evaluation_count=9,
        )
        distiller._principles = [p]

        distiller.update_ab_test(p.id, is_treatment=True, coverage=0.72)
        assert p.status == "retired"

    def test_enforce_max_active(
        self, distiller: ActionPrincipleDistiller,
    ) -> None:
        """Overflow active principles get retired 🗑️."""
        # Create MAX + 1 active principles
        distiller._principles = [
            ActionPrinciple(
                text=f"Principle {i}",
                status="active",
                treatment_coverage=0.50 + i * 0.01,
            )
            for i in range(distiller.MAX_ACTIVE_PRINCIPLES + 2)
        ]
        distiller._enforce_max_active()

        active = [
            p for p in distiller._principles if p.status == "active"
        ]
        retired = [
            p for p in distiller._principles if p.status == "retired"
        ]
        assert len(active) == distiller.MAX_ACTIVE_PRINCIPLES
        assert len(retired) == 2

    def test_disabled_inject(self) -> None:
        """Disabled distiller returns None 🔇."""

        async def noop(p: str) -> str:
            return ""

        d = ActionPrincipleDistiller(
            store=None,  # type: ignore[arg-type]
            llm_fn=noop,
            namespace="test",
            enabled=False,
        )
        assert d.inject({}) is None


class TestActionPrinciple:
    """Tests for ActionPrinciple model 📜."""

    def test_default_status(self) -> None:
        """Default status is 'candidate' 🏷️."""
        p = ActionPrinciple(text="Test")
        assert p.status == "candidate"
        assert p.evaluation_count == 0

    def test_has_uuid(self) -> None:
        """Auto-generates UUID id 🆔."""
        p = ActionPrinciple(text="Test")
        assert len(p.id) == 36  # UUID format
