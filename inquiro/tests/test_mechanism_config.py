"""Tests for MechanismConfig and MECHANISM_PRESETS 🧪."""

from inquiro.evolution.mechanism_config import (
    MECHANISM_PRESETS,
    TOTAL_INJECTION_BUDGET,
    MechanismBudget,
    MechanismConfig,
)
from inquiro.evolution.types import MechanismType


class TestMechanismBudget:
    """Tests for MechanismBudget model 📊."""

    def test_default_budget_is_disabled(self) -> None:
        """Budget defaults to disabled with zero share 🔒."""
        budget = MechanismBudget()
        assert budget.enabled is False
        assert budget.token_share == 0.0
        assert budget.token_budget == 0

    def test_token_budget_computation(self) -> None:
        """Token budget computed from share * total 📏."""
        budget = MechanismBudget(enabled=True, token_share=0.45)
        expected = int(TOTAL_INJECTION_BUDGET * 0.45)
        assert budget.token_budget == expected

    def test_full_share_budget(self) -> None:
        """Full share returns total budget 📏."""
        budget = MechanismBudget(enabled=True, token_share=1.0)
        assert budget.token_budget == TOTAL_INJECTION_BUDGET


class TestMechanismConfig:
    """Tests for MechanismConfig model 🎛️."""

    def test_get_budget_all_types(self) -> None:
        """Every MechanismType has a corresponding budget 🔍."""
        config = MechanismConfig()
        for mt in MechanismType:
            budget = config.get_budget(mt)
            assert isinstance(budget, MechanismBudget)

    def test_enabled_types_empty_by_default(self) -> None:
        """Default config has no enabled types 📋."""
        config = MechanismConfig()
        assert config.enabled_types() == []

    def test_enabled_types_with_overrides(self) -> None:
        """Enabled types reflect overrides 📋."""
        config = MechanismConfig(
            experience_extraction=MechanismBudget(
                enabled=True, token_share=0.5,
            ),
            tool_selection=MechanismBudget(
                enabled=True, token_share=0.5,
            ),
        )
        enabled = config.enabled_types()
        assert MechanismType.EXPERIENCE_EXTRACTION in enabled
        assert MechanismType.TOOL_SELECTION in enabled
        assert MechanismType.ROUND_REFLECTION not in enabled


class TestMechanismPresets:
    """Tests for MECHANISM_PRESETS per IntensityLevel 📋."""

    def test_standard_has_experience_tool_and_reflection(self) -> None:
        """STANDARD enables experience + tool selection + reflection 📊."""
        config = MECHANISM_PRESETS["STANDARD"]
        enabled = config.enabled_types()
        assert MechanismType.EXPERIENCE_EXTRACTION in enabled
        assert MechanismType.TOOL_SELECTION in enabled
        assert MechanismType.ROUND_REFLECTION in enabled
        assert MechanismType.ACTION_PRINCIPLES not in enabled

    def test_discovery_all_active(self) -> None:
        """DISCOVERY enables all 4 mechanisms 🧬."""
        config = MECHANISM_PRESETS["DISCOVERY"]
        enabled = config.enabled_types()
        assert len(enabled) == 4
        for mt in MechanismType:
            assert mt in enabled

    def test_all_presets_shares_sum_to_one(self) -> None:
        """Token shares sum to ~1.0 for each preset with mechanisms 📏."""
        for name, config in MECHANISM_PRESETS.items():
            enabled = config.enabled_types()
            if not enabled:
                continue
            total_share = sum(
                config.get_budget(mt).token_share for mt in MechanismType
            )
            assert abs(total_share - 1.0) < 0.01, (
                f"{name}: shares sum to {total_share}"
            )
