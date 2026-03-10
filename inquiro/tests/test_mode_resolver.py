"""Tests for Inquiro ModeResolver — evaluation mode preset resolution 🧪.

Tests cover:
- Loading both presets (standard, discovery) from YAML
- Default mode fallback when no mode is specified
- Invalid mode name raises ModeResolverError
- ModeConfig field validation (Pydantic constraints)
- Conversion helpers (to_agent_config, to_quality_gate_config, to_cost_guard_config)
- Serialization round-trip for ModeConfig
- Reload behavior clears cache
- Available modes listing

Uses Google Python Style Guide. English comments with emojis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inquiro.core.types import (
    AgentConfig,
    CostGuardConfig,
    OverspendStrategy,
    QualityGateConfig,
)
from inquiro.infrastructure.config_loader import ConfigLoader
from inquiro.infrastructure.mode_resolver import (
    ModeAgentConfig,
    ModeConfig,
    ModeCostGuardConfig,
    ModeQualityGateConfig,
    ModeResolver,
    ModeResolverError,
)


# ============================================================
# 🔧 Fixtures
# ============================================================


@pytest.fixture
def config_loader() -> ConfigLoader:
    """ConfigLoader pointing to real configs directory 📋."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    return ConfigLoader(config_dir)


@pytest.fixture
def resolver(config_loader: ConfigLoader) -> ModeResolver:
    """ModeResolver backed by real evaluation_modes.yaml 🎚️."""
    return ModeResolver(config_loader)


# ============================================================
# ⚖️ Standard Mode Tests
# ============================================================


class TestStandardMode:
    """Tests for the 'standard' evaluation mode ⚖️."""

    def test_standard_mode_agent_config(self, resolver: ModeResolver) -> None:
        """Standard mode should have 15 max_turns and 600s timeout 🔧."""
        mode = resolver.resolve_mode("standard")
        assert mode.agent_config.max_turns == 15
        assert mode.agent_config.timeout_seconds == 600
        assert mode.agent_config.temperature == 0.3

    def test_standard_mode_quality_gate(self, resolver: ModeResolver) -> None:
        """Standard mode should have 1 retry and QG enabled."""
        mode = resolver.resolve_mode("standard")
        assert mode.quality_gate.enabled is True
        assert mode.quality_gate.max_retries == 1

    def test_standard_mode_cost_guard(self, resolver: ModeResolver) -> None:
        """Standard mode should have $3.0 max cost per task."""
        mode = resolver.resolve_mode("standard")
        assert mode.cost_guard.max_cost_per_task == 3.0

    def test_standard_mode_quality_checks_enabled(self, resolver: ModeResolver) -> None:
        """Standard mode should have all core checks enabled."""
        mode = resolver.resolve_mode("standard")
        checks = mode.quality_gate.checks
        assert checks.schema_validation is True
        assert checks.coverage_check is True
        assert checks.evidence_reference_check is True


# ============================================================
# 🔬 Discovery Mode Tests
# ============================================================


class TestDiscoveryMode:
    """Tests for the 'discovery' evaluation mode 🔬."""

    def test_discovery_mode_agent_config(self, resolver: ModeResolver) -> None:
        """Discovery mode should have 30 max_turns and 900s timeout 🔧."""
        mode = resolver.resolve_mode("discovery")
        assert mode.agent_config.max_turns == 30
        assert mode.agent_config.timeout_seconds == 900
        assert mode.agent_config.temperature == 0.2

    def test_discovery_mode_quality_gate(self, resolver: ModeResolver) -> None:
        """Discovery mode should have 2 retries and QG enabled."""
        mode = resolver.resolve_mode("discovery")
        assert mode.quality_gate.enabled is True
        assert mode.quality_gate.max_retries == 2

    def test_discovery_mode_cost_guard(self, resolver: ModeResolver) -> None:
        """Discovery mode should have $10.0 max cost per task."""
        mode = resolver.resolve_mode("discovery")
        assert mode.cost_guard.max_cost_per_task == 10.0


# ============================================================
# 🎯 Default Mode & Fallback Tests
# ============================================================


class TestDefaultMode:
    """Tests for default mode resolution 🎯."""

    def test_none_resolves_to_default(self, resolver: ModeResolver) -> None:
        """Passing None should return the default (standard) mode."""
        mode = resolver.resolve_mode(None)
        assert mode.name == "standard"

    def test_no_argument_resolves_to_default(self, resolver: ModeResolver) -> None:
        """Calling resolve_mode() without args returns standard."""
        mode = resolver.resolve_mode()
        assert mode.name == "standard"

    def test_default_mode_name_is_standard(self, resolver: ModeResolver) -> None:
        """get_default_mode_name should return 'standard'."""
        assert resolver.get_default_mode_name() == "standard"


# ============================================================
# ❌ Error Handling Tests
# ============================================================


class TestModeResolverErrors:
    """Tests for error conditions in ModeResolver ❌."""

    def test_unknown_mode_raises_error(self, resolver: ModeResolver) -> None:
        """Requesting a non-existent mode should raise ModeResolverError."""
        with pytest.raises(ModeResolverError, match="Unknown evaluation mode"):
            resolver.resolve_mode("nonexistent_mode")

    def test_error_message_lists_available_modes(self, resolver: ModeResolver) -> None:
        """Error message should list available modes for guidance."""
        with pytest.raises(ModeResolverError) as exc_info:
            resolver.resolve_mode("invalid")
        error_msg = str(exc_info.value)
        assert "standard" in error_msg
        assert "discovery" in error_msg


# ============================================================
# 🔄 Conversion Helper Tests
# ============================================================


class TestModeConfigConversion:
    """Tests for ModeConfig to canonical type conversions 🔄."""

    def test_to_agent_config_returns_correct_type(self, resolver: ModeResolver) -> None:
        """to_agent_config should return an AgentConfig instance."""
        mode = resolver.resolve_mode("standard")
        agent_config = mode.to_agent_config()
        assert isinstance(agent_config, AgentConfig)
        assert agent_config.max_turns == 15
        assert agent_config.temperature == 0.3

    def test_to_quality_gate_config_returns_correct_type(
        self, resolver: ModeResolver
    ) -> None:
        """to_quality_gate_config should return a QualityGateConfig."""
        mode = resolver.resolve_mode("discovery")
        qg_config = mode.to_quality_gate_config()
        assert isinstance(qg_config, QualityGateConfig)
        assert qg_config.max_retries == 2
        assert qg_config.enabled is True

    def test_to_cost_guard_config_returns_correct_type(
        self, resolver: ModeResolver
    ) -> None:
        """to_cost_guard_config should return a CostGuardConfig."""
        mode = resolver.resolve_mode("standard")
        cost_config = mode.to_cost_guard_config()
        assert isinstance(cost_config, CostGuardConfig)
        assert cost_config.max_cost_per_task == 3.0
        assert cost_config.overspend_strategy == OverspendStrategy.SOFT_STOP

    def test_agent_config_preserves_default_model(self, resolver: ModeResolver) -> None:
        """Converted AgentConfig should keep default model name."""
        mode = resolver.resolve_mode("standard")
        agent_config = mode.to_agent_config()
        # ✨ Model defaults to claude-sonnet-4 from AgentConfig
        assert agent_config.model == "claude-sonnet-4-20250514"


# ============================================================
# 📦 Serialization Tests
# ============================================================


class TestModeConfigSerialization:
    """Tests for ModeConfig serialization round-trip 📦."""

    def test_model_dump_round_trip(self, resolver: ModeResolver) -> None:
        """ModeConfig should survive model_dump/reconstruct."""
        mode = resolver.resolve_mode("standard")
        data = mode.model_dump()
        restored = ModeConfig(**data)
        assert restored.name == mode.name
        assert restored.agent_config.max_turns == mode.agent_config.max_turns
        assert restored.quality_gate.max_retries == mode.quality_gate.max_retries
        assert (
            restored.cost_guard.max_cost_per_task == mode.cost_guard.max_cost_per_task
        )

    def test_json_round_trip(self, resolver: ModeResolver) -> None:
        """ModeConfig should survive JSON serialize/deserialize."""
        mode = resolver.resolve_mode("discovery")
        json_str = mode.model_dump_json()
        restored = ModeConfig.model_validate_json(json_str)
        assert restored.name == "discovery"
        assert restored.agent_config.max_turns == 30
        assert restored.cost_guard.max_cost_per_task == 10.0


# ============================================================
# 📋 Available Modes & Reload Tests
# ============================================================


class TestAvailableModesAndReload:
    """Tests for mode listing and cache reload 📋."""

    def test_get_available_modes_returns_both(
        self, resolver: ModeResolver
    ) -> None:
        """get_available_modes should return all defined modes."""
        modes = resolver.get_available_modes()
        assert "standard" in modes
        assert "discovery" in modes
        assert len(modes) == 2

    def test_available_modes_are_sorted(self, resolver: ModeResolver) -> None:
        """get_available_modes should return sorted names."""
        modes = resolver.get_available_modes()
        assert modes == sorted(modes)

    def test_reload_clears_cache(self, resolver: ModeResolver) -> None:
        """reload() should clear internal cache so next call reloads."""
        # 🔄 Load once to populate cache
        _ = resolver.resolve_mode("standard")
        assert resolver._modes is not None

        # 🔄 Reload clears cache
        resolver.reload()
        assert resolver._modes is None

        # 🔄 Next resolve re-loads from YAML
        mode = resolver.resolve_mode("standard")
        assert mode.name == "standard"
        assert resolver._modes is not None


# ============================================================
# 🔍 ModeConfig Pydantic Validation Tests
# ============================================================


class TestModeConfigValidation:
    """Tests for ModeConfig Pydantic field validation 🔍."""

    def test_max_turns_must_be_positive(self) -> None:
        """ModeAgentConfig max_turns must be > 0."""
        with pytest.raises(Exception):
            ModeAgentConfig(max_turns=0)

    def test_timeout_must_be_positive(self) -> None:
        """ModeAgentConfig timeout_seconds must be > 0."""
        with pytest.raises(Exception):
            ModeAgentConfig(timeout_seconds=0)

    def test_temperature_bounds(self) -> None:
        """ModeAgentConfig temperature must be in [0.0, 2.0]."""
        with pytest.raises(Exception):
            ModeAgentConfig(temperature=-0.1)
        with pytest.raises(Exception):
            ModeAgentConfig(temperature=2.5)

    def test_max_retries_non_negative(self) -> None:
        """ModeQualityGateConfig max_retries must be >= 0."""
        with pytest.raises(Exception):
            ModeQualityGateConfig(max_retries=-1)

    def test_cost_must_be_positive(self) -> None:
        """ModeCostGuardConfig max_cost_per_task must be > 0."""
        with pytest.raises(Exception):
            ModeCostGuardConfig(max_cost_per_task=0.0)
        with pytest.raises(Exception):
            ModeCostGuardConfig(max_cost_per_task=-1.0)

    def test_valid_mode_config_construction(self) -> None:
        """ModeConfig should accept valid fields directly."""
        config = ModeConfig(
            name="custom",
            description="Custom test mode",
            agent_config=ModeAgentConfig(
                max_turns=10,
                timeout_seconds=120,
                temperature=0.5,
            ),
            quality_gate=ModeQualityGateConfig(
                enabled=True,
                max_retries=1,
            ),
            cost_guard=ModeCostGuardConfig(
                max_cost_per_task=2.0,
            ),
        )
        assert config.name == "custom"
        assert config.agent_config.max_turns == 10
        assert config.quality_gate.max_retries == 1
        assert config.cost_guard.max_cost_per_task == 2.0


# ============================================================
# 🔗 Cross-Mode Comparison Tests
# ============================================================


class TestCrossModeComparison:
    """Tests verifying correct ordering across modes 🔗."""

    def test_max_turns_ordering(self, resolver: ModeResolver) -> None:
        """standard < discovery for max_turns."""
        standard = resolver.resolve_mode("standard")
        discovery = resolver.resolve_mode("discovery")
        assert (
            standard.agent_config.max_turns
            < discovery.agent_config.max_turns
        )

    def test_timeout_ordering(self, resolver: ModeResolver) -> None:
        """standard <= discovery for timeout_seconds 🕐."""
        standard = resolver.resolve_mode("standard")
        discovery = resolver.resolve_mode("discovery")
        assert (
            standard.agent_config.timeout_seconds
            <= discovery.agent_config.timeout_seconds
        )

    def test_cost_ordering(self, resolver: ModeResolver) -> None:
        """standard < discovery for max_cost_per_task."""
        standard = resolver.resolve_mode("standard")
        discovery = resolver.resolve_mode("discovery")
        assert (
            standard.cost_guard.max_cost_per_task
            < discovery.cost_guard.max_cost_per_task
        )

    def test_retries_ordering(self, resolver: ModeResolver) -> None:
        """standard <= discovery for max_retries."""
        standard = resolver.resolve_mode("standard")
        discovery = resolver.resolve_mode("discovery")
        assert (
            standard.quality_gate.max_retries
            <= discovery.quality_gate.max_retries
        )
