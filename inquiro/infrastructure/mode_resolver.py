"""Inquiro ModeResolver — evaluation mode preset loading and resolution 🎚️.

Loads ``evaluation_modes.yaml`` via :class:`ConfigLoader` and provides
typed access to predefined execution profiles (standard / discovery).

Each mode encapsulates agent configuration, quality gate settings, and
cost guard parameters so that callers can switch execution profiles
with a single string parameter.

Example::

    resolver = ModeResolver(config_loader)
    mode = resolver.resolve_mode("standard")
    print(mode.agent_config.max_turns)  # 15

The default mode is ``"standard"`` unless overridden in the YAML file.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from inquiro.core.types import (
    AgentConfig,
    CostGuardConfig,
    OverspendStrategy,
    QualityChecks,
    QualityGateConfig,
)
from inquiro.infrastructure.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

# 🎯 Sentinel for unresolvable mode names
_DEFAULT_MODE = "standard"


class ModeAgentConfig(BaseModel):
    """Agent configuration subset within an evaluation mode 🤖.

    Contains only the fields that vary between modes. Fields not
    specified here (e.g., ``model``, ``context``) keep their
    :class:`AgentConfig` defaults.
    """

    max_turns: int = Field(
        default=15,
        description="Maximum number of agent turns (LLM round-trips)",
        gt=0,
    )
    timeout_seconds: int = Field(
        default=300,
        description="Per-task execution timeout in seconds",
        gt=0,
    )
    temperature: float = Field(
        default=0.3,
        description="LLM sampling temperature",
        ge=0.0,
        le=2.0,
    )


class ModeQualityGateConfig(BaseModel):
    """Quality gate configuration subset within an evaluation mode 🔍.

    Mirrors :class:`QualityGateConfig` with mode-specific defaults.
    """

    enabled: bool = Field(
        default=True,
        description="Whether quality gate validation is active",
    )
    max_retries: int = Field(
        default=1,
        description="Maximum retry attempts on hard quality failures",
        ge=0,
    )
    checks: QualityChecks = Field(
        default_factory=QualityChecks,
        description="Individual check toggles",
    )


class ModeCostGuardConfig(BaseModel):
    """Cost guard configuration subset within an evaluation mode 💰.

    Mirrors :class:`CostGuardConfig` with mode-specific defaults.
    """

    max_cost_per_task: float = Field(
        default=3.0,
        description="Maximum cost in USD for a single task",
        gt=0.0,
    )
    overspend_strategy: OverspendStrategy = Field(
        default=OverspendStrategy.SOFT_STOP,
        description="Behavior when budget is exceeded",
    )


class ModeConfig(BaseModel):
    """Complete evaluation mode configuration 🎚️.

    Encapsulates a single named preset combining agent, quality gate,
    and cost guard settings. Use :meth:`to_agent_config`,
    :meth:`to_quality_gate_config`, and :meth:`to_cost_guard_config`
    to convert into the canonical :mod:`inquiro.core.types` models.

    Attributes:
        name: Mode identifier (e.g., "standard", "discovery").
        description: Human-readable description of the mode.
        agent_config: Agent execution parameters for this mode.
        quality_gate: Quality gate parameters for this mode.
        cost_guard: Cost budget parameters for this mode.
    """

    name: str = Field(
        description="Mode identifier (e.g., 'fast', 'standard', 'deep')",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the mode",
    )
    agent_config: ModeAgentConfig = Field(
        default_factory=ModeAgentConfig,
        description="Agent execution parameters for this mode",
    )
    quality_gate: ModeQualityGateConfig = Field(
        default_factory=ModeQualityGateConfig,
        description="Quality gate parameters for this mode",
    )
    cost_guard: ModeCostGuardConfig = Field(
        default_factory=ModeCostGuardConfig,
        description="Cost budget parameters for this mode",
    )

    def to_agent_config(self) -> AgentConfig:
        """Convert mode agent settings to a canonical AgentConfig 🤖.

        Fields not controlled by the mode (e.g., ``model``, ``context``)
        retain their :class:`AgentConfig` defaults.

        Returns:
            A fully populated AgentConfig instance.
        """
        return AgentConfig(
            max_turns=self.agent_config.max_turns,
            temperature=self.agent_config.temperature,
        )

    def to_quality_gate_config(self) -> QualityGateConfig:
        """Convert mode QG settings to a canonical QualityGateConfig 🔍.

        Returns:
            A fully populated QualityGateConfig instance.
        """
        return QualityGateConfig(
            enabled=self.quality_gate.enabled,
            max_retries=self.quality_gate.max_retries,
            checks=self.quality_gate.checks,
        )

    def to_cost_guard_config(self) -> CostGuardConfig:
        """Convert mode cost settings to a canonical CostGuardConfig 💰.

        Returns:
            A fully populated CostGuardConfig instance.
        """
        return CostGuardConfig(
            max_cost_per_task=self.cost_guard.max_cost_per_task,
            overspend_strategy=self.cost_guard.overspend_strategy,
        )


class ModeResolverError(Exception):
    """Raised when mode resolution fails ❌."""


class ModeResolver:
    """Load and resolve evaluation mode presets from YAML config 🎚️.

    Reads ``evaluation_modes.yaml`` via :class:`ConfigLoader` and
    provides typed :class:`ModeConfig` instances for each named preset.

    Thread-safe: the underlying ConfigLoader cache is lock-protected.

    Attributes:
        config_loader: The ConfigLoader instance used for YAML access.
    """

    _FILENAME = "evaluation_modes.yaml"

    def __init__(self, config_loader: ConfigLoader) -> None:
        """Initialize ModeResolver 🔧.

        Args:
            config_loader: ConfigLoader pointing to the configs directory.
        """
        self._config_loader = config_loader
        self._modes: dict[str, ModeConfig] | None = None
        self._default_mode: str = _DEFAULT_MODE
        logger.info("🎚️ ModeResolver initialized")

    def _load_modes(self) -> dict[str, ModeConfig]:
        """Load and parse all modes from YAML on first access 📄.

        Returns:
            Dict mapping mode names to ModeConfig instances.

        Raises:
            ModeResolverError: If the YAML structure is invalid.
        """
        if self._modes is not None:
            return self._modes

        raw = self._config_loader._load_yaml(self._FILENAME)

        # 🔑 Extract default mode name
        self._default_mode = raw.get("default_mode", _DEFAULT_MODE)

        modes_raw = raw.get("modes")
        if not isinstance(modes_raw, dict):
            raise ModeResolverError(
                "evaluation_modes.yaml must contain a 'modes' mapping"
            )

        parsed: dict[str, ModeConfig] = {}
        for mode_name, mode_data in modes_raw.items():
            if not isinstance(mode_data, dict):
                raise ModeResolverError(
                    f"Mode '{mode_name}' must be a mapping, "
                    f"got {type(mode_data).__name__}"
                )
            try:
                parsed[mode_name] = ModeConfig(
                    name=mode_name,
                    **mode_data,
                )
            except Exception as exc:
                raise ModeResolverError(
                    f"Failed to parse mode '{mode_name}': {exc}"
                ) from exc

        if not parsed:
            raise ModeResolverError("evaluation_modes.yaml 'modes' mapping is empty")

        self._modes = parsed
        logger.info(
            "✅ Loaded %d evaluation modes: %s",
            len(parsed),
            list(parsed.keys()),
        )
        return self._modes

    def resolve_mode(self, mode_name: str | None = None) -> ModeConfig:
        """Resolve a named evaluation mode to its configuration 🎯.

        Args:
            mode_name: Name of the mode to resolve (e.g., "fast",
                "standard", "deep"). If None, returns the default mode.

        Returns:
            The corresponding ModeConfig instance.

        Raises:
            ModeResolverError: If the mode name is not found.
        """
        modes = self._load_modes()

        effective_name = mode_name or self._default_mode
        if effective_name not in modes:
            available = list(modes.keys())
            raise ModeResolverError(
                f"Unknown evaluation mode '{effective_name}'. "
                f"Available modes: {available}"
            )

        logger.info("🎚️ Resolved evaluation mode: %s", effective_name)
        return modes[effective_name]

    def get_available_modes(self) -> list[str]:
        """List all available mode names 📋.

        Returns:
            Sorted list of mode identifiers.
        """
        modes = self._load_modes()
        return sorted(modes.keys())

    def get_default_mode_name(self) -> str:
        """Return the configured default mode name 🔑.

        Returns:
            The default mode identifier string.
        """
        self._load_modes()  # ✨ Ensure YAML is loaded
        return self._default_mode

    def reload(self) -> None:
        """Clear cached modes and force reload on next access 🔄."""
        self._modes = None
        self._default_mode = _DEFAULT_MODE
        logger.info("🔄 ModeResolver cache cleared")
