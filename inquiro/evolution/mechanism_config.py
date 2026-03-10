"""Mechanism configuration and intensity presets 🎛️.

Maps ``IntensityLevel`` (STANDARD / DISCOVERY) to the set of enabled
mechanisms and their token budget allocations. Upper layers can override
individual mechanism settings via ``mechanism_overrides`` in
EvolutionProfile.

Token budget: 800 tokens total for all mechanism injection text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from inquiro.evolution.types import MechanismType


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = [
    "MechanismConfig",
    "MechanismBudget",
    "MECHANISM_PRESETS",
    "TOTAL_INJECTION_BUDGET",
]


# ============================================================================
# 📊 Constants
# ============================================================================

TOTAL_INJECTION_BUDGET: int = 800
"""Total token budget for all mechanism injection text 📊."""


# ============================================================================
# 🎛️ Configuration Models
# ============================================================================


class MechanismBudget(BaseModel):
    """Token budget allocation for a single mechanism 📊.

    Attributes:
        enabled: Whether the mechanism is active.
        token_share: Fraction of total budget (0.0-1.0).
    """

    enabled: bool = Field(
        default=False,
        description="Whether this mechanism is active",
    )
    token_share: float = Field(
        default=0.0,
        description="Fraction of total injection budget (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @property
    def token_budget(self) -> int:
        """Compute absolute token budget from share 📏.

        Returns:
            Integer token count allocated to this mechanism.
        """
        return int(TOTAL_INJECTION_BUDGET * self.token_share)


class MechanismConfig(BaseModel):
    """Complete mechanism configuration for an evaluation 🎛️.

    Maps each MechanismType to its budget and enabled state.
    """

    experience_extraction: MechanismBudget = Field(
        default_factory=MechanismBudget,
        description="ExpeL-style experience extraction config",
    )
    tool_selection: MechanismBudget = Field(
        default_factory=MechanismBudget,
        description="Thompson Sampling tool selection config",
    )
    round_reflection: MechanismBudget = Field(
        default_factory=MechanismBudget,
        description="Reflexion-style round reflection config",
    )
    action_principles: MechanismBudget = Field(
        default_factory=MechanismBudget,
        description="PRAct-style action principle distillation config",
    )

    def get_budget(self, mechanism_type: MechanismType) -> MechanismBudget:
        """Look up budget by mechanism type 🔍.

        Args:
            mechanism_type: The mechanism to look up.

        Returns:
            MechanismBudget for the requested mechanism.

        Raises:
            ValueError: If mechanism type is unknown.
        """
        mapping = {
            MechanismType.EXPERIENCE_EXTRACTION: self.experience_extraction,
            MechanismType.TOOL_SELECTION: self.tool_selection,
            MechanismType.ROUND_REFLECTION: self.round_reflection,
            MechanismType.ACTION_PRINCIPLES: self.action_principles,
        }
        budget = mapping.get(mechanism_type)
        if budget is None:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}")
        return budget

    def enabled_types(self) -> list[MechanismType]:
        """Return list of enabled mechanism types 📋.

        Returns:
            List of MechanismType values that are enabled.
        """
        result = []
        for mt in MechanismType:
            if self.get_budget(mt).enabled:
                result.append(mt)
        return result


# ============================================================================
# 📋 Intensity Presets
# ============================================================================


MECHANISM_PRESETS: dict[str, MechanismConfig] = {
    # STANDARD: Experience + Tool selection + Round reflection
    "STANDARD": MechanismConfig(
        experience_extraction=MechanismBudget(enabled=True, token_share=0.50),
        tool_selection=MechanismBudget(enabled=True, token_share=0.35),
        round_reflection=MechanismBudget(enabled=True, token_share=0.15),
        action_principles=MechanismBudget(enabled=False, token_share=0.0),
    ),
    # DISCOVERY: All mechanisms active
    "DISCOVERY": MechanismConfig(
        experience_extraction=MechanismBudget(enabled=True, token_share=0.45),
        tool_selection=MechanismBudget(enabled=True, token_share=0.15),
        round_reflection=MechanismBudget(enabled=True, token_share=0.25),
        action_principles=MechanismBudget(enabled=True, token_share=0.15),
    ),
}
