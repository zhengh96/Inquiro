"""Evolution mechanisms — pluggable learning strategies 🧬.

Each mechanism implements the ``BaseMechanism`` protocol:
- ``produce()`` — extract experiences from a round snapshot
- ``inject()`` — format experiences for prompt injection

Available mechanisms:
- ``ExperienceExtractionMechanism`` — wraps existing ExpeL pipeline
- ``ToolSelectionBandit`` — Thompson Sampling for MCP tool allocation
- ``RoundReflectionMechanism`` — Reflexion-style inter-round self-critique
- ``ActionPrincipleDistiller`` — PRAct-style cross-task principle distillation
"""

from inquiro.evolution.mechanisms.action_principles import ActionPrincipleDistiller
from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.mechanisms.experience_extraction import (
    ExperienceExtractionMechanism,
)
from inquiro.evolution.mechanisms.round_reflection import RoundReflectionMechanism
from inquiro.evolution.mechanisms.tool_selection import ToolSelectionBandit

__all__ = [
    "ActionPrincipleDistiller",
    "BaseMechanism",
    "ExperienceExtractionMechanism",
    "RoundReflectionMechanism",
    "ToolSelectionBandit",
]
