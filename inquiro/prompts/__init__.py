"""Inquiro prompt template management 📝.

Provides externalized prompt templates and a loader utility for
rendering system and user prompts for Research and Synthesis agents.

Templates are stored as Markdown files in this package directory
and loaded via PromptLoader at runtime.

PromptSectionBuilder offers stateless, independently testable methods
for building each section of the system prompt.
"""

from inquiro.prompts.loader import PromptLoader
from inquiro.prompts.section_builder import PromptSectionBuilder

__all__ = [
    "PromptLoader",
    "PromptSectionBuilder",
]
