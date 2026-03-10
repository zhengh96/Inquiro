"""Prompt template loader for Inquiro agents 📝.

Reads Markdown template files from the prompts package directory and
renders them with keyword arguments via Python str.format().

Templates are cached after first load to avoid repeated disk I/O.

Usage::

    loader = PromptLoader()
    prompt = loader.render(
        "research_system",
        rules="...",
        search_checklist="...",
        output_schema="...",
        prior_context="",
    )
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 📁 Directory containing the .md template files
_TEMPLATE_DIR = Path(__file__).resolve().parent


class PromptLoader:
    """Thread-safe prompt template loader with caching 🔧.

    Loads .md template files from the ``inquiro/prompts/`` directory
    and renders them using Python ``str.format()``.

    Attributes:
        _cache: In-memory cache mapping template names to raw content.
        _lock: Thread lock protecting cache writes.
        _template_dir: Directory containing template files.
    """

    def __init__(
        self,
        template_dir: Path | None = None,
    ) -> None:
        """Initialize PromptLoader 🔧.

        Args:
            template_dir: Custom template directory. Defaults to the
                package directory containing the built-in .md templates.
        """
        self._template_dir = template_dir or _TEMPLATE_DIR
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()

    def load(self, template_name: str) -> str:
        """Load a raw template by name (without .md extension) 📖.

        Reads from disk on first access, then returns cached content
        on subsequent calls. Thread-safe.

        Args:
            template_name: Template file name without the .md extension
                (e.g., "research_system", "synthesis_user").

        Returns:
            Raw template string with {placeholder} markers intact.

        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        # 🚀 Fast path: check cache without lock
        if template_name in self._cache:
            return self._cache[template_name]

        # 🔒 Slow path: load from disk under lock
        with self._lock:
            # Double-check after acquiring lock
            if template_name in self._cache:
                return self._cache[template_name]

            file_path = self._template_dir / f"{template_name}.md"
            if not file_path.is_file():
                raise FileNotFoundError(f"Prompt template not found: {file_path}")

            content = file_path.read_text(encoding="utf-8")
            self._cache[template_name] = content
            logger.debug(
                "📝 Loaded prompt template: %s",
                template_name,
            )
            return content

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Load and render a template with keyword substitution 🎯.

        Uses Python's ``str.format()`` to replace ``{placeholder}``
        markers with the provided keyword arguments.

        Args:
            template_name: Template file name without the .md extension.
            **kwargs: Key-value pairs for template placeholder substitution.

        Returns:
            Rendered prompt string with all placeholders replaced.

        Raises:
            FileNotFoundError: If the template file does not exist.
            KeyError: If a required placeholder is missing from kwargs.
        """
        raw = self.load(template_name)
        return raw.format(**kwargs)

    def clear_cache(self) -> None:
        """Clear the template cache 🗑️.

        Useful for testing or when templates are modified at runtime.
        """
        with self._lock:
            self._cache.clear()
            logger.debug("🗑️ Prompt template cache cleared")
