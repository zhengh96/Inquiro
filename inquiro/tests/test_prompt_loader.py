"""Unit tests for PromptLoader 📝.

Tests template loading, rendering, caching, and error handling.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from inquiro.prompts.loader import PromptLoader


@pytest.fixture
def tmp_template_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample templates 📁."""
    (tmp_path / "greeting.md").write_text("Hello, {name}!")
    (tmp_path / "multi.md").write_text("Topic: {topic}\nRules: {rules}")
    return tmp_path


@pytest.fixture
def loader(tmp_template_dir: Path) -> PromptLoader:
    """Create a PromptLoader with temp directory 🔧."""
    return PromptLoader(template_dir=tmp_template_dir)


class TestPromptLoaderLoad:
    """Tests for PromptLoader.load() 📖."""

    def test_load_valid_template(
        self,
        loader: PromptLoader,
    ) -> None:
        """Load an existing template returns raw content."""
        content = loader.load("greeting")
        assert content == "Hello, {name}!"

    def test_load_missing_template_raises(
        self,
        loader: PromptLoader,
    ) -> None:
        """Load a nonexistent template raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not_exist"):
            loader.load("not_exist")

    def test_load_caches_on_second_call(
        self,
        loader: PromptLoader,
    ) -> None:
        """Second load returns cached content without disk read."""
        content1 = loader.load("greeting")
        content2 = loader.load("greeting")
        assert content1 == content2
        assert content1 is content2  # Same object (cached)

    def test_load_multiple_templates(
        self,
        loader: PromptLoader,
    ) -> None:
        """Load different templates independently."""
        assert loader.load("greeting") == "Hello, {name}!"
        assert loader.load("multi") == ("Topic: {topic}\nRules: {rules}")


class TestPromptLoaderRender:
    """Tests for PromptLoader.render() 🎯."""

    def test_render_with_valid_kwargs(
        self,
        loader: PromptLoader,
    ) -> None:
        """Render replaces placeholders correctly."""
        result = loader.render("greeting", name="World")
        assert result == "Hello, World!"

    def test_render_multiple_placeholders(
        self,
        loader: PromptLoader,
    ) -> None:
        """Render handles multiple placeholders."""
        result = loader.render("multi", topic="AI Safety", rules="Be careful")
        assert result == "Topic: AI Safety\nRules: Be careful"

    def test_render_missing_placeholder_raises(
        self,
        loader: PromptLoader,
    ) -> None:
        """Render with missing kwargs raises KeyError."""
        with pytest.raises(KeyError):
            loader.render("greeting")  # Missing 'name'

    def test_render_extra_kwargs_ignored(
        self,
        loader: PromptLoader,
    ) -> None:
        """Render ignores extra kwargs not in template."""
        result = loader.render("greeting", name="World", extra="unused")
        assert result == "Hello, World!"


class TestPromptLoaderCacheManagement:
    """Tests for cache clearing and thread safety 🔒."""

    def test_clear_cache_forces_reload(
        self,
        loader: PromptLoader,
        tmp_template_dir: Path,
    ) -> None:
        """clear_cache forces re-read from disk."""
        # 📖 Load and cache
        content1 = loader.load("greeting")
        assert content1 == "Hello, {name}!"

        # 🔧 Modify template on disk
        (tmp_template_dir / "greeting.md").write_text("Hi, {name}!")

        # ❌ Still returns cached version
        assert loader.load("greeting") == "Hello, {name}!"

        # 🗑️ Clear cache
        loader.clear_cache()

        # ✅ Now returns updated version
        assert loader.load("greeting") == "Hi, {name}!"

    def test_thread_safety_concurrent_loads(
        self,
        loader: PromptLoader,
    ) -> None:
        """Concurrent loads do not corrupt cache 🔒."""
        results: list[str] = []
        errors: list[Exception] = []

        def load_template() -> None:
            try:
                result = loader.load("greeting")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_template) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20
        assert all(r == "Hello, {name}!" for r in results)


class TestPromptLoaderDefaultDir:
    """Tests for default template directory behavior 📁."""

    def test_default_dir_loads_real_templates(self) -> None:
        """Default PromptLoader can load built-in templates."""
        loader = PromptLoader()
        # ✅ These are real templates in inquiro/prompts/
        content = loader.load("search_system")
        assert len(content) > 100  # Non-trivial content
