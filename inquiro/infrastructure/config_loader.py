"""Inquiro ConfigLoader — YAML config loading with env var interpolation 📋.

Loads YAML configuration files from a specified directory and resolves
``${ENV_VAR}`` placeholders against the process environment. Provides
typed accessors for MCP, LLM, and service configuration sections.

Example::

    loader = ConfigLoader(Path("inquiro/configs"))
    mcp_config = loader.get_mcp_config()
    llm_config = loader.get_llm_config()
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 🔍 Pattern for ${ENV_VAR} or ${ENV_VAR:-default}
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

# 🔄 Backward-compatibility env var aliases (old -> canonical)
_ENV_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "OPENAI_BASE_URL": ("OPENAI_API_BASE",),
    "INQUIRO_BASE_URL": ("DIMSENSE_BASE_URL",),
    "BRAVE_SEARCH_API_KEY": ("BRAVE_API_KEY",),
}


def _resolve_env_var(var_name: str) -> str | None:
    """Resolve env var with alias fallback and deprecation warnings 🔍."""
    value = os.environ.get(var_name)
    if value not in (None, ""):
        return value

    aliases = _ENV_VAR_ALIASES.get(var_name, ())
    for alias in aliases:
        alias_value = os.environ.get(alias)
        if alias_value not in (None, ""):
            logger.warning(
                "⚠️ Environment variable %s is deprecated; use %s instead",
                alias,
                var_name,
            )
            return alias_value
    return None


def _interpolate_env_vars(value: Any) -> Any:
    """Recursively resolve ``${ENV_VAR}`` placeholders in a config tree 🔍.

    Supports default values via ``${VAR:-default}`` syntax.
    Non-string leaves are returned unchanged.

    Args:
        value: Config value — may be str, dict, list, or primitive.

    Returns:
        Resolved config value with all env vars expanded.
    """
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = _resolve_env_var(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            logger.warning(
                "⚠️ Environment variable %s not set and no default provided",
                var_name,
            )
            return ""  # Return empty string for unresolvable vars

        return _ENV_VAR_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


class ConfigLoadError(Exception):
    """Raised when configuration loading or parsing fails ❌."""


class ConfigLoader:
    """Load and resolve YAML configurations with env var interpolation 📋.

    Reads ``mcp_servers.yaml``, ``llm_providers.yaml``, and ``service.yaml``
    from *config_dir* at construction time. All ``${ENV_VAR}`` placeholders
    are resolved against ``os.environ``.

    Thread-safe: internal caches are protected by a lock.

    Attributes:
        config_dir: Root directory containing YAML config files.
    """

    def __init__(self, config_dir: str | Path) -> None:
        """Initialize ConfigLoader 🔧.

        Args:
            config_dir: Path to the directory containing YAML config files.

        Raises:
            ConfigLoadError: If the directory does not exist.
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise ConfigLoadError(f"Config directory does not exist: {self.config_dir}")
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        logger.info("📋 ConfigLoader initialized with dir: %s", self.config_dir)

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        """Load and cache a single YAML file with env var interpolation 📄.

        Args:
            filename: YAML filename relative to *config_dir*.

        Returns:
            Parsed and interpolated config dict.

        Raises:
            ConfigLoadError: If the file is missing or malformed.
        """
        with self._lock:
            if filename in self._cache:
                return self._cache[filename]

        filepath = self.config_dir / filename
        if not filepath.exists():
            raise ConfigLoadError(f"Config file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigLoadError(
                f"Failed to parse YAML file {filepath}: {exc}"
            ) from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigLoadError(
                f"Config file {filepath} root must be a mapping, "
                f"got {type(raw).__name__}"
            )

        resolved = _interpolate_env_vars(raw)

        with self._lock:
            self._cache[filename] = resolved

        logger.info("✅ Loaded config: %s", filename)
        return resolved

    # -- Public accessors ---------------------------------------------------

    def get_mcp_config(self) -> dict[str, Any]:
        """Load MCP server configuration 🔌.

        Returns:
            Resolved ``mcp_servers.yaml`` content.
        """
        return self._load_yaml("mcp_servers.yaml")

    def get_llm_config(self) -> dict[str, Any]:
        """Load LLM provider configuration 🤖.

        Returns:
            Resolved ``llm_providers.yaml`` content.
        """
        return self._load_yaml("llm_providers.yaml")

    def get_service_config(self) -> dict[str, Any]:
        """Load service configuration ⚙️.

        Returns:
            Resolved ``service.yaml`` content.
        """
        return self._load_yaml("service.yaml")

    def get_ensemble_config(self) -> dict[str, Any]:
        """Load ensemble configuration 🎭.

        Returns:
            Resolved ``ensemble.yaml`` content, or empty dict
            if the file does not exist (ensemble is optional).
        """
        try:
            return self._load_yaml("ensemble.yaml")
        except ConfigLoadError:
            logger.info("ℹ️ ensemble.yaml not found — ensemble disabled")
            return {}

    def reload(self) -> None:
        """Clear cache and force reload on next access 🔄."""
        with self._lock:
            self._cache.clear()
        logger.info("🔄 Config cache cleared")
