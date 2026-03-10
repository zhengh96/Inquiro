"""Inquiro LLMProviderPool — service-level LLM provider management 🤖.

Manages LLM client instances with lazy initialization and caching.
Supports multiple providers (OpenAI-compatible, Anthropic) via
EvoMaster's BaseLLM interface.

Example::

    pool = LLMProviderPool(config={
        "providers": {
            "claude-sonnet": {
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-20250514",
                "api_key": "sk-...",
            }
        },
        "default_model": "claude-sonnet",
    })
    llm = pool.get_llm("claude-sonnet")
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from evomaster.utils.llm import BaseLLM, LLMConfig, create_llm

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when LLM provider initialization or lookup fails ❌."""


class LLMProviderPool:
    """Service-level LLM provider pool 🤖.

    Manages LLM client instances with lazy initialization and reuse.
    Each model identifier maps to a provider configuration, and the
    actual BaseLLM instance is created on first use.

    Thread-safe: all mutable state is protected by a lock.

    Attributes:
        default_model: Model name returned when no specific model requested.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize LLMProviderPool 🔧.

        Args:
            config: Provider configuration dict with structure::

                {
                    "providers": {
                        "<model_name>": {
                            "provider": "openai" | "anthropic" | "deepseek",
                            "model_id": "<model identifier>",
                            "api_key": "<API key>",
                            "base_url": "<optional base URL>",
                            "temperature": 0.7,
                            "max_tokens": 4096,
                            "timeout": 300,
                        },
                        ...
                    },
                    "default_model": "<model_name>",
                }

                If None, pool starts empty (stub mode).
        """
        self._config = config or {}
        self._providers: dict[str, dict[str, Any]] = self._config.get("providers", {})
        self.default_model: str = self._config.get("default_model", "")
        self._instances: dict[str, BaseLLM] = {}
        self._lock = threading.Lock()
        logger.info(
            "🤖 LLMProviderPool initialized with %d provider(s), default=%s",
            len(self._providers),
            self.default_model or "(none)",
        )

    def get_llm(self, model: str = "") -> BaseLLM:
        """Get or create an LLM instance for the specified model 🔧.

        Lazy initialization: the client is only created on first request.
        Subsequent calls for the same model return the cached instance.

        Args:
            model: Model name matching a key in ``providers``. If empty
                or not found, falls back to ``default_model``.

        Returns:
            BaseLLM instance ready for ``query()`` calls.

        Raises:
            LLMProviderError: If no provider config found for the model.
        """
        effective_model = model or self.default_model
        if not effective_model:
            raise LLMProviderError("No model specified and no default_model configured")

        with self._lock:
            if effective_model in self._instances:
                return self._instances[effective_model]

        # 🏗️ Build LLM instance outside the lock to avoid blocking
        provider_config = self._providers.get(effective_model)
        if provider_config is None:
            # 🔄 Try fallback to default
            if effective_model != self.default_model and self.default_model:
                logger.warning(
                    "⚠️ Model '%s' not found, falling back to default '%s'",
                    effective_model,
                    self.default_model,
                )
                return self.get_llm(self.default_model)
            raise LLMProviderError(
                f"No provider configuration for model '{effective_model}'. "
                f"Available: {list(self._providers.keys())}"
            )

        llm = self._create_llm(effective_model, provider_config)

        with self._lock:
            # ✅ Double-check: another thread may have created it
            if effective_model not in self._instances:
                self._instances[effective_model] = llm
            return self._instances[effective_model]

    def _create_llm(self, model_name: str, provider_config: dict[str, Any]) -> BaseLLM:
        """Create a BaseLLM instance from provider config 🏗️.

        Args:
            model_name: Logical model name for logging.
            provider_config: Provider-specific config dict.

        Returns:
            Configured BaseLLM instance.

        Raises:
            LLMProviderError: If creation fails.
        """
        try:
            provider = provider_config.get("provider", "openai")

            # 🌩️ Bedrock requires special handling (AWS credentials).
            if provider == "bedrock":
                from inquiro.infrastructure.bedrock_llm import (
                    create_bedrock_llm,
                )

                llm = create_bedrock_llm(provider_config)
                logger.info("✅ Created Bedrock LLM instance: %s", model_name)
                return llm

            # 🔧 Normalize base_url to prevent double-slash issues
            base_url = provider_config.get("base_url")
            if base_url:
                base_url = base_url.rstrip("/")
                # Validate URL has proper protocol
                if not base_url.startswith(("http://", "https://")):
                    logger.warning(
                        "⚠️ Invalid base_url for '%s': %r — skipping model",
                        model_name,
                        base_url[:30],
                    )
                    raise LLMProviderError(
                        f"Invalid base_url for '{model_name}': "
                        f"must start with http:// or https://"
                    )

            llm_config = LLMConfig(
                provider=provider,
                model=provider_config.get("model_id", model_name),
                api_key=provider_config.get("api_key", ""),
                base_url=base_url,
                temperature=provider_config.get("temperature", 0.7),
                max_tokens=provider_config.get("max_tokens"),
                timeout=provider_config.get("timeout", 300),
                max_retries=provider_config.get("max_retries", 3),
            )
            llm = create_llm(llm_config)
            logger.info(
                "✅ Created LLM instance: %s (%s)", model_name, llm_config.provider
            )
            return llm
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to create LLM for '{model_name}': {exc}"
            ) from exc

    def get_available_models(self) -> list[str]:
        """Return the list of configured model names 📋.

        Returns:
            Sorted list of model name strings.
        """
        return sorted(self._providers.keys())

    def close(self) -> None:
        """Clean up all cached LLM instances 🧹.

        Calls ``close()`` on any instance that supports it.
        """
        with self._lock:
            for name, instance in self._instances.items():
                if hasattr(instance, "close"):
                    try:
                        instance.close()
                    except Exception as exc:
                        logger.warning("⚠️ Error closing LLM '%s': %s", name, exc)
            self._instances.clear()
        logger.info("🧹 LLMProviderPool closed")
