"""Inquiro DegradationManager -- graceful fallback strategy 🛡️.

Implements three-level degradation when components fail:
LLM -> MCP -> Functional, with event emission and metadata tracking.

Degradation Levels:
    1. LLM degradation: Opus -> Sonnet -> Haiku (on timeout/error)
    2. MCP degradation: persistent -> per-call -> disabled (web-search-only)
    3. Functional degradation: partial result with "limited_information" flag

Thread-safe: mutable state protected by ``threading.Lock``.
"""

from __future__ import annotations

import enum
import logging
import threading
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class DegradationLevel(str, enum.Enum):
    """Degradation severity levels 📉.

    Attributes:
        NONE: No degradation, running at full capacity.
        LLM_DOWNGRADE: Using a cheaper/faster LLM model.
        MCP_FALLBACK: Using fallback MCP connection mode.
        FUNCTIONAL: Limited functionality, returning partial results.
    """

    NONE = "none"
    LLM_DOWNGRADE = "llm_downgrade"
    MCP_FALLBACK = "mcp_fallback"
    FUNCTIONAL = "functional"


class DegradationEvent(BaseModel):
    """Record of a single degradation event 📋.

    Attributes:
        level: The degradation level triggered.
        component: Which component degraded (e.g., "llm", "mcp").
        original: The original configuration/value before fallback.
        fallback: The fallback configuration/value after degradation.
        reason: Why degradation was triggered.
    """

    level: DegradationLevel = Field(description="The degradation level triggered.")
    component: str = Field(description="Which component degraded (e.g., 'llm', 'mcp').")
    original: str = Field(
        description="The original configuration/value before fallback."
    )
    fallback: str = Field(
        description="The fallback configuration/value after degradation."
    )
    reason: str = Field(description="Why degradation was triggered.")


# ---------------------------------------------------------------------------
# DegradationManager
# ---------------------------------------------------------------------------


class DegradationManager:
    """Manages graceful degradation across LLM, MCP, and functional layers 🛡️.

    Tracks degradation state and provides fallback suggestions when
    components fail. Emits events to ``EventEmitter`` for observability
    and records all degradation history for result metadata.

    LLM Fallback Chain::

        claude-opus-4-20250514 -> claude-sonnet-4-20250514
                               -> claude-haiku-4-5-20251001

    MCP Fallback Chain::

        persistent -> per-call -> disabled (web-search-only)

    Functional Fallback::

        Full result -> partial result with "limited_information" flag

    Thread-safe: all mutable state is protected by a lock.

    Example::

        mgr = DegradationManager(event_emitter=emitter)
        fallback = mgr.suggest_llm_fallback("timeout after 30s")
        # fallback == "claude-sonnet-4-20250514"
        meta = mgr.degradation_metadata
        # meta["is_degraded"] == True
    """

    # 🤖 Default LLM fallback chain (can be configured)
    DEFAULT_LLM_CHAIN: list[str] = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ]

    # 🔌 Default MCP fallback chain
    DEFAULT_MCP_CHAIN: list[str] = [
        "persistent",
        "per-call",
        "disabled",
    ]

    def __init__(
        self,
        llm_chain: list[str] | None = None,
        mcp_chain: list[str] | None = None,
        event_emitter: Any | None = None,
    ) -> None:
        """Initialize DegradationManager 🔧.

        Args:
            llm_chain: LLM model fallback chain. Defaults to
                Opus -> Sonnet -> Haiku.
            mcp_chain: MCP connection mode fallback chain. Defaults to
                persistent -> per-call -> disabled.
            event_emitter: Optional ``EventEmitter`` instance for
                degradation event emission.
        """
        self._llm_chain = list(llm_chain) if llm_chain else list(self.DEFAULT_LLM_CHAIN)
        self._mcp_chain = list(mcp_chain) if mcp_chain else list(self.DEFAULT_MCP_CHAIN)
        self._event_emitter = event_emitter
        self._degradation_history: list[DegradationEvent] = []
        self._current_llm_index: int = 0
        self._current_mcp_index: int = 0
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    # -- Properties ----------------------------------------------------------

    @property
    def current_llm(self) -> str:
        """Get the currently active LLM model 🤖.

        Returns:
            Model name string from the LLM fallback chain.
        """
        with self._lock:
            idx = self._current_llm_index
        if idx < len(self._llm_chain):
            return self._llm_chain[idx]
        return self._llm_chain[-1]

    @property
    def current_mcp_mode(self) -> str:
        """Get the currently active MCP connection mode 🔌.

        Returns:
            MCP mode string from the MCP fallback chain.
        """
        with self._lock:
            idx = self._current_mcp_index
        if idx < len(self._mcp_chain):
            return self._mcp_chain[idx]
        return self._mcp_chain[-1]

    @property
    def is_degraded(self) -> bool:
        """Check if any degradation has occurred 📉.

        Returns:
            True if LLM or MCP has been downgraded from default.
        """
        with self._lock:
            return (
                self._current_llm_index > 0
                or self._current_mcp_index > 0
                or any(
                    e.level == DegradationLevel.FUNCTIONAL
                    for e in self._degradation_history
                )
            )

    @property
    def degradation_metadata(self) -> dict[str, Any]:
        """Get degradation status for inclusion in result metadata 📋.

        Returns:
            Dict containing degradation state summary including
            current LLM, current MCP mode, event count, and
            serialized event history.
        """
        with self._lock:
            history_snapshot = list(self._degradation_history)
            llm_idx = self._current_llm_index
            mcp_idx = self._current_mcp_index

        # ✨ Compute current values from indices
        current_llm = (
            self._llm_chain[llm_idx]
            if llm_idx < len(self._llm_chain)
            else self._llm_chain[-1]
        )
        current_mcp = (
            self._mcp_chain[mcp_idx]
            if mcp_idx < len(self._mcp_chain)
            else self._mcp_chain[-1]
        )

        return {
            "is_degraded": (
                llm_idx > 0
                or mcp_idx > 0
                or any(e.level == DegradationLevel.FUNCTIONAL for e in history_snapshot)
            ),
            "current_llm": current_llm,
            "current_mcp_mode": current_mcp,
            "degradation_count": len(history_snapshot),
            "degradation_events": [e.model_dump() for e in history_snapshot],
        }

    # -- LLM Degradation -----------------------------------------------------

    def suggest_llm_fallback(self, reason: str) -> str | None:
        """Suggest next LLM in fallback chain 🤖.

        Advances the LLM index by one step and records a degradation
        event. Returns None if the chain is already exhausted.

        Args:
            reason: Why the current LLM failed (e.g., "timeout",
                "rate_limit", "server_error").

        Returns:
            Next LLM model name, or None if chain is exhausted.
        """
        with self._lock:
            next_idx = self._current_llm_index + 1
            if next_idx >= len(self._llm_chain):
                self._logger.warning("⚠️ LLM fallback chain exhausted, no more options")
                return None

            original = self._llm_chain[self._current_llm_index]
            self._current_llm_index = next_idx
            fallback = self._llm_chain[next_idx]

        event = DegradationEvent(
            level=DegradationLevel.LLM_DOWNGRADE,
            component="llm",
            original=original,
            fallback=fallback,
            reason=reason,
        )
        self._record_event(event)

        self._logger.warning(
            "🤖 LLM degradation: %s -> %s (reason: %s)",
            original,
            fallback,
            reason,
        )
        return fallback

    # -- MCP Degradation -----------------------------------------------------

    def suggest_mcp_fallback(self, reason: str) -> str | None:
        """Suggest next MCP mode in fallback chain 🔌.

        Advances the MCP index by one step and records a degradation
        event. Returns None if the chain is already exhausted.

        Args:
            reason: Why the current MCP mode failed (e.g.,
                "connection_reset", "timeout").

        Returns:
            Next MCP mode name, or None if chain is exhausted.
        """
        with self._lock:
            next_idx = self._current_mcp_index + 1
            if next_idx >= len(self._mcp_chain):
                self._logger.warning("⚠️ MCP fallback chain exhausted, no more options")
                return None

            original = self._mcp_chain[self._current_mcp_index]
            self._current_mcp_index = next_idx
            fallback = self._mcp_chain[next_idx]

        event = DegradationEvent(
            level=DegradationLevel.MCP_FALLBACK,
            component="mcp",
            original=original,
            fallback=fallback,
            reason=reason,
        )
        self._record_event(event)

        self._logger.warning(
            "🔌 MCP degradation: %s -> %s (reason: %s)",
            original,
            fallback,
            reason,
        )
        return fallback

    # -- Functional Degradation ----------------------------------------------

    def mark_functional_degradation(
        self,
        reason: str,
    ) -> DegradationEvent:
        """Mark that functional degradation is in effect 📉.

        This is the last resort: the system returns a partial result
        with a ``limited_information`` flag rather than failing outright.

        Args:
            reason: Why functional degradation was triggered (e.g.,
                "all MCP servers unavailable", "LLM chain exhausted").

        Returns:
            The ``DegradationEvent`` that was recorded.
        """
        event = DegradationEvent(
            level=DegradationLevel.FUNCTIONAL,
            component="functional",
            original="full",
            fallback="limited",
            reason=reason,
        )
        self._record_event(event)

        self._logger.warning(
            "📉 Functional degradation: %s",
            reason,
        )
        return event

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Reset all degradation state to initial values 🔄.

        Clears LLM/MCP indices back to 0 and empties event history.
        Useful between task runs or after recovery.
        """
        with self._lock:
            self._current_llm_index = 0
            self._current_mcp_index = 0
            self._degradation_history.clear()

        self._logger.info("🔄 Degradation state reset")

    # -- Internal helpers ----------------------------------------------------

    def _record_event(self, event: DegradationEvent) -> None:
        """Record a degradation event and emit to EventEmitter 📝.

        Args:
            event: The degradation event to record and emit.
        """
        with self._lock:
            self._degradation_history.append(event)

        if self._event_emitter is not None:
            try:
                self._event_emitter.emit(
                    "degradation",
                    "",  # 📝 No specific task_id for degradation events
                    event.model_dump(),
                )
            except Exception as exc:
                self._logger.warning(
                    "⚠️ Failed to emit degradation event: %s",
                    exc,
                )
