"""Inquiro trace ID propagation via contextvars 🔍.

Provides cross-layer trace correlation so that every log line from a
single task execution can be filtered by ``trace_id``.  Uses Python's
:mod:`contextvars` module, which automatically propagates context into
``asyncio.to_thread()`` on Python 3.12+ and can be manually propagated
via ``copy_context().run()`` on earlier versions.

Usage::

    from inquiro.infrastructure.tracing import (
        install_trace_filter,
        set_trace_context,
    )

    # 🔧 Install once at startup (idempotent)
    install_trace_filter()

    # 🎯 Set at task start (Runner layer)
    set_trace_context(trace_id="task-abc-123", evaluation_id="eval-42")

    # 📝 All subsequent log records now carry trace_id / evaluation_id
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

# ============================================================
# 🔍 Context Variables
# ============================================================

_trace_id_var: ContextVar[str] = ContextVar(
    "trace_id",
    default="",
)
_evaluation_id_var: ContextVar[str] = ContextVar(
    "evaluation_id",
    default="",
)


# ============================================================
# 🎯 Public API — getters / setters
# ============================================================


def set_trace_context(
    trace_id: str,
    evaluation_id: str = "",
) -> None:
    """Set the trace context for the current execution scope 🎯.

    Should be called at the beginning of each task in the Runner layer.
    The values propagate automatically through ``asyncio.to_thread()``
    on Python 3.12+ and through ``copy_context().run()`` on earlier
    versions.

    Args:
        trace_id: Unique identifier for the task execution, typically
            the ``task_id`` or a generated UUID.
        evaluation_id: Optional evaluation-level identifier for
            grouping multiple sub-tasks under one evaluation run.
    """
    _trace_id_var.set(trace_id)
    _evaluation_id_var.set(evaluation_id)


def get_trace_id() -> str:
    """Return the current trace ID from context 🔍.

    Returns:
        The trace ID string, or empty string if no context is set.
    """
    return _trace_id_var.get()


def get_evaluation_id() -> str:
    """Return the current evaluation ID from context 🔍.

    Returns:
        The evaluation ID string, or empty string if not set.
    """
    return _evaluation_id_var.get()


# ============================================================
# 📝 Logging Filter
# ============================================================

# 🏷️ Sentinel attribute used for idempotent installation
_FILTER_INSTALLED_ATTR = "_inquiro_trace_filter_installed"


class TraceFilter(logging.Filter):
    """Logging filter that auto-attaches trace context to records 📝.

    Adds ``trace_id`` and ``evaluation_id`` attributes to every
    :class:`logging.LogRecord` that passes through the filter.
    Formatters can then reference these fields via ``%(trace_id)s``
    and ``%(evaluation_id)s``.

    The filter never rejects records — it only enriches them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich the log record with trace context attributes 📝.

        Args:
            record: The log record to enrich.

        Returns:
            Always True (record is never suppressed).
        """
        record.trace_id = _trace_id_var.get()  # type: ignore[attr-defined]
        record.evaluation_id = _evaluation_id_var.get()  # type: ignore[attr-defined]
        return True


# ============================================================
# 🔧 Installation Utility
# ============================================================


def install_trace_filter(
    logger_name: str = "inquiro",
) -> None:
    """Install :class:`TraceFilter` on the specified logger 🔧.

    Safe to call multiple times — skips installation if the filter
    is already present on the target logger.

    Args:
        logger_name: Name of the logger to attach the filter to.
            Defaults to ``"inquiro"`` (the engine root logger).
    """
    logger = logging.getLogger(logger_name)

    # 🛡️ Idempotent: skip if already installed
    if getattr(logger, _FILTER_INSTALLED_ATTR, False):
        return

    logger.addFilter(TraceFilter())
    setattr(logger, _FILTER_INSTALLED_ATTR, True)
