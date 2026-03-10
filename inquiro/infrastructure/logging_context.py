"""Inquiro Logging Context -- task-aware structured logging 📋.

Uses ``contextvars`` to propagate task identity through async and
threaded call stacks, and provides a custom formatter that includes
context fields in every log line.

Typical usage::

    from inquiro.infrastructure.logging_context import (
        set_logging_context,
        clear_logging_context,
        install_context_logging,
    )

    # 🔧 Install once at startup
    install_context_logging()

    # 🏷️ Set context at Runner / Exp level
    set_logging_context(
        task_id="task-abc123",
        evaluation_id="eval-001",
        sub_item="novelty",
    )

    # ... all downstream log calls automatically include context ...

    # 🧹 Clean up when task completes
    clear_logging_context()
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any


# ============================================================
# 🏷️ Context variables for task identification
# ============================================================

_task_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "task_id", default=""
)
_evaluation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "evaluation_id", default=""
)
_sub_item_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sub_item", default=""
)


# ============================================================
# 🔧 Context accessors
# ============================================================


def set_logging_context(
    task_id: str = "",
    evaluation_id: str = "",
    sub_item: str = "",
) -> None:
    """Set the logging context for the current execution scope 🏷️.

    Only non-empty values are applied, allowing partial updates
    (e.g. setting ``sub_item`` without changing ``task_id``).

    Args:
        task_id: Unique task identifier.
        evaluation_id: Parent evaluation identifier.
        sub_item: Sub-item being evaluated (e.g., checklist key).
    """
    if task_id:
        _task_id_var.set(task_id)
    if evaluation_id:
        _evaluation_id_var.set(evaluation_id)
    if sub_item:
        _sub_item_var.set(sub_item)


def get_logging_context() -> dict[str, str]:
    """Get the current logging context as a dict 📋.

    Returns:
        Dict with ``task_id``, ``evaluation_id``, ``sub_item`` values.
    """
    return {
        "task_id": _task_id_var.get(),
        "evaluation_id": _evaluation_id_var.get(),
        "sub_item": _sub_item_var.get(),
    }


def clear_logging_context() -> None:
    """Reset all logging context variables to empty strings 🧹."""
    _task_id_var.set("")
    _evaluation_id_var.set("")
    _sub_item_var.set("")


# ============================================================
# 🔍 Logging filter — injects context into LogRecords
# ============================================================


class ContextFilter(logging.Filter):
    """Logging filter that injects task context into log records 🔍.

    Adds ``task_id``, ``evaluation_id``, and ``sub_item`` attributes
    to every log record so formatters can include them in output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to the log record 📝.

        Args:
            record: The log record to augment.

        Returns:
            Always True — this filter never suppresses records.
        """
        record.task_id = _task_id_var.get()  # type: ignore[attr-defined]
        record.evaluation_id = (  # type: ignore[attr-defined]
            _evaluation_id_var.get()
        )
        record.sub_item = _sub_item_var.get()  # type: ignore[attr-defined]
        return True


# ============================================================
# 🎨 Context-aware formatters
# ============================================================


class ContextConsoleFormatter(logging.Formatter):
    """Console formatter that includes task context when available 🎨.

    Extends the base format with ``[task_id=X sub_item=Y]`` prefix
    when context is set. Falls back to standard format when no
    context is active.

    Format::

        TIMESTAMP | LEVEL | LOGGER [task_id=X sub_item=Y] | MESSAGE
    """

    # 🎨 ANSI color codes for log levels
    LEVEL_COLORS: dict[int, str] = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        """Initialize ContextConsoleFormatter 🔧.

        Args:
            use_color: Whether to use ANSI color codes. Defaults to
                True; set False for non-terminal output.
        """
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with context for console display 📝.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with task context prefix.
        """
        ts = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")

        level = record.levelname.ljust(8)
        if self._use_color:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            level = f"{color}{level}{self.RESET}"

        # 🏷️ Build context prefix from injected attributes
        ctx_parts: list[str] = []
        task_id = getattr(record, "task_id", "")
        sub_item = getattr(record, "sub_item", "")
        if task_id:
            ctx_parts.append(f"task_id={task_id}")
        if sub_item:
            ctx_parts.append(f"sub_item={sub_item}")

        ctx_prefix = f" [{' '.join(ctx_parts)}]" if ctx_parts else ""
        msg = record.getMessage()

        return f"{ts} | {level} | {record.name}{ctx_prefix} | {msg}"


class ContextJSONFormatter(logging.Formatter):
    """JSON formatter that includes task context fields 🏭.

    Outputs one JSON object per line with task context included
    when present. Suitable for log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON line with context 📝.

        Args:
            record: The log record to format.

        Returns:
            Single-line JSON string with context fields.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 🏷️ Add context fields if present
        task_id = getattr(record, "task_id", "")
        evaluation_id = getattr(record, "evaluation_id", "")
        sub_item = getattr(record, "sub_item", "")
        if task_id:
            log_entry["task_id"] = task_id
        if evaluation_id:
            log_entry["evaluation_id"] = evaluation_id
        if sub_item:
            log_entry["sub_item"] = sub_item

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(
                record.exc_info,
            )

        return json.dumps(log_entry, ensure_ascii=False)


# ============================================================
# 🔧 Installation helper
# ============================================================


def install_context_logging(
    logger_name: str = "inquiro",
    json_output: bool = False,
) -> None:
    """Install context-aware logging on the specified logger 🔧.

    Adds ``ContextFilter`` and optionally replaces the formatter on
    existing handlers with a context-aware version. Safe to call
    multiple times — subsequent calls are no-ops.

    Args:
        logger_name: Logger name to configure.
        json_output: Use JSON formatter if True, console otherwise.
    """
    logger = logging.getLogger(logger_name)

    # 🔍 Check if filter already installed (idempotent guard)
    for f in logger.filters:
        if isinstance(f, ContextFilter):
            return

    logger.addFilter(ContextFilter())
