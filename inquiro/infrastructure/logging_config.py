"""Inquiro structured logging configuration 📝.

Provides centralized logging setup for the Inquiro engine with:
- Console handler with human-readable formatting (dev mode)
- JSON handler with structured output (production mode)
- Configurable log levels per module
- Emoji-prefixed log format for readability

Usage::

    from inquiro.infrastructure.logging_config import setup_logging

    # 🎯 Development mode (default)
    setup_logging()

    # 🏭 Production mode with JSON output
    setup_logging(json_output=True, level="INFO")

    # 🔧 Custom level for specific loggers
    setup_logging(level="INFO", module_levels={
        "inquiro.core.runner": "DEBUG",
        "inquiro.infrastructure.mcp_pool": "WARNING",
    })
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


# ============================================================
# 📝 Custom Formatters
# ============================================================


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors and structure 🎨.

    Format: ``TIMESTAMP | LEVEL | LOGGER | MESSAGE``
    Uses ANSI colors for level names when outputting to a terminal.
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
        """Initialize ConsoleFormatter 🔧.

        Args:
            use_color: Whether to use ANSI color codes. Defaults to
                True; set False for non-terminal output.
        """
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for console display 📝.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        ts = datetime.fromtimestamp(
            record.created,
            tz=timezone.utc,
        ).strftime("%Y-%m-%d %H:%M:%S")

        level = record.levelname.ljust(8)

        if self._use_color:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            level = f"{color}{level}{self.RESET}"

        name = record.name
        msg = record.getMessage()

        return f"{ts} | {level} | {name} | {msg}"


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production environments 🏭.

    Outputs one JSON object per line with standardized fields:
    ``timestamp``, ``level``, ``logger``, ``message``, and optional
    ``exc_info`` for exceptions.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON line 📝.

        Args:
            record: The log record to format.

        Returns:
            Single-line JSON string.
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

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(
                record.exc_info,
            )

        return json.dumps(log_entry, ensure_ascii=False)


# ============================================================
# 🔧 Setup Function
# ============================================================

# 📋 Default log levels for noisy third-party loggers
_QUIET_LOGGERS: dict[str, int] = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "uvicorn": logging.INFO,
    "uvicorn.access": logging.WARNING,
    "asyncio": logging.WARNING,
    "urllib3": logging.WARNING,
}


def setup_logging(
    level: str | int = "INFO",
    json_output: bool = False,
    module_levels: dict[str, str | int] | None = None,
) -> None:
    """Configure structured logging for the Inquiro engine 🔧.

    Sets up the root ``inquiro`` logger with appropriate handlers
    and formatters. Safe to call multiple times — existing handlers
    are removed before reconfiguration.

    Args:
        level: Default log level for the ``inquiro`` logger hierarchy.
            Accepts string (e.g., "DEBUG", "INFO") or int constants.
        json_output: If True, use JSON formatter for structured output
            suitable for log aggregation systems. If False, use
            human-readable console format with ANSI colors.
        module_levels: Optional per-module log level overrides.
            Keys are logger names (e.g., "inquiro.core.runner"),
            values are level strings or int constants.

    Example::

        setup_logging(
            level="INFO",
            json_output=False,
            module_levels={
                "inquiro.core.runner": "DEBUG",
                "inquiro.infrastructure.mcp_pool": "WARNING",
            },
        )
    """
    # 🎯 Resolve string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # 📦 Configure the inquiro root logger
    ds_logger = logging.getLogger("inquiro")
    ds_logger.setLevel(level)

    # 🧹 Remove existing handlers to allow reconfiguration
    ds_logger.handlers.clear()

    # 🔧 Create handler with appropriate formatter
    handler = logging.StreamHandler(sys.stderr)

    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        handler.setFormatter(ConsoleFormatter(use_color=use_color))

    handler.setLevel(level)
    ds_logger.addHandler(handler)

    # 🔇 Prevent propagation to root logger (avoid duplicate output)
    ds_logger.propagate = False

    # 🔇 Quiet down noisy third-party loggers
    for logger_name, quiet_level in _QUIET_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(quiet_level)

    # 🎛️ Apply per-module level overrides
    if module_levels:
        for logger_name, mod_level in module_levels.items():
            if isinstance(mod_level, str):
                mod_level = getattr(
                    logging,
                    mod_level.upper(),
                    logging.INFO,
                )
            logging.getLogger(logger_name).setLevel(mod_level)

    ds_logger.debug(
        "📝 Logging configured: level=%s, json=%s",
        logging.getLevelName(level),
        json_output,
    )
