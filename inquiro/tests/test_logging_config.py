"""Unit tests for Inquiro logging configuration 📝.

Tests ConsoleFormatter, JSONFormatter, and setup_logging().
"""

from __future__ import annotations

import json
import logging


from inquiro.infrastructure.logging_config import (
    ConsoleFormatter,
    JSONFormatter,
    setup_logging,
)


class TestConsoleFormatter:
    """Tests for ConsoleFormatter 🎨."""

    def test_format_output_structure(self) -> None:
        """Output contains timestamp, level, logger name, message."""
        formatter = ConsoleFormatter(use_color=False)
        record = logging.LogRecord(
            name="inquiro.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "inquiro.test" in output
        assert "Hello world" in output
        assert "|" in output  # Pipe separator

    def test_format_with_color(self) -> None:
        """Color mode includes ANSI escape codes."""
        formatter = ConsoleFormatter(use_color=True)
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[31m" in output  # Red for ERROR
        assert "\033[0m" in output  # Reset code

    def test_format_without_color(self) -> None:
        """No-color mode omits ANSI escape codes."""
        formatter = ConsoleFormatter(use_color=False)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[" not in output


class TestJSONFormatter:
    """Tests for JSONFormatter 🏭."""

    def test_produces_valid_json(self) -> None:
        """Output is valid JSON with required fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="inquiro.core",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Task started",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "inquiro.core"
        assert data["message"] == "Task started"
        assert "timestamp" in data

    def test_includes_exception_info(self) -> None:
        """Exception info is captured when present."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Something failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_no_exception_field_when_none(self) -> None:
        """No 'exception' key when exc_info is None."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="OK",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" not in data

    def test_lazy_formatting(self) -> None:
        """Formatter handles %-style lazy formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Count: %d items",
            args=(42,),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Count: 42 items"


class TestSetupLogging:
    """Tests for setup_logging() function 🔧."""

    def teardown_method(self) -> None:
        """Reset inquiro logger after each test 🧹."""
        logger = logging.getLogger("inquiro")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_configures_inquiro_logger(self) -> None:
        """setup_logging configures the inquiro logger."""
        setup_logging(level="DEBUG")
        logger = logging.getLogger("inquiro")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

    def test_json_output_mode(self) -> None:
        """json_output=True uses JSONFormatter."""
        setup_logging(json_output=True)
        logger = logging.getLogger("inquiro")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_console_output_mode(self) -> None:
        """json_output=False uses ConsoleFormatter."""
        setup_logging(json_output=False)
        logger = logging.getLogger("inquiro")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, ConsoleFormatter)

    def test_module_level_overrides(self) -> None:
        """Per-module level overrides work correctly."""
        setup_logging(
            level="INFO",
            module_levels={
                "inquiro.core.runner": "DEBUG",
                "inquiro.infrastructure.mcp_pool": "WARNING",
            },
        )
        runner_logger = logging.getLogger("inquiro.core.runner")
        mcp_logger = logging.getLogger("inquiro.infrastructure.mcp_pool")
        assert runner_logger.level == logging.DEBUG
        assert mcp_logger.level == logging.WARNING

    def test_idempotent_reconfiguration(self) -> None:
        """Calling setup_logging twice is safe (no duplicate handlers)."""
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")
        logger = logging.getLogger("inquiro")
        assert len(logger.handlers) == 1  # Not 2
        assert logger.level == logging.DEBUG

    def test_string_level_parsing(self) -> None:
        """String levels are correctly parsed to int."""
        setup_logging(level="WARNING")
        logger = logging.getLogger("inquiro")
        assert logger.level == logging.WARNING
