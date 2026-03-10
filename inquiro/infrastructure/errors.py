"""Inquiro error classification system 🔍.

Provides a hierarchy of errors enabling different retry strategies:
- TransientError: temporary failures that may succeed on retry (exponential backoff)
- PermanentError: failures that will not succeed on retry (immediate fail)

The ``classify_error`` function inspects any exception and returns
a classification string ("transient", "permanent", or "unknown")
that retry logic can use to decide whether to re-attempt an operation.

Typical usage::

    from inquiro.infrastructure.errors import classify_error

    try:
        result = do_something()
    except Exception as exc:
        kind = classify_error(exc)
        if kind == "transient":
            schedule_retry(exc)
        elif kind == "permanent":
            fail_immediately(exc)
        else:
            log_and_escalate(exc)
"""

from __future__ import annotations


# ====================================================================
# 🏗️ Base hierarchy
# ====================================================================


class InquiroError(Exception):
    """Base class for all Inquiro errors 🏗️.

    All domain-specific exceptions in Inquiro inherit from this class,
    enabling a single ``except InquiroError`` catch when broad handling
    is desired.
    """


class TransientError(InquiroError):
    """Temporary failure that may succeed on retry 🔄.

    Examples: timeout, connection reset, rate limit, MCP server down.
    Retry strategy: exponential backoff with jitter.
    """


class PermanentError(InquiroError):
    """Failure that will not succeed on retry ❌.

    Examples: validation error, auth failure, invalid config,
    schema mismatch, missing required field.
    Retry strategy: immediate fail, do not retry.
    """


# ====================================================================
# 🔄 Specific transient errors
# ====================================================================


class MCPTransientError(TransientError):
    """MCP server connection failure (transient) 🔌.

    Raised when an MCP server is temporarily unreachable or returns
    a retriable error (e.g., 503 Service Unavailable, connection reset).

    Note:
        The existing ``MCPConnectionError`` in ``persistent_connection.py``
        is preserved for backward compatibility. This class provides
        retry-aware classification on top of it.
    """


class RateLimitError(TransientError):
    """API rate limit exceeded ⏳.

    Raised when an upstream API (LLM provider, search service) returns
    HTTP 429 or an equivalent throttling signal.
    """


class LLMTimeoutError(TransientError):
    """LLM inference timeout ⏰.

    Raised when an LLM call exceeds its configured timeout without
    returning a response.
    """


# ====================================================================
# ❌ Specific permanent errors
# ====================================================================


class SchemaValidationError(PermanentError):
    """Output schema validation failure 📋.

    Raised when the agent output cannot be validated against the
    expected JSON schema (missing fields, wrong types, etc.).
    """


class ConfigurationError(PermanentError):
    """Invalid configuration ⚙️.

    Raised when the system configuration is invalid or incomplete
    (e.g., missing required config keys, malformed YAML).
    """


class AuthenticationError(PermanentError):
    """Authentication or authorization failure 🔒.

    Raised when an API key is invalid, expired, or lacks the
    required permissions.
    """


# ====================================================================
# 🔍 Classification function
# ====================================================================


def classify_error(exc: Exception) -> str:
    """Classify an exception as transient or permanent 🔍.

    Inspects the exception type hierarchy to determine whether the
    error is likely transient (worth retrying) or permanent (should
    fail immediately).

    Checks in order:
    1. Inquiro-specific hierarchy (``TransientError`` / ``PermanentError``).
    2. EvoMaster tool errors — ``ToolParameterError`` (permanent, bad params
       never self-heal) is checked **before** ``ToolError`` (transient,
       general tool failures may be retriable) because ``ToolParameterError``
       is a subclass of ``ToolError`` and the more specific check must come
       first.
    3. Well-known built-in exception patterns.

    A local import is used for EvoMaster types to avoid circular imports
    between the Inquiro infrastructure layer and EvoMaster core.

    Args:
        exc: The exception to classify.

    Returns:
        One of:
        - ``"transient"`` — the operation may succeed on retry.
        - ``"permanent"`` — the operation will not succeed on retry.
        - ``"unknown"`` — classification could not be determined;
          callers should apply a conservative retry policy.
    """
    # ✅ Inquiro-specific hierarchy takes precedence
    if isinstance(exc, TransientError):
        return "transient"
    if isinstance(exc, PermanentError):
        return "permanent"

    # 🔧 EvoMaster tool errors — local import avoids circular imports.
    # ToolParameterError MUST be checked before ToolError because it is a
    # subclass; reversing the order would incorrectly classify parameter
    # errors as transient.
    try:
        from evomaster.agent.tools.base import ToolError, ToolParameterError

        if isinstance(exc, ToolParameterError):
            return "permanent"
        if isinstance(exc, ToolError):
            return "transient"
    except ImportError:
        pass

    # 🔄 Well-known built-in transient patterns
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "transient"

    # ❌ Well-known built-in permanent patterns
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "permanent"

    # ❓ Cannot determine — caller decides policy
    return "unknown"
