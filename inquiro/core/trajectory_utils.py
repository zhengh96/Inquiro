"""Shared trajectory utility functions for Inquiro 🔧.

Provides reusable helpers for extracting structured data from EvoMaster
agent trajectories. These utilities are consumed by both the Exp layer
(InquiroBaseExp) and the evolution layer (TrajectoryCollector) to avoid
duplicating finish-tool parsing logic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_finish_result(trajectory: Any) -> dict[str, Any]:
    """Extract the structured result dict from a finish tool call 📝.

    Searches trajectory steps in reverse order (most recent first) to
    locate the "finish" tool call, then applies a two-level JSON parse:

    1. Outer parse: ``tool_call.function.arguments`` → dict (may be a
       JSON string or already a dict when the LLM returns pre-parsed args).
    2. Inner parse: ``args["result_json"]`` → dict (the actual result
       payload, encoded as a nested JSON string by the finish tool schema).

    Fallback: if the inner ``result_json`` parse fails, attempts to return
    the outer args dict directly (handles agents that skip the double-encode
    pattern).

    Supports both object-style trajectories (with ``.steps`` and
    ``.assistant_message`` attributes, as produced by EvoMaster's
    ``Trajectory`` class) and dict-style trajectories (with ``"steps"``
    and ``"assistant_message"`` keys, as may appear in serialised records).

    ``strict=False`` is used throughout so that control characters
    (newlines, tabs) embedded in LLM-generated strings do not cause
    ``JSONDecodeError``.

    Args:
        trajectory: Agent execution trajectory. May be an object with a
            ``.steps`` attribute or a dict with a ``"steps"`` key. ``None``
            and empty trajectories are handled gracefully.

    Returns:
        The parsed result dictionary extracted from the finish tool call.
        Returns an empty dict if no finish tool call is found, if parsing
        fails at all levels, or if the trajectory is ``None`` / empty.
    """
    if trajectory is None:
        return {}

    # 🔄 Resolve steps — support both object-style and dict-style trajectories
    if isinstance(trajectory, dict):
        steps = trajectory.get("steps") or []
    else:
        steps = getattr(trajectory, "steps", None) or []

    if not steps:
        return {}

    # 🔍 Search in reverse for the most recent finish tool call
    for step in reversed(steps):
        # -- Resolve assistant_message (object or dict) --------------------
        if isinstance(step, dict):
            assistant_message = step.get("assistant_message")
        else:
            assistant_message = getattr(step, "assistant_message", None)

        if not assistant_message:
            continue

        # -- Resolve tool_calls (object or dict) ---------------------------
        if isinstance(assistant_message, dict):
            tool_calls = assistant_message.get("tool_calls") or []
        else:
            tool_calls = getattr(assistant_message, "tool_calls", None) or []

        if not tool_calls:
            continue

        for tool_call in tool_calls:
            # -- Resolve function name and arguments ----------------------
            if isinstance(tool_call, dict):
                function = tool_call.get("function") or {}
                tool_name = (
                    function.get("name", "")
                    if isinstance(function, dict)
                    else getattr(function, "name", "")
                )
                raw_arguments = (
                    function.get("arguments", "")
                    if isinstance(function, dict)
                    else getattr(function, "arguments", "")
                )
            else:
                function = getattr(tool_call, "function", None)
                if function is None:
                    continue
                tool_name = getattr(function, "name", "")
                raw_arguments = getattr(function, "arguments", "")

            if tool_name != "finish":
                continue

            # -- Two-level JSON parse: outer args → result_json -----------
            try:
                # 🎯 Parse outer tool-call arguments
                # strict=False tolerates control chars in LLM-generated strings
                if isinstance(raw_arguments, str):
                    args = json.loads(raw_arguments, strict=False)
                elif isinstance(raw_arguments, dict):
                    args = raw_arguments
                else:
                    # 🔄 Unknown type — skip this tool call
                    continue

                result_json_str = args.get("result_json", "")

                # 📝 Parse inner result_json payload
                if isinstance(result_json_str, str) and result_json_str:
                    return json.loads(result_json_str, strict=False)
                elif isinstance(result_json_str, dict):
                    return result_json_str

                # 🔄 No result_json key — return outer args directly
                return args

            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                logger.debug(
                    "⚠️ Failed to parse finish tool result_json, "
                    "attempting raw-arguments fallback: %s",
                    exc,
                )
                # 🔄 Fallback: try returning outer args parsed directly
                try:
                    if isinstance(raw_arguments, str):
                        return json.loads(raw_arguments, strict=False)
                    if isinstance(raw_arguments, dict):
                        return raw_arguments
                except (json.JSONDecodeError, ValueError) as fallback_exc:
                    logger.debug(
                        "⚠️ Raw-arguments fallback also failed: %s",
                        fallback_exc,
                    )
                return {}

    return {}
