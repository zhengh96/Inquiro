"""Shared LLM response utilities 🔧.

Provides common helper functions for JSON extraction and cost estimation
that are reused across AnalysisExp, EnsembleRunner, and DiscoverySynthesisExp.
Centralising them here eliminates duplication and ensures consistent behaviour.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 💰 Conservative average cost estimates per token across providers (USD)
_DEFAULT_INPUT_COST_PER_TOKEN: float = 3.0 / 1_000_000
_DEFAULT_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000


def extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract a JSON object from potentially wrapped LLM response text 🔍.

    Handles three common LLM output formats in order:
        1. Pure JSON string.
        2. JSON enclosed in a Markdown fenced code block (```json ... ```).
        3. First ``{...}`` block found anywhere in the text.

    Args:
        text: Raw text from an LLM response.

    Returns:
        Parsed JSON dict.  Returns an empty dict if all strategies fail.
    """
    text = text.strip()

    # ✨ Try 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ✨ Try 2: Extract from Markdown code block
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            # ⏭️ Remove optional language tag (e.g., "json\n")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                result = json.loads(cleaned)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    # ✨ Try 3: Extract from XML-like tags (e.g., <json>...</json>)
    xml_match = re.search(r"<json\s*>(.*?)</json>", text, re.DOTALL)
    if xml_match:
        try:
            result = json.loads(xml_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # ✨ Try 4: Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # ✨ Try 4b: Fix trailing commas before ] or }
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    logger.warning("⚠️ Failed to extract JSON from LLM response (length=%d)", len(text))
    return {}


def extract_cost_from_response(
    response: Any,
    input_cost_per_token: float = _DEFAULT_INPUT_COST_PER_TOKEN,
    output_cost_per_token: float = _DEFAULT_OUTPUT_COST_PER_TOKEN,
) -> float:
    """Extract estimated cost from LLM response metadata 💰.

    Reads ``response.meta.usage`` for token counts and multiplies by
    the per-token rates.  Falls back to 0.0 when metadata is absent.

    Args:
        response: AssistantMessage returned by ``BaseLLM.query()``.
        input_cost_per_token: Cost per input (prompt) token in USD.
            Defaults to ~$3/M tokens.
        output_cost_per_token: Cost per output (completion) token in USD.
            Defaults to ~$15/M tokens.

    Returns:
        Estimated cost in USD.  Returns 0.0 on any metadata error.
    """
    meta = getattr(response, "meta", {}) or {}
    usage = meta.get("usage", {})

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    # 💰 Conservative average cost estimate across providers
    return input_tokens * input_cost_per_token + output_tokens * output_cost_per_token
