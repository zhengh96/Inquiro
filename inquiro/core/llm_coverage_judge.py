"""LLMCoverageJudge — LLM-based checklist coverage assessment 🤖.

Replaces MockCoverageJudge with semantic LLM judgment for accurate
coverage assessment.  Uses a fast/cheap model (e.g. Haiku) to evaluate
whether claims and evidence adequately cover each checklist item.

Falls back to MockCoverageJudge on any LLM error to ensure system
availability.

Reference pattern: runner.py _LLMGroupSummarizer (lines 96-200).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from inquiro.core.gap_analysis import CoverageResult, MockCoverageJudge
from inquiro.core.llm_utils import extract_cost_from_response, extract_json_from_text
from inquiro.prompts.loader import PromptLoader

if TYPE_CHECKING:
    from inquiro.infrastructure.llm_pool import LLMProviderPool

logger = logging.getLogger(__name__)

# 📏 Evidence truncation limits
_MAX_EVIDENCE_ITEMS = 20
_MAX_SUMMARY_LENGTH = 150


class LLMCoverageJudge:
    """LLM-based checklist coverage judge 🤖.

    Sends checklist items, claims, and evidence summaries to a fast LLM
    model for semantic coverage assessment.  Falls back to
    MockCoverageJudge on any error.

    Attributes:
        _llm_pool: LLM provider pool for model access.
        _model: Model name to request (e.g. 'haiku').
        _fallback: MockCoverageJudge used when LLM call fails.
        _prompt_loader: Cached prompt template loader.
    """

    def __init__(
        self,
        llm_pool: LLMProviderPool,
        model: str = "haiku",
    ) -> None:
        """Initialize LLMCoverageJudge 🔧.

        Args:
            llm_pool: LLM provider pool for model access.
            model: Model name to request from the pool.
        """
        self._llm_pool = llm_pool
        self._model = model
        self._fallback = MockCoverageJudge()
        self._prompt_loader = PromptLoader()

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Assess which checklist items are covered by claims 🔍.

        Builds a prompt with checklist, claims, and evidence summaries,
        sends it to the LLM, and parses the structured JSON response.
        Falls back to MockCoverageJudge on any error.

        Args:
            checklist: Checklist item descriptions to assess.
            claims: Current analysis claims with evidence references.
            evidence: Current evidence pool items.

        Returns:
            CoverageResult with LLM-assessed coverage.
        """
        if not checklist:
            return CoverageResult(covered=[], uncovered=[])

        try:
            return await self._llm_judge(checklist, claims, evidence)
        except Exception as exc:
            logger.warning(
                "⚠️ LLMCoverageJudge failed, falling back to mock: %s",
                exc,
            )
            return await self._fallback.judge_coverage(
                checklist,
                claims,
                evidence,
            )

    async def _llm_judge(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Core LLM judgment logic 🧠.

        Args:
            checklist: Checklist items to assess.
            claims: Analysis claims with evidence refs.
            evidence: Evidence pool items.

        Returns:
            CoverageResult from LLM response.

        Raises:
            ValueError: If LLM response cannot be parsed.
        """
        from evomaster.utils.llm import Dialog
        from evomaster.utils.types import UserMessage

        system_prompt = self._prompt_loader.load("coverage_judge_system")
        user_prompt = self._build_user_prompt(checklist, claims, evidence)

        dialog = Dialog(
            system=system_prompt,
            messages=[UserMessage(content=user_prompt)],
        )

        llm = self._llm_pool.get_llm(self._model)
        response = await asyncio.to_thread(llm.query, dialog)
        content = getattr(response, "content", "") or ""

        # 💰 Extract cost
        cost = extract_cost_from_response(response)

        # 📊 Parse JSON response
        data = extract_json_from_text(content)
        if not data:
            logger.warning(
                "🔍 LLMCoverageJudge raw response (first 2000 chars): %s",
                content[:2000],
            )
            raise ValueError(
                f"Empty JSON from LLM coverage judge (content_len={len(content)})"
            )

        # 🔧 Adaptive schema extraction — models rarely follow the
        # exact schema, so we extract covered/uncovered from whatever
        # JSON structure they return.
        covered_raw, uncovered_raw, conflicts_raw = self._extract_coverage(
            data, checklist,
        )

        # ✅ Validate: every checklist item must appear in exactly one list
        covered, uncovered, conflicts = self._validate_and_reconcile(
            checklist,
            covered_raw,
            uncovered_raw,
            conflicts_raw,
        )

        logger.info(
            "🤖 LLMCoverageJudge: %d/%d covered, %d conflicts, cost=$%.4f",
            len(covered),
            len(checklist),
            len(conflicts),
            cost,
        )

        return CoverageResult(
            covered=covered,
            uncovered=uncovered,
            conflict_signals=conflicts,
            judge_cost_usd=cost,
        )

    def _build_user_prompt(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> str:
        """Build the user message with checklist, claims, and evidence 📝.

        Evidence is filtered to only include items referenced by claims,
        truncated to summary form, and capped at _MAX_EVIDENCE_ITEMS.

        Args:
            checklist: Checklist items.
            claims: Analysis claims with evidence_ids.
            evidence: Full evidence pool.

        Returns:
            Formatted user prompt string.
        """
        # 📋 Checklist section
        checklist_text = "\n".join(
            f"  {i}. {item}" for i, item in enumerate(checklist, 1)
        )

        # 🧠 Claims section
        claims_lines: list[str] = []
        referenced_ids: set[str] = set()
        for i, claim in enumerate(claims, 1):
            claim_text = claim.get("claim", "")
            evidence_ids = claim.get("evidence_ids", [])
            strength = claim.get("strength", "")
            referenced_ids.update(evidence_ids)
            refs = ", ".join(evidence_ids) if evidence_ids else "none"
            line = f"  [{i}] {claim_text}"
            if strength:
                line += f" (strength: {strength})"
            line += f"\n      Evidence: {refs}"
            claims_lines.append(line)
        claims_text = "\n".join(claims_lines) if claims_lines else "  (no claims)"

        # 🔗 Evidence section — only referenced items, summary form
        evidence_text = self._build_evidence_section(evidence, referenced_ids)

        return (
            f"## CHECKLIST\n\n{checklist_text}\n\n"
            f"## CLAIMS\n\n{claims_text}\n\n"
            f"## EVIDENCE\n\n{evidence_text}\n\n"
            f"---\n"
            f"CLASSIFY each checklist item as covered or uncovered. "
            f"Copy each item's EXACT text into the correct list.\n"
            f'Output ONLY: {{"covered":[...],"uncovered":[...],'
            f'"conflict_signals":[],"reasoning":"..."}}'
        )

    def _build_evidence_section(
        self,
        evidence: list[Any],
        referenced_ids: set[str],
    ) -> str:
        """Build truncated evidence section for the prompt 📝.

        Only includes evidence items referenced by claims.
        Each item is summarized to ≤150 characters.
        Capped at _MAX_EVIDENCE_ITEMS items.

        Args:
            evidence: Full evidence pool.
            referenced_ids: Evidence IDs referenced by claims.

        Returns:
            Formatted evidence text.
        """
        if not referenced_ids:
            return "  (no evidence referenced)"

        # 🔍 Filter to referenced items only
        referenced: list[Any] = [
            ev
            for ev in evidence
            if getattr(ev, "id", None) in referenced_ids
        ]

        # 📏 Cap at max items
        if len(referenced) > _MAX_EVIDENCE_ITEMS:
            referenced = referenced[:_MAX_EVIDENCE_ITEMS]

        lines: list[str] = []
        for ev in referenced:
            ev_id = getattr(ev, "id", "?")
            summary = getattr(ev, "summary", "") or ""
            tag = getattr(ev, "evidence_tag", "") or ""
            quality = getattr(ev, "quality_label", "") or ""

            # 📏 Truncate summary
            if len(summary) > _MAX_SUMMARY_LENGTH:
                summary = summary[: _MAX_SUMMARY_LENGTH - 3] + "..."

            parts = [f"  [{ev_id}]"]
            if tag:
                parts.append(f"tag={tag}")
            if quality:
                parts.append(f"quality={quality}")
            parts.append(f"— {summary}")
            lines.append(" ".join(parts))

        if len(evidence) > len(referenced):
            lines.append(
                f"  ({len(evidence) - len(referenced)} additional evidence "
                f"items not referenced by claims)"
            )

        return "\n".join(lines) if lines else "  (no evidence referenced)"

    def _extract_coverage(
        self,
        data: dict[str, Any],
        checklist: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract covered/uncovered from any JSON schema the LLM returns 🔄.

        Models rarely follow the exact output schema. This method handles
        common alternative formats:
        1. Standard: {"covered": [...], "uncovered": [...]}
        2. Checklist dict: {"checklist_analysis": {"item": {"status": "addressed"}}}
        3. Checklist list: {"checklist": [{"question": "...", "addressed": true}]}

        Args:
            data: Parsed JSON dict from LLM response.
            checklist: Original checklist items for matching.

        Returns:
            Tuple of (covered_items, uncovered_items, conflict_signals).
        """
        # 📊 Log actual JSON structure for debugging
        import json as _json

        logger.info(
            "🔍 Coverage JSON keys=%s, sample=%s",
            list(data.keys()),
            _json.dumps(data, ensure_ascii=False)[:500],
        )

        # ✅ Strategy 1: Standard schema with covered/uncovered lists
        if "covered" in data and isinstance(data["covered"], list):
            return (
                data.get("covered", []),
                data.get("uncovered", []),
                data.get("conflict_signals", []),
            )

        # ✅ Strategy 1B: checklist_answers or similar keyed dict
        for ca_key in (
            "checklist_answers", "checklist_coverage",
            "checklist_results", "items",
        ):
            ca = data.get(ca_key)
            if isinstance(ca, dict):
                c, u = [], []
                for ik, iv in ca.items():
                    resolved = self._resolve_item_key(
                        ik, checklist,
                    )
                    if isinstance(iv, dict):
                        v = self._classify_item_val(iv)
                    elif isinstance(iv, str):
                        low = iv.lower()
                        v = (
                            "uncovered"
                            if self._has_negative_signal(low)
                            else "covered"
                        )
                    elif isinstance(iv, bool):
                        v = "covered" if iv else "uncovered"
                    else:
                        v = "uncovered"
                    (c if v == "covered" else u).append(resolved)
                if c or u:
                    logger.info(
                        "🔄 Extracted via %s: %d cov, %d uncov",
                        ca_key, len(c), len(u),
                    )
                    return c, u, []
            elif isinstance(ca, list):
                c, u = [], []
                for idx, item in enumerate(ca):
                    if isinstance(item, dict):
                        q = (
                            item.get("item", "")
                            or item.get("question", "")
                            or item.get("text", "")
                            or ""
                        )
                        if not q and idx < len(checklist):
                            q = checklist[idx]
                        v = self._classify_item_val(item)
                    else:
                        q = (
                            checklist[idx]
                            if idx < len(checklist) else str(item)
                        )
                        v = "covered"
                    (c if v == "covered" else u).append(q)
                if c or u:
                    logger.info(
                        "🔄 Extracted via %s list: %d cov, %d uncov",
                        ca_key, len(c), len(u),
                    )
                    return c, u, []

        # ✅ Strategy 2A: Nested dict keyed by index/text
        # e.g., {"checklist_scores": {"1": {"score": 7, ...}}}
        covered: list[str] = []
        uncovered: list[str] = []

        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            # Check if value is a nested dict of items
            inner_dicts = {
                k: v for k, v in value.items()
                if isinstance(v, dict)
            }
            if inner_dicts:
                for item_key, item_val in inner_dicts.items():
                    resolved = self._resolve_item_key(
                        item_key, checklist,
                    )
                    verdict = self._classify_item_val(item_val)
                    if verdict == "covered":
                        covered.append(resolved)
                    else:
                        uncovered.append(resolved)
                if covered or uncovered:
                    logger.info(
                        "🔄 Extracted via nested dict: %d cov, %d uncov",
                        len(covered), len(uncovered),
                    )
                    return covered, uncovered, []

        # ✅ Strategy 2B: Flat dict — top-level keys are items
        # e.g., {"item_name": {"score": 7, "reasoning": "..."}}
        covered.clear()
        uncovered.clear()
        flat_items = {
            k: v for k, v in data.items()
            if isinstance(v, dict)
            and any(
                f in v for f in (
                    "score", "status", "coverage",
                    "addressed", "reasoning",
                )
            )
        }
        if len(flat_items) >= 2:
            for item_key, item_val in flat_items.items():
                resolved = self._resolve_item_key(
                    item_key, checklist,
                )
                verdict = self._classify_item_val(item_val)
                if verdict == "covered":
                    covered.append(resolved)
                else:
                    uncovered.append(resolved)
            if covered or uncovered:
                logger.info(
                    "🔄 Extracted via flat dict: %d cov, %d uncov",
                    len(covered), len(uncovered),
                )
                return covered, uncovered, []

            # ✅ Strategy 3: List of items with question/addressed
            if (
                isinstance(value, list)
                and value
                and isinstance(value[0], dict)
            ):
                for idx, item in enumerate(value):
                    question = (
                        item.get("question", "")
                        or item.get("item", "")
                        or item.get("text", "")
                        or ""
                    )
                    if not question:
                        # Use index to map to checklist
                        if idx < len(checklist):
                            question = checklist[idx]
                        else:
                            continue
                    verdict = self._classify_item_val(item)
                    if verdict == "covered":
                        covered.append(question)
                    else:
                        uncovered.append(question)
                if covered or uncovered:
                    logger.info(
                        "🔄 Extracted via list schema: %d covered, %d uncov",
                        len(covered), len(uncovered),
                    )
                    return covered, uncovered, []

        # ✅ Strategy 4: Fallback — search for each checklist item in the
        # stringified JSON and check immediate context for explicit coverage
        # verdicts (NOT vague negative phrases).
        full_text = _json.dumps(data, ensure_ascii=False).lower()
        for item in checklist:
            item_lower = item.lower()
            # Search for first 50 chars of the item in the JSON text
            idx = full_text.find(item_lower[:50])
            if idx == -1:
                # 🔍 Try shorter prefix (items may be truncated)
                idx = full_text.find(item_lower[:25])
            if idx == -1:
                uncovered.append(item)
                continue
            # Check 600 chars after match for explicit verdict signals
            context = full_text[idx : idx + 600]
            if self._has_explicit_uncovered_verdict(context):
                uncovered.append(item)
            else:
                covered.append(item)

        if covered or uncovered:
            logger.info(
                "🔄 Extracted coverage via text fallback: %d covered, %d uncovered",
                len(covered), len(uncovered),
            )
            return covered, uncovered, []

        # ❌ Could not extract from any known pattern
        logger.warning(
            "⚠️ Could not extract coverage from JSON (keys=%s): %s",
            list(data.keys()),
            _json.dumps(data, ensure_ascii=False)[:1000],
        )
        raise ValueError(
            f"Unrecognized JSON schema from LLM (keys={list(data.keys())})"
        )

    @staticmethod
    def _resolve_item_key(
        key: str,
        checklist: list[str],
    ) -> str:
        """Resolve a dict key to a checklist item text 🔑.

        If the key is a numeric index (1-based), returns the
        corresponding checklist item. Otherwise returns the key as-is.

        Args:
            key: Dict key from LLM response (may be "1", "2", etc.).
            checklist: Original checklist items.

        Returns:
            Resolved item text.
        """
        # Try 1-based index (most common LLM pattern)
        try:
            idx = int(key) - 1
            if 0 <= idx < len(checklist):
                return checklist[idx]
        except ValueError:
            pass
        return key

    def _classify_item_val(
        self,
        item_val: dict[str, Any],
    ) -> str:
        """Classify an item as covered or uncovered from its dict 🔍.

        Handles multiple formats: status fields, addressed booleans,
        score values, and answer text with negative signal detection.

        Args:
            item_val: Dict with status/score/answer fields.

        Returns:
            "covered" or "uncovered".
        """
        # Check explicit status fields
        status = (
            item_val.get("status", "")
            or item_val.get("coverage", "")
            or ""
        ).lower()
        addressed = item_val.get("addressed", None)

        if addressed is True or status in (
            "covered", "addressed", "adequate",
            "fully covered", "partially covered",
        ):
            return "covered"
        if addressed is False or status in (
            "uncovered", "not addressed", "missing", "not covered",
        ):
            return "uncovered"

        # Check score (covered if > 3 on any scale)
        score = item_val.get("score", None)
        if score is not None:
            try:
                return "covered" if float(score) > 3 else "uncovered"
            except (ValueError, TypeError):
                pass

        # Fallback: check answer/reasoning for negative signals
        answer_text = (
            item_val.get("answer", "")
            or item_val.get("findings", "")
            or item_val.get("response", "")
            or item_val.get("reasoning", "")
            or ""
        ).lower()
        if answer_text and not self._has_negative_signal(answer_text):
            return "covered"
        return "uncovered"

    @staticmethod
    def _has_negative_signal(text: str) -> bool:
        """Check if answer text indicates lack of coverage 🔍.

        Used by Strategy 2/3 for items with answer fields but no
        explicit status. Moderately strict.

        Args:
            text: Lowercased answer/response text from LLM.

        Returns:
            True if the text contains signals of no coverage.
        """
        # Explicit coverage-absence phrases
        negative_phrases = (
            "not covered",
            "not addressed",
            "no evidence",
            "no information",
            "no claims",
            "no relevant",
            "none found",
            "not found",
            "not mentioned",
            "not included",
            "not provided",
            "does not contain",
        )
        return any(phrase in text for phrase in negative_phrases)

    @staticmethod
    def _has_explicit_uncovered_verdict(context: str) -> bool:
        """Check if context around a checklist item shows explicit uncovered verdict 🔍.

        Used by Strategy 4 (text fallback). Very strict — only triggers
        on unambiguous verdict words that indicate the LLM explicitly
        marked the item as not covered.

        Args:
            context: ~600 chars of lowercased text after the item match.

        Returns:
            True only if there is an unambiguous uncovered verdict.
        """
        import re

        # Only match explicit verdict-level phrases
        verdict_phrases = (
            "not covered",
            "not addressed",
            "uncovered",
            '"uncovered"',
            '"not covered"',
            '"not addressed"',
            '"missing"',
            "status\": \"uncovered",
            "status\": \"not",
            "addressed\": false",
            "covered\": false",
        )
        if any(phrase in context for phrase in verdict_phrases):
            return True

        # Match "no <0-3 words> evidence/data/information/claims found"
        if re.search(
            r"\bno\s+(?:\w+\s+){0,3}(?:evidence|data|information|claims)\s+"
            r"(?:found|available|provided)\b",
            context,
        ):
            return True

        return False

    def _validate_and_reconcile(
        self,
        checklist: list[str],
        covered_raw: list[str],
        uncovered_raw: list[str],
        conflicts_raw: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Validate LLM output and reconcile with original checklist 🔧.

        Ensures every checklist item appears in exactly one of
        covered/uncovered. Items missing from both are added to uncovered
        (conservative). Items in both are kept in covered only.

        Args:
            checklist: Original checklist items.
            covered_raw: LLM-reported covered items.
            uncovered_raw: LLM-reported uncovered items.
            conflicts_raw: LLM-reported conflict signals.

        Returns:
            Tuple of (covered, uncovered, conflict_signals) reconciled
            against the original checklist.
        """
        # 🔧 Normalize: use original checklist text as canonical form
        # LLM may rephrase items or use snake_case keys
        covered_set: set[str] = set()
        uncovered_set: set[str] = set()

        def _norm(s: str) -> str:
            return s.lower().strip().replace("_", " ")

        covered_lower = {_norm(c) for c in covered_raw}
        uncovered_lower = {_norm(u) for u in uncovered_raw}

        for item in checklist:
            item_norm = _norm(item)
            in_covered = item_norm in covered_lower or any(
                item_norm in c or c in item_norm
                for c in covered_lower
            )
            in_uncovered = item_norm in uncovered_lower or any(
                item_norm in u or u in item_norm
                for u in uncovered_lower
            )

            if in_covered and not in_uncovered:
                covered_set.add(item)
            elif in_uncovered and not in_covered:
                uncovered_set.add(item)
            elif in_covered and in_uncovered:
                # ⚠️ Conflict: keep as covered (LLM confused)
                covered_set.add(item)
                logger.debug(
                    "⚠️ Item in both covered and uncovered, keeping covered: %s",
                    item[:80],
                )
            else:
                # ❌ Missing from both: conservatively mark uncovered
                uncovered_set.add(item)
                logger.debug(
                    "⚠️ Item missing from LLM output, marking uncovered: %s",
                    item[:80],
                )

        # 🔍 Reconcile conflicts against covered items
        conflict_lower = {_norm(c) for c in conflicts_raw}
        conflicts: list[str] = [
            item
            for item in covered_set
            if _norm(item) in conflict_lower
            or any(
                _norm(item) in c or c in _norm(item)
                for c in conflict_lower
            )
        ]

        return list(covered_set), list(uncovered_set), conflicts
