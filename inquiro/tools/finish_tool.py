"""Inquiro Finish Tool — schema-enforcing task completion 📋.

Extends EvoMaster's BaseTool to validate agent output against a
caller-defined JSON Schema before accepting the result. Shared by
both SearchAgent and SynthesisAgent.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class InquiroFinishToolParams(BaseToolParams):
    """Signals the completion of a Inquiro research or synthesis task.

    Use this tool when you have completed your research/synthesis and are
    ready to submit your structured result. Your output **must** conform
    to the JSON Schema described in your system instructions.

    The ``result_json`` field should contain the full structured result
    as a JSON string.

    The ``task_completed`` field indicates whether you believe the task
    is fully complete or only partially finished.
    """

    name: ClassVar[str] = "finish"

    result_json: str = Field(
        description=(
            "The structured result as a JSON string. Must conform to "
            "the output schema described in your instructions."
        )
    )
    task_completed: bool = Field(
        default=True, description="Whether the task is fully completed"
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class InquiroFinishTool(BaseTool):
    """Finish tool with configurable JSON Schema validation 📋.

    Unlike the default ``FinishTool`` which accepts any JSON,
    this tool validates the output against a caller-defined schema.
    Schema errors are returned as observations so the agent can
    self-correct.

    Example::

        tool = InquiroFinishTool(output_schema=my_schema)
        observation, info = tool.execute(session, args_json)
        if "error" in info:
            # agent should fix and retry ✨
            ...
    """

    name: ClassVar[str] = "finish"
    params_class: ClassVar[type[BaseToolParams]] = InquiroFinishToolParams

    def __init__(self, output_schema: dict[str, Any]) -> None:
        """Initialize with the output schema for validation 🔧.

        Args:
            output_schema: JSON Schema dict that the agent output
                must conform to.
        """
        super().__init__()
        self.output_schema = output_schema
        # ✨ Lazy-import jsonschema to keep module-level deps light
        self._validator: Any | None = None
        # 🔍 Self-review config (optional, off by default)
        self._self_review_checklist: list[dict[str, Any]] = []
        self._self_review_threshold: float = 0.8

    def _get_validator(self) -> Any:
        """Lazily create a JSON Schema validator instance 🔒.

        Returns:
            ``jsonschema.Draft7Validator`` for ``self.output_schema``.

        Raises:
            ImportError: If jsonschema package is not installed.
        """
        if self._validator is None:
            try:
                import jsonschema
            except ImportError:
                raise ImportError(
                    "jsonschema package is required for Inquiro output "
                    "validation. Install it with: pip install jsonschema"
                )
            self._validator = jsonschema.Draft7Validator(self.output_schema)
        return self._validator

    # -- Self-review configuration & logic ----------------------------------

    def set_self_review_config(
        self,
        checklist_items: list[dict[str, Any]] | None = None,
        coverage_threshold: float = 0.8,
    ) -> None:
        """Configure pre-finish self-review parameters 🔍.

        When configured, the tool performs a mini quality check before
        accepting the finish. If checklist coverage is below threshold
        or evidence references are broken, a warning observation is
        returned instead of completing, giving the agent a chance to
        improve.

        Args:
            checklist_items: Required checklist items with ``id`` keys.
            coverage_threshold: Minimum coverage ratio to allow finish
                (0.0-1.0).
        """
        self._self_review_checklist = checklist_items or []
        self._self_review_threshold = coverage_threshold

    def _run_self_review(
        self,
        result_data: dict[str, Any],
    ) -> tuple[bool, str]:
        """Run pre-finish self-review checks on the result 🔍.

        Checks two dimensions:
        1. **Coverage**: ratio of covered checklist items vs total.
        2. **Reference integrity**: evidence IDs in reasoning must
           exist in ``evidence_index``.

        Self-review is skipped entirely when no checklist items have
        been configured via ``set_self_review_config()``.

        Args:
            result_data: Parsed result dictionary to inspect.

        Returns:
            Tuple of ``(passed, warning_text)``. When ``passed`` is
            ``True``, ``warning_text`` is empty. When ``False``,
            ``warning_text`` contains actionable guidance.
        """
        if not self._self_review_checklist:
            return True, ""

        warnings: list[str] = []

        # 📊 Check 1: Checklist coverage
        coverage_data = result_data.get("checklist_coverage", {})
        checklist_ids = [item["id"] for item in self._self_review_checklist]
        covered = [cid for cid in checklist_ids if coverage_data.get(cid) is True]
        total = len(checklist_ids)
        ratio = len(covered) / total if total > 0 else 1.0

        if ratio < self._self_review_threshold:
            missing = [cid for cid in checklist_ids if cid not in covered]
            warnings.append(
                f"Checklist coverage {ratio:.0%} is below the "
                f"{self._self_review_threshold:.0%} threshold. "
                f"Missing items: {', '.join(missing)}. "
                "Please address these items before finishing."
            )

        # 🔗 Check 2: Evidence reference integrity
        reasoning = result_data.get("reasoning", [])
        evidence_index = result_data.get("evidence_index", [])

        valid_ids: set[str] = set()
        for entry in evidence_index:
            if isinstance(entry, dict):
                eid = entry.get("id") or entry.get("evidence_id")
                if eid:
                    valid_ids.add(str(eid))
            elif isinstance(entry, str):
                valid_ids.add(entry)

        orphans: list[str] = []
        for entry in reasoning:
            if not isinstance(entry, dict):
                continue
            for eid in entry.get("evidence_ids", []):
                if str(eid) not in valid_ids:
                    orphans.append(str(eid))

        if orphans:
            unique_orphans = sorted(set(orphans))
            warnings.append(
                f"Orphan evidence references found: "
                f"{', '.join(unique_orphans)}. These IDs do not "
                "exist in evidence_index. Please fix or remove "
                "them before finishing."
            )

        if warnings:
            return False, "\n\n".join(warnings)
        return True, ""

    # -- Main execution ------------------------------------------------------

    def execute(
        self,
        session: "BaseSession",
        args_json: str,
    ) -> tuple[str, dict[str, Any]]:
        """Validate output against schema before accepting 🔍.

        Execution steps:
        1. Parse the outer tool-call JSON (``result_json`` +
           ``task_completed``).
        2. Parse the inner ``result_json`` string as JSON.
        3. Validate the parsed result against ``self.output_schema``.
        4. Run optional pre-finish self-review (coverage + refs).
        5. Return validation errors, self-review warning, or success.

        Args:
            session: Session instance (unused by this tool).
            args_json: JSON string containing
                ``InquiroFinishToolParams`` fields.

        Returns:
            Tuple of ``(observation, info)``:
            - On success: ``("Task completed successfully.",
              {"result": ...})``
            - On schema error: ``("Output does not match...",
              {"error": ...})``
            - On self-review warning: ``("..warning text..",
              {"self_review_warning": True})``
            - On JSON parse error: ``("Invalid JSON: ...",
              {"error": ...})``
        """
        logger = logging.getLogger(self.__class__.__name__)

        # 🎯 Step 1: Parse outer tool-call parameters
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            msg = f"Parameter validation error: {e}"
            logger.warning("❌ %s", msg)
            return msg, {"error": msg}

        assert isinstance(params, InquiroFinishToolParams)

        # 🎯 Step 2: Parse inner result_json string as JSON
        # strict=False tolerates control chars (newlines, tabs)
        # that LLMs frequently embed in long-form text fields
        try:
            result_data = json.loads(params.result_json, strict=False)
        except json.JSONDecodeError as e:
            msg = (
                f"Invalid JSON in result_json: {e}. "
                "Please fix the JSON syntax and try again."
            )
            logger.warning("❌ %s", msg)
            return msg, {"error": msg}

        # 🎯 Step 3: Validate against output_schema using jsonschema
        if self.output_schema:
            validator = self._get_validator()
            errors = list(validator.iter_errors(result_data))
            if errors:
                # 📝 Build detailed error messages for self-correction
                error_details = []
                for i, err in enumerate(errors, 1):
                    path = " → ".join(str(p) for p in err.absolute_path) or "(root)"
                    error_details.append(f"  {i}. Path: {path} | Error: {err.message}")
                error_list = "\n".join(error_details)
                msg = (
                    f"Output does not match the required schema. "
                    f"Found {len(errors)} validation error(s):\n{error_list}\n\n"
                    "Please fix these issues and call finish again."
                )
                logger.warning(
                    "❌ Schema validation failed with %s error(s)",
                    len(errors),
                )
                return msg, {"error": msg, "validation_errors": error_details}

        # 🔍 Step 4: Optional pre-finish self-review
        passed, warning_text = self._run_self_review(result_data)
        if not passed:
            logger.info("⚠️ Self-review found issues, returning warning")
            return warning_text, {"self_review_warning": True}

        # ✅ Step 5: Success — return validated result
        logger.info(
            "✅ Task completed (task_completed=%s)",
            params.task_completed,
        )
        return "Task completed successfully.", {
            "result": result_data,
            "task_completed": params.task_completed,
        }

    def get_tool_spec(self) -> Any:
        """Override tool spec to include output schema hint in description 📝.

        Returns:
            ToolSpec with enriched description.
        """
        spec = super().get_tool_spec()

        # ✨ Enrich description with required schema keys
        if self.output_schema:
            required = self.output_schema.get("required", [])
            if required:
                keys_hint = ", ".join(required)
                schema_note = (
                    f"\n\nYour result_json MUST include these required "
                    f"fields: {keys_hint}."
                )
                spec.function.description += schema_note

        return spec
