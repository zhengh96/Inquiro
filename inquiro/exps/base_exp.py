"""Inquiro BaseExp — shared lifecycle logic for Exp classes 📋.

Provides common functionality shared across Exp classes (SynthesisExp, etc.):
    - Finish tool result extraction from trajectories 📝
    - Quality gate creation and configuration 🔍
    - Decision/confidence/reasoning/evidence parsing 🧠
    - Cost budget checking 💰
    - Retry event emission 🔁
    - BaseExp interface bridging 🔄
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evomaster.core.exp import BaseExp

from inquiro.infrastructure.event_emitter import InquiroEvent
from inquiro.core.trajectory_utils import extract_finish_result
from inquiro.core.types import ExpPhase, _VALID_TRANSITIONS
from inquiro.infrastructure.quality_gate import (
    QualityGate,
    QualityGateConfig as InfraQualityGateConfig,
    QualityGateResult,
)


class _SentinelAgent:
    """Minimal placeholder agent for BaseExp.__init__() compatibility 🎭.

    Satisfies BaseExp's expectation of an agent parameter while
    Inquiro creates real agents per-attempt inside the retry loop.
    """

    def __init__(self) -> None:
        """Initialize sentinel agent with minimal attributes 🔧."""
        self.name = "sentinel"
        self.logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker
    from inquiro.infrastructure.event_emitter import EventEmitter
    from inquiro.core.types import (
        Decision,
        Evidence,
        QualityGateConfig,
        ReasoningClaim,
    )


class InquiroBaseExp(BaseExp):
    """Shared base class for Inquiro experiment lifecycle 📋.

    Encapsulates common logic for Inquiro Exp classes:
        - Finish tool result extraction from agent trajectories
        - Quality gate creation from Inquiro config
        - Decision/confidence/reasoning/evidence parsing
        - Cost budget checking
        - Retry event handling
        - BaseExp.run() interface bridge

    Subclasses MUST implement:
        - ``exp_name`` property
        - ``run_sync()`` method
        - ``_build_result()`` method

    Attributes:
        task: The task definition (EvaluationTask or SynthesisTask).
        llm: LLM instance for the agent.
        quality_gate: Quality gate validator.
        cost_tracker: Cost tracking instance.
        event_emitter: SSE event emitter.
        cancellation_token: Cancellation signal.
        results: List of result dicts (BaseExp compatibility).
        run_dir: Run directory path (BaseExp compatibility).
        _enrichment_result: Evolution context data (optional).
    """

    def _init_base(
        self,
        task: Any,
        llm: BaseLLM,
        quality_gate_config: QualityGateConfig,
        cost_tracker: CostTracker,
        event_emitter: EventEmitter,
        cancellation_token: CancellationToken,
    ) -> None:
        """Initialize common Exp attributes via proper BaseExp.__init__() 🔧.

        Uses the Sentinel Agent pattern to satisfy BaseExp's requirement
        for an agent parameter, while allowing Inquiro to create fresh
        agents per-attempt inside the retry loop.

        Args:
            task: Task definition (EvaluationTask or SynthesisTask).
            llm: LLM instance for the agent.
            quality_gate_config: Quality validation configuration.
            cost_tracker: Cost tracking instance.
            event_emitter: SSE event emitter.
            cancellation_token: Cancellation signal.
        """
        # 🎭 Create sentinel agent and minimal config for BaseExp
        sentinel_agent = _SentinelAgent()
        minimal_config = {
            "exp_name": self.__class__.__name__,
            "task_id": getattr(task, "task_id", "unknown"),
        }

        # ✅ Properly call BaseExp.__init__() with sentinel agent
        super().__init__(sentinel_agent, minimal_config)

        # 📋 Set Inquiro-specific attributes
        self.task = task
        self.llm = llm
        self.quality_gate = self._create_quality_gate(
            quality_gate_config, task.output_schema
        )
        self.cost_tracker = cost_tracker
        self.event_emitter = event_emitter
        self.cancellation_token = cancellation_token

        # 🔄 Lifecycle state machine
        self._phase = ExpPhase.INIT

        # 🧬 Evolution infrastructure
        self._enrichment_result: Any = None

    @property
    def phase(self) -> ExpPhase:
        """Return current lifecycle phase 🔄.

        Returns:
            The current ``ExpPhase`` value.
        """
        return self._phase

    def _transition_phase(self, new_phase: ExpPhase) -> None:
        """Transition to a new lifecycle phase 🔄.

        Validates the transition against ``_VALID_TRANSITIONS`` and emits
        an ``InquiroEvent.PHASE_CHANGED`` event on success.

        Args:
            new_phase: Target phase to transition to.

        Raises:
            ValueError: If the transition from the current phase to
                ``new_phase`` is not allowed.
        """
        allowed = _VALID_TRANSITIONS.get(self._phase, set())
        if new_phase not in allowed:
            raise ValueError(
                f"Invalid phase transition: "
                f"{self._phase.value!r} -> {new_phase.value!r}. "
                f"Allowed targets from {self._phase.value!r}: "
                f"{sorted(p.value for p in allowed)}"
            )

        old_phase = self._phase
        self._phase = new_phase

        self.logger.info(
            "🔄 Phase transition: %s -> %s (task %s)",
            old_phase.value,
            new_phase.value,
            self.task.task_id,
        )

        # 📡 Emit phase change event
        self.event_emitter.emit(
            InquiroEvent.PHASE_CHANGED,
            self.task.task_id,
            {
                "from_phase": old_phase.value,
                "to_phase": new_phase.value,
                "task_id": self.task.task_id,
            },
        )

    def _safe_transition(self, target: ExpPhase) -> None:
        """Attempt a phase transition, logging instead of raising on invalid 🔄.

        Used in exception handlers where the current phase may not have a
        valid path to the target. Logs a warning if the transition is
        invalid rather than raising ValueError.

        Args:
            target: Desired target phase.
        """
        try:
            self._transition_phase(target)
        except ValueError:
            self.logger.warning(
                "⚠️ Skipping invalid phase transition: %s -> %s",
                self._phase.value,
                target.value,
            )

    def run(
        self,
        task_description: str,
        task_id: str = "exp_001",
    ) -> dict[str, Any]:
        """Run experiment via BaseExp interface (delegates to run_sync) 🔄.

        This override bridges the BaseExp.run() interface to Inquiro's
        run_sync() pattern. Callers should prefer run_sync() directly.

        Args:
            task_description: Task description (unused, task already set).
            task_id: Task ID (unused, task already set).

        Returns:
            Result dictionary compatible with BaseExp expectations.
        """
        result = self.run_sync()
        result_dict = result.model_dump()
        self.results.append(result_dict)
        return {
            "trajectory": None,
            "status": "completed",
            "result": result_dict,
        }

    # -- Shared utility methods ------------------------------------------------

    def _extract_result(self, trajectory: Any) -> dict[str, Any]:
        """Extract structured result from agent trajectory 📝.

        Delegates finish-tool parsing to the shared
        ``extract_finish_result`` utility, then falls back to plain-text
        JSON extraction when no finish tool call is found.

        Args:
            trajectory: Agent execution trajectory.

        Returns:
            Raw result dictionary from the finish tool. Empty dict if
            no finish tool call was found and the fallback also fails.
        """
        # 🔍 Try shared finish-tool extractor first
        result = extract_finish_result(trajectory)
        if result:
            return result

        # 🔄 Fallback: attempt to extract JSON from last assistant message
        self.logger.warning(
            "⚠️ No finish tool call found in trajectory, attempting fallback extraction"
        )
        return self._fallback_extract_from_text(trajectory)

    def _fallback_extract_from_text(
        self,
        trajectory: Any,
    ) -> dict[str, Any]:
        """Fallback: extract JSON from assistant message text 🔄.

        When the agent fails to call finish but outputs JSON in plain
        text, attempt to salvage the result. Scans from the last step
        backwards (most recent output is most likely the final result).

        Args:
            trajectory: Agent execution trajectory.

        Returns:
            Parsed result dict, or empty dict if extraction fails.
        """
        for step in reversed(trajectory.steps):
            msg = step.assistant_message
            if msg is None or not msg.content:
                continue

            content = msg.content

            # 🔍 Look for JSON block in markdown code fence
            json_match = re.search(
                r"```(?:json)?\s*\n(.*?)\n```",
                content,
                re.DOTALL,
            )
            if json_match:
                try:
                    result = json.loads(
                        json_match.group(1),
                        strict=False,
                    )
                    if isinstance(result, dict):
                        self.logger.info(
                            "✅ Fallback extraction succeeded (code fence JSON)"
                        )
                        return result
                except json.JSONDecodeError:
                    pass

            # 🔍 Look for raw JSON object (last brace pair in content)
            brace_end = content.rfind("}")
            if brace_end >= 0:
                # 📝 Find matching opening brace by scanning backwards
                depth = 0
                brace_start = -1
                for i in range(brace_end, -1, -1):
                    if content[i] == "}":
                        depth += 1
                    elif content[i] == "{":
                        depth -= 1
                        if depth == 0:
                            brace_start = i
                            break

                if brace_start >= 0:
                    try:
                        result = json.loads(
                            content[brace_start : brace_end + 1],
                            strict=False,
                        )
                        if isinstance(result, dict):
                            self.logger.info(
                                "✅ Fallback extraction succeeded (raw JSON object)"
                            )
                            return result
                    except json.JSONDecodeError:
                        pass

        self.logger.warning("⚠️ Fallback extraction failed — no valid JSON found")
        return {}

    def _create_quality_gate(
        self,
        config: QualityGateConfig,
        output_schema: dict[str, Any],
    ) -> QualityGate:
        """Create a QualityGate instance from config 🔍.

        InfraQualityGateConfig now inherits from CoreQualityGateConfig,
        so we can pass core fields directly without field-by-field
        translation.

        Args:
            config: Quality gate configuration from the task.
            output_schema: JSON Schema for output validation.

        Returns:
            QualityGate instance ready for validation.
        """
        # ✅ Core config fields pass through via inheritance
        infra_config = InfraQualityGateConfig(
            **config.model_dump(),
        )

        return QualityGate(config=infra_config, output_schema=output_schema)

    def _handle_retry(
        self,
        raw_result: dict[str, Any],
        failures: list[str],
        attempt: int,
    ) -> None:
        """Handle Quality Gate retry by emitting events 🔁.

        Emits a quality_gate_retry event and logs the failure details
        for observability.

        Args:
            raw_result: The failed result from the current attempt.
            failures: List of hard failure descriptions.
            attempt: The current attempt number (1-based).
        """
        self.logger.warning(
            "🔁 Quality gate retry %s for task %s: %s",
            attempt,
            self.task.task_id,
            failures,
        )

        # 📡 Emit retry event
        self.event_emitter.emit(
            InquiroEvent.QUALITY_GATE_RETRY,
            self.task.task_id,
            {
                "attempt": attempt,
                "failures": failures,
                "will_retry": True,
            },
        )

    def _record_trajectory_costs(self, trajectory: Any) -> None:
        """Extract token usage from trajectory and record to CostTracker 💰.

        Iterates through all steps in the trajectory, extracts the
        usage metadata from each AssistantMessage, and records it
        to the cost tracker for accurate cost accounting.

        Args:
            trajectory: Agent execution trajectory containing steps
                with AssistantMessage metadata.
        """
        if not trajectory or not trajectory.steps:
            return

        task_id = self.task.task_id

        for step in trajectory.steps:
            msg = step.assistant_message
            if msg is None:
                continue

            meta = getattr(msg, "meta", {}) or {}
            usage = meta.get("usage", {})
            model = meta.get("model", "")

            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            if input_tokens > 0 or output_tokens > 0:
                self.cost_tracker.record(
                    task_id=task_id,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

    def _check_cost(self) -> bool:
        """Check if the task is within cost budget 💰.

        Returns:
            True if within budget, False if over budget.
        """
        from inquiro.infrastructure.cost_tracker import CostStatus

        status = self.cost_tracker.check_budget(self.task.task_id)
        return status not in (
            CostStatus.TASK_EXCEEDED,
            CostStatus.TOTAL_EXCEEDED,
        )

    def _persist_trajectory(self, trajectory: Any) -> None:
        """Persist agent trajectory to a JSONL file for post-hoc analysis 💾.

        Writes trajectory data as JSONL (one JSON object per line):
            - Line 1: metadata header (task_id, status, timestamps, step count)
            - Lines 2+: one StepRecord per line (assistant_message + tool_responses)

        This method is non-blocking: failures are logged as warnings and
        do NOT affect the task result.

        NOTE (R9): Does NOT use EvoMaster's ``set_trajectory_file_path()``
        because it is a ClassVar shared across all BaseAgent instances.
        Concurrent tasks would overwrite each other's paths. Instead, we
        serialize at the Inquiro Exp layer using per-task file paths.

        Args:
            trajectory: Agent execution trajectory (EvoMaster Trajectory).
                If None or empty, this method is a no-op.
        """
        import json
        from pathlib import Path

        trajectory_dir = getattr(self.task, "trajectory_dir", None)
        if not trajectory_dir or not trajectory:
            return

        try:
            traj_path = Path(trajectory_dir) / f"{self.task.task_id}.jsonl"
            traj_path.parent.mkdir(parents=True, exist_ok=True)

            with open(traj_path, "w", encoding="utf-8") as f:
                # 📝 Line 1: metadata header
                meta = {
                    "type": "meta",
                    "task_id": getattr(trajectory, "task_id", self.task.task_id),
                    "status": getattr(trajectory, "status", "unknown"),
                    "total_steps": len(trajectory.steps),
                    "start_time": (
                        trajectory.start_time.isoformat()
                        if getattr(trajectory, "start_time", None)
                        else None
                    ),
                    "end_time": (
                        trajectory.end_time.isoformat()
                        if getattr(trajectory, "end_time", None)
                        else None
                    ),
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                # 📝 Lines 2+: one step per line
                for step in trajectory.steps:
                    step_data = step.model_dump(mode="json")
                    step_data["type"] = "step"
                    f.write(json.dumps(step_data, ensure_ascii=False) + "\n")

            self.logger.info(
                "💾 Trajectory persisted to %s (%d steps)",
                traj_path,
                len(trajectory.steps),
            )
        except Exception as e:
            self.logger.warning(
                "⚠️ Failed to persist trajectory (non-blocking): %s",
                e,
            )

    def _init_trajectory_stream(self, trajectory: Any) -> None:
        """Initialize streaming trajectory file with first meta line 💾.

        When task.trajectory_dir and task.trajectory_streaming are set,
        creates {trajectory_dir}/{task_id}.jsonl and writes one meta line
        (type: meta, status: "running", total_steps: null, start_time,
        end_time: null). Otherwise no-op. Failures are logged only.

        Args:
            trajectory: Agent trajectory (used for start_time if present).
        """
        trajectory_dir = getattr(self.task, "trajectory_dir", None)
        streaming = getattr(self.task, "trajectory_streaming", False)
        if not trajectory_dir or not streaming:
            return
        try:
            traj_path = Path(trajectory_dir) / f"{self.task.task_id}.jsonl"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            start_time = None
            if trajectory is not None and getattr(trajectory, "start_time", None):
                start_time = trajectory.start_time.isoformat()
            if start_time is None:
                start_time = datetime.now().isoformat()
            meta = {
                "type": "meta",
                "task_id": getattr(trajectory, "task_id", self.task.task_id)
                if trajectory
                else self.task.task_id,
                "status": "running",
                "total_steps": None,
                "start_time": start_time,
                "end_time": None,
            }
            with open(traj_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                f.flush()
            self.logger.debug(
                "💾 Trajectory stream initialized: %s",
                traj_path,
            )
        except Exception as e:
            self.logger.warning(
                "⚠️ Failed to init trajectory stream (non-blocking): %s",
                e,
            )

    def _persist_trajectory_step(
        self,
        trajectory: Any,
        step_index: int,
    ) -> None:
        """Append one step line to the streaming trajectory file 💾.

        When task.trajectory_dir and task.trajectory_streaming are set
        and 0 <= step_index < len(trajectory.steps), appends a single
        step line (type: step, step.model_dump(mode=\"json\")) to the
        existing file. Otherwise no-op. Failures are logged only.

        Args:
            trajectory: Agent execution trajectory.
            step_index: Index of the step to persist (0-based).
        """
        trajectory_dir = getattr(self.task, "trajectory_dir", None)
        streaming = getattr(self.task, "trajectory_streaming", False)
        if not trajectory_dir or not streaming or not trajectory:
            return
        steps = getattr(trajectory, "steps", None) or []
        if not (0 <= step_index < len(steps)):
            return
        try:
            traj_path = Path(trajectory_dir) / f"{self.task.task_id}.jsonl"
            step = steps[step_index]
            step_data = step.model_dump(mode="json")
            step_data["type"] = "step"
            with open(traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(step_data, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:
            self.logger.warning(
                "⚠️ Failed to persist trajectory step (non-blocking): %s",
                e,
            )

    def _finalize_trajectory_stream(self, trajectory: Any) -> None:
        """Append meta_final line to the streaming trajectory file 💾.

        When task.trajectory_dir and task.trajectory_streaming are set,
        appends one line (type: meta_final, total_steps, end_time, status)
        to the existing file. Otherwise no-op. Failures are logged only.

        Args:
            trajectory: Agent execution trajectory (for total_steps,
                end_time, status).
        """
        trajectory_dir = getattr(self.task, "trajectory_dir", None)
        streaming = getattr(self.task, "trajectory_streaming", False)
        if not trajectory_dir or not streaming:
            return
        try:
            traj_path = Path(trajectory_dir) / f"{self.task.task_id}.jsonl"
            total_steps = len(trajectory.steps) if trajectory else 0
            end_time = None
            if trajectory and getattr(trajectory, "end_time", None):
                end_time = trajectory.end_time.isoformat()
            if end_time is None:
                end_time = datetime.now().isoformat()
            status = getattr(trajectory, "status", "unknown")
            meta_final = {
                "type": "meta_final",
                "total_steps": total_steps,
                "end_time": end_time,
                "status": status,
            }
            with open(traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta_final, ensure_ascii=False) + "\n")
                f.flush()
            self.logger.info(
                "💾 Trajectory stream finalized: %s (%d steps)",
                traj_path,
                total_steps,
            )
        except Exception as e:
            self.logger.warning(
                "⚠️ Failed to finalize trajectory stream (non-blocking): %s",
                e,
            )

    def _parse_decision(
        self,
        raw_result: dict[str, Any],
    ) -> Decision:
        """Parse decision from raw result, defaulting to cautious 🎯.

        Args:
            raw_result: Raw result dictionary from agent.

        Returns:
            Parsed Decision enum value.
        """
        from inquiro.core.types import Decision

        decision_str = raw_result.get("decision", "cautious")
        try:
            return Decision(decision_str)
        except ValueError:
            return Decision.CAUTIOUS

    def _parse_confidence(
        self,
        raw_result: dict[str, Any],
        qg_result: QualityGateResult | None,
    ) -> float:
        """Parse confidence and apply QG cap 📊.

        Args:
            raw_result: Raw result dictionary from agent.
            qg_result: Quality gate result (may cap confidence).

        Returns:
            Clamped confidence value in [0.0, 1.0].
        """
        confidence = float(raw_result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        # 📊 Use module-level logger for visibility with uvicorn
        _conf_logger = logging.getLogger(__name__)
        _conf_logger.info("📊 Agent raw confidence: %.2f", confidence)

        if qg_result and qg_result.confidence_cap is not None:
            confidence = min(confidence, qg_result.confidence_cap)
            _conf_logger.info(
                "📊 Confidence capped: %.2f → %.2f (soft failures)",
                float(raw_result.get("confidence", 0.5)),
                confidence,
            )

        return confidence

    def _parse_reasoning(
        self,
        raw_result: dict[str, Any],
    ) -> list[ReasoningClaim]:
        """Parse reasoning claims from raw result 🧠.

        Args:
            raw_result: Raw result dictionary from agent.

        Returns:
            List of parsed ReasoningClaim objects.
        """
        from inquiro.core.types import ReasoningClaim

        reasoning_raw = raw_result.get("reasoning", [])
        reasoning: list[ReasoningClaim] = []
        for claim_data in reasoning_raw:
            if isinstance(claim_data, dict):
                try:
                    reasoning.append(ReasoningClaim(**claim_data))
                except Exception as exc:
                    # ⚠️ Skip malformed claims
                    logging.getLogger(__name__).debug(
                        "⚠️ Skipped malformed claim: %s",
                        exc,
                    )
        return reasoning

    def _parse_evidence(
        self,
        raw_result: dict[str, Any],
    ) -> list[Evidence]:
        """Parse evidence index from raw result 🔗.

        Args:
            raw_result: Raw result dictionary from agent.

        Returns:
            List of parsed Evidence objects.
        """
        from inquiro.core.types import Evidence

        evidence_raw = raw_result.get("evidence_index", [])
        evidence_index: list[Evidence] = []
        for ev_data in evidence_raw:
            if isinstance(ev_data, dict):
                try:
                    evidence_index.append(Evidence(**ev_data))
                except Exception as exc:
                    logging.getLogger(__name__).debug(
                        "⚠️ Skipped malformed evidence: %s",
                        exc,
                    )
        return evidence_index

    def _get_evolution_context(self) -> dict[str, Any]:
        """Get evolution context for agent injection 🧬.

        Hook method for subclasses to provide evolution data.
        SearchExp can override this to return enriched context
        from prior experiences.

        Returns:
            Dictionary with evolution context data. Empty by default.
        """
        return {}
