"""ActionPrincipleDistiller — PRAct-style cross-task principle distillation 📜.

Periodically analyzes accumulated high-fitness experiences to distill
reusable operating principles. New principles undergo A/B testing
before promotion to active status.

Trigger: Every 10 sub-item evaluations (batch processing).
Cost: ~1 LLM call per 10 evaluations (~$0.01/evaluation amortized).
Max active principles: 10 (oldest/lowest fitness retired on overflow).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.store import ExperienceStore
from inquiro.evolution.types import (
    ActionPrinciple,
    Experience,
    ExperienceQuery,
    MechanismType,
    TrajectorySnapshot,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ✨ Public API
# ============================================================================

__all__ = ["ActionPrincipleDistiller"]


# ============================================================================
# 📜 Default Prompt Template
# ============================================================================

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "principle_distillation.md"
)

_DEFAULT_DISTILLATION_PROMPT: str = (
    _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""
)


# ============================================================================
# 📜 ActionPrincipleDistiller
# ============================================================================


class ActionPrincipleDistiller(BaseMechanism):
    """PRAct-style cross-task action principle distillation 📜.

    Periodically analyzes accumulated high-fitness experiences to
    distill reusable operating principles. New principles undergo
    A/B testing before promotion to active status.

    Trigger: Every 10 sub-item evaluations (batch processing).
    Cost: 1 LLM call per 10 evaluations (~$0.01/evaluation amortized).
    Max active principles: 10 (oldest/lowest fitness retired).

    Attributes:
        MAX_ACTIVE_PRINCIPLES: Hard cap on simultaneously active principles.
        DISTILLATION_BATCH_SIZE: Evaluations between distillation runs.
        AB_TEST_EVALUATIONS: Evaluations before a candidate is judged.
        PROMOTION_THRESHOLD: Min treatment_coverage delta for promotion.
    """

    MAX_ACTIVE_PRINCIPLES: int = 10
    DISTILLATION_BATCH_SIZE: int = 10
    AB_TEST_EVALUATIONS: int = 10
    PROMOTION_THRESHOLD: float = 0.05

    # Minimum fitness to query experiences for distillation input
    _DISTILLATION_MIN_FITNESS: float = 0.4
    # Max source insights passed to LLM (context budget)
    _MAX_SOURCE_INSIGHTS: int = 50

    def __init__(
        self,
        store: ExperienceStore,
        llm_fn: Callable[[str], Awaitable[str]],
        namespace: str,
        distillation_prompt_template: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        """Initialize ActionPrincipleDistiller 🔧.

        Args:
            store: ExperienceStore instance for querying accumulated insights.
            llm_fn: Async callable that accepts a prompt string and returns
                the LLM's text response.
            namespace: Namespace for store isolation (e.g., "targetmaster").
            distillation_prompt_template: Override the default distillation
                prompt template. Uses the bundled template if None.
            enabled: Whether this mechanism is active.
        """
        super().__init__(enabled=enabled)
        self._store = store
        self._llm_fn = llm_fn
        self._namespace = namespace
        self._prompt_template = (
            distillation_prompt_template or _DEFAULT_DISTILLATION_PROMPT
        )
        self._principles: list[ActionPrinciple] = []
        self._evaluation_counter: int = 0

    # ========================================================================
    # 🏷️ Identity
    # ========================================================================

    @property
    def mechanism_type(self) -> MechanismType:
        """Return ACTION_PRINCIPLES mechanism type 🏷️.

        Returns:
            MechanismType.ACTION_PRINCIPLES
        """
        return MechanismType.ACTION_PRINCIPLES

    # ========================================================================
    # 📦 BaseMechanism interface
    # ========================================================================

    async def produce(
        self,
        snapshot: TrajectorySnapshot,
        round_context: dict[str, Any],
    ) -> list[Experience]:
        """Track evaluations and trigger distillation when batch is ready 📦.

        Increments the evaluation counter on each call. When the counter
        reaches a multiple of ``DISTILLATION_BATCH_SIZE``, runs distillation
        and returns the new candidate principles as Experience objects.

        Args:
            snapshot: Structured execution data from the completed round.
            round_context: Additional context including round_num,
                gap_items, coverage, sub_item_id, etc.

        Returns:
            List of Experience objects representing newly distilled candidate
            principles, or empty list if the batch is not yet full.
        """
        if not self.enabled:
            return []

        self._evaluation_counter += 1
        logger.debug(
            "📜 ActionPrincipleDistiller counter=%d (batch_size=%d)",
            self._evaluation_counter,
            self.DISTILLATION_BATCH_SIZE,
        )

        if self._evaluation_counter % self.DISTILLATION_BATCH_SIZE != 0:
            return []

        logger.info(
            "📜 Distillation triggered at evaluation_counter=%d",
            self._evaluation_counter,
        )

        # 🧪 Distill candidate principles from top insights
        new_principles = await self._distill_principles()

        # 🔄 Convert to Experience objects for storage
        experiences: list[Experience] = []
        for principle in new_principles:
            exp = Experience(
                namespace=self._namespace,
                category="action_principle",
                insight=principle.text,
                mechanism_type=MechanismType.ACTION_PRINCIPLES,
                source="principle_distillation",
                fitness_score=0.5,
            )
            experiences.append(exp)

        logger.info(
            "📜 Distillation produced %d candidate principles "
            "(total principles in memory=%d)",
            len(new_principles),
            len(self._principles),
        )
        return experiences

    def inject(
        self,
        round_context: dict[str, Any],
    ) -> str | None:
        """Inject active principles into the agent prompt 💉.

        Formats all principles with ``status == "active"`` as a numbered
        markdown list under the ``## OPERATING PRINCIPLES`` header.

        Args:
            round_context: Context including round_num, sub_item_id, etc.
                Not used directly — injection is based on in-memory state.

        Returns:
            Formatted markdown text, or None if no active principles exist.
        """
        if not self.enabled:
            return None

        active = [p for p in self._principles if p.status == "active"]
        if not active:
            return None

        lines = ["## OPERATING PRINCIPLES"]
        for i, principle in enumerate(active, 1):
            lines.append(f"{i}. {principle.text}")

        text = "\n".join(lines)
        logger.debug(
            "💉 Injecting %d active principles (%d chars)",
            len(active),
            len(text),
        )
        return text

    async def periodic_maintenance(self) -> None:
        """Load principles from store and persist updated state 🔧.

        Called between evaluations (not between rounds). Synchronizes
        in-memory principle list with the persisted store state.
        """
        if not self.enabled:
            return

        await self._load_from_store()
        logger.debug(
            "🔧 Periodic maintenance complete: %d principles loaded",
            len(self._principles),
        )

    # ========================================================================
    # 🔬 A/B Testing
    # ========================================================================

    def update_ab_test(
        self,
        principle_id: str,
        is_treatment: bool,
        coverage: float,
    ) -> None:
        """Update A/B test metrics for a candidate principle 🔬.

        Updates the running average of treatment or control coverage for
        the specified principle. Once ``AB_TEST_EVALUATIONS`` are complete,
        triggers promotion evaluation.

        Args:
            principle_id: ID of the ActionPrinciple to update.
            is_treatment: True if the principle was injected (treatment group),
                False if it was withheld (control group).
            coverage: Checklist coverage fraction (0.0–1.0) observed in this
                evaluation.
        """
        for principle in self._principles:
            if principle.id != principle_id:
                continue

            principle.evaluation_count += 1

            if is_treatment:
                # 📊 Running average for treatment arm
                old_avg = principle.treatment_coverage
                n = principle.evaluation_count
                principle.treatment_coverage = (old_avg * (n - 1) + coverage) / n
            else:
                # 📊 Running average for control arm
                old_avg = principle.control_coverage
                n = principle.evaluation_count
                principle.control_coverage = (old_avg * (n - 1) + coverage) / n

            logger.debug(
                "🔬 A/B update for principle %s: is_treatment=%s, "
                "coverage=%.3f, count=%d, "
                "treatment_avg=%.3f, control_avg=%.3f",
                principle_id[:8],
                is_treatment,
                coverage,
                principle.evaluation_count,
                principle.treatment_coverage,
                principle.control_coverage,
            )

            # 🎓 Evaluate promotion when A/B test is complete
            if principle.evaluation_count >= self.AB_TEST_EVALUATIONS:
                self._evaluate_promotion(principle)
            break

    def _evaluate_promotion(self, principle: ActionPrinciple) -> None:
        """Promote or discard a candidate principle based on A/B results 🎓.

        Promotes to ``"active"`` if the treatment arm outperforms the
        control arm by at least ``PROMOTION_THRESHOLD``. Otherwise retires.

        Args:
            principle: Candidate principle to evaluate.
        """
        delta = principle.treatment_coverage - principle.control_coverage

        if delta >= self.PROMOTION_THRESHOLD:
            principle.status = "active"
            logger.info(
                "🎓 Principle promoted to active: id=%s, delta=%.3f, "
                "text='%.60s...'",
                principle.id[:8],
                delta,
                principle.text,
            )
            self._enforce_max_active()
        else:
            principle.status = "retired"
            logger.info(
                "🗑️ Principle retired (insufficient lift): id=%s, "
                "delta=%.3f (threshold=%.3f)",
                principle.id[:8],
                delta,
                self.PROMOTION_THRESHOLD,
            )

    def _enforce_max_active(self) -> None:
        """Retire lowest-coverage active principles if over max 🗑️.

        When the number of active principles exceeds ``MAX_ACTIVE_PRINCIPLES``,
        retires the weakest ones (sorted by treatment_coverage ascending).
        """
        active = [p for p in self._principles if p.status == "active"]
        overflow = len(active) - self.MAX_ACTIVE_PRINCIPLES

        if overflow <= 0:
            return

        # 🗑️ Sort ascending so we retire the weakest first
        active.sort(key=lambda p: p.treatment_coverage)
        for principle in active[:overflow]:
            principle.status = "retired"
            logger.info(
                "🗑️ Retired overflow principle: id=%s, "
                "treatment_coverage=%.3f",
                principle.id[:8],
                principle.treatment_coverage,
            )

    # ========================================================================
    # 🧪 Distillation
    # ========================================================================

    async def _distill_principles(self) -> list[ActionPrinciple]:
        """Distill candidate principles from top insights 🧪.

        Queries the store for the top-50 high-fitness experiences, groups
        them by category, and sends them to the LLM for distillation.
        The LLM response is parsed as a JSON array of principle objects.
        Successful candidates are appended to ``_principles`` as candidates.

        Returns:
            List of newly created ActionPrinciple objects (status="candidate").
            Returns empty list on LLM failure or parse error.
        """
        # 1️⃣ Query top-50 high-fitness experiences
        query = ExperienceQuery(
            namespace=self._namespace,
            min_fitness=self._DISTILLATION_MIN_FITNESS,
            max_results=self._MAX_SOURCE_INSIGHTS,
        )
        try:
            experiences = await self._store.query(query)
        except Exception as exc:
            logger.warning("⚠️ Store query failed during distillation: %s", exc)
            return []

        if not experiences:
            logger.info("📜 No eligible insights for distillation yet")
            return []

        # 2️⃣ Group insights by category for the prompt
        grouped: dict[str, list[Experience]] = defaultdict(list)
        for exp in experiences:
            grouped[exp.category].append(exp)

        insights_text = self._format_insights_for_prompt(grouped)

        # 3️⃣ Render distillation prompt
        prompt = self._prompt_template.format(
            insight_count=len(experiences),
            insights_by_category=insights_text,
        )

        # 4️⃣ Call LLM
        try:
            raw_response = await self._llm_fn(prompt)
        except Exception as exc:
            logger.warning("⚠️ LLM call failed during distillation: %s", exc)
            return []

        # 5️⃣ Parse LLM output as JSON array
        parsed = self._parse_distillation_response(raw_response)
        if not parsed:
            return []

        # 6️⃣ Create ActionPrinciple objects and register them
        new_principles: list[ActionPrinciple] = []
        for item in parsed:
            text = item.get("text", "").strip()
            source_ids = item.get("source_insight_ids", [])

            if not text:
                continue

            principle = ActionPrinciple(
                text=text,
                status="candidate",
                source_insight_ids=source_ids,
            )
            self._principles.append(principle)
            new_principles.append(principle)

            logger.info(
                "📜 New candidate principle: id=%s, text='%.80s'",
                principle.id[:8],
                principle.text,
            )

        return new_principles

    def _format_insights_for_prompt(
        self,
        grouped: dict[str, list[Experience]],
    ) -> str:
        """Format grouped insights as structured text for the LLM prompt 📝.

        Args:
            grouped: Insights grouped by category, each sorted by fitness.

        Returns:
            Formatted markdown text listing insights per category.
        """
        sections: list[str] = []
        for category, exps in sorted(grouped.items()):
            # 📊 Sort by fitness descending within each category
            exps_sorted = sorted(exps, key=lambda e: e.fitness_score, reverse=True)
            section_lines = [f"### Category: {category}"]
            for exp in exps_sorted:
                section_lines.append(
                    f"- [id={exp.id}] (fitness={exp.fitness_score:.2f}) {exp.insight}"
                )
            sections.append("\n".join(section_lines))

        return "\n\n".join(sections)

    def _parse_distillation_response(
        self,
        raw: str,
    ) -> list[dict[str, Any]]:
        """Parse LLM distillation response as JSON array 🔍.

        Attempts to locate and parse a JSON array within the raw response.
        Handles common LLM formatting artifacts (markdown fences, leading text).

        Args:
            raw: Raw text response from the LLM.

        Returns:
            List of parsed dicts, each with "text" and "source_insight_ids".
            Returns empty list on parse failure.
        """
        # 🔍 Strip markdown fences if present
        text = raw.strip()
        if "```" in text:
            # Extract content between first ``` and last ```
            parts = text.split("```")
            # Look for the JSON block (may start with "json")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("["):
                    text = candidate
                    break

        # 🔍 Locate JSON array boundaries
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning(
                "⚠️ Could not locate JSON array in distillation response: "
                "'%.100s'",
                raw,
            )
            return []

        json_str = text[start : end + 1]
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "⚠️ JSON parse error in distillation response: %s — "
                "raw snippet: '%.100s'",
                exc,
                json_str,
            )
            return []

        if not isinstance(parsed, list):
            logger.warning(
                "⚠️ Distillation response is not a list: %s", type(parsed)
            )
            return []

        return parsed

    # ========================================================================
    # 💾 Persistence
    # ========================================================================

    async def _load_from_store(self) -> None:
        """Load persisted principles from ExperienceStore 📂.

        Queries for experiences with ``category="action_principle"`` and
        reconstructs in-memory ``ActionPrinciple`` objects. Existing
        in-memory principles are replaced to stay in sync with the store.

        Principles stored as Experience records use the ``insight`` field
        for principle text and ``context_tags`` for status encoding.
        """
        query = ExperienceQuery(
            namespace=self._namespace,
            category="action_principle",
            min_fitness=0.0,  # Load all regardless of fitness
            max_results=self.MAX_ACTIVE_PRINCIPLES * 3,  # Active + candidates
        )
        try:
            experiences = await self._store.query(query)
        except Exception as exc:
            logger.warning("⚠️ Failed to load principles from store: %s", exc)
            return

        # 🔄 Reconstruct ActionPrinciple list from Experience records
        loaded: list[ActionPrinciple] = []
        for exp in experiences:
            # 📦 Decode status from context_tags (e.g., "status:active")
            status = "candidate"
            for tag in exp.context_tags:
                if tag.startswith("status:"):
                    status = tag[len("status:"):]
                    break

            principle = ActionPrinciple(
                id=exp.id,
                text=exp.insight,
                status=status,
                treatment_coverage=exp.fitness_score,  # Reuse fitness as coverage proxy
            )
            loaded.append(principle)

        self._principles = loaded
        logger.info(
            "📂 Loaded %d principles from store (namespace=%s)",
            len(loaded),
            self._namespace,
        )

    async def _save_to_store(self) -> None:
        """Persist current principles to ExperienceStore 💾.

        Upserts each in-memory ActionPrinciple as an Experience record.
        Uses ``context_tags`` to encode the principle's lifecycle status
        so it can be restored accurately on the next load.
        """
        for principle in self._principles:
            exp = Experience(
                id=principle.id,
                namespace=self._namespace,
                category="action_principle",
                insight=principle.text,
                context_tags=[
                    f"status:{principle.status}",
                    *[f"src:{sid}" for sid in principle.source_insight_ids[:5]],
                ],
                mechanism_type=MechanismType.ACTION_PRINCIPLES,
                source="principle_distillation",
                fitness_score=principle.treatment_coverage,
            )
            try:
                # 🔄 Try update first, fall back to add for new principles
                existing = await self._store.get_by_id(principle.id)
                if existing is not None:
                    exp.times_used = existing.times_used
                    exp.times_helpful = existing.times_helpful
                    await self._store.update(exp)
                else:
                    await self._store.add(exp)
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to persist principle %s: %s",
                    principle.id[:8],
                    exc,
                )

        logger.info(
            "💾 Saved %d principles to store (namespace=%s)",
            len(self._principles),
            self._namespace,
        )
