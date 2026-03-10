"""Inquiro CostTracker — real-time token cost tracking & budget 💰.

Provides thread-safe cost accounting with per-model pricing,
per-task budgets, and configurable overspend strategies.
"""

from __future__ import annotations

import enum
import logging
import threading
from datetime import datetime, timezone

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums & Models
# ---------------------------------------------------------------------------


class OverspendStrategy(str, enum.Enum):
    """Strategy when a task exceeds its cost budget ⚠️.

    Attributes:
        SOFT_STOP: Warn the agent and request graceful completion.
        HARD_STOP: Immediately terminate the task.
        WARN: Log a warning but allow the task to continue.
    """

    SOFT_STOP = "SoftStop"
    HARD_STOP = "HardStop"
    WARN = "Warn"


class CostStatus(str, enum.Enum):
    """Budget check result after recording a cost increment 📊.

    Statuses are ordered by severity. Callers that only check for
    TASK_EXCEEDED / TOTAL_EXCEEDED (the original two "hard-stop"
    values) remain fully backward-compatible — the new intermediate
    tiers (MODEL_DOWNGRADE, BUDGET_CRITICAL) are informational and
    never returned unless explicitly enabled via threshold config.

    Attributes:
        OK: Within budget.
        WARNING: Approaching budget limit (default >50%).
        MODEL_DOWNGRADE: Budget pressure suggests cheaper model (>80%).
        BUDGET_CRITICAL: Near budget exhaustion (>90%).
        TASK_EXCEEDED: Per-task budget exceeded.
        TOTAL_EXCEEDED: Total cross-task budget exceeded.
    """

    OK = "ok"
    WARNING = "warning"
    MODEL_DOWNGRADE = "model_downgrade"
    BUDGET_CRITICAL = "budget_critical"
    TASK_EXCEEDED = "task_exceeded"
    TOTAL_EXCEEDED = "total_exceeded"


class CostRecord(BaseModel):
    """Single cost record for one LLM call 📝.

    Attributes:
        task_id: ID of the task that incurred this cost.
        model: LLM model identifier (e.g. "claude-sonnet-4-20250514").
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        cost_usd: Computed cost in USD.
        timestamp: UTC timestamp of the record.
    """

    task_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CostSummary(BaseModel):
    """Aggregated cost summary 📊.

    Attributes:
        total_cost_usd: Sum of all recorded costs.
        task_costs: Per-task cost breakdown ``{task_id: cost_usd}``.
        total_input_tokens: Total input tokens across all tasks.
        total_output_tokens: Total output tokens across all tasks.
        budget_remaining: Remaining budget in USD (may be negative).
        records: Optional full list of cost records for auditing.
    """

    total_cost_usd: float = 0.0
    task_costs: dict[str, float] = Field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    budget_remaining: float = 0.0
    records: list[CostRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Default per-model pricing (USD per 1 K tokens) 💵
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    # 🤖 Anthropic models (direct API IDs)
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-haiku-3.5": {"input": 0.0008, "output": 0.004},
    # 🤖 Anthropic models (short names from Bedrock responses)
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250929": {"input": 0.003, "output": 0.015},
    # 🤖 Anthropic models (AWS Bedrock ARN-style IDs)
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input": 0.003,
        "output": 0.015,
    },
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 0.003,
        "output": 0.015,
    },
    "anthropic.claude-sonnet-4-20250514-v1:0": {
        "input": 0.003,
        "output": 0.015,
    },
    "us.anthropic.claude-opus-4-20250514-v1:0": {
        "input": 0.015,
        "output": 0.075,
    },
    # 🤖 OpenAI models
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "azure/gpt-5.2": {"input": 0.005, "output": 0.015},
    # 🤖 Google models
    "gemini-3-pro": {"input": 0.00125, "output": 0.005},
}


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Real-time cost tracking with budget enforcement 💰.

    Thread-safe cost tracking across concurrent tasks.
    Uses ``task_id`` as the primary accounting unit.

    Example::

        tracker = CostTracker(max_per_task=1.5, max_total=10.0)
        status = tracker.record(
            task_id="eval_123",
            model="claude-sonnet-4-20250514",
            input_tokens=5000,
            output_tokens=2000,
        )
        if status == CostStatus.WARNING:
            # approaching budget ⚠️
            ...

    Attributes:
        max_per_task: Maximum cost per individual task (USD).
        max_total: Maximum total cost across all tasks (USD).
        overspend_strategy: What to do when budget is exceeded.
    """

    def __init__(
        self,
        max_per_task: float = 1.5,
        max_total: float = 10.0,
        session_budget: float | None = None,
        warning_threshold: float = 0.5,
        downgrade_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        overspend_strategy: OverspendStrategy = OverspendStrategy.SOFT_STOP,
        model_pricing: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize CostTracker 🔧.

        Args:
            max_per_task: Maximum cost per individual task (USD).
            max_total: Maximum total cost across all tasks (USD).
            session_budget: Optional session-level budget that spans
                across all tasks (USD). When set, threshold tiers are
                computed against this value instead of ``max_total``.
            warning_threshold: Fraction of effective budget at which
                WARNING status is returned (default 0.5 = 50%).
            downgrade_threshold: Fraction of effective budget at which
                MODEL_DOWNGRADE status is returned (default 0.8 = 80%).
            critical_threshold: Fraction of effective budget at which
                BUDGET_CRITICAL status is returned (default 0.9 = 90%).
            overspend_strategy: Strategy when budget is exceeded.
            model_pricing: Custom per-model pricing map. Falls back to
                ``DEFAULT_MODEL_PRICING`` when ``None``.
        """
        self.max_per_task = max_per_task
        self.max_total = max_total
        self.session_budget = session_budget
        self.warning_threshold = warning_threshold
        self.downgrade_threshold = downgrade_threshold
        self.critical_threshold = critical_threshold
        self.overspend_strategy = overspend_strategy
        self._model_pricing = model_pricing or DEFAULT_MODEL_PRICING

        self._records: list[CostRecord] = []
        self._task_costs: dict[str, float] = {}
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    # -- Recording -----------------------------------------------------------

    def record(
        self,
        task_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostStatus:
        """Record a cost increment and check budget 📊.

        Computes cost from token counts using the pricing table,
        appends a ``CostRecord``, and returns the budget status.

        Args:
            task_id: The task that incurred the cost.
            model: LLM model identifier.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.

        Returns:
            CostStatus indicating whether to continue, warn, or stop.
        """
        cost_usd = self._compute_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            task_id=task_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

        with self._lock:
            self._records.append(record)
            self._task_costs[task_id] = self._task_costs.get(task_id, 0.0) + cost_usd
            self._total_cost += cost_usd
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # 🔍 Check budget status
            return self._check_budget_locked(task_id)

    # -- Queries -------------------------------------------------------------

    def get_task_cost(self, task_id: str) -> float:
        """Get current accumulated cost for a specific task 💰.

        Args:
            task_id: The task to query.

        Returns:
            Current cost in USD (0.0 if task has no records).
        """
        with self._lock:
            return self._task_costs.get(task_id, 0.0)

    def get_total_cost(self) -> float:
        """Get total cost across all tasks 💰.

        Returns:
            Total accumulated cost in USD.
        """
        with self._lock:
            return self._total_cost

    def get_summary(self) -> CostSummary:
        """Get full cost summary with per-task breakdown 📊.

        Returns:
            CostSummary model with totals, breakdown, and records.
        """
        with self._lock:
            return CostSummary(
                total_cost_usd=self._total_cost,
                task_costs=dict(self._task_costs),
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
                budget_remaining=(
                    (self.session_budget or self.max_total) - self._total_cost
                ),
                records=list(self._records),
            )

    def check_budget(self, task_id: str) -> CostStatus:
        """Check budget status for a given task without recording 🔍.

        Args:
            task_id: The task to check.

        Returns:
            Current CostStatus for the task.
        """
        with self._lock:
            return self._check_budget_locked(task_id)

    # -- Helpers -------------------------------------------------------------

    def _check_budget_locked(self, task_id: str) -> CostStatus:
        """Check budget status while holding the lock 🔒.

        Priority order (highest to lowest):
        1. Per-task limit exceeded  → TASK_EXCEEDED
        2. Total limit exceeded     → TOTAL_EXCEEDED
        3. Session budget exceeded  → TOTAL_EXCEEDED
        4. Critical threshold       → BUDGET_CRITICAL
        5. Downgrade threshold      → MODEL_DOWNGRADE
        6. Warning threshold        → WARNING
        7. Otherwise                → OK

        Args:
            task_id: The task to check.

        Returns:
            CostStatus based on current cost vs limits.
        """
        task_cost = self._task_costs.get(task_id, 0.0)

        # ❌ Per-task limit check (highest priority)
        if task_cost > self.max_per_task:
            self._logger.warning(
                "💸 Task '%s' exceeded per-task budget: $%s > $%s",
                task_id,
                f"{task_cost:.4f}",
                f"{self.max_per_task:.4f}",
            )
            return CostStatus.TASK_EXCEEDED

        # ❌ Total limit check (max_total hard ceiling)
        if self._total_cost > self.max_total:
            self._logger.warning(
                "💸 Total budget exceeded: $%s > $%s",
                f"{self._total_cost:.4f}",
                f"{self.max_total:.4f}",
            )
            return CostStatus.TOTAL_EXCEEDED

        # ❌ Session budget hard ceiling (if configured)
        if self.session_budget is not None and self._total_cost > self.session_budget:
            self._logger.warning(
                "💸 Session budget exceeded: $%s > $%s",
                f"{self._total_cost:.4f}",
                f"{self.session_budget:.4f}",
            )
            return CostStatus.TOTAL_EXCEEDED

        # 📊 Tiered threshold checks against effective budget
        effective_budget = (
            self.session_budget if self.session_budget is not None else self.max_total
        )

        if effective_budget > 0:
            budget_ratio = self._total_cost / effective_budget
        else:
            budget_ratio = 0.0

        # 🔴 Critical threshold (default ≥90%)
        if budget_ratio >= self.critical_threshold:
            self._logger.warning(
                "🔴 Budget critical: $%s / $%s (%.0f%%)",
                f"{self._total_cost:.4f}",
                f"{effective_budget:.4f}",
                budget_ratio * 100,
            )
            return CostStatus.BUDGET_CRITICAL

        # 🟠 Downgrade threshold (default ≥80%)
        if budget_ratio >= self.downgrade_threshold:
            self._logger.info(
                "🟠 Model downgrade suggested: $%s / $%s (%.0f%%)",
                f"{self._total_cost:.4f}",
                f"{effective_budget:.4f}",
                budget_ratio * 100,
            )
            return CostStatus.MODEL_DOWNGRADE

        # 🟡 Warning threshold (default ≥50%)
        if budget_ratio >= self.warning_threshold:
            self._logger.info(
                "⚠️ Approaching budget: $%s / $%s (%.0f%%)",
                f"{self._total_cost:.4f}",
                f"{effective_budget:.4f}",
                budget_ratio * 100,
            )
            return CostStatus.WARNING

        # ✅ Within budget
        return CostStatus.OK

    def _compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute USD cost from token counts and model pricing 🧮.

        Falls back to zero cost if the model is not in the pricing
        table (with a warning log).

        Args:
            model: LLM model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.

        Returns:
            Cost in USD.
        """
        pricing = self._model_pricing.get(model)
        if pricing is None:
            # 🔍 Fallback: try prefix-based matching for model variants
            for known_model, known_pricing in self._model_pricing.items():
                if model.startswith(known_model) or known_model.startswith(model):
                    self._logger.info(
                        "💰 Model '%s' matched to '%s' via prefix",
                        model,
                        known_model,
                    )
                    pricing = known_pricing
                    break

        if pricing is None:
            self._logger.warning(
                "⚠️ Unknown model '%s' — defaulting to zero cost",
                model,
            )
            return 0.0

        # 💰 Pricing is per 1K tokens
        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        return input_cost + output_cost
