"""Discovery-specific trajectory collector for evolution system 🔬.

Unlike TrajectoryCollector (which requires EvoMaster Trajectory objects),
this collector works with DiscoveryRoundRecord which captures the
multi-phase Discovery pipeline output (search → cleaning → analysis → gap).
"""

from __future__ import annotations

import logging

from inquiro.core.trajectory.models import (
    DiscoveryRoundRecord,
    SearchPhaseRecord,
)
from inquiro.core.types import EvaluationTask
from inquiro.evolution.types import (
    ResultMetrics,
    ToolCallRecord,
    TrajectorySnapshot,
)

logger = logging.getLogger(__name__)


class DiscoveryTrajectoryCollector:
    """Collects trajectory data from Discovery round records 📊.

    Unlike TrajectoryCollector (which requires EvoMaster Trajectory),
    this collector works with DiscoveryRoundRecord which may not have
    a full agent trajectory object.  It extracts tool calls from
    SearchPhaseRecord.queries and builds ResultMetrics from the
    analysis and gap phases.

    All context classification (context_tags, sub_item_id) is provided
    by the caller, not derived by the collector.
    """

    def collect_from_round(
        self,
        round_record: DiscoveryRoundRecord,
        task: EvaluationTask,
        context_tags: list[str],
    ) -> TrajectorySnapshot:
        """Build snapshot from a single Discovery round record 🔄.

        Args:
            round_record: Complete record of one Discovery round.
            task: The evaluation task being processed.
            context_tags: Tags for experience matching context.

        Returns:
            TrajectorySnapshot suitable for experience extraction.

        Raises:
            ValueError: If round_record or task is None.
        """
        if round_record is None:
            raise ValueError("round_record cannot be None")
        if task is None:
            raise ValueError("task cannot be None")

        # 🔧 Extract tool calls from search phase queries
        tool_calls = self._extract_search_tool_calls(round_record)

        # 📊 Build result metrics from analysis and gap phases
        metrics = self._build_metrics(round_record)

        return TrajectorySnapshot(
            evaluation_id=getattr(task, "evaluation_id", ""),
            task_id=task.task_id,
            topic=getattr(task, "topic", ""),
            context_tags=context_tags,
            sub_item_id=getattr(task, "sub_item_id", "") or "",
            tool_calls=tool_calls,
            metrics=metrics,
            wall_time_seconds=round_record.round_duration_seconds,
        )

    def _extract_search_tool_calls(
        self,
        round_record: DiscoveryRoundRecord,
    ) -> list[ToolCallRecord]:
        """Extract tool call records from the search phase 🔍.

        Each QueryRecord in the search phase maps to one ToolCallRecord.
        The MCP tool name and query text become the tool name and
        arguments summary respectively.

        Args:
            round_record: Complete record of one Discovery round.

        Returns:
            List of ToolCallRecord objects from the search phase.
        """
        tool_calls: list[ToolCallRecord] = []
        search_phase: SearchPhaseRecord = round_record.search_phase

        for query in search_phase.queries:
            tool_name = query.mcp_tool or "unknown_search_tool"
            # 📏 Summarize query text (first 200 chars)
            arguments_summary = query.query_text[:200] if query.query_text else ""

            tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    arguments_summary=arguments_summary,
                    result_size=query.result_count,
                    success=query.result_count > 0,
                    round_number=round_record.round_number,
                )
            )

        return tool_calls

    def _build_metrics(
        self,
        round_record: DiscoveryRoundRecord,
    ) -> ResultMetrics:
        """Build result metrics from analysis and gap phases 📈.

        Combines data from the analysis consensus, gap coverage, and
        cleaning output to produce a unified ResultMetrics object.
        Includes enriched search effectiveness and quality data when
        available.

        Args:
            round_record: Complete record of one Discovery round.

        Returns:
            ResultMetrics with evidence count, confidence, coverage, etc.
        """
        analysis = round_record.analysis_phase
        gap = round_record.gap_phase
        cleaning = round_record.cleaning_phase
        search = round_record.search_phase

        # 📊 Evidence count from cleaning output
        evidence_count = cleaning.output_count

        # 🤝 Confidence from analysis consensus ratio
        confidence = analysis.consensus.consensus_ratio

        # 🏷️ Decision from analysis consensus
        decision = analysis.consensus.consensus_decision

        # 🎯 Checklist coverage from gap phase
        checklist_coverage = gap.coverage_ratio

        # 💰 Total cost across all phases
        cost_usd = self._sum_phase_costs(round_record)

        # 📡 Aggregate server effectiveness into extra metadata
        extra: dict[str, float] = {}
        if search.server_effectiveness:
            total_hit = sum(
                s.hit_rate for s in search.server_effectiveness.values()
            )
            n_servers = len(search.server_effectiveness)
            extra["avg_server_hit_rate"] = (
                total_hit / n_servers if n_servers else 0.0
            )
        if search.query_diversity_score > 0:
            extra["query_diversity"] = search.query_diversity_score
        if gap.newly_covered_items:
            extra["newly_covered_count"] = float(
                len(gap.newly_covered_items)
            )

        metrics = ResultMetrics(
            evidence_count=evidence_count,
            confidence=confidence,
            search_rounds=1,  # ✅ One round record = one search round
            cost_usd=cost_usd,
            decision=decision,
            checklist_coverage=checklist_coverage,
        )
        # ✅ Attach extra metrics if ResultMetrics supports it
        if extra and hasattr(metrics, "extra"):
            metrics.extra = extra  # type: ignore[attr-defined]
        return metrics

    def _sum_phase_costs(
        self,
        round_record: DiscoveryRoundRecord,
    ) -> float:
        """Sum costs across all phases of the round 💰.

        Uses round_cost_usd if available, otherwise sums individual
        phase costs from queries and model results.

        Args:
            round_record: Complete record of one Discovery round.

        Returns:
            Total cost in USD for the round.
        """
        # 🎯 Use pre-calculated round cost if available
        if round_record.round_cost_usd > 0:
            return round_record.round_cost_usd

        # 🔍 Sum search query costs
        total = sum(q.cost_usd for q in round_record.search_phase.queries)

        # 🧠 Sum analysis model costs
        total += sum(m.cost_usd for m in round_record.analysis_phase.model_results)

        # 🎯 Sum gap phase focus prompt cost
        if round_record.gap_phase.focus_prompt:
            total += round_record.gap_phase.focus_prompt.cost_usd

        return total
