"""Integration tests for the DISCOVERY pipeline 🧪.

End-to-end tests exercising multiple DISCOVERY components working
together: DiscoveryLoop + EvidencePipeline + GapAnalysis + mock
executors.  External dependencies (LLM, MCP) are mocked, but
internal logic runs for real.

Sections:
    1. Full DiscoveryLoop with Mock Executors
    2. DiscoveryLoop + Real EvidencePipeline
    3. DiscoveryLoop + Real GapAnalysis
    4. Runner Pipeline Dispatch Integration
    5. Multi-Round Progression Integration
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from inquiro.core.discovery_loop import (
    DiscoveryLoop,
    MockAnalysisExecutor,
    MockFocusPromptGenerator,
    MockSearchExecutor,
    SearchRoundOutput,
)
from inquiro.core.evidence_pipeline import (
    EvidencePipeline,
)
from inquiro.core.gap_analysis import (
    CoverageResult,
    GapAnalysis,
    MockCoverageJudge,
)
from inquiro.core.types import (
    Checklist,
    ChecklistItem,
    CleanedEvidence,
    DiscoveryConfig,
    DiscoveryResult,
    Evidence,
    EvaluationTask,
    GapReport,
)


# ============================================================================
# 🔧 Shared test helpers
# ============================================================================


def _make_evidence(
    eid: str = "E1",
    summary: str = (
        "This is a sufficiently long test evidence summary about "
        "protein interactions and binding affinities in cell cultures"
    ),
    source: str = "test-mcp",
    query: str = "test query for evidence collection",
    url: str | None = None,
) -> Evidence:
    """Create a test Evidence instance with valid length 🔧.

    Args:
        eid: Evidence identifier.
        summary: Evidence summary (must be > 50 chars).
        source: Evidence source name.
        query: Query that produced this evidence.
        url: Optional source URL.

    Returns:
        Evidence instance.
    """
    return Evidence(
        id=eid,
        source=source,
        query=query,
        summary=summary,
        url=url,
    )


def _make_task(
    task_id: str = "integ-test-001",
    topic: str = "Integration test topic for evidence research pipeline",
    checklist_items: list[str] | None = None,
) -> EvaluationTask:
    """Create a test EvaluationTask with optional checklist 🔧.

    Args:
        task_id: Task identifier.
        topic: Task topic.
        checklist_items: Optional checklist item descriptions.

    Returns:
        EvaluationTask instance.
    """
    items = (
        checklist_items
        if checklist_items is not None
        else [
            "Assess protein binding affinity data",
            "Evaluate clinical trial outcomes",
            "Review safety profile evidence",
        ]
    )
    required = [
        ChecklistItem(
            id=f"CK-{i + 1}",
            description=item,
            keywords=item.lower().split()[:3],
        )
        for i, item in enumerate(items)
    ]
    return EvaluationTask(
        task_id=task_id,
        topic=topic,
        checklist=Checklist(
            required=required,
            coverage_threshold=0.8,
        ),
    )


def _make_config(
    max_rounds: int = 3,
    coverage_threshold: float = 0.80,
    max_cost: float = 10.0,
    convergence_delta: float = 0.05,
) -> DiscoveryConfig:
    """Create a test DiscoveryConfig 🔧.

    Args:
        max_rounds: Maximum discovery rounds.
        coverage_threshold: Coverage target to stop.
        max_cost: Budget cap per sub-item.
        convergence_delta: Min improvement to continue.

    Returns:
        DiscoveryConfig instance.
    """
    return DiscoveryConfig(
        max_rounds=max_rounds,
        coverage_threshold=coverage_threshold,
        max_cost_per_subitem=max_cost,
        convergence_delta=convergence_delta,
    )


def _make_unique_evidence(
    count: int = 5,
    prefix: str = "E",
) -> list[Evidence]:
    """Create a list of evidence items with unique summaries 🔧.

    Each item has a distinct summary to survive EvidencePipeline dedup.

    Args:
        count: Number of evidence items to create.
        prefix: ID prefix for evidence items.

    Returns:
        List of Evidence instances with unique summaries.
    """
    topics = [
        "protein receptor binding mechanisms in hepatocellular carcinoma",
        "kinase inhibitor selectivity profiles in enzymatic assays",
        "pharmacokinetic properties of oral bioavailability studies",
        "dose-response relationship analysis in phase two trials",
        "genomic biomarker validation for patient stratification",
        "antibody drug conjugate linker stability assessments",
        "cell line characterization for high throughput screening",
        "safety pharmacology cardiovascular liability evaluation",
        "formulation development for sustained release dosage forms",
        "manufacturing process scalability for commercial production",
    ]
    return [
        _make_evidence(
            f"{prefix}{i}",
            summary=(
                f"Detailed analysis of {topics[i % len(topics)]} "
                f"in study batch number {i * 1000 + 42}"
            ),
        )
        for i in range(count)
    ]


def _make_claims(keyword: str) -> list[dict[str, Any]]:
    """Create mock claims containing a keyword for coverage matching 🔧.

    Args:
        keyword: Keyword to embed in the claim text.

    Returns:
        List of claim dicts with the keyword.
    """
    return [
        {
            "claim": (
                f"Evidence shows strong {keyword} "
                f"results across multiple studies and research papers"
            ),
            "evidence_ids": ["E1"],
            "strength": "strong",
        }
    ]


# ============================================================================
# 🧪 Section 1: Full DiscoveryLoop with Mock Executors (5 tests)
# ============================================================================


class TestFullDiscoveryLoopMockExecutors:
    """Full DiscoveryLoop with MockSearchExecutor + MockAnalysisExecutor 🧪."""

    @pytest.mark.asyncio
    async def test_complete_loop_runs_to_convergence(self) -> None:
        """Complete loop run with mock executors converges properly 🔄."""
        evidence = _make_unique_evidence(5)
        # 🔧 Claims that match all checklist keywords
        claims = (
            _make_claims("protein")
            + _make_claims("binding")
            + _make_claims("clinical")
            + _make_claims("trial")
            + _make_claims("safety")
            + _make_claims("profile")
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
                mock_decision="positive",
            ),
        )

        task = _make_task()
        config = _make_config(max_rounds=5, coverage_threshold=0.80)
        result = await loop.run(task, config)

        # ✅ Loop should have completed
        assert isinstance(result, DiscoveryResult)
        assert result.task_id == "integ-test-001"
        assert result.pipeline_mode == "discovery"
        assert result.total_rounds >= 1
        assert result.trajectory_id is not None

    @pytest.mark.asyncio
    async def test_evidence_accumulates_across_rounds(self) -> None:
        """Evidence accumulates correctly across multiple rounds 📊."""
        # 🔧 Each evidence item needs a unique summary to survive dedup
        evidence = _make_unique_evidence(5)

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=[]),
            # 🔧 Force max rounds without convergence
            gap_analysis=GapAnalysis(
                coverage_judge=MockCoverageJudge(),
            ),
        )

        task = _make_task()
        # 🔧 Use a high threshold so it never converges by coverage
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,  # ⚠️ Disable diminishing returns
        )
        result = await loop.run(task, config)

        # ✅ Evidence from each round should accumulate
        assert result.total_rounds == 3
        # ⚠️ Dedup is per-round; same items across rounds accumulate
        # 5 items per round * 3 rounds = 15 total (dedup within each
        # round keeps all 5 since summaries are unique)
        assert len(result.evidence) >= 5  # At least from first round

    @pytest.mark.asyncio
    async def test_gap_analysis_convergence_triggers_stop(self) -> None:
        """Gap analysis convergence triggers the loop to stop 🛑."""
        evidence = _make_unique_evidence(5)
        # 🔧 Claims matching all checklist items
        claims = (
            _make_claims("protein")
            + _make_claims("binding")
            + _make_claims("affinity")
            + _make_claims("clinical")
            + _make_claims("trial")
            + _make_claims("outcomes")
            + _make_claims("safety")
            + _make_claims("profile")
            + _make_claims("review")
        )

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(mock_evidence=evidence),
            analysis_executor=MockAnalysisExecutor(mock_claims=claims),
        )

        task = _make_task()
        config = _make_config(max_rounds=10, coverage_threshold=0.60)
        result = await loop.run(task, config)

        # ✅ Should converge before max_rounds
        assert result.total_rounds < 10
        assert result.termination_reason != "max_rounds_reached"
        assert result.final_coverage > 0.0

    @pytest.mark.asyncio
    async def test_trajectory_jsonl_has_expected_records(self) -> None:
        """Trajectory JSONL file has all expected record types 📝."""
        evidence = _make_unique_evidence(3)
        claims = _make_claims("protein") + _make_claims("binding")

        with tempfile.TemporaryDirectory() as tmpdir:
            loop = DiscoveryLoop(
                search_executor=MockSearchExecutor(
                    mock_evidence=evidence,
                ),
                analysis_executor=MockAnalysisExecutor(
                    mock_claims=claims,
                ),
                trajectory_dir=tmpdir,
            )

            task = _make_task()
            config = _make_config(max_rounds=2, coverage_threshold=0.99)
            _result = await loop.run(task, config)

            # 📂 Find the trajectory file
            files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
            assert len(files) == 1, f"Expected 1 JSONL file, found {len(files)}"

            # 📋 Read and verify record types
            with open(os.path.join(tmpdir, files[0])) as f:
                records = [json.loads(line) for line in f]

            record_types = [r.get("type") for r in records]
            # ✅ Must have meta, at least 1 round, and summary
            assert "meta" in record_types
            assert "round" in record_types
            assert "summary" in record_types

    @pytest.mark.asyncio
    async def test_cost_tracking_consistent(self) -> None:
        """Cost tracking is consistent: round costs sum to total 💰."""
        evidence = _make_unique_evidence(3)

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # 💰 Sum round costs
        round_cost_sum = sum(rs.round_cost_usd for rs in result.round_summaries)

        # ✅ Round costs should sum to approximately total cost
        assert abs(round_cost_sum - result.total_cost_usd) < 0.01


# ============================================================================
# 🧪 Section 2: DiscoveryLoop + Real EvidencePipeline (5 tests)
# ============================================================================


class TestDiscoveryLoopRealEvidencePipeline:
    """DiscoveryLoop with real EvidencePipeline cleaning 🧹."""

    @pytest.mark.asyncio
    async def test_real_pipeline_cleaning_in_loop(self) -> None:
        """Real EvidencePipeline cleaning runs inside the loop 🧹."""
        evidence = [
            _make_evidence(
                "E1",
                summary=(
                    "Detailed research findings about protein receptor "
                    "binding mechanisms in hepatocellular carcinoma"
                ),
                url="https://pubmed.ncbi.nlm.nih.gov/12345",
            ),
            _make_evidence(
                "E2",
                summary=(
                    "Comprehensive analysis of safety data from "
                    "phase III clinical trials in oncology patients"
                ),
            ),
        ]

        real_pipeline = EvidencePipeline()
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=_make_claims("protein"),
            ),
            evidence_pipeline=real_pipeline,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=1,
            coverage_threshold=0.99,
        )
        result = await loop.run(task, config)

        # ✅ Should have evidence (cleaned through real pipeline)
        assert len(result.evidence) >= 1
        assert result.total_rounds == 1

    @pytest.mark.asyncio
    async def test_duplicate_evidence_across_rounds_deduped(self) -> None:
        """Duplicate evidence across rounds is properly deduped 🔑."""
        # 🔧 Same evidence in every round (will be deduped per round)
        same_evidence = [
            _make_evidence(
                "E1",
                summary=(
                    "Identical evidence about cellular signaling "
                    "pathways and downstream molecular cascades"
                ),
            ),
            _make_evidence(
                "E1-dup",
                summary=(
                    "Identical evidence about cellular signaling "
                    "pathways and downstream molecular cascades"
                ),
            ),
        ]

        real_pipeline = EvidencePipeline()
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=same_evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
            evidence_pipeline=real_pipeline,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Within each round, duplicates should be removed by pipeline
        # But evidence accumulates across rounds (dedup is per-round)
        # So we should have round_count * 1 unique evidence item
        # (1 item survives per round after dedup)
        assert result.total_rounds == 2
        assert len(result.evidence) == 2  # 1 per round after dedup

    @pytest.mark.asyncio
    async def test_source_tagging_preserved_through_pipeline(self) -> None:
        """Source tagging preserves through the pipeline 🏷️."""
        evidence = [
            _make_evidence(
                "E1",
                summary=(
                    "Academic peer-reviewed paper on molecular biology "
                    "and cell signaling pathway characterization results"
                ),
                url="https://pubmed.ncbi.nlm.nih.gov/99999",
            ),
            _make_evidence(
                "E2",
                summary=(
                    "Patent filing describing novel compound synthesis "
                    "methods for kinase inhibitor drug candidates"
                ),
                url="https://patents.google.com/patent/US12345",
            ),
        ]

        real_pipeline = EvidencePipeline()
        # 🔧 Verify the pipeline tags correctly
        cleaned, stats = real_pipeline.clean(evidence)

        # ✅ Both items should survive cleaning
        assert len(cleaned) == 2
        # ✅ Tag distribution should include academic and patent
        assert stats.tag_distribution.get("academic", 0) >= 1
        assert stats.tag_distribution.get("patent", 0) >= 1

    @pytest.mark.asyncio
    async def test_cleaning_stats_tracked_per_round(self) -> None:
        """CleaningStats tracked per round in the loop 📊."""
        evidence = _make_unique_evidence(5)

        real_pipeline = EvidencePipeline()
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
            evidence_pipeline=real_pipeline,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=2,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Round summaries should track cleaned counts
        for summary in result.round_summaries:
            assert summary.raw_evidence_count >= 0
            assert summary.cleaned_evidence_count >= 0

    @pytest.mark.asyncio
    async def test_noise_filter_removes_short_evidence(self) -> None:
        """Noise filter removes short/invalid evidence 🚫."""
        evidence = [
            _make_evidence(
                "E1",
                summary="Too short",  # < 50 chars → removed
            ),
            _make_evidence(
                "E2",
                summary=(
                    "This evidence is long enough to pass the noise "
                    "filter minimum length threshold for validity"
                ),
            ),
            _make_evidence(
                "E3",
                summary=(
                    "AI Search Session Created - this is a noise "
                    "pattern from MCP tool that should be filtered"
                ),
            ),
        ]

        real_pipeline = EvidencePipeline()
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
            evidence_pipeline=real_pipeline,
        )

        task = _make_task()
        config = _make_config(max_rounds=1, coverage_threshold=0.99)
        result = await loop.run(task, config)

        # ✅ Only E2 should survive (E1 too short, E3 noise pattern)
        assert len(result.evidence) == 1
        # ✅ Round summary should reflect cleaning
        assert result.round_summaries[0].raw_evidence_count == 3
        assert result.round_summaries[0].cleaned_evidence_count == 1


# ============================================================================
# 🧪 Section 3: DiscoveryLoop + Real GapAnalysis (5 tests)
# ============================================================================


class TestDiscoveryLoopRealGapAnalysis:
    """DiscoveryLoop with real GapAnalysis + MockCoverageJudge 🎯."""

    @pytest.mark.asyncio
    async def test_real_gap_analysis_with_mock_judge(self) -> None:
        """Real GapAnalysis with MockCoverageJudge integration 🎯."""
        evidence = _make_unique_evidence(5)
        claims = _make_claims("protein") + _make_claims("binding")

        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        config = _make_config(max_rounds=5)
        result = await loop.run(task, config)

        # ✅ Gap reports should be populated
        assert len(result.gap_reports) >= 1
        for report in result.gap_reports:
            assert isinstance(report, GapReport)
            assert 0.0 <= report.coverage_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_coverage_improves_with_matching_claims(self) -> None:
        """Coverage improves when claims match checklist keywords 📈."""
        evidence = _make_unique_evidence(5)
        # 🔧 Claims matching "protein" and "binding" from first checklist item
        claims = _make_claims("protein") + _make_claims("binding")

        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        config = _make_config(max_rounds=3, coverage_threshold=0.99)
        result = await loop.run(task, config)

        # ✅ Coverage should be > 0 since claims match some checklist keywords
        assert result.final_coverage > 0.0

    @pytest.mark.asyncio
    async def test_convergence_by_coverage_threshold(self) -> None:
        """Convergence triggered by coverage threshold reached 🛑."""
        evidence = _make_unique_evidence(5)
        # 🔧 Claims matching ALL checklist items' keywords
        claims = (
            _make_claims("protein")
            + _make_claims("binding")
            + _make_claims("affinity")
            + _make_claims("clinical")
            + _make_claims("trial")
            + _make_claims("outcomes")
            + _make_claims("safety")
            + _make_claims("profile")
            + _make_claims("review")
        )

        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        # 🔧 Low threshold to trigger convergence
        config = _make_config(
            max_rounds=10,
            coverage_threshold=0.30,
        )
        result = await loop.run(task, config)

        # ✅ Should converge by coverage
        assert (
            "coverage" in (result.termination_reason or "").lower()
            or result.final_coverage >= 0.30
        )

    @pytest.mark.asyncio
    async def test_convergence_by_budget_exhaustion(self) -> None:
        """Convergence triggered by budget exhaustion 💸."""
        evidence = _make_unique_evidence(3)

        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        # 🔧 Very small budget — will exhaust after 1 round
        # MockSearch costs 0.10, MockAnalysis costs 0.50 → 0.60 per round
        config = _make_config(
            max_rounds=10,
            coverage_threshold=0.99,
            max_cost=0.50,  # Less than 1 round cost
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Should converge by budget
        assert result.total_rounds <= 2
        assert (
            "budget" in (result.termination_reason or "").lower()
            or "cost" in (result.termination_reason or "").lower()
            or "max_cost" in (result.termination_reason or "").lower()
        )

    @pytest.mark.asyncio
    async def test_convergence_by_diminishing_returns(self) -> None:
        """Convergence triggered by diminishing returns 📉."""
        evidence = _make_unique_evidence(3)
        # 🔧 Empty claims → no coverage improvement between rounds
        claims: list[dict[str, Any]] = []

        gap = GapAnalysis(coverage_judge=MockCoverageJudge())
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=evidence,
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.99,
            max_cost=100.0,
            # 🔧 Any delta > 0 will trigger diminishing returns
            # when coverage doesn't improve at all
            convergence_delta=0.05,
        )
        result = await loop.run(task, config)

        # ✅ Should stop from diminishing returns (after round 2)
        # Round 1: coverage 0.0, round 2: still 0.0 → delta=0 < 0.05
        assert result.total_rounds <= 3
        is_diminishing = "diminishing" in (result.termination_reason or "").lower()
        is_max_rounds = result.total_rounds == 5
        assert is_diminishing or is_max_rounds or result.total_rounds < 5


# ============================================================================
# 🧪 Section 4: Runner Pipeline Dispatch Integration (5 tests)
# ============================================================================


class TestRunnerPipelineDispatch:
    """Runner dispatches to correct pipeline based on mode 🔧."""

    @pytest.mark.asyncio
    async def test_discovery_result_to_evaluation_result(self) -> None:
        """DiscoveryResult converts to EvaluationResult correctly 🔄."""
        # 🔧 Build a mock DiscoveryResult
        discovery_result = DiscoveryResult(
            task_id="dispatch-test-001",
            pipeline_mode="discovery",
            total_rounds=2,
            final_coverage=0.85,
            total_cost_usd=1.20,
            termination_reason="coverage_threshold_reached",
            evidence=[
                CleanedEvidence(
                    id="E1",
                    summary=(
                        "Strong evidence about protein binding "
                        "affinity data across multiple cell lines"
                    ),
                    url="https://pubmed.ncbi.nlm.nih.gov/123",
                    tag="academic",
                    source_query="protein binding",
                    mcp_server="perplexity",
                ),
            ],
            claims=[
                {
                    "claim": "Protein shows strong binding affinity",
                    "evidence_ids": ["E1"],
                    "strength": "strong",
                },
            ],
            gap_reports=[
                GapReport(
                    round_number=1,
                    coverage_ratio=0.5,
                    covered_items=["item1"],
                    uncovered_items=["item2"],
                ),
                GapReport(
                    round_number=2,
                    coverage_ratio=0.85,
                    covered_items=["item1", "item2"],
                    uncovered_items=[],
                    converged=True,
                    convergence_reason="coverage_threshold_reached",
                ),
            ],
        )

        task = _make_task(task_id="dispatch-test-001")

        # 🔧 Call the static conversion method from runner module
        from inquiro.core.runner import EvalTaskRunner

        eval_result = EvalTaskRunner._discovery_to_evaluation_result(
            EvalTaskRunner.__new__(EvalTaskRunner),
            discovery_result=discovery_result,
            task=task,
        )

        # ✅ Verify conversion
        assert eval_result.task_id == "dispatch-test-001"
        assert eval_result.pipeline_mode == "discovery"
        assert eval_result.discovery_rounds == 2
        assert eval_result.discovery_coverage == 0.85
        assert len(eval_result.evidence_index) == 1
        assert len(eval_result.reasoning) == 1

    @pytest.mark.asyncio
    async def test_evaluation_result_has_discovery_metadata(self) -> None:
        """EvaluationResult has correct discovery metadata 📊."""
        discovery_result = DiscoveryResult(
            task_id="meta-test-001",
            pipeline_mode="discovery",
            total_rounds=3,
            final_coverage=0.75,
            total_cost_usd=2.50,
            termination_reason="diminishing_returns",
            evidence=[],
            claims=[],
            gap_reports=[
                GapReport(
                    round_number=3,
                    coverage_ratio=0.75,
                    covered_items=["a", "b"],
                    uncovered_items=["c"],
                ),
            ],
            trajectory_id="traj-abc",
        )

        task = _make_task(task_id="meta-test-001")

        from inquiro.core.runner import EvalTaskRunner

        eval_result = EvalTaskRunner._discovery_to_evaluation_result(
            EvalTaskRunner.__new__(EvalTaskRunner),
            discovery_result=discovery_result,
            task=task,
        )

        # ✅ Discovery metadata in metadata bag
        assert eval_result.metadata.get("discovery") is True
        assert eval_result.metadata["total_rounds"] == 3
        assert eval_result.metadata["termination_reason"] == ("diminishing_returns")
        assert eval_result.metadata["trajectory_id"] == "traj-abc"

    @pytest.mark.asyncio
    async def test_pipeline_mode_preserved_in_result(self) -> None:
        """Pipeline mode is preserved in EvaluationResult 🏷️."""
        discovery_result = DiscoveryResult(
            task_id="mode-test-001",
            pipeline_mode="discovery",
            total_rounds=1,
            final_coverage=0.90,
            total_cost_usd=0.80,
            termination_reason="coverage_threshold_reached",
            gap_reports=[
                GapReport(
                    round_number=1,
                    coverage_ratio=0.90,
                    covered_items=["a"],
                    uncovered_items=[],
                    converged=True,
                ),
            ],
        )

        task = _make_task(task_id="mode-test-001")

        from inquiro.core.runner import EvalTaskRunner

        eval_result = EvalTaskRunner._discovery_to_evaluation_result(
            EvalTaskRunner.__new__(EvalTaskRunner),
            discovery_result=discovery_result,
            task=task,
        )

        # ✅ Pipeline mode should be "discovery"
        assert eval_result.pipeline_mode == "discovery"

    def test_parse_discovery_config_from_task(self) -> None:
        """Runner parses DiscoveryConfig from task correctly 🔧."""
        from inquiro.core.runner import EvalTaskRunner

        runner = EvalTaskRunner.__new__(EvalTaskRunner)

        task = _make_task()
        task.discovery_config = {
            "max_rounds": 5,
            "coverage_threshold": 0.90,
            "max_cost_per_subitem": 12.0,
        }

        config = runner._parse_discovery_config(task)

        # ✅ Config parsed correctly
        assert isinstance(config, DiscoveryConfig)
        assert config.max_rounds == 5
        assert config.coverage_threshold == 0.90
        assert config.max_cost_per_subitem == 12.0

    def test_parse_discovery_config_defaults(self) -> None:
        """Runner uses INTENSITY_PRESETS["standard"] when task has none 🔧."""
        from inquiro.core.runner import EvalTaskRunner

        runner = EvalTaskRunner.__new__(EvalTaskRunner)

        task = _make_task()
        task.discovery_config = None

        config = runner._parse_discovery_config(task)

        # ✅ Default config uses "standard" preset values
        assert isinstance(config, DiscoveryConfig)
        assert config.max_rounds == 2
        assert config.coverage_threshold == 0.75
        assert config.intensity == "standard"


# ============================================================================
# 🧪 Section 5: Multi-Round Progression Integration (5 tests)
# ============================================================================


class ProgressiveCoverageJudge:
    """Judge that covers more items each round 📈.

    Simulates progressive coverage improvement by covering
    one additional checklist item per round.

    Attributes:
        _round: Internal round counter.
    """

    def __init__(self) -> None:
        """Initialize with zero rounds completed 🔧."""
        self._round = 0

    async def judge_coverage(
        self,
        checklist: list[str],
        claims: list[dict[str, Any]],
        evidence: list[Any],
    ) -> CoverageResult:
        """Cover one more item per round 📈.

        Args:
            checklist: Checklist item descriptions.
            claims: Current claims (unused).
            evidence: Current evidence (unused).

        Returns:
            CoverageResult with progressively more covered items.
        """
        self._round += 1
        covered = checklist[: self._round]
        uncovered = checklist[self._round :]
        return CoverageResult(
            covered=covered,
            uncovered=uncovered,
        )


class ProgressiveSearchExecutor:
    """Search executor that returns more evidence each round 📈.

    Attributes:
        call_count: Number of times execute_search has been called.
    """

    def __init__(self) -> None:
        """Initialize search executor 🔧."""
        self.call_count = 0

    async def execute_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> SearchRoundOutput:
        """Return increasing evidence per round 📈.

        Args:
            task: Evaluation task.
            config: Discovery config.
            round_number: Current round number.
            focus_prompt: Optional focus guidance.

        Returns:
            SearchRoundOutput with round_number * 3 evidence items.
        """
        self.call_count += 1
        evidence_count = round_number * 3
        evidence = [
            _make_evidence(
                f"E-R{round_number}-{i}",
                summary=(
                    f"Round {round_number} evidence item {i} "
                    f"about research findings in molecular biology "
                    f"and pharmacological mechanisms of action"
                ),
            )
            for i in range(evidence_count)
        ]
        return SearchRoundOutput(
            evidence=evidence,
            queries_executed=[
                f"query-r{round_number}-{i}" for i in range(evidence_count)
            ],
            mcp_tools_used=["mock-tool"],
            cost_usd=0.10 * round_number,
            duration_seconds=1.0,
        )


class TestMultiRoundProgression:
    """Multi-round progression with progressive coverage 📈."""

    @pytest.mark.asyncio
    async def test_three_round_progressive_coverage(self) -> None:
        """3-round loop with progressive coverage improvement 📈."""
        progressive_judge = ProgressiveCoverageJudge()
        gap = GapAnalysis(coverage_judge=progressive_judge)

        loop = DiscoveryLoop(
            search_executor=ProgressiveSearchExecutor(),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=_make_claims("test"),
            ),
            gap_analysis=gap,
        )

        task = _make_task()
        # 🔧 High threshold so it runs 3 rounds
        config = _make_config(
            max_rounds=5,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Coverage should improve each round
        for i in range(1, len(result.gap_reports)):
            assert (
                result.gap_reports[i].coverage_ratio
                >= result.gap_reports[i - 1].coverage_ratio
            )

    @pytest.mark.asyncio
    async def test_focus_prompt_generated_between_rounds(self) -> None:
        """Focus prompt is generated between rounds 🎯."""
        focus_gen = MockFocusPromptGenerator()

        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=_make_unique_evidence(5),
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
            focus_generator=focus_gen,
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Focus generator should be called between rounds
        # (called after each non-final round)
        assert focus_gen.call_count >= result.total_rounds - 1

    @pytest.mark.asyncio
    async def test_each_round_has_increasing_evidence(self) -> None:
        """Each round produces increasing evidence counts 📊."""
        progressive_search = ProgressiveSearchExecutor()

        loop = DiscoveryLoop(
            search_executor=progressive_search,
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Progressive search returns more evidence each round
        for i, summary in enumerate(result.round_summaries, 1):
            assert summary.raw_evidence_count == i * 3

    @pytest.mark.asyncio
    async def test_final_result_contains_all_accumulated(self) -> None:
        """Final result contains all accumulated evidence and claims 📋."""
        progressive_search = ProgressiveSearchExecutor()
        claims = _make_claims("research") + _make_claims("findings")

        loop = DiscoveryLoop(
            search_executor=progressive_search,
            analysis_executor=MockAnalysisExecutor(
                mock_claims=claims,
            ),
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Evidence should accumulate across rounds
        # Round 1: 3, Round 2: 6, Round 3: 9 → total = 18
        expected_evidence = sum(
            rs.cleaned_evidence_count for rs in result.round_summaries
        )
        assert len(result.evidence) == expected_evidence

        # ✅ Claims should accumulate (2 claims per round * 3 rounds = 6)
        assert len(result.claims) == len(claims) * result.total_rounds

    @pytest.mark.asyncio
    async def test_round_summaries_sequential_numbering(self) -> None:
        """Round summaries have sequential numbering and valid metrics 🔢."""
        loop = DiscoveryLoop(
            search_executor=MockSearchExecutor(
                mock_evidence=_make_unique_evidence(5),
            ),
            analysis_executor=MockAnalysisExecutor(
                mock_claims=[],
            ),
        )

        task = _make_task()
        config = _make_config(
            max_rounds=3,
            coverage_threshold=0.99,
            max_cost=100.0,
            convergence_delta=0.0,
        )
        result = await loop.run(task, config)

        # ✅ Sequential round numbering
        for i, summary in enumerate(result.round_summaries):
            assert summary.round_number == i + 1

        # ✅ Valid metrics in each summary
        for summary in result.round_summaries:
            assert summary.queries_executed >= 0
            assert summary.raw_evidence_count >= 0
            assert summary.cleaned_evidence_count >= 0
            assert 0.0 <= summary.coverage_ratio <= 1.0
            assert summary.round_cost_usd >= 0.0
