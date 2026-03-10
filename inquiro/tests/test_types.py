"""Tests for Inquiro core data models 🧪.

Tests all Pydantic v2 data models in ``inquiro.core.types`` for:
- Valid construction with required and optional fields
- Validation errors for invalid inputs
- Serialization round-trip (model_dump → reconstruct)
- Default value behaviour
- Enum constraints
- Helper methods (get_evidence_by_id, get_covered_ratio)

Uses Google Python Style Guide. English comments with emojis.
"""

from __future__ import annotations


import pytest
from pydantic import ValidationError

from inquiro.core.types import (
    # 🏷️ Enums
    CostStatus,
    Decision,
    EvidenceStrength,
    OverspendStrategy,
    TaskPhase,
    TaskStatus,
    TaskType,
    TruncationStrategy,
    # ⚙️ Config Models
    AgentConfig,
    Checklist,
    ChecklistItem,
    CostGuardConfig,
    DecisionGuidance,
    EnsembleConfig,
    EnsembleModelConfig,
    InputReport,
    QualityGateConfig,
    ToolsConfig,
    # 📥 Input Models
    EvaluationTask,
    SynthesisTask,
    # 📤 Output Models
    ChecklistCoverage,
    Evidence,
    EvaluationResult,
    ReasoningClaim,
    SynthesisResult,
    # 🔧 Infrastructure
    CostSummary,
    QualityGateResult,
)


# ============================================================
# 🏷️ Enum Tests
# ============================================================


class TestEnums:
    """Tests for Inquiro enum types 🏷️."""

    def test_task_type_values(self) -> None:
        """TaskType should have research and synthesis values."""
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.SYNTHESIS.value == "synthesis"

    def test_task_status_values(self) -> None:
        """TaskStatus should have all lifecycle statuses."""
        statuses = [s.value for s in TaskStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "cancelled" in statuses

    def test_decision_values(self) -> None:
        """Decision should have positive, cautious, negative."""
        assert Decision.POSITIVE.value == "positive"
        assert Decision.CAUTIOUS.value == "cautious"
        assert Decision.NEGATIVE.value == "negative"

    def test_evidence_strength_values(self) -> None:
        """EvidenceStrength should have weak, moderate, strong."""
        assert EvidenceStrength.WEAK.value == "weak"
        assert EvidenceStrength.MODERATE.value == "moderate"
        assert EvidenceStrength.STRONG.value == "strong"

    def test_overspend_strategy_values(self) -> None:
        """OverspendStrategy should have SoftStop and HardStop."""
        assert OverspendStrategy.SOFT_STOP.value == "SoftStop"
        assert OverspendStrategy.HARD_STOP.value == "HardStop"

    def test_task_phase_values(self) -> None:
        """TaskPhase should have all execution phases."""
        phases = {p.value for p in TaskPhase}
        assert phases == {"searching", "reasoning", "synthesizing", "quality_check"}

    def test_truncation_strategy(self) -> None:
        """TruncationStrategy should have latest_half."""
        assert TruncationStrategy.LATEST_HALF.value == "latest_half"

    def test_cost_status_values(self) -> None:
        """CostStatus should have all budget health statuses."""
        values = {s.value for s in CostStatus}
        assert values == {
            "ok",
            "warning",
            "model_downgrade",
            "budget_critical",
            "task_exceeded",
            "total_exceeded",
        }


# ============================================================
# 📌 ChecklistItem Tests
# ============================================================


class TestChecklistItem:
    """Tests for ChecklistItem model 📌."""

    def test_valid_construction_with_all_fields(self) -> None:
        """ChecklistItem should accept all valid fields."""
        item = ChecklistItem(
            id="item_1",
            description="Market size analysis",
            keywords=["EGFR", "market"],
            suggested_sources=["perplexity", "biomcp"],
        )
        assert item.id == "item_1"
        assert item.description == "Market size analysis"
        assert item.keywords == ["EGFR", "market"]
        assert item.suggested_sources == ["perplexity", "biomcp"]

    def test_valid_construction_minimal(self) -> None:
        """ChecklistItem should work with only required fields."""
        item = ChecklistItem(id="item_1", description="Test item")
        assert item.id == "item_1"
        assert item.description == "Test item"
        assert item.keywords == []
        assert item.suggested_sources == []

    def test_keywords_default_to_empty_list(self) -> None:
        """Keywords should default to empty list when not provided."""
        item = ChecklistItem(id="item_1", description="Test")
        assert item.keywords == []

    def test_serialization_round_trip(self) -> None:
        """ChecklistItem should serialize and deserialize correctly."""
        item = ChecklistItem(
            id="item_1",
            description="Market size",
            keywords=["EGFR", "market"],
            suggested_sources=["perplexity"],
        )
        data = item.model_dump()
        restored = ChecklistItem(**data)
        assert restored == item


# ============================================================
# 📋 Checklist Tests
# ============================================================


class TestChecklist:
    """Tests for Checklist model 📋."""

    def test_valid_with_required_and_optional(self) -> None:
        """Checklist should accept both required and optional items."""
        config = Checklist(
            required=[
                ChecklistItem(id="r1", description="Required item"),
            ],
            optional=[
                ChecklistItem(id="o1", description="Optional item"),
            ],
            coverage_threshold=0.9,
        )
        assert len(config.required) == 1
        assert len(config.optional) == 1
        assert config.coverage_threshold == 0.9

    def test_default_coverage_threshold(self) -> None:
        """Default coverage threshold should be 0.8."""
        config = Checklist()
        assert config.coverage_threshold == 0.8

    def test_coverage_threshold_validation_too_high(self) -> None:
        """Coverage threshold > 1.0 should raise validation error."""
        with pytest.raises(ValidationError):
            Checklist(coverage_threshold=1.5)

    def test_coverage_threshold_validation_too_low(self) -> None:
        """Coverage threshold < 0.0 should raise validation error."""
        with pytest.raises(ValidationError):
            Checklist(coverage_threshold=-0.1)

    def test_empty_required_list_allowed(self) -> None:
        """Empty required list should be valid."""
        config = Checklist(required=[], optional=[])
        assert config.required == []


# ============================================================
# 🤖 AgentConfig Tests
# ============================================================


class TestAgentConfig:
    """Tests for AgentConfig model 🤖."""

    def test_default_values(self) -> None:
        """AgentConfig should have sensible defaults."""
        config = AgentConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_turns == 30
        assert config.temperature == 0.3
        assert config.system_prompt_template is None

    def test_max_turns_must_be_positive(self) -> None:
        """max_turns must be > 0."""
        with pytest.raises(ValidationError):
            AgentConfig(max_turns=0)

    def test_temperature_too_high(self) -> None:
        """temperature > 2.0 should be rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.5)

    def test_temperature_too_low(self) -> None:
        """temperature < 0.0 should be rejected."""
        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)

    def test_context_config_defaults(self) -> None:
        """ContextConfig should have correct defaults."""
        config = AgentConfig()
        assert config.context.max_tokens == 128000
        assert config.context.truncation_strategy == TruncationStrategy.LATEST_HALF

    def test_custom_system_prompt_template(self) -> None:
        """Custom system prompt template should override default."""
        config = AgentConfig(system_prompt_template="Custom template: {topic}")
        assert config.system_prompt_template == "Custom template: {topic}"

    def test_serialization_round_trip(self) -> None:
        """AgentConfig should survive serialize/deserialize."""
        config = AgentConfig(
            model="claude-opus-4-20250514",
            max_turns=50,
            temperature=0.7,
        )
        data = config.model_dump()
        restored = AgentConfig(**data)
        assert restored.model == config.model
        assert restored.max_turns == config.max_turns
        assert restored.temperature == config.temperature


# ============================================================
# 🔧 ToolsConfig Tests
# ============================================================


class TestToolsConfig:
    """Tests for ToolsConfig model 🔧."""

    def test_default_empty(self) -> None:
        """ToolsConfig should default to empty server list."""
        config = ToolsConfig()
        assert config.mcp_servers == []
        assert config.mcp_config_override == {}

    def test_with_mcp_servers(self) -> None:
        """ToolsConfig should accept a list of MCP server names."""
        config = ToolsConfig(mcp_servers=["perplexity", "biomcp"])
        assert config.mcp_servers == ["perplexity", "biomcp"]


# ============================================================
# 🎭 EnsembleConfig Tests
# ============================================================


class TestEnsembleConfig:
    """Tests for EnsembleConfig model 🎭."""

    def test_disabled_by_default(self) -> None:
        """Ensemble should be disabled by default."""
        config = EnsembleConfig()
        assert config.enabled is False
        assert config.models == []

    def test_enabled_with_models(self) -> None:
        """Ensemble should accept models with weights."""
        config = EnsembleConfig(
            enabled=True,
            models=[
                EnsembleModelConfig(name="gpt-4o", weight=0.5),
                EnsembleModelConfig(name="claude-sonnet", weight=0.5),
            ],
            consensus_threshold=0.8,
        )
        assert config.enabled is True
        assert len(config.models) == 2
        assert config.consensus_threshold == 0.8

    def test_consensus_threshold_too_high(self) -> None:
        """consensus_threshold > 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            EnsembleConfig(consensus_threshold=1.5)

    def test_consensus_threshold_too_low(self) -> None:
        """consensus_threshold < 0.0 should be rejected."""
        with pytest.raises(ValidationError):
            EnsembleConfig(consensus_threshold=-0.1)

    def test_model_weight_bounds(self) -> None:
        """Model weight must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            EnsembleModelConfig(name="test", weight=1.5)


# ============================================================
# ✅ QualityGateConfig Tests
# ============================================================


class TestQualityGateConfig:
    """Tests for QualityGateConfig model ✅."""

    def test_enabled_by_default(self) -> None:
        """Quality gate should be enabled by default."""
        config = QualityGateConfig()
        assert config.enabled is True
        assert config.max_retries == 2

    def test_all_checks_enabled_by_default(self) -> None:
        """Most quality checks should be enabled by default."""
        config = QualityGateConfig()
        assert config.checks.schema_validation is True
        assert config.checks.coverage_check is True
        assert config.checks.evidence_reference_check is True

    def test_cross_reference_check_disabled_by_default(self) -> None:
        """Cross-reference check should be disabled by default."""
        config = QualityGateConfig()
        assert config.checks.cross_reference_check is False

    def test_max_retries_non_negative(self) -> None:
        """max_retries must be >= 0."""
        with pytest.raises(ValidationError):
            QualityGateConfig(max_retries=-1)


# ============================================================
# 💰 CostGuardConfig Tests
# ============================================================


class TestCostGuardConfig:
    """Tests for CostGuardConfig model 💰."""

    def test_default_values(self) -> None:
        """CostGuardConfig should have sensible defaults."""
        config = CostGuardConfig()
        assert config.max_cost_per_task == 1.5
        assert config.overspend_strategy == OverspendStrategy.SOFT_STOP

    def test_hard_stop_strategy(self) -> None:
        """HardStop strategy should be accepted."""
        config = CostGuardConfig(overspend_strategy=OverspendStrategy.HARD_STOP)
        assert config.overspend_strategy == OverspendStrategy.HARD_STOP

    def test_negative_cost_rejected(self) -> None:
        """Negative max_cost should be rejected (gt=0.0)."""
        with pytest.raises(ValidationError):
            CostGuardConfig(max_cost_per_task=-1.0)

    def test_zero_cost_rejected(self) -> None:
        """Zero max_cost should be rejected (gt=0.0)."""
        with pytest.raises(ValidationError):
            CostGuardConfig(max_cost_per_task=0.0)


# ============================================================
# 🔬 EvaluationTask Tests
# ============================================================


class TestEvaluationTask:
    """Tests for EvaluationTask (research request) model 🔬."""

    def test_minimal_construction(self) -> None:
        """EvaluationTask with only required fields."""
        task = EvaluationTask(
            task_id="test-001",
            topic="Test topic",
        )
        assert task.task_id == "test-001"
        assert task.topic == "Test topic"
        assert task.rules == ""
        assert task.output_schema == {}
        assert task.agent_config.max_turns == 30

    def test_full_construction(self) -> None:
        """EvaluationTask with all fields populated."""
        schema = {"type": "object", "required": ["decision"]}
        task = EvaluationTask(
            task_id="test-002",
            topic="EGFR market size",
            rules="# Rules\nFocus on 2020-2025.",
            checklist=Checklist(
                required=[
                    ChecklistItem(id="c1", description="Market size"),
                ],
            ),
            decision_guidance=DecisionGuidance(
                positive=["Large market"],
            ),
            output_schema=schema,
            agent_config=AgentConfig(max_turns=20),
            cost_guard=CostGuardConfig(max_cost_per_task=2.0),
        )
        assert task.topic == "EGFR market size"
        assert len(task.checklist.required) == 1
        assert task.output_schema == schema
        assert task.agent_config.max_turns == 20
        assert task.cost_guard.max_cost_per_task == 2.0

    def test_serialization_round_trip(self) -> None:
        """EvaluationTask should survive serialize/deserialize."""
        task = EvaluationTask(
            task_id="rt-001",
            topic="Round trip test",
            output_schema={"type": "object"},
        )
        data = task.model_dump()
        restored = EvaluationTask(**data)
        assert restored.task_id == task.task_id
        assert restored.topic == task.topic

    def test_json_serialization(self) -> None:
        """EvaluationTask should serialize to/from JSON string."""
        task = EvaluationTask(task_id="json-001", topic="JSON test")
        json_str = task.model_dump_json()
        restored = EvaluationTask.model_validate_json(json_str)
        assert restored.task_id == "json-001"


# ============================================================
# 📊 SynthesisTask Tests
# ============================================================


class TestSynthesisTask:
    """Tests for SynthesisTask model 📊."""

    def test_minimal_construction(self) -> None:
        """SynthesisTask with required fields."""
        task = SynthesisTask(
            task_id="synth-001",
            topic="Synthesis test",
            input_reports=[
                InputReport(
                    report_id="r1",
                    label="Report 1",
                    content={"decision": "positive"},
                ),
            ],
        )
        assert task.task_id == "synth-001"
        assert len(task.input_reports) == 1

    def test_allow_additional_research_defaults_true(self) -> None:
        """W-4 fix: allow_additional_research should default to True ✅."""
        task = SynthesisTask(
            task_id="synth-002",
            topic="Test defaults",
            input_reports=[
                InputReport(
                    report_id="r1",
                    label="Report",
                    content={},
                ),
            ],
        )
        assert task.allow_additional_research is True

    def test_input_reports_minimum_one(self) -> None:
        """At least one input report is required (min_length=1)."""
        with pytest.raises(ValidationError):
            SynthesisTask(
                task_id="synth-003",
                topic="No reports",
                input_reports=[],
            )

    def test_serialization_round_trip(self) -> None:
        """SynthesisTask should survive serialize/deserialize."""
        task = SynthesisTask(
            task_id="synth-rt",
            topic="Round trip",
            input_reports=[
                InputReport(
                    report_id="r1",
                    label="Report 1",
                    content={"key": "value"},
                ),
            ],
        )
        data = task.model_dump()
        restored = SynthesisTask(**data)
        assert restored.task_id == task.task_id
        assert len(restored.input_reports) == len(task.input_reports)


# ============================================================
# 📖 InputReport Tests
# ============================================================


class TestInputReport:
    """Tests for InputReport model 📖."""

    def test_valid_construction(self) -> None:
        """InputReport should accept valid fields."""
        report = InputReport(
            report_id="uuid-report-1",
            label="Target Biology",
            content={"decision": "positive", "confidence": 0.85},
        )
        assert report.report_id == "uuid-report-1"
        assert report.label == "Target Biology"

    def test_content_preserves_structure(self) -> None:
        """Report content dict should preserve nested structure."""
        nested = {
            "decision": "positive",
            "reasoning": [
                {"claim": "test", "evidence_ids": ["E1"]},
            ],
        }
        report = InputReport(
            report_id="r1",
            label="Test",
            content=nested,
        )
        assert report.content["reasoning"][0]["claim"] == "test"

    def test_serialization_round_trip(self) -> None:
        """InputReport should survive serialize/deserialize."""
        report = InputReport(
            report_id="r1",
            label="Test",
            content={"key": [1, 2, 3]},
        )
        data = report.model_dump()
        restored = InputReport(**data)
        assert restored == report


# ============================================================
# 📋 EvaluationResult Tests (with helper methods)
# ============================================================


class TestEvaluationResult:
    """Tests for EvaluationResult model and helper methods 📋."""

    @pytest.fixture
    def sample_result(self) -> EvaluationResult:
        """Build a realistic EvaluationResult for testing 🔬."""
        return EvaluationResult(
            task_id="res-001",
            decision=Decision.POSITIVE,
            confidence=0.85,
            reasoning=[
                ReasoningClaim(
                    claim="Market exceeds $20B",
                    evidence_ids=["E1", "E2"],
                    strength=EvidenceStrength.STRONG,
                ),
            ],
            evidence_index=[
                Evidence(
                    id="E1",
                    source="perplexity",
                    query="EGFR market size",
                    summary="$22.5B global market",
                ),
                Evidence(
                    id="E2",
                    source="biomcp",
                    query="EGFR growth",
                    summary="8% CAGR",
                ),
            ],
            search_rounds=2,
            checklist_coverage=ChecklistCoverage(
                required_covered=["c1", "c2"],
                required_missing=["c3"],
            ),
        )

    def test_valid_construction(self, sample_result: EvaluationResult) -> None:
        """EvaluationResult should construct with valid data."""
        assert sample_result.task_id == "res-001"
        assert sample_result.decision == Decision.POSITIVE
        assert sample_result.confidence == 0.85

    def test_confidence_validation_too_high(self) -> None:
        """Confidence > 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                task_id="bad",
                decision=Decision.POSITIVE,
                confidence=1.5,
            )

    def test_confidence_validation_too_low(self) -> None:
        """Confidence < 0.0 should be rejected."""
        with pytest.raises(ValidationError):
            EvaluationResult(
                task_id="bad",
                decision=Decision.POSITIVE,
                confidence=-0.1,
            )

    def test_confidence_boundary_zero(self) -> None:
        """Confidence == 0.0 should be accepted."""
        result = EvaluationResult(
            task_id="edge",
            decision=Decision.NEGATIVE,
            confidence=0.0,
        )
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """Confidence == 1.0 should be accepted."""
        result = EvaluationResult(
            task_id="edge",
            decision=Decision.POSITIVE,
            confidence=1.0,
        )
        assert result.confidence == 1.0

    def test_get_evidence_by_id_found(self, sample_result: EvaluationResult) -> None:
        """get_evidence_by_id should return matching Evidence 🔍."""
        ev = sample_result.get_evidence_by_id("E1")
        assert ev is not None
        assert ev.source == "perplexity"

    def test_get_evidence_by_id_not_found(
        self, sample_result: EvaluationResult
    ) -> None:
        """get_evidence_by_id should return None for unknown ID."""
        ev = sample_result.get_evidence_by_id("E999")
        assert ev is None

    def test_get_covered_ratio(self, sample_result: EvaluationResult) -> None:
        """get_covered_ratio should calculate correct coverage 📊."""
        # 2 covered, 1 missing → 2/3 ≈ 0.6667
        ratio = sample_result.get_covered_ratio()
        assert abs(ratio - 2 / 3) < 1e-9

    def test_get_covered_ratio_all_covered(self) -> None:
        """get_covered_ratio should return 1.0 when all covered."""
        result = EvaluationResult(
            task_id="full",
            decision=Decision.POSITIVE,
            confidence=0.9,
            checklist_coverage=ChecklistCoverage(
                required_covered=["c1", "c2"],
                required_missing=[],
            ),
        )
        assert result.get_covered_ratio() == 1.0

    def test_get_covered_ratio_empty(self) -> None:
        """get_covered_ratio should return 1.0 when no items exist."""
        result = EvaluationResult(
            task_id="empty",
            decision=Decision.POSITIVE,
            confidence=0.5,
        )
        assert result.get_covered_ratio() == 1.0

    def test_serialization_round_trip(self, sample_result: EvaluationResult) -> None:
        """EvaluationResult should survive serialize/deserialize."""
        data = sample_result.model_dump()
        restored = EvaluationResult(**data)
        assert restored.task_id == sample_result.task_id
        assert restored.confidence == sample_result.confidence
        assert len(restored.evidence_index) == len(sample_result.evidence_index)

    def test_json_serialization_round_trip(
        self, sample_result: EvaluationResult
    ) -> None:
        """EvaluationResult should survive JSON string round-trip."""
        json_str = sample_result.model_dump_json()
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.task_id == sample_result.task_id


# ============================================================
# 📊 SynthesisResult Tests
# ============================================================


class TestSynthesisResult:
    """Tests for SynthesisResult model 📊."""

    def test_valid_construction(self) -> None:
        """SynthesisResult should construct with valid data."""
        result = SynthesisResult(
            task_id="synth-001",
            decision=Decision.CAUTIOUS,
            confidence=0.7,
            source_reports=["r1", "r2"],
        )
        assert result.task_id == "synth-001"
        assert result.decision == Decision.CAUTIOUS

    def test_confidence_validation(self) -> None:
        """Confidence > 1.0 should be rejected."""
        with pytest.raises(ValidationError):
            SynthesisResult(
                task_id="bad",
                decision=Decision.POSITIVE,
                confidence=1.1,
            )

    def test_get_evidence_by_id(self) -> None:
        """SynthesisResult.get_evidence_by_id should work 🔍."""
        result = SynthesisResult(
            task_id="synth-002",
            decision=Decision.POSITIVE,
            confidence=0.8,
            evidence_index=[
                Evidence(
                    id="SE1",
                    source="perplexity",
                    query="test",
                    summary="test summary",
                ),
            ],
        )
        assert result.get_evidence_by_id("SE1") is not None
        assert result.get_evidence_by_id("SE999") is None


# ============================================================
# 📬 Response & API Model Tests
# ============================================================


# ============================================================
# 💰 EvaluationResult / SynthesisResult cost field tests
# ============================================================


class TestEvaluationResultCost:
    """Tests for cost field on EvaluationResult 💰."""

    def test_evaluation_result_has_cost_field(self) -> None:
        """Verify EvaluationResult has cost field 💰."""
        result = EvaluationResult(
            task_id="test",
            decision=Decision.POSITIVE,
            confidence=0.8,
            cost=1.23,
        )
        assert result.cost == 1.23
        dumped = result.model_dump()
        assert dumped["cost"] == 1.23

    def test_evaluation_result_cost_default(self) -> None:
        """Verify EvaluationResult cost defaults to 0.0 💰."""
        result = EvaluationResult(
            task_id="test",
            decision=Decision.POSITIVE,
            confidence=0.8,
        )
        assert result.cost == 0.0

    def test_evaluation_result_cost_serialization_round_trip(
        self,
    ) -> None:
        """Verify cost survives serialize/deserialize round-trip 📦."""
        result = EvaluationResult(
            task_id="cost-rt",
            decision=Decision.CAUTIOUS,
            confidence=0.6,
            cost=3.14,
        )
        data = result.model_dump()
        restored = EvaluationResult(**data)
        assert restored.cost == 3.14


class TestSynthesisResultCost:
    """Tests for cost field on SynthesisResult 💰."""

    def test_synthesis_result_has_cost_field(self) -> None:
        """Verify SynthesisResult has cost field 💰."""
        result = SynthesisResult(
            task_id="test",
            decision=Decision.CAUTIOUS,
            confidence=0.6,
            cost=2.50,
        )
        assert result.cost == 2.50

    def test_synthesis_result_cost_default(self) -> None:
        """Verify SynthesisResult cost defaults to 0.0 💰."""
        result = SynthesisResult(
            task_id="test",
            decision=Decision.POSITIVE,
            confidence=0.9,
        )
        assert result.cost == 0.0

    def test_synthesis_result_cost_serialization_round_trip(
        self,
    ) -> None:
        """Verify cost survives serialize/deserialize round-trip 📦."""
        result = SynthesisResult(
            task_id="cost-rt",
            decision=Decision.NEGATIVE,
            confidence=0.4,
            cost=5.67,
        )
        data = result.model_dump()
        restored = SynthesisResult(**data)
        assert restored.cost == 5.67


class TestCostSummary:
    """Tests for CostSummary model 💰."""

    def test_cost_summary(self) -> None:
        """CostSummary should track per-task costs."""
        summary = CostSummary(
            task_costs={"t1": 0.5, "t2": 1.0},
            total_cost=1.5,
            budget_remaining=8.5,
        )
        assert summary.total_cost == 1.5
        assert summary.budget_remaining == 8.5


class TestQualityGateResult:
    """Tests for QualityGateResult model ✅."""

    def test_passed(self) -> None:
        """QualityGateResult should report passing status."""
        result = QualityGateResult(passed=True)
        assert result.passed is True
        assert result.hard_failures == []
        assert result.soft_failures == []

    def test_failed(self) -> None:
        """QualityGateResult should report failures."""
        result = QualityGateResult(
            passed=False,
            hard_failures=["Missing required field: decision"],
            soft_failures=["Low coverage: 60%"],
            confidence_cap=0.5,
        )
        assert result.passed is False
        assert len(result.hard_failures) == 1
        assert result.confidence_cap == 0.5
