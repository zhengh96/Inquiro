"""Tests for UnifiedEvolutionProvider 🧪."""

import pytest

from inquiro.evolution.mechanism_config import MECHANISM_PRESETS, MechanismConfig
from inquiro.evolution.mechanisms.base import BaseMechanism
from inquiro.evolution.types import (
    MechanismType,
    TrajectorySnapshot,
)
from inquiro.evolution.unified_provider import UnifiedEvolutionProvider


# ============================================================================
# 🔧 Test Fixtures
# ============================================================================


class MockMechanism(BaseMechanism):
    """Mock mechanism for testing orchestration 🧪."""

    def __init__(
        self,
        mtype: MechanismType = MechanismType.EXPERIENCE_EXTRACTION,
        inject_text: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self._mtype = mtype
        self._inject_text = inject_text
        self.produce_called = False
        self.on_round_start_called = False
        self.on_round_end_called = False

    @property
    def mechanism_type(self) -> MechanismType:
        return self._mtype

    async def produce(self, snapshot, round_context):
        self.produce_called = True
        return []

    def inject(self, round_context):
        return self._inject_text

    async def on_round_start(self, round_num):
        self.on_round_start_called = True

    async def on_round_end(self, round_num, round_record, metrics):
        self.on_round_end_called = True


class MockTask:
    """Minimal mock EvaluationTask 🧪."""

    task_id = "test_task"
    evaluation_id = "test_eval"
    evolution_profile = None
    discovery_config = None
    context_tags = []


class MockStore:
    """Minimal mock ExperienceStore 🧪."""

    async def add(self, exp):
        return exp.id

    async def deduplicate(self, namespace, new_insight):
        return False

    async def query(self, q):
        return []


class MockCollector:
    """Minimal mock DiscoveryTrajectoryCollector 🧪."""

    def collect_from_round(self, round_record, task, context_tags):
        return TrajectorySnapshot(
            evaluation_id="eval1",
            task_id="task1",
        )


class MockFitnessEvaluator:
    """Minimal mock FitnessEvaluator 🧪."""

    async def evaluate(self, enrichment, before, after, config):
        return []

    async def apply_updates(self, updates, lr):
        pass


# ============================================================================
# 🧪 Tests
# ============================================================================


class TestUnifiedEvolutionProvider:
    """Tests for UnifiedEvolutionProvider orchestration 🧬."""

    @pytest.fixture()
    def provider(self) -> UnifiedEvolutionProvider:
        """Create a provider with mock mechanisms 🔧."""
        mechanisms = [
            MockMechanism(
                MechanismType.EXPERIENCE_EXTRACTION,
                inject_text="## EXPERIENCE INSIGHTS\nUse PubMed first.",
            ),
            MockMechanism(
                MechanismType.TOOL_SELECTION,
                inject_text="## TOOL GUIDANCE\n| Tool | Priority |",
            ),
            MockMechanism(
                MechanismType.ROUND_REFLECTION,
                inject_text=None,  # No reflection yet
            ),
            MockMechanism(
                MechanismType.ACTION_PRINCIPLES,
                enabled=False,  # DISCOVERY only
            ),
        ]
        return UnifiedEvolutionProvider(
            mechanisms=mechanisms,
            collector=MockCollector(),
            fitness_evaluator=MockFitnessEvaluator(),
            store=MockStore(),
            profile_config={"namespace": "test"},
            mechanism_config=MECHANISM_PRESETS["STANDARD"],
            context_tags=["modality:SmallMolecule"],
            sub_item_id="safety_1a",
            task=MockTask(),
        )

    def test_enabled_mechanisms_filter(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """enabled_mechanisms excludes disabled ones 📋."""
        enabled = provider.enabled_mechanisms
        assert len(enabled) == 3  # 4 total - 1 disabled

    def test_ab_group_default_treatment(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """Default ab_group is 'treatment' 🔬."""
        assert provider.ab_group == "treatment"

    def test_get_search_enrichment_combines_sections(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """get_search_enrichment combines mechanism outputs 🔍."""
        result = provider.get_search_enrichment(1, [])
        assert result is not None
        assert "EXPERIENCE INSIGHTS" in result
        assert "TOOL GUIDANCE" in result

    def test_get_search_enrichment_control_group(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """Control group returns None 🔬."""
        provider.ab_group = "control"
        result = provider.get_search_enrichment(1, [])
        assert result is None

    def test_get_analysis_enrichment(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """get_analysis_enrichment wraps with header 🔬."""
        result = provider.get_analysis_enrichment()
        assert result is not None
        assert "# ANALYSIS INSIGHTS" in result

    def test_get_synthesis_enrichment(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """get_synthesis_enrichment wraps with header 📝."""
        result = provider.get_synthesis_enrichment()
        assert result is not None
        assert "# SYNTHESIS INSIGHTS" in result

    @pytest.mark.asyncio()
    async def test_prepare_enrichment_calls_on_round_start(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """prepare_enrichment triggers on_round_start 🔄."""
        await provider.prepare_enrichment()
        for m in provider.enabled_mechanisms:
            assert m.on_round_start_called  # type: ignore[union-attr]

    def test_budget_truncation(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """Long text is truncated to fit token budget ✂️."""
        long_text = "x" * 10000
        result = provider._budget_truncate(long_text)
        assert len(result) < 10000
        assert "truncated" in result

    def test_budget_no_truncation_needed(
        self, provider: UnifiedEvolutionProvider,
    ) -> None:
        """Short text is not truncated 📏."""
        short_text = "Hello"
        result = provider._budget_truncate(short_text)
        assert result == "Hello"


class TestUnifiedProviderProtocolCompat:
    """Test that UnifiedEvolutionProvider matches the old API 🔄."""

    def test_has_get_search_enrichment(self) -> None:
        """Has get_search_enrichment(round_num, gap_items) 🔍."""
        assert hasattr(UnifiedEvolutionProvider, "get_search_enrichment")

    def test_has_get_analysis_enrichment(self) -> None:
        """Has get_analysis_enrichment() 🔬."""
        assert hasattr(UnifiedEvolutionProvider, "get_analysis_enrichment")

    def test_has_get_synthesis_enrichment(self) -> None:
        """Has get_synthesis_enrichment() 📝."""
        assert hasattr(UnifiedEvolutionProvider, "get_synthesis_enrichment")

    def test_has_on_round_complete(self) -> None:
        """Has on_round_complete() 🔄."""
        assert hasattr(UnifiedEvolutionProvider, "on_round_complete")

    def test_has_on_synthesis_complete(self) -> None:
        """Has on_synthesis_complete() 🏁."""
        assert hasattr(UnifiedEvolutionProvider, "on_synthesis_complete")

    def test_has_prepare_enrichment(self) -> None:
        """Has prepare_enrichment() 🔄."""
        assert hasattr(UnifiedEvolutionProvider, "prepare_enrichment")

    def test_has_ab_group(self) -> None:
        """Has ab_group attribute 🔬."""
        p = UnifiedEvolutionProvider(
            mechanisms=[],
            collector=MockCollector(),
            fitness_evaluator=MockFitnessEvaluator(),
            store=MockStore(),
            profile_config={"namespace": "test"},
            mechanism_config=MechanismConfig(),
            context_tags=[],
            sub_item_id="test",
            task=MockTask(),
        )
        assert hasattr(p, "ab_group")
