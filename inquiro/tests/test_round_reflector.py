"""Tests for RoundReflectionMechanism 🧪."""

import json

import pytest

from inquiro.evolution.mechanisms.round_reflection import RoundReflectionMechanism
from inquiro.evolution.types import MechanismType, ReflectionRecord


class TestRoundReflectionMechanism:
    """Tests for RoundReflectionMechanism lifecycle 🪞."""

    @pytest.fixture()
    def mock_llm_fn(self):
        """Create an async mock LLM function 🤖."""
        async def _llm_fn(prompt: str) -> str:
            return json.dumps({
                "what_worked": "PubMed search yielded high-quality results",
                "what_failed": "fetch_url returned too many irrelevant pages",
                "strategy": "Focus on clinical trial databases next round",
                "tool_recommendations": {
                    "pubmed_search": "maintain",
                    "fetch_url": "decrease",
                },
                "priority_gaps": ["mechanism_of_action", "safety_profile"],
            })
        return _llm_fn

    @pytest.fixture()
    def mechanism(self, mock_llm_fn) -> RoundReflectionMechanism:
        """Create a RoundReflectionMechanism with mock LLM 🔧."""
        return RoundReflectionMechanism(
            llm_fn=mock_llm_fn,
            enabled=True,
        )

    def test_mechanism_type(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """Returns ROUND_REFLECTION mechanism type 🏷️."""
        assert mechanism.mechanism_type == MechanismType.ROUND_REFLECTION

    @pytest.mark.asyncio()
    async def test_produce_skips_round_1(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """produce() returns empty for Round 1 📦."""
        from inquiro.evolution.types import TrajectorySnapshot

        snapshot = TrajectorySnapshot(
            evaluation_id="eval1", task_id="task1",
        )
        result = await mechanism.produce(
            snapshot, {"round_num": 1},
        )
        assert result == []

    @pytest.mark.asyncio()
    async def test_on_round_end_generates_reflection(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """on_round_end generates a ReflectionRecord 🔴."""

        class FakeMetrics:
            evidence_count = 42
            checklist_coverage = 0.65

        class FakeRecord:
            tool_calls = []

        await mechanism.on_round_end(1, FakeRecord(), FakeMetrics())
        assert mechanism._latest_reflection is not None
        assert mechanism._latest_reflection.round_number == 1
        assert "PubMed" in mechanism._latest_reflection.what_worked

    @pytest.mark.asyncio()
    async def test_inject_returns_none_before_reflection(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """inject() returns None when no reflection exists 💉."""
        result = mechanism.inject({"round_num": 1})
        assert result is None

    @pytest.mark.asyncio()
    async def test_inject_returns_markdown_after_reflection(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """inject() returns markdown after on_round_end 💉."""

        class FakeMetrics:
            evidence_count = 42
            checklist_coverage = 0.65

        class FakeRecord:
            tool_calls = []

        await mechanism.on_round_end(1, FakeRecord(), FakeMetrics())
        result = mechanism.inject({"round_num": 2})
        assert result is not None
        assert "## ROUND 1 REFLECTION" in result
        assert "PubMed" in result
        assert "Strategy adjustment" in result

    @pytest.mark.asyncio()
    async def test_produce_returns_experience_after_round_2(
        self, mechanism: RoundReflectionMechanism,
    ) -> None:
        """produce() returns Experience after Round 2+ 📦."""
        from inquiro.evolution.types import TrajectorySnapshot

        class FakeMetrics:
            evidence_count = 42
            checklist_coverage = 0.65

        class FakeRecord:
            tool_calls = []

        # Generate reflection for Round 1
        await mechanism.on_round_end(1, FakeRecord(), FakeMetrics())

        # produce() for Round 2 should return experience
        snapshot = TrajectorySnapshot(
            evaluation_id="eval1", task_id="task1",
        )
        result = await mechanism.produce(
            snapshot, {"round_num": 2, "namespace": "test"},
        )
        assert len(result) == 1
        assert result[0].category == "round_reflection"
        assert result[0].mechanism_type == MechanismType.ROUND_REFLECTION

    def test_inject_disabled(self) -> None:
        """Disabled mechanism returns None 🔇."""

        async def noop(p: str) -> str:
            return ""

        mech = RoundReflectionMechanism(llm_fn=noop, enabled=False)
        assert mech.inject({"round_num": 1}) is None


class TestReflectionRecord:
    """Tests for ReflectionRecord model 📊."""

    def test_default_values(self) -> None:
        """ReflectionRecord has sensible defaults 🔒."""
        rec = ReflectionRecord(round_number=1)
        assert rec.what_worked == ""
        assert rec.what_failed == ""
        assert rec.strategy == ""
        assert rec.tool_recommendations == {}
        assert rec.priority_gaps == []

    def test_full_record(self) -> None:
        """Full ReflectionRecord with all fields 📋."""
        rec = ReflectionRecord(
            round_number=2,
            what_worked="Effective PubMed queries",
            what_failed="Too many duplicate results",
            strategy="Use more specific MeSH terms",
            tool_recommendations={"pubmed_search": "maintain"},
            priority_gaps=["safety_1a", "efficacy_2b"],
        )
        assert rec.round_number == 2
        assert len(rec.priority_gaps) == 2
