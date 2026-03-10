"""Tests for QueryStrategy data models and EvaluationTask/DiscoveryConfig extensions 🧪.

Tests cover:
- QuerySection, ToolAllocation, QueryStrategy serialization round-trips.
- Default value correctness.
- EvaluationTask backward compatibility with query_strategy field.
- DiscoveryConfig backward compatibility and validation for new fields.

Uses Google Python Style Guide. English comments with emojis.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inquiro.core.query_strategy import QuerySection, QueryStrategy, ToolAllocation
from inquiro.core.types import DiscoveryConfig, EvaluationTask


# ============================================================
# 📋 QuerySection Tests
# ============================================================


class TestQuerySection:
    """Serialization and default-value tests for QuerySection 📋."""

    def test_round_trip_with_all_fields(self) -> None:
        """QuerySection with all fields survives model_dump / model_validate."""
        # Arrange
        section = QuerySection(
            id="Q1",
            priority=1,
            tool_name="perplexity",
            description="overview of mechanism",
            content="Search for {entity} mechanism of action",
        )

        # Act
        data = section.model_dump()
        reconstructed = QuerySection.model_validate(data)

        # Assert
        assert reconstructed.id == "Q1"
        assert reconstructed.priority == 1
        assert reconstructed.tool_name == "perplexity"
        assert reconstructed.description == "overview of mechanism"
        assert reconstructed.content == "Search for {entity} mechanism of action"

    def test_default_description_is_empty_string(self) -> None:
        """QuerySection defaults description to empty string."""
        section = QuerySection(
            id="Q2",
            priority=2,
            tool_name="bohrium",
            content="body",
        )
        assert section.description == ""

    def test_round_trip_with_only_required_fields(self) -> None:
        """QuerySection with only required fields serializes and reconstructs."""
        section = QuerySection(
            id="Q3",
            priority=3,
            tool_name="web_search",
            content="query body",
        )
        data = section.model_dump()
        reconstructed = QuerySection.model_validate(data)
        assert reconstructed.id == "Q3"
        assert reconstructed.priority == 3
        assert reconstructed.tool_name == "web_search"


# ============================================================
# 🎯 ToolAllocation Tests
# ============================================================


class TestToolAllocation:
    """Serialization and validation tests for ToolAllocation 🎯."""

    def test_round_trip_with_all_fields(self) -> None:
        """ToolAllocation with all fields survives model_dump / model_validate."""
        # Arrange
        allocation = ToolAllocation(
            tool_name="perplexity",
            percentage=60,
        )

        # Act
        data = allocation.model_dump()
        reconstructed = ToolAllocation.model_validate(data)

        # Assert
        assert reconstructed.tool_name == "perplexity"
        assert reconstructed.percentage == 60

    def test_percentage_lower_bound_zero(self) -> None:
        """ToolAllocation accepts percentage of exactly 0."""
        allocation = ToolAllocation(tool_name="tool_a", percentage=0)
        assert allocation.percentage == 0

    def test_percentage_upper_bound_hundred(self) -> None:
        """ToolAllocation accepts percentage of exactly 100."""
        allocation = ToolAllocation(tool_name="tool_a", percentage=100)
        assert allocation.percentage == 100

    def test_percentage_below_zero_raises(self) -> None:
        """ToolAllocation rejects percentage below 0."""
        with pytest.raises(ValidationError):
            ToolAllocation(tool_name="tool_a", percentage=-1)

    def test_percentage_above_hundred_raises(self) -> None:
        """ToolAllocation rejects percentage above 100."""
        with pytest.raises(ValidationError):
            ToolAllocation(tool_name="tool_a", percentage=101)


# ============================================================
# 🗺️ QueryStrategy Tests
# ============================================================


class TestQueryStrategy:
    """Serialization and default-value tests for QueryStrategy 🗺️."""

    def test_minimal_query_strategy_is_valid(self) -> None:
        """QueryStrategy with only required sub_item_id + alias_expansion works."""
        strategy = QueryStrategy(
            sub_item_id="test_item",
            alias_expansion="alias text",
        )
        assert strategy.sub_item_id == "test_item"
        assert strategy.alias_expansion == "alias text"
        assert strategy.query_sections == []
        assert strategy.tool_allocations == []
        assert strategy.follow_up_rules == ""
        assert strategy.evidence_tiers == ""

    def test_round_trip_with_all_fields(self) -> None:
        """Full QueryStrategy survives model_dump / model_validate round-trip."""
        # Arrange
        strategy = QueryStrategy(
            sub_item_id="genetic_evidence",
            alias_expansion="Entity X also known as Y, Z",
            query_sections=[
                QuerySection(
                    id="Q1",
                    priority=1,
                    tool_name="perplexity",
                    content="search body 1",
                ),
                QuerySection(
                    id="Q2",
                    priority=2,
                    tool_name="bohrium",
                    content="search body 2",
                ),
            ],
            tool_allocations=[
                ToolAllocation(tool_name="perplexity", percentage=70),
                ToolAllocation(tool_name="bohrium", percentage=30),
            ],
            follow_up_rules="## Follow-up\n- Check gaps first",
            evidence_tiers="## Tier 1: RCT\n## Tier 2: Observational",
        )

        # Act
        data = strategy.model_dump()
        reconstructed = QueryStrategy.model_validate(data)

        # Assert
        assert reconstructed.alias_expansion == "Entity X also known as Y, Z"
        assert len(reconstructed.query_sections) == 2
        assert reconstructed.query_sections[0].id == "Q1"
        assert reconstructed.query_sections[1].tool_name == "bohrium"
        assert len(reconstructed.tool_allocations) == 2
        assert reconstructed.tool_allocations[0].percentage == 70
        assert reconstructed.follow_up_rules == "## Follow-up\n- Check gaps first"
        assert "Tier 1" in reconstructed.evidence_tiers

    def test_nested_query_sections_serialized_correctly(self) -> None:
        """QuerySection nested inside QueryStrategy is serialized as a dict."""
        strategy = QueryStrategy(
            sub_item_id="test",
            alias_expansion="aliases",
            query_sections=[
                QuerySection(
                    id="Q1",
                    priority=1,
                    tool_name="tool_x",
                    description="test",
                    content="body",
                )
            ],
        )
        data = strategy.model_dump()
        # 📋 Nested section must appear as a dict in the dump output
        assert isinstance(data["query_sections"], list)
        assert data["query_sections"][0]["id"] == "Q1"
        assert data["query_sections"][0]["description"] == "test"


# ============================================================
# 🔬 EvaluationTask backward-compatibility Tests
# ============================================================


class TestEvaluationTaskQueryStrategy:
    """EvaluationTask backward-compatibility for query_strategy field 🔬."""

    def _minimal_task_kwargs(self) -> dict:
        """Return minimal keyword arguments for constructing an EvaluationTask."""
        return {
            "task_id": "t-001",
            "topic": "test topic",
            "rules": "no rules",
            "checklist": {},
            "output_schema": {},
        }

    def test_query_strategy_defaults_to_none(self) -> None:
        """EvaluationTask created without query_strategy has query_strategy=None."""
        task = EvaluationTask(**self._minimal_task_kwargs())
        assert task.query_strategy is None

    def test_query_strategy_none_explicit(self) -> None:
        """EvaluationTask accepts explicit query_strategy=None."""
        task = EvaluationTask(**self._minimal_task_kwargs(), query_strategy=None)
        assert task.query_strategy is None

    def test_query_strategy_dict_round_trip(self) -> None:
        """EvaluationTask with query_strategy dict survives round-trip."""
        # Arrange — pass strategy as plain dict (opaque to Inquiro)
        strategy_dict = {
            "sub_item_id": "genetic_evidence",
            "alias_expansion": "Target A / Alias B",
            "query_sections": [
                {
                    "id": "Q1",
                    "priority": 1,
                    "tool_name": "perplexity",
                    "content": "query body",
                }
            ],
            "tool_allocations": [],
            "follow_up_rules": "",
            "evidence_tiers": "",
        }
        task = EvaluationTask(
            **self._minimal_task_kwargs(),
            query_strategy=strategy_dict,
        )

        # Act
        data = task.model_dump()
        reconstructed = EvaluationTask.model_validate(data)

        # Assert
        assert reconstructed.query_strategy is not None
        assert reconstructed.query_strategy["alias_expansion"] == "Target A / Alias B"
        assert reconstructed.query_strategy["query_sections"][0]["id"] == "Q1"

    def test_existing_tasks_without_query_strategy_still_construct(self) -> None:
        """Existing EvaluationTask construction without query_strategy is unaffected."""
        task = EvaluationTask(
            task_id="t-002",
            topic="legacy topic",
            rules="legacy rules",
            checklist={},
            output_schema={},
            evolution_profile={"fitness": "coverage"},
        )
        assert task.query_strategy is None
        assert task.evolution_profile == {"fitness": "coverage"}


# ============================================================
# 🔧 DiscoveryConfig backward-compatibility Tests
# ============================================================


class TestDiscoveryConfigNewFields:
    """DiscoveryConfig backward-compatibility and validation for new fields 🔧."""

    def test_enable_parallel_search_defaults_to_true(self) -> None:
        """DiscoveryConfig defaults enable_parallel_search to True."""
        config = DiscoveryConfig()
        assert config.enable_parallel_search is True

    def test_max_parallel_agents_defaults_to_three(self) -> None:
        """DiscoveryConfig defaults max_parallel_agents to 3."""
        config = DiscoveryConfig()
        assert config.max_parallel_agents == 3

    def test_existing_config_without_new_fields_still_constructs(self) -> None:
        """DiscoveryConfig constructed with pre-existing fields is unaffected."""
        config = DiscoveryConfig(
            max_rounds=5,
            coverage_threshold=0.90,
        )
        assert config.max_rounds == 5
        assert config.coverage_threshold == 0.90
        assert config.enable_parallel_search is True
        assert config.max_parallel_agents == 3

    def test_enable_parallel_search_can_be_set_false(self) -> None:
        """DiscoveryConfig accepts enable_parallel_search=False."""
        config = DiscoveryConfig(enable_parallel_search=False)
        assert config.enable_parallel_search is False

    def test_max_parallel_agents_minimum_is_one(self) -> None:
        """DiscoveryConfig accepts max_parallel_agents=1 (ge=1 boundary)."""
        config = DiscoveryConfig(max_parallel_agents=1)
        assert config.max_parallel_agents == 1

    def test_max_parallel_agents_maximum_is_eight(self) -> None:
        """DiscoveryConfig accepts max_parallel_agents=8 (le=8 boundary)."""
        config = DiscoveryConfig(max_parallel_agents=8)
        assert config.max_parallel_agents == 8

    def test_max_parallel_agents_below_one_raises(self) -> None:
        """DiscoveryConfig rejects max_parallel_agents=0 (below ge=1)."""
        with pytest.raises(ValidationError):
            DiscoveryConfig(max_parallel_agents=0)

    def test_max_parallel_agents_above_eight_raises(self) -> None:
        """DiscoveryConfig rejects max_parallel_agents=9 (above le=8)."""
        with pytest.raises(ValidationError):
            DiscoveryConfig(max_parallel_agents=9)

    def test_round_trip_with_new_fields(self) -> None:
        """DiscoveryConfig with new fields survives model_dump / model_validate."""
        config = DiscoveryConfig(
            enable_parallel_search=True,
            max_parallel_agents=5,
        )
        data = config.model_dump()
        reconstructed = DiscoveryConfig.model_validate(data)
        assert reconstructed.enable_parallel_search is True
        assert reconstructed.max_parallel_agents == 5
