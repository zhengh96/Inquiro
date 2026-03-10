"""Tests for Inquiro PromptEnricher 🧪.

Tests the prompt enrichment engine:
- ExperienceStore querying
- Category grouping
- Jinja2 template rendering
- Token budget management
- Truncation by fitness_score priority
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.types import EnrichmentResult, Experience, ExperienceQuery


# ============================================================
# 🏗️ Fixtures
# ============================================================


@pytest.fixture
def sample_experiences() -> list[Experience]:
    """Sample experiences for testing 🧬."""
    return [
        Experience(
            id="exp_001",
            namespace="targetmaster",
            category="search_strategy",
            insight="PubMed queries with specific gene names yield better results",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["evidence_1a"],
            fitness_score=0.9,
            times_used=10,
            times_helpful=9,
            source="trajectory_extraction",
            created_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
        ),
        Experience(
            id="exp_002",
            namespace="targetmaster",
            category="evidence_quality",
            insight="Clinical trial data provides strong validation",
            context_tags=["indication:Obesity"],
            applicable_sub_items=["*"],
            fitness_score=0.8,
            times_used=8,
            times_helpful=7,
            source="trajectory_extraction",
            created_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
        ),
        Experience(
            id="exp_003",
            namespace="targetmaster",
            category="search_strategy",
            insight="Cross-referencing multiple databases improves coverage",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["evidence_1a"],
            fitness_score=0.7,
            times_used=5,
            times_helpful=3,
            source="trajectory_extraction",
            created_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
        ),
        Experience(
            id="exp_004",
            namespace="targetmaster",
            category="reasoning_pattern",
            insight="Low fitness experience that should be dropped first",
            context_tags=["modality:SmallMolecule"],
            applicable_sub_items=["evidence_1a"],
            fitness_score=0.3,
            times_used=2,
            times_helpful=0,
            source="trajectory_extraction",
            created_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def sample_profile_config() -> dict[str, Any]:
    """Sample evolution profile configuration 📋."""
    return {
        "enrichment_prompt_template": """
# RELEVANT EXPERIENCES

{% for category, exps in experiences_by_category.items() %}
## {{ category }}
{% for exp in exps %}
- [{{ exp.fitness_score }}] {{ exp.insight }}
{% endfor %}
{% endfor %}
""".strip(),
        "enrichment_max_tokens": 1000,
        "enrichment_max_items": 10,
        "namespace": "targetmaster",
        "prune_min_fitness": 0.3,
    }


@pytest.fixture
def mock_store(sample_experiences: list[Experience]) -> Any:
    """Mock ExperienceStore that returns sample experiences 🗄️."""
    store = MagicMock()
    store.query = AsyncMock(return_value=sample_experiences)
    return store


# ============================================================
# ✅ Success Path Tests
# ============================================================


@pytest.mark.asyncio
async def test_enrich_success(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
    sample_experiences: list[Experience],
) -> None:
    """Test successful prompt enrichment ✅."""
    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=sample_profile_config,
    )

    # Should return non-empty enrichment
    assert isinstance(result, EnrichmentResult)
    assert len(result.injected_experience_ids) == 4  # All 4 experiences
    assert len(result.enrichment_text) > 0
    assert result.token_count > 0
    assert not result.truncated  # Should fit within budget

    # Check that all experience IDs are included
    assert "exp_001" in result.injected_experience_ids
    assert "exp_002" in result.injected_experience_ids
    assert "exp_003" in result.injected_experience_ids
    assert "exp_004" in result.injected_experience_ids


@pytest.mark.asyncio
async def test_enrich_category_grouping(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test that experiences are grouped by category 📊."""
    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=sample_profile_config,
    )

    # Check that enrichment text contains category headers
    assert "search_strategy" in result.enrichment_text
    assert "evidence_quality" in result.enrichment_text
    assert "reasoning_pattern" in result.enrichment_text

    # Check that insights are included
    assert "PubMed" in result.enrichment_text
    assert "Clinical trial" in result.enrichment_text
    assert "Cross-referencing" in result.enrichment_text


@pytest.mark.asyncio
async def test_enrich_store_query(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test that ExperienceStore is queried correctly 🔍."""
    enricher = PromptEnricher(store=mock_store)

    await enricher.enrich(
        task_context_tags=["modality:SmallMolecule", "indication:Obesity"],
        sub_item="evidence_1a",
        profile_config=sample_profile_config,
    )

    # Verify store.query was called
    mock_store.query.assert_called_once()

    # Check query parameters
    call_args = mock_store.query.call_args[0][0]
    assert isinstance(call_args, ExperienceQuery)
    assert call_args.namespace == "targetmaster"
    assert call_args.context_tags == ["modality:SmallMolecule", "indication:Obesity"]
    assert call_args.sub_item == "evidence_1a"
    assert call_args.min_fitness == 0.3
    assert call_args.max_results == 20  # 10 * 2


# ============================================================
# ✂️ Token Budget and Truncation Tests
# ============================================================


@pytest.mark.asyncio
async def test_enrich_truncation_by_fitness(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test truncation when enrichment exceeds token budget ✂️."""
    # Set very low token budget to force truncation
    config = sample_profile_config.copy()
    config["enrichment_max_tokens"] = 50  # Very small budget

    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=config,
    )

    # Should be truncated
    assert result.truncated
    assert len(result.injected_experience_ids) < 4  # Some experiences dropped

    # Should keep highest-fitness experiences
    # exp_001 (0.9) and exp_002 (0.8) should be prioritized over exp_004 (0.3)
    if "exp_001" in result.injected_experience_ids:
        # If we kept at least one, it should be high-fitness
        if len(result.injected_experience_ids) >= 1:
            # exp_004 (lowest fitness) should be dropped first
            assert "exp_004" not in result.injected_experience_ids


@pytest.mark.asyncio
async def test_enrich_token_estimation(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test token count estimation 📏."""
    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=sample_profile_config,
    )

    # Token count should be roughly 1/4 of character count
    expected_tokens = len(result.enrichment_text) // 4
    assert result.token_count == expected_tokens


@pytest.mark.asyncio
async def test_enrich_empty_experiences(
    sample_profile_config: dict[str, Any],
) -> None:
    """Test enrichment with empty experience list 📭."""
    # Mock store that returns no experiences
    empty_store = MagicMock()
    empty_store.query = AsyncMock(return_value=[])

    enricher = PromptEnricher(store=empty_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=sample_profile_config,
    )

    # Should return valid but empty result
    assert isinstance(result, EnrichmentResult)
    assert result.injected_experience_ids == []
    # Text might be empty or just template structure
    assert result.token_count >= 0
    assert not result.truncated


# ============================================================
# ⚠️ Error Handling Tests
# ============================================================


@pytest.mark.asyncio
async def test_enrich_template_rendering_error(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test handling of Jinja2 template rendering errors ⚠️."""
    # Create invalid template
    invalid_config = sample_profile_config.copy()
    invalid_config["enrichment_prompt_template"] = "{{ undefined_variable.foo }}"

    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=invalid_config,
    )

    # Should return empty result on template error
    assert isinstance(result, EnrichmentResult)
    assert result.injected_experience_ids == []
    assert result.enrichment_text == ""
    assert result.token_count == 0
    assert not result.truncated


@pytest.mark.asyncio
async def test_enrich_extreme_truncation(
    mock_store: Any,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test truncation with impossibly small token budget 🚫."""
    # Set token budget so small that even 1 experience won't fit
    config = sample_profile_config.copy()
    config["enrichment_max_tokens"] = 1  # Impossibly small

    enricher = PromptEnricher(store=mock_store)

    result = await enricher.enrich(
        task_context_tags=["modality:SmallMolecule"],
        sub_item="evidence_1a",
        profile_config=config,
    )

    # Should return empty result when nothing fits
    assert result.truncated
    assert result.injected_experience_ids == []
    assert result.enrichment_text == ""


# ============================================================
# 🧪 Internal Method Tests
# ============================================================


def test_group_by_category(
    mock_store: Any,
    sample_experiences: list[Experience],
) -> None:
    """Test _group_by_category internal method 📊."""
    enricher = PromptEnricher(store=mock_store)

    groups = enricher._group_by_category(sample_experiences)

    # Should have 3 categories
    assert len(groups) == 3
    assert "search_strategy" in groups
    assert "evidence_quality" in groups
    assert "reasoning_pattern" in groups

    # Check group sizes
    assert len(groups["search_strategy"]) == 2  # exp_001, exp_003
    assert len(groups["evidence_quality"]) == 1  # exp_002
    assert len(groups["reasoning_pattern"]) == 1  # exp_004

    # Check that experiences are in correct groups
    assert groups["search_strategy"][0].id in ["exp_001", "exp_003"]
    assert groups["evidence_quality"][0].id == "exp_002"
    assert groups["reasoning_pattern"][0].id == "exp_004"


def test_estimate_tokens(mock_store: Any) -> None:
    """Test _estimate_tokens internal method 📏."""
    enricher = PromptEnricher(store=mock_store)

    # Test simple cases
    assert enricher._estimate_tokens("") == 0
    assert enricher._estimate_tokens("a" * 4) == 1
    assert enricher._estimate_tokens("a" * 8) == 2
    assert enricher._estimate_tokens("a" * 100) == 25
