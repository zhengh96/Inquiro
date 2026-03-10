"""Tests for Inquiro ExperienceExtractor 🧪.

Tests the LLM-powered experience extraction engine:
- Template rendering with Jinja2
- LLM call and JSON parsing
- Category validation
- Auto-field assignment (namespace, source, source_evaluation_id)
- Error handling (LLM failure, JSON parse error, invalid data)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.types import (
    ResultMetrics,
    ToolCallRecord,
    TrajectorySnapshot,
)


# ============================================================
# 🏗️ Fixtures
# ============================================================


@pytest.fixture
def sample_snapshot() -> TrajectorySnapshot:
    """Sample trajectory snapshot for testing 📸."""
    return TrajectorySnapshot(
        evaluation_id="eval_12345",
        task_id="task_67890",
        topic="GPR75 obesity",
        context_tags=["modality:SmallMolecule", "indication:Obesity"],
        sub_item_id="evidence_1a",
        tool_calls=[
            ToolCallRecord(
                tool_name="pubmed_search",
                arguments_summary='query="GPR75 obesity"',
                result_size=5000,
                success=True,
                round_number=1,
            ),
            ToolCallRecord(
                tool_name="clinicaltrials_search",
                arguments_summary='condition="obesity"',
                result_size=3000,
                success=True,
                round_number=2,
            ),
        ],
        metrics=ResultMetrics(
            evidence_count=15,
            confidence=0.85,
            search_rounds=2,
            cost_usd=0.05,
            decision="approve",
            checklist_coverage=0.9,
        ),
        started_at=datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 2, 19, 10, 5, 0, tzinfo=timezone.utc),
        wall_time_seconds=300.0,
    )


@pytest.fixture
def sample_profile_config() -> dict[str, Any]:
    """Sample evolution profile configuration 📋."""
    return {
        "extraction_prompt_template": """
Extract up to {{ max_experiences }} experiences from this trajectory:

Evaluation ID: {{ snapshot.evaluation_id }}
Topic: {{ snapshot.topic }}
Evidence Count: {{ snapshot.metrics.evidence_count }}
Confidence: {{ snapshot.metrics.confidence }}

Valid categories: {{ valid_categories }}

Return JSON array of experiences with fields: category, insight, context_tags, applicable_sub_items.
""".strip(),
        "experience_categories": [
            "search_strategy",
            "evidence_quality",
            "reasoning_pattern",
        ],
        "max_experiences_per_extraction": 3,
        "namespace": "targetmaster",
    }


@pytest.fixture
def mock_llm_success() -> Any:
    """Mock LLM function that returns valid JSON 🤖."""

    async def llm_fn(prompt: str) -> str:
        return """
[
  {
    "category": "search_strategy",
    "insight": "PubMed search with specific gene names yields high-quality evidence",
    "context_tags": ["modality:SmallMolecule"],
    "applicable_sub_items": ["evidence_1a", "evidence_1b"]
  },
  {
    "category": "evidence_quality",
    "insight": "Clinical trial data provides strong validation for obesity targets",
    "context_tags": ["indication:Obesity"],
    "applicable_sub_items": ["*"]
  }
]
""".strip()

    return llm_fn


@pytest.fixture
def mock_llm_invalid_json() -> Any:
    """Mock LLM function that returns invalid JSON 🚫."""

    async def llm_fn(prompt: str) -> str:
        return "This is not valid JSON"

    return llm_fn


@pytest.fixture
def mock_llm_failure() -> Any:
    """Mock LLM function that raises an exception 💥."""

    async def llm_fn(prompt: str) -> str:
        raise RuntimeError("LLM service unavailable")

    return llm_fn


# ============================================================
# ✅ Success Path Tests
# ============================================================


@pytest.mark.asyncio
async def test_extract_success(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
    mock_llm_success: Any,
) -> None:
    """Test successful experience extraction ✅."""
    extractor = ExperienceExtractor(llm_fn=mock_llm_success)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Should extract 2 experiences
    assert len(experiences) == 2

    # Check first experience
    exp1 = experiences[0]
    assert exp1.category == "search_strategy"
    assert "PubMed" in exp1.insight
    assert exp1.namespace == "targetmaster"
    assert exp1.source == "trajectory_extraction"
    assert exp1.source_evaluation_id == "eval_12345"
    assert exp1.context_tags == ["modality:SmallMolecule"]
    assert exp1.applicable_sub_items == ["evidence_1a", "evidence_1b"]

    # Check second experience
    exp2 = experiences[1]
    assert exp2.category == "evidence_quality"
    assert "Clinical trial" in exp2.insight
    assert exp2.namespace == "targetmaster"
    assert exp2.source == "trajectory_extraction"
    assert exp2.source_evaluation_id == "eval_12345"
    assert exp2.applicable_sub_items == ["*"]


@pytest.mark.asyncio
async def test_extract_template_rendering(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test that Jinja2 template rendering works correctly 🎨."""
    captured_prompt: str = ""

    async def capture_llm_fn(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "[]"

    extractor = ExperienceExtractor(llm_fn=capture_llm_fn)

    await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Check that template variables were rendered
    assert "eval_12345" in captured_prompt
    assert "GPR75 obesity" in captured_prompt
    assert "15" in captured_prompt  # evidence_count
    assert "0.85" in captured_prompt  # confidence
    assert "search_strategy" in captured_prompt
    assert "evidence_quality" in captured_prompt


# ============================================================
# 🚫 Category Validation Tests
# ============================================================


@pytest.mark.asyncio
async def test_extract_invalid_category(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test that experiences with invalid categories are filtered out ❌."""

    async def llm_with_invalid_category(prompt: str) -> str:
        return """
[
  {
    "category": "invalid_category",
    "insight": "This should be filtered out",
    "context_tags": [],
    "applicable_sub_items": ["*"]
  },
  {
    "category": "search_strategy",
    "insight": "This should be kept",
    "context_tags": [],
    "applicable_sub_items": ["*"]
  }
]
""".strip()

    extractor = ExperienceExtractor(llm_fn=llm_with_invalid_category)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Only the valid experience should be kept
    assert len(experiences) == 1
    assert experiences[0].category == "search_strategy"


# ============================================================
# ⚠️ Error Handling Tests
# ============================================================


@pytest.mark.asyncio
async def test_extract_llm_failure(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
    mock_llm_failure: Any,
) -> None:
    """Test graceful handling of LLM call failure 💥."""
    extractor = ExperienceExtractor(llm_fn=mock_llm_failure)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Should return empty list on LLM failure
    assert experiences == []


@pytest.mark.asyncio
async def test_extract_invalid_json(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
    mock_llm_invalid_json: Any,
) -> None:
    """Test graceful handling of invalid JSON output 🚫."""
    extractor = ExperienceExtractor(llm_fn=mock_llm_invalid_json)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Should return empty list on JSON parse error
    assert experiences == []


@pytest.mark.asyncio
async def test_extract_non_array_json(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test handling of LLM returning non-array JSON 🚫."""

    async def llm_returns_object(prompt: str) -> str:
        return '{"error": "not an array"}'

    extractor = ExperienceExtractor(llm_fn=llm_returns_object)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Should return empty list when JSON is not an array
    assert experiences == []


@pytest.mark.asyncio
async def test_extract_malformed_experience_data(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
) -> None:
    """Test handling of malformed experience data in array 🚫."""

    async def llm_returns_malformed(prompt: str) -> str:
        return """
[
  "this is not a dict",
  {
    "category": "search_strategy",
    "insight": "Valid experience"
  }
]
""".strip()

    extractor = ExperienceExtractor(llm_fn=llm_returns_malformed)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=sample_profile_config,
    )

    # Should skip the string and keep the valid experience
    assert len(experiences) == 1
    assert experiences[0].insight == "Valid experience"


@pytest.mark.asyncio
async def test_extract_template_rendering_error(
    sample_snapshot: TrajectorySnapshot,
    sample_profile_config: dict[str, Any],
    mock_llm_success: Any,
) -> None:
    """Test handling of Jinja2 template rendering errors ⚠️."""
    # Create invalid template
    invalid_config = sample_profile_config.copy()
    invalid_config["extraction_prompt_template"] = "{{ undefined_variable.foo }}"

    extractor = ExperienceExtractor(llm_fn=mock_llm_success)

    experiences = await extractor.extract(
        snapshot=sample_snapshot,
        profile_config=invalid_config,
    )

    # Should return empty list on template rendering error
    assert experiences == []
