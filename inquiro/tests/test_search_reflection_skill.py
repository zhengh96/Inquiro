"""Tests for search-reflection Skill and SearchExp skill subset wiring 🧪.

Verifies:
- search-reflection SKILL.md parses correctly (frontmatter, meta_info).
- search-reflection is loaded by SkillRegistry from inquiro/skills/.
- SearchExp includes search-reflection in the skill subset for SearchAgent.
- discovery-convergence-rules SKILL.md contains checklist awareness section.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from evomaster.agent.tools.base import ToolRegistry
from evomaster.skills import SkillRegistry

_SKILLS_ROOT = Path(__file__).parent.parent / "skills"
_REFLECTION_SKILL_PATH = _SKILLS_ROOT / "search-reflection" / "SKILL.md"
_CONVERGENCE_SKILL_PATH = (
    _SKILLS_ROOT / "discovery-convergence-rules" / "SKILL.md"
)


# ============================================================
# 🧪 SKILL.md parsing tests
# ============================================================


class TestSearchReflectionSkillMd:
    """Tests that search-reflection SKILL.md is valid 🎯."""

    def test_skill_md_exists(self) -> None:
        """SKILL.md file should exist at expected path ✅."""
        assert _REFLECTION_SKILL_PATH.exists(), (
            f"SKILL.md not found: {_REFLECTION_SKILL_PATH}"
        )

    def test_frontmatter_parses(self) -> None:
        """SKILL.md frontmatter should parse as valid YAML ✅."""
        content = _REFLECTION_SKILL_PATH.read_text()
        # Extract YAML frontmatter between --- markers
        parts = content.split("---", 2)
        assert len(parts) >= 3, "SKILL.md must have --- delimited frontmatter"
        frontmatter = yaml.safe_load(parts[1])
        assert isinstance(frontmatter, dict)

    def test_frontmatter_has_required_fields(self) -> None:
        """Frontmatter must have name, description, license ✅."""
        content = _REFLECTION_SKILL_PATH.read_text()
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["name"] == "search-reflection"
        assert "description" in frontmatter
        assert len(frontmatter["description"]) > 50
        assert "license" in frontmatter

    def test_description_contains_trigger(self) -> None:
        """Description must contain trigger condition ✅."""
        content = _REFLECTION_SKILL_PATH.read_text()
        parts = content.split("---", 2)
        frontmatter = yaml.safe_load(parts[1])
        desc = frontmatter["description"].lower()
        # Should have usage trigger guidance
        assert "use" in desc or "during" in desc

    def test_body_has_required_sections(self) -> None:
        """SKILL.md body should have Overview, When to Use, Details ✅."""
        content = _REFLECTION_SKILL_PATH.read_text()
        assert "## Overview" in content
        assert "## When to Use" in content
        assert "## Details" in content
        assert "## Constraints" in content

    def test_body_has_reflection_protocol(self) -> None:
        """SKILL.md should document the reflection protocol steps ✅."""
        content = _REFLECTION_SKILL_PATH.read_text()
        assert "Checklist Coverage" in content
        assert "Gap Identification" in content
        assert "Strategy Adjustment" in content
        assert "Confidence" in content

    def test_no_domain_terms(self) -> None:
        """Inquiro skills must not contain domain-specific terms ✅."""
        content = _REFLECTION_SKILL_PATH.read_text().lower()
        # Pharma-specific terms that should NOT appear in Inquiro skills
        for term in ["drug", "pharma", "clinical trial", "fda", "therapeutic"]:
            assert term not in content, (
                f"Domain term '{term}' found in Inquiro skill"
            )


# ============================================================
# 🧪 SkillRegistry loading tests
# ============================================================


class TestSearchReflectionRegistryLoading:
    """Tests that SkillRegistry loads search-reflection correctly 🎯."""

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        """Create SkillRegistry from inquiro/skills/ directory."""
        assert _SKILLS_ROOT.exists()
        return SkillRegistry(_SKILLS_ROOT)

    def test_skill_loaded_by_name(self, registry: SkillRegistry) -> None:
        """search-reflection should be discoverable by name ✅."""
        skill = registry.get_skill("search-reflection")
        assert skill is not None
        assert skill.meta_info.name == "search-reflection"

    def test_skill_in_all_skills(self, registry: SkillRegistry) -> None:
        """search-reflection should appear in get_all_skills() ✅."""
        names = {s.meta_info.name for s in registry.get_all_skills()}
        assert "search-reflection" in names

    def test_skill_has_no_scripts(self, registry: SkillRegistry) -> None:
        """search-reflection is a knowledge skill -- no scripts ✅."""
        skill = registry.get_skill("search-reflection")
        assert skill is not None
        assert skill.available_scripts == []

    def test_meta_info_context_includes_skill(
        self, registry: SkillRegistry
    ) -> None:
        """Meta info context should mention search-reflection ✅."""
        context = registry.get_meta_info_context()
        assert "search-reflection" in context

    def test_create_subset_includes_skill(
        self, registry: SkillRegistry
    ) -> None:
        """create_subset with search-reflection should work ✅."""
        subset = registry.create_subset(["search-reflection"])
        assert len(subset.get_all_skills()) == 1
        assert subset.get_skill("search-reflection") is not None


# ============================================================
# 🧪 SearchExp skill subset wiring tests
# ============================================================


class TestSearchExpSkillSubset:
    """Tests that SearchExp includes search-reflection in subset 🎯."""

    def test_create_subset_called_with_search_reflection(self) -> None:
        """SearchExp._create_search_agent should include search-reflection ✅."""
        from inquiro.core.types import EvaluationTask
        from inquiro.exps.search_exp import SearchExp

        mock_registry = MagicMock()
        exp = SearchExp(
            llm=MagicMock(),
            tools=ToolRegistry(),
            skill_registry=mock_registry,
        )

        task = EvaluationTask(
            task_id="test-reflection-subset",
            topic="Test topic",
            rules="Test rules",
            output_schema={"type": "object", "properties": {}},
        )

        with patch("inquiro.agents.search_agent.SearchAgent") as mock_cls:
            mock_cls.return_value = MagicMock()
            exp._create_search_agent(
                task=task,
                system_prompt="sys",
                user_prompt="usr",
            )

            # Verify create_subset was called with search-reflection
            mock_registry.create_subset.assert_called_once()
            subset_names = mock_registry.create_subset.call_args[0][0]
            assert "search-reflection" in subset_names
            assert "query-templates" in subset_names
            assert "alias-expansion" in subset_names
            assert "evidence-grader" in subset_names


# ============================================================
# 🧪 discovery-convergence-rules enhancement tests
# ============================================================


class TestConvergenceRulesEnhancement:
    """Tests that convergence-rules SKILL.md has checklist awareness 🎯."""

    def test_convergence_skill_md_exists(self) -> None:
        """discovery-convergence-rules SKILL.md should exist ✅."""
        assert _CONVERGENCE_SKILL_PATH.exists()

    def test_has_checklist_aware_section(self) -> None:
        """Should contain Checklist-Aware Convergence section ✅."""
        content = _CONVERGENCE_SKILL_PATH.read_text()
        assert "Checklist-Aware Convergence" in content

    def test_has_per_round_tracking(self) -> None:
        """Should document per-round checklist tracking ✅."""
        content = _CONVERGENCE_SKILL_PATH.read_text()
        assert "Per-Round Checklist Tracking" in content
        assert "coverage_ratio" in content

    def test_has_prioritization_guidance(self) -> None:
        """Should include guidance on prioritizing uncovered items ✅."""
        content = _CONVERGENCE_SKILL_PATH.read_text()
        assert "Prioritizing Uncovered Items" in content
        assert "Required before optional" in content

    def test_has_convergence_decision_guide(self) -> None:
        """Should include convergence vs continuing decision guide ✅."""
        content = _CONVERGENCE_SKILL_PATH.read_text()
        assert "Convergence vs Continuing" in content

    def test_has_multi_round_example(self) -> None:
        """Should include a multi-round checklist progression example ✅."""
        content = _CONVERGENCE_SKILL_PATH.read_text()
        assert "Multi-Round Checklist Progression" in content
