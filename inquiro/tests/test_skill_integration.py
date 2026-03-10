"""Tests for SkillRegistry integration 🧪.

Verifies that the Inquiro pipeline correctly loads, registers,
and injects EvoMaster Skills into Agent context.

Updated for EvoMaster v0.0.2: unified Skill class, flat directory
structure, no more KnowledgeSkill/OperatorSkill distinction.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evomaster.skills import Skill, SkillRegistry


_SKILLS_ROOT = Path(__file__).parent.parent / "skills"


class TestSkillRegistryLoading:
    """Tests that SkillRegistry loads Inquiro skills correctly 🎯."""

    @pytest.fixture
    def registry(self) -> SkillRegistry:
        """Create SkillRegistry from inquiro/skills/ directory."""
        assert _SKILLS_ROOT.exists(), f"Skills dir not found: {_SKILLS_ROOT}"
        return SkillRegistry(_SKILLS_ROOT)

    def test_loads_all_skills(self, registry: SkillRegistry) -> None:
        """Should load all skills from flat directory structure ✅."""
        all_skills = registry.get_all_skills()
        names = {s.meta_info.name for s in all_skills}
        # ✅ Skills at root level (flat structure)
        assert "discovery-convergence-rules" in names
        assert "alias-expansion" in names
        assert "query-templates" in names
        assert "evidence-grader" in names
        assert "evidence-condenser" in names
        assert "evidence-source-classification" in names
        assert "search-reflection" in names
        # ❌ Deleted skills should NOT be present
        assert "search-methodology" not in names
        assert "evidence-rules" not in names
        assert "output-format-guide" not in names
        assert "evidence-validator" not in names

    def test_get_skill_by_name(self, registry: SkillRegistry) -> None:
        """Should retrieve skill by name ✅."""
        skill = registry.get_skill("discovery-convergence-rules")
        assert skill is not None
        assert skill.meta_info.name == "discovery-convergence-rules"

    def test_meta_info_context_non_empty(self, registry: SkillRegistry) -> None:
        """Meta info context should contain skill descriptions ✅."""
        context = registry.get_meta_info_context()
        assert "discovery-convergence-rules" in context
        assert "evidence-condenser" in context
        # ❌ Deleted skills should NOT appear as registered skills
        assert "[Skill: search-methodology]" not in context
        assert "[Skill: evidence-rules]" not in context
        assert "[Skill: evidence-validator]" not in context

    def test_skill_has_scripts(self, registry: SkillRegistry) -> None:
        """Skills with scripts/ dir should have available_scripts ✅."""
        skill = registry.get_skill("evidence-condenser")
        assert skill is not None
        assert isinstance(skill, Skill)
        script_names = [s.name for s in skill.available_scripts]
        assert "condense.py" in script_names

    def test_skill_without_scripts(self, registry: SkillRegistry) -> None:
        """Skills without scripts/ dir should have empty scripts list ✅."""
        skill = registry.get_skill("discovery-convergence-rules")
        assert skill is not None
        assert isinstance(skill, Skill)
        assert skill.available_scripts == []

    def test_create_subset(self, registry: SkillRegistry) -> None:
        """create_subset should return filtered registry ✅."""
        subset = registry.create_subset(["discovery-convergence-rules", "evidence-grader"])
        all_names = {s.meta_info.name for s in subset.get_all_skills()}
        assert all_names == {"discovery-convergence-rules", "evidence-grader"}
        # 🔍 Should not contain non-requested skills
        assert subset.get_skill("evidence-validator") is None

    def test_skill_meta_info_no_skill_type(self, registry: SkillRegistry) -> None:
        """SkillMetaInfo should NOT have skill_type field (v0.0.2) ✅."""
        skill = registry.get_skill("discovery-convergence-rules")
        assert skill is not None
        assert not hasattr(skill.meta_info, "skill_type")


class TestEvalTaskRunnerSkillInit:
    """Tests that EvalTaskRunner initializes SkillRegistry 🎯."""

    def test_runner_has_skill_registry(self) -> None:
        """Runner should have skill_registry property ✅."""
        from inquiro.core.runner import EvalTaskRunner

        mock_mcp = MagicMock()
        mock_llm = MagicMock()
        runner = EvalTaskRunner(mock_mcp, mock_llm)
        # 🎯 skill_registry property should exist (None without service)
        assert hasattr(runner, "skill_registry")
        assert runner.skill_registry is None

    def test_runner_with_skill_service(self) -> None:
        """Runner with SkillService returns registry after setup ✅."""
        from inquiro.core.runner import EvalTaskRunner
        from inquiro.infrastructure.skill_service import SkillService

        mock_mcp = MagicMock()
        mock_llm = MagicMock()

        service = SkillService()
        service.setup()

        runner = EvalTaskRunner(
            mock_mcp,
            mock_llm,
            skill_service=service,
        )
        # 🎯 skill_registry should delegate to SkillService
        registry = runner.skill_registry
        if _SKILLS_ROOT.exists():
            assert registry is not None
        else:
            assert registry is None


class TestSkillService:
    """Tests for SkillService lifecycle 🎯."""

    def test_constructor_no_file_io(self) -> None:
        """Constructor should NOT do file IO ✅."""
        from inquiro.infrastructure.skill_service import SkillService

        # 🔍 Use a non-existent path — should NOT raise
        service = SkillService(skills_root=Path("/nonexistent/path"))
        assert service.get_registry() is None
        assert not service._initialized

    def test_setup_loads_registry(self) -> None:
        """setup() should load SkillRegistry from skills dir ✅."""
        from inquiro.infrastructure.skill_service import SkillService

        service = SkillService(skills_root=_SKILLS_ROOT)
        service.setup()
        registry = service.get_registry()
        if _SKILLS_ROOT.exists():
            assert registry is not None
        else:
            assert registry is None

    def test_setup_idempotent(self) -> None:
        """Calling setup() twice is safe ✅."""
        from inquiro.infrastructure.skill_service import SkillService

        service = SkillService(skills_root=_SKILLS_ROOT)
        service.setup()
        first = service.get_registry()
        service.setup()
        second = service.get_registry()
        assert first is second

    def test_missing_dir_returns_none(self) -> None:
        """Non-existent skills dir returns None registry ✅."""
        from inquiro.infrastructure.skill_service import SkillService

        service = SkillService(skills_root=Path("/tmp/nonexistent_skills_dir"))
        service.setup()
        assert service.get_registry() is None


class TestCompositeSkillRegistry:
    """Tests for CompositeSkillRegistry methods 🎯."""

    def test_create_subset(self) -> None:
        """CompositeSkillRegistry.create_subset returns filtered view ✅."""
        from inquiro.infrastructure.skill_service import (
            CompositeSkillRegistry,
            SkillService,
        )

        service = SkillService(skills_root=_SKILLS_ROOT)
        service.setup()
        registry = service.get_registry()
        if registry is None:
            pytest.skip("No skills directory found")

        # 🔍 If single registry, wrap in Composite for testing
        if not isinstance(registry, CompositeSkillRegistry):
            composite = CompositeSkillRegistry([registry])
        else:
            composite = registry

        subset = composite.create_subset(["evidence-condenser"])
        assert len(subset.get_all_skills()) == 1
        assert subset.get_skill("evidence-condenser") is not None
        assert subset.get_skill("discovery-convergence-rules") is None


class TestSearchExpSkillInjection:
    """Tests that SearchExp injects skill meta_info into prompt 🎯."""

    def test_system_prompt_includes_skills(self) -> None:
        """System prompt should include skill metadata when registry exists ✅."""
        from inquiro.exps.search_exp import SearchExp
        from inquiro.core.types import EvaluationTask

        # 🏗️ Create minimal task (uses defaults for most fields)
        task = EvaluationTask(
            task_id="test-skill-inject",
            topic="Test topic",
            rules="Test rules",
            output_schema={"type": "object", "properties": {}},
        )

        # 🔧 Create registry
        registry = SkillRegistry(_SKILLS_ROOT) if _SKILLS_ROOT.exists() else None

        exp = SearchExp(
            llm=MagicMock(),
            tools=MagicMock(),
            skill_registry=registry,
        )

        prompt = exp._render_system_prompt(task)

        if registry:
            assert "Available Skills" in prompt or "discovery-convergence-rules" in prompt
            assert "get_reference" in prompt
