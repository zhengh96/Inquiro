"""Tests for additional skill root resolution in Inquiro app 🧪."""

from __future__ import annotations

from pathlib import Path

from inquiro.api.app import _resolve_additional_skill_roots


class TestResolveAdditionalSkillRoots:
    """Validate INQUIRO additional skill roots behavior 🎯."""

    def test_auto_discovers_targetmaster_skills(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Auto-discover targetmaster/skills in monorepo layout ✅."""
        project_root = tmp_path / "project"
        auto_root = project_root / "targetmaster" / "skills"
        auto_root.mkdir(parents=True)
        monkeypatch.delenv("INQUIRO_ADDITIONAL_SKILLS_ROOTS", raising=False)

        roots = _resolve_additional_skill_roots(project_root)
        assert roots == [auto_root.resolve()]

    def test_merges_env_roots_and_auto_root(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Combine env roots (relative/absolute) with auto-discovery ✅."""
        project_root = tmp_path / "project"
        rel_root = project_root / "extra_skills"
        abs_root = tmp_path / "absolute_skills"
        auto_root = project_root / "targetmaster" / "skills"
        rel_root.mkdir(parents=True)
        abs_root.mkdir(parents=True)
        auto_root.mkdir(parents=True)

        monkeypatch.setenv(
            "INQUIRO_ADDITIONAL_SKILLS_ROOTS",
            f"extra_skills,{abs_root}",
        )

        roots = _resolve_additional_skill_roots(project_root)
        assert roots == [
            rel_root.resolve(),
            abs_root.resolve(),
            auto_root.resolve(),
        ]

    def test_skips_missing_roots(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Missing env roots are skipped and do not break resolution ✅."""
        project_root = tmp_path / "project"
        monkeypatch.setenv(
            "INQUIRO_ADDITIONAL_SKILLS_ROOTS",
            "missing_skills",
        )

        roots = _resolve_additional_skill_roots(project_root)
        assert roots == []
