"""Inquiro SkillService — manages agent skill lifecycle 🎯.

Encapsulates skill registry initialization and external reference
injection, extracted from EvalTaskRunner to enforce the principle
that constructors should have no side effects (no file IO).

Supports multi-root skill loading via ``additional_roots`` for
upper-layer platforms (e.g., TargetMaster) to contribute skills.

Usage::

    service = SkillService(
        additional_roots=[Path("targetmaster/skills")],
    )
    service.setup()          # ← File IO happens here
    registry = service.get_registry()  # CompositeSkillRegistry or None
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 📁 Default skills directory relative to inquiro package root
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


class CompositeSkillRegistry:
    """Merges multiple SkillRegistries into a unified interface 🔀.

    Wraps one or more ``SkillRegistry`` instances loaded from
    different roots (e.g., ``inquiro/skills/`` + ``targetmaster/skills/``).
    All public methods mirror the ``SkillRegistry`` API so it is a
    drop-in replacement.

    Attributes:
        _registries: Internal list of wrapped SkillRegistry instances.
    """

    def __init__(self, registries: list[Any]) -> None:
        """Initialize with one or more SkillRegistry instances 🔧.

        Args:
            registries: List of SkillRegistry objects to merge.
        """
        self._registries = registries

    def get_skill(self, name: str) -> Any | None:
        """Look up a skill by name across all registries 🔍.

        Args:
            name: Skill name to search for.

        Returns:
            First matching BaseSkill, or None if not found.
        """
        for registry in self._registries:
            skill = registry.get_skill(name)
            if skill is not None:
                return skill
        return None

    def get_all_skills(self) -> list[Any]:
        """Return all skills from all registries 📋.

        Returns:
            Combined list of BaseSkill instances.
        """
        seen: set[str] = set()
        result: list[Any] = []
        for registry in self._registries:
            for skill in registry.get_all_skills():
                if skill.meta_info.name not in seen:
                    seen.add(skill.meta_info.name)
                    result.append(skill)
        return result

    def get_meta_info_context(self) -> str:
        """Build combined meta-info context string 📝.

        Returns:
            Merged context string from all registries.
        """
        all_skills = self.get_all_skills()
        if not all_skills:
            return ""
        lines = ["# Available Skills\n"]
        for skill in all_skills:
            lines.append(skill.to_context_string())
        lines.append("")
        return "\n".join(lines)

    def create_subset(self, skill_names: list[str]) -> CompositeSkillRegistry:
        """Create a filtered registry containing only named skills 🔍.

        Args:
            skill_names: List of skill names to retain.

        Returns:
            New CompositeSkillRegistry with only matching skills.
        """
        filtered = [
            s for s in self.get_all_skills()
            if s.meta_info.name in set(skill_names)
        ]
        # Wrap in a _FilteredSkillRegistry that implements the same interface
        return _FilteredSkillRegistry(filtered)

    def search_skills(self, query: str) -> list[Any]:
        """Search skills by keyword across all registries 🔍.

        Args:
            query: Search keyword.

        Returns:
            List of matching BaseSkill instances.
        """
        query_lower = query.lower()
        seen: set[str] = set()
        results: list[Any] = []
        for registry in self._registries:
            for skill in registry.search_skills(query_lower):
                if skill.meta_info.name not in seen:
                    seen.add(skill.meta_info.name)
                    results.append(skill)
        return results


class _FilteredSkillRegistry:
    """Lightweight registry wrapping a pre-filtered skill list 🔍.

    Used by ``CompositeSkillRegistry.create_subset()`` to provide a
    reduced view of skills for per-agent filtering.
    """

    def __init__(self, skills: list[Any]) -> None:
        self._skills = {s.meta_info.name: s for s in skills}

    def get_skill(self, name: str) -> Any | None:
        return self._skills.get(name)

    def get_all_skills(self) -> list[Any]:
        return list(self._skills.values())

    def get_meta_info_context(self) -> str:
        if not self._skills:
            return ""
        lines = ["# Available Skills\n"]
        for skill in self._skills.values():
            lines.append(skill.to_context_string())
        lines.append("")
        return "\n".join(lines)

    def search_skills(self, query: str) -> list[Any]:
        q = query.lower()
        return [
            s for s in self._skills.values()
            if q in s.meta_info.name.lower()
            or q in s.meta_info.description.lower()
        ]

    def create_subset(self, skill_names: list[str]) -> _FilteredSkillRegistry:
        names = set(skill_names)
        filtered = [s for s in self._skills.values() if s.meta_info.name in names]
        return _FilteredSkillRegistry(filtered)


class SkillService:
    """Manages skill discovery, reference injection, and registry 🎯.

    Separated from EvalTaskRunner so that file IO only occurs
    during an explicit ``setup()`` call (not in a constructor).

    Supports loading skills from multiple roots via
    ``additional_roots``. When multiple roots are provided,
    returns a ``CompositeSkillRegistry`` that merges them.

    Attributes:
        skills_root: Primary skills directory path.
        additional_roots: Extra skills directories to merge.
    """

    def __init__(
        self,
        skills_root: Path | None = None,
        additional_roots: list[Path] | None = None,
    ) -> None:
        """Initialize SkillService (no file IO) 🔧.

        Args:
            skills_root: Primary skills directory path.
                Defaults to ``inquiro/skills/``.
            additional_roots: Additional skill directories to merge.
                Skills from these directories are combined with the
                primary root via CompositeSkillRegistry.
        """
        self.skills_root = skills_root or _DEFAULT_SKILLS_DIR
        self.additional_roots = additional_roots or []
        self._registry: Any = None
        self._initialized = False

    def setup(self) -> None:
        """Perform file IO to load skill registry 📂.

        1. Inject external skill references from ``INQUIRO_EXTERNAL_SKILL_REFS``.
        2. Scan ``skills_root`` and all ``additional_roots``.
        3. If multiple roots found, wrap in CompositeSkillRegistry.

        Safe to call multiple times (idempotent on the first call).
        """
        if self._initialized:
            return

        self._initialized = True

        # 🔗 Inject external references before loading registry
        if self.skills_root.exists():
            self._inject_external_skill_refs()

        try:
            from evomaster.skills import SkillRegistry

            registries: list[Any] = []

            # 📁 Load primary skills root
            if self.skills_root.exists():
                primary = SkillRegistry(self.skills_root)
                count = len(primary.get_all_skills())
                if count > 0:
                    logger.info(
                        "🎯 Loaded %d skill(s) from %s",
                        count,
                        self.skills_root,
                    )
                registries.append(primary)

            # 📁 Load additional roots
            for root in self.additional_roots:
                if not root.exists():
                    logger.debug(
                        "📁 Additional skills root not found: %s",
                        root,
                    )
                    continue
                extra = SkillRegistry(root)
                count = len(extra.get_all_skills())
                if count > 0:
                    logger.info(
                        "🎯 Loaded %d skill(s) from %s",
                        count,
                        root,
                    )
                registries.append(extra)

            # 🔀 Build composite or single registry
            if len(registries) > 1:
                self._registry = CompositeSkillRegistry(registries)
                total = len(self._registry.get_all_skills())
                logger.info(
                    "🔀 CompositeSkillRegistry: %d total skill(s) from %d root(s)",
                    total,
                    len(registries),
                )
            elif len(registries) == 1:
                self._registry = registries[0]
            else:
                logger.debug("📁 No skills directories found")

        except Exception as e:
            logger.warning("⚠️ Failed to initialize SkillRegistry: %s", e)

    def get_registry(self) -> Any:
        """Return the loaded SkillRegistry (or None) 📋.

        Returns:
            SkillRegistry or CompositeSkillRegistry instance, or None
            if setup was not called, no skills found, or loading failed.
        """
        return self._registry

    # -- Internal helpers ---------------------------------------------------

    def _inject_external_skill_refs(self) -> None:
        """Inject external reference files into skill directories 🔗.

        Reads ``INQUIRO_EXTERNAL_SKILL_REFS`` env var and copies files
        from external directories into the corresponding skill's
        ``references/`` folder. This enables upper-layer platforms
        (e.g., TargetMaster) to provide domain-specific content to
        domain-agnostic skills.

        Format: ``skill_name:source_dir`` (comma-separated).
        Example: ``query-templates:/path/to/targetmaster/configs/query_templates``
        """
        env_val = os.environ.get("INQUIRO_EXTERNAL_SKILL_REFS", "")
        if not env_val.strip():
            return

        for mapping in env_val.split(","):
            mapping = mapping.strip()
            if ":" not in mapping:
                continue

            skill_name, source_dir_str = mapping.split(":", 1)
            skill_name = skill_name.strip()
            source_dir = Path(source_dir_str.strip())

            if not source_dir.is_dir():
                logger.warning(
                    "⚠️ External skill refs source not found: %s",
                    source_dir,
                )
                continue

            # 🔍 Find skill directory (flat structure since v0.0.2)
            target_refs = None
            candidate = self.skills_root / skill_name / "references"
            if candidate.parent.exists():
                target_refs = candidate

            if target_refs is None:
                logger.warning(
                    "⚠️ Skill directory not found for: %s",
                    skill_name,
                )
                continue

            target_refs.mkdir(parents=True, exist_ok=True)

            # 📋 Copy all .md files from source to target
            copied = 0
            for src_file in source_dir.glob("*.md"):
                dst_file = target_refs / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied += 1

            if copied > 0:
                logger.info(
                    "🔗 Injected %d reference(s) into skill '%s' from %s",
                    copied,
                    skill_name,
                    source_dir,
                )
