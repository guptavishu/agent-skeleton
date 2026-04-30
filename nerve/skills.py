from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import Tool


@dataclass
class Skill:
    name: str
    prompt: str
    description: str = ""
    tools: list[Tool] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> Skill:
        """Load a skill from a Python file.

        The file must define a module-level `skill` variable of type Skill.
        """
        p = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(f"skill_{p.stem}", p)
        if not spec or not spec.loader:
            raise ImportError(f"Cannot load skill from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        skill = getattr(mod, "skill", None)
        if not isinstance(skill, Skill):
            raise ValueError(f"{path} must define a module-level `skill` variable of type Skill")
        return skill


SKILL_DIRS = [
    Path.home() / ".nerve" / "skills",
    Path.cwd() / "skills",
]


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def discover(self, dirs: list[Path] | None = None) -> list[Skill]:
        found = []
        for d in (dirs or SKILL_DIRS):
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                try:
                    skill = Skill.load(str(f))
                    self.register(skill)
                    found.append(skill)
                except Exception:
                    pass  # skip broken skill files
        return found

    def get_tools(self, skill_names: list[str] | None = None) -> list[Tool]:
        tools = []
        targets = skill_names or list(self._skills.keys())
        for name in targets:
            skill = self._skills.get(name)
            if skill:
                tools.extend(skill.tools)
        return tools

    def get_prompts(self, skill_names: list[str] | None = None) -> str:
        parts = []
        targets = skill_names or list(self._skills.keys())
        for name in targets:
            skill = self._skills.get(name)
            if skill and skill.prompt:
                parts.append(f"## Skill: {skill.name}\n{skill.prompt}")
        return "\n\n".join(parts)
