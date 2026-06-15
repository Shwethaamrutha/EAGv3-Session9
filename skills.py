"""Skill catalogue — loads skills from agent_config.yaml and renders prompts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    prompt_file: str
    tools_allowed: list[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 1024
    critic: bool = False
    internal_successors: list[str] = field(default_factory=list)

    _prompt_cache: str | None = field(default=None, repr=False)

    @property
    def prompt_text(self) -> str:
        if self._prompt_cache is None:
            path = Path(__file__).parent / self.prompt_file
            self._prompt_cache = path.read_text()
        return self._prompt_cache

    def render_prompt(self, inputs: dict[str, str], memory_hits: str = "", question: str = "") -> str:
        parts = [self.prompt_text]

        if memory_hits:
            parts.append(f"\n\nMEMORY HITS:\n{memory_hits}")

        if question:
            parts.append(f"\n\nQUESTION: {question}")

        if inputs:
            parts.append("\n\nINPUTS:")
            for key, value in inputs.items():
                parts.append(f"\n--- {key} ---\n{value}")

        return "\n".join(parts)


def load_skills(config_path: str | None = None) -> dict[str, Skill]:
    if config_path is None:
        config_path = str(Path(__file__).parent / "agent_config.yaml")
    with open(config_path) as f:
        data = yaml.safe_load(f)

    skills = {}
    for name, spec in data.get("skills", {}).items():
        skills[name] = Skill(
            name=name,
            prompt_file=spec["prompt"],
            tools_allowed=spec.get("tools_allowed", []),
            temperature=spec.get("temperature", 0.7),
            max_tokens=spec.get("max_tokens", 1024),
            critic=spec.get("critic", False),
            internal_successors=spec.get("internal_successors", []),
        )
    return skills
