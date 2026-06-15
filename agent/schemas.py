from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

# Shared constant — single source of truth for synthesis detection across all modules
SYNTHESIS_KEYWORDS: frozenset[str] = frozenset({
    "synthesize", "synthesise", "extract", "list", "compare",
    "decide", "choose", "summarize", "common", "agree", "advice",
    "tell me", "which one", "appropriate", "recommend", "most",
})


class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    value: dict
    artifact_id: str | None = None
    embedding: list[float] | None = None
    source: str
    run_id: str
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.now)


class Artifact(BaseModel):
    id: str
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


class Goal(BaseModel):
    id: str
    text: str
    done: bool = False
    attach_artifact_id: str | None = None


class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        return all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    name: str
    arguments: dict


class DecisionOutput(BaseModel):
    answer: str | None = None
    tool_call: ToolCall | None = None
    is_error: bool = False

    @property
    def is_answer(self) -> bool:
        return self.answer is not None and not self.is_error
