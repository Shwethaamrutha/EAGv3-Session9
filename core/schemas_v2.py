"""Typed contracts for the DAG orchestrator (Session 8 v2).

Every node returns an AgentResult. Every planner node is validated through NodeSpec.
Downstream nodes receive structured data, not raw strings.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """One node the planner emits. Validated at graph-extension time."""
    skill: str
    inputs: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class AgentResult(BaseModel):
    """What every skill returns. The boundary between flow.py and a skill."""
    success: bool
    agent_name: str
    output: dict = Field(default_factory=dict)
    successors: list[NodeSpec] = Field(default_factory=list)
    elapsed_s: float = 0.0
    provider: str = ""
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def text(self) -> str:
        return self.output.get("text", "") or self.output.get("findings", "") or ""


class NodeState(BaseModel):
    """Per-node persistent record with prompt for replay."""
    node_id: str
    skill: str
    status: Literal["pending", "running", "complete", "failed", "skipped"]
    inputs: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    result: AgentResult | None = None
    prompt_sent: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


class RunBudget(BaseModel):
    """Token budget for a single run. Circuit breaker."""
    max_input_tokens: int = 100_000
    max_output_tokens: int = 20_000
    max_wall_clock_s: float = 300.0
    used_input: int = 0
    used_output: int = 0

    @property
    def exhausted(self) -> bool:
        return (self.used_input >= self.max_input_tokens or
                self.used_output >= self.max_output_tokens)

    def record(self, tokens_in: int, tokens_out: int):
        self.used_input += tokens_in
        self.used_output += tokens_out
