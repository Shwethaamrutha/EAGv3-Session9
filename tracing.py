"""Observability — structured span logging per node execution.

Emits JSON-lines to state/sessions/<sid>/traces.jsonl for each node.
Queryable for: token cost per skill, P95 latency per skill, provider per call.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    session_id: str
    node_id: str
    skill: str
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    provider: str = ""
    status: str = "pending"
    error: str | None = None
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "skill": self.skill,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_s": round(self.elapsed_s, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "provider": self.provider,
            "status": self.status,
            "error": self.error,
            "tool_calls": self.tool_calls,
        }


class TraceLog:
    def __init__(self, session_id: str, base_dir: str = "state/sessions"):
        self.session_id = session_id
        self.path = Path(base_dir) / session_id / "traces.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._spans: list[Span] = []

    def start_span(self, node_id: str, skill: str) -> Span:
        span = Span(
            session_id=self.session_id,
            node_id=node_id,
            skill=skill,
            start_time=time.time(),
        )
        self._spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "complete", error: str | None = None):
        span.end_time = time.time()
        span.status = status
        span.error = error
        self._flush(span)

    def _flush(self, span: Span):
        with open(self.path, "a") as f:
            f.write(json.dumps(span.to_dict(), default=str) + "\n")

    def summary(self) -> dict:
        by_skill: dict[str, dict] = {}
        for span in self._spans:
            if span.skill not in by_skill:
                by_skill[span.skill] = {"calls": 0, "total_s": 0.0, "tokens_in": 0, "tokens_out": 0}
            entry = by_skill[span.skill]
            entry["calls"] += 1
            entry["total_s"] += span.elapsed_s
            entry["tokens_in"] += span.tokens_in
            entry["tokens_out"] += span.tokens_out
        return {
            "session_id": self.session_id,
            "total_spans": len(self._spans),
            "by_skill": by_skill,
            "total_tokens_in": sum(s.tokens_in for s in self._spans),
            "total_tokens_out": sum(s.tokens_out for s in self._spans),
        }
