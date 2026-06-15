"""Agent trace logger — captures full step-by-step execution to state/trace.jsonl.

Every LLM call, tool dispatch, memory operation, and decision is logged with
full inputs and outputs for debugging and replay.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path("logs")
TRACE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_FILE = TRACE_DIR / "trace.jsonl"


def _write(event: dict):
    event["timestamp"] = datetime.now().isoformat()
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def trace_llm_call(role: str, model: str, messages: list[dict], tools: list | None,
                   response_text: str | None, tool_calls: list | None,
                   is_error: bool, latency_ms: float, tokens_in: int, tokens_out: int):
    _write({
        "event": "llm_call",
        "role": role,
        "model": model,
        "input_messages": [{"role": m.get("role"), "content": m.get("content", "")} for m in messages],
        "tools_provided": [t["name"] for t in tools] if tools else None,
        "output_text": response_text if response_text else None,
        "output_tool_calls": tool_calls,
        "is_error": is_error,
        "latency_ms": round(latency_ms),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    })


def trace_tool_call(tool_name: str, arguments: dict, result_text: str,
                    artifact_id: str | None, error: bool):
    _write({
        "event": "tool_call",
        "tool": tool_name,
        "arguments": arguments,
        "result_preview": result_text[:500],
        "artifact_id": artifact_id,
        "error": error,
    })


def trace_memory_op(operation: str, kind: str | None = None, descriptor: str | None = None,
                    keywords: list | None = None, hit_count: int | None = None):
    _write({
        "event": "memory",
        "operation": operation,
        "kind": kind,
        "descriptor": descriptor,
        "keywords": keywords,
        "hit_count": hit_count,
    })


def trace_perception(goals: list[dict], all_done: bool, iteration: int):
    _write({
        "event": "perception",
        "iteration": iteration,
        "goals": goals,
        "all_done": all_done,
    })


def trace_decision(goal_text: str, action_type: str, detail: str):
    _write({
        "event": "decision",
        "goal": goal_text,
        "action_type": action_type,
        "detail": detail[:500],
    })


def trace_run_start(run_id: str, query: str):
    _write({
        "event": "run_start",
        "run_id": run_id,
        "query": query,
    })


def trace_run_end(run_id: str, final_answer: str, iterations: int):
    _write({
        "event": "run_end",
        "run_id": run_id,
        "final_answer": final_answer[:1000],
        "iterations": iterations,
    })
