"""Logging for agent6 — matches session trace format from the course."""
from __future__ import annotations

import logging
import structlog
from config import settings

# Suppress all third-party noise
logging.getLogger("mcp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("crawl4ai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


def _agent_renderer(logger, name, event_dict):
    """Renderer matching the course's trace format."""
    event = event_dict.pop("event", "")
    level = event_dict.pop("level", "info")

    if level == "debug" and settings.log_format == "console":
        return ""

    if event == "run_start":
        query = event_dict.get("query", "")
        run_id = event_dict.get("run_id", "")
        return f"\n{'='*60}\n  AGENT6 | run_id={run_id}\n  QUERY: {query}\n{'='*60}\n"

    elif event == "run_complete":
        return ""

    elif event == "iteration_start":
        it = event_dict.get("iter", "?")
        return f"\n{'─'*3} iter {it} {'─'*3}"

    elif event == "mcp_connected":
        return f"[mcp]           {event_dict.get('tool_count', 0)} tools loaded"

    elif event == "memory_read":
        return f"[memory.read]   {event_dict.get('count', 0)} hits"

    elif event == "goal":
        status = event_dict.get("status", "")
        text = event_dict.get("text", "")
        attach = event_dict.get("attach", "")
        is_first = event_dict.get("first", False)
        prefix = "[perception]    " if is_first else "                "
        marker = "[done]" if status == "done" else "[open]"
        line = f"{prefix}{marker} {text}"
        if attach:
            line += f"\n                  attach={attach}"
        return line

    elif event == "decision_tool_call":
        tool = event_dict.get("tool", "")
        args = event_dict.get("args", "")
        return f"[decision]      TOOL_CALL: {tool}({args})"

    elif event == "decision_answer":
        text = event_dict.get("text", "")[:200]
        return f"[decision]      ANSWER: {text}"

    elif event == "tool_result":
        tool = event_dict.get("tool", "")
        size = event_dict.get("size", 0)
        result = event_dict.get("result", "")[:80]
        return f"[action]        -> {result}"

    elif event == "artifact_stored":
        art_id = event_dict.get("art_id", "")
        size = event_dict.get("size", 0)
        preview = event_dict.get("preview", "")[:60]
        return f"[action]        -> [artifact {art_id}, {size} bytes] preview: {preview}..."

    elif event == "single_attach":
        art_id = event_dict.get("art_id", "")
        size = event_dict.get("size", 0)
        return f"[attach]        {art_id} ({size} bytes)"

    elif event == "multi_attach":
        count = event_dict.get("count", 1)
        return f"[attach]        {count} artifacts for synthesis"

    elif event == "memory_stored":
        kind = event_dict.get("kind", "")
        desc = event_dict.get("descriptor", "")
        return f"[memory.remember] stored [{kind}] {desc}"

    elif event == "all_goals_done":
        count = event_dict.get("goal_count", 0)
        return f"\n[done] all {count} goals satisfied"

    elif event == "generating_answer_from_memory":
        return ""

    elif event == "decision_error_skipped":
        return f"[decision]      (transient error, retrying...)"

    elif event == "rate_limit_wait":
        wait = event_dict.get("wait_sec", 0)
        return f"[gateway]       rate limit pause ({wait}s)"

    elif event == "final_answer":
        text = event_dict.get("text", "")
        return f"\nFINAL: {text}"

    elif event.startswith("memory_dedup") or event.startswith("memory_evict"):
        return ""

    elif "error" in event or level == "error":
        error = event_dict.get("error", str(event_dict))
        return f"[error]         {event}: {error}"

    elif level == "warning":
        msg = event_dict.get("msg", event)
        return f"[warn]          {msg}"

    return ""


class AgentRenderer:
    def __call__(self, logger, name, event_dict):
        result = _agent_renderer(logger, name, event_dict)
        if result:
            return result
        raise structlog.DropEvent


def setup_logging():
    if settings.log_format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            AgentRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.get_logger(name)


setup_logging()
