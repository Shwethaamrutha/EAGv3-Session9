"""Action — pure MCP dispatch. No LLM calls."""
from __future__ import annotations

from mcp import ClientSession
from mcp.shared.exceptions import McpError

from artifacts import artifact_store
from config import settings
from logger import get_logger
from schemas import ToolCall

log = get_logger("action")


async def execute(session: ClientSession, tool_call: ToolCall) -> tuple[str, str | None]:
    # Guard: reject artifact handles passed as paths or URLs
    for key in ("path", "url", "file_path"):
        val = tool_call.arguments.get(key, "")
        if isinstance(val, str) and val.startswith("art:"):
            log.warning("artifact_handle_rejected", tool=tool_call.name, key=key, val=val)
            return (
                f"[error] artifact handles are not paths. "
                f"'{val}' is an internal reference, not a file path or URL.",
                None,
            )

    try:
        result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
    except (McpError, ConnectionError, TimeoutError, OSError) as e:
        log.error("tool_dispatch_failed", tool=tool_call.name, error=str(e))
        return f"[error] tool dispatch failed: {e}", None

    # Collapse content blocks into text
    text_parts = []
    for block in result.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    full_text = "\n".join(text_parts)

    # Threshold check: large results go to artifact store
    blob = full_text.encode("utf-8")
    if len(blob) > settings.artifact_threshold_bytes:
        art_id = artifact_store.put(
            blob,
            content_type="text/plain",
            source=f"tool:{tool_call.name}",
            descriptor=f"{tool_call.name} result ({len(blob)} bytes)",
        )
        preview = full_text[:200].replace("\n", " ")
        return f"[artifact {art_id}, {len(blob)} bytes] preview: {preview}...", art_id

    # Trace tool call
    try:
        from tracer import trace_tool_call
        trace_tool_call(tool_call.name, tool_call.arguments, full_text, None, False)
    except Exception:
        pass

    return full_text, None
