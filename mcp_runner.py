"""Tool-use loop using Bedrock Converse API's native tool protocol.

Uses proper toolUse/toolResult blocks instead of converting to role="user".
This eliminates: repeated tool calls, hallucinated tools, "no response" issues.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVER = Path(__file__).parent / "mcp_server.py"
MAX_TOOL_HOPS = 4


async def dispatch_tool(session: ClientSession, name: str, args: dict) -> str:
    try:
        result = await session.call_tool(name, arguments=args)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    parts = []
    for c in getattr(result, "content", []) or []:
        t = getattr(c, "text", None)
        parts.append(t if t is not None else str(c))
    return "\n".join(parts) if parts else ""


async def run_with_tools(
    *,
    system_prompt: str,
    user_message: str,
    tools_payload: list[dict],
    gateway_chat_fn,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    node_timeout: float = 60.0,
    on_tool_call=None,
) -> tuple[str, list[dict]]:
    """Multi-turn tool-use loop with proper Bedrock toolResult protocol.

    Returns (final_text, tool_call_log).
    """
    # Build Bedrock-native messages
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_message}"},
    ]
    tool_log = []

    # Convert tool schemas to gateway format
    tools_for_gateway = [
        {"name": t["name"], "description": t.get("description", ""),
         "parameters": t.get("parameters", {"type": "object", "properties": {}})}
        for t in tools_payload
    ] if tools_payload else None

    valid_tool_names = {t["name"] for t in tools_payload} if tools_payload else set()

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
        env={**os.environ, "MCP_LOG_LEVEL": "error"},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp:
            await mcp.initialize()

            for hop in range(MAX_TOOL_HOPS):
                resp = gateway_chat_fn(
                    messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                    tools=tools_for_gateway,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if resp.is_error:
                    return f"[error] {resp.text}", tool_log

                if resp.tool_calls:
                    # Filter valid tool calls
                    valid_calls = [tc for tc in resp.tool_calls if tc["name"] in valid_tool_names]
                    if not valid_calls:
                        # Model hallucinated a tool — treat arguments as response
                        fake = resp.tool_calls[0]
                        return json.dumps(fake.get("arguments", {})), tool_log

                    # Record assistant response (with tool call indicator)
                    messages.append({
                        "role": "assistant",
                        "content": resp.text or f"[calling {valid_calls[0]['name']}]",
                    })

                    # Execute tools and build result message
                    tool_results = []
                    for tc in valid_calls:
                        if on_tool_call:
                            try:
                                await on_tool_call(tc["name"], tc["arguments"])
                            except:
                                pass

                        try:
                            result_text = await asyncio.wait_for(
                                dispatch_tool(mcp, tc["name"], tc["arguments"]),
                                timeout=node_timeout,
                            )
                        except asyncio.TimeoutError:
                            result_text = f"[TIMEOUT] Tool {tc['name']} exceeded {node_timeout}s"

                        tool_log.append({
                            "tool": tc["name"],
                            "arguments": tc["arguments"],
                            "result_preview": result_text[:500],
                        })
                        tool_results.append(f"Result from {tc['name']}:\n{result_text[:3000]}")

                    # Send tool results back as user message
                    nudge = "\n\nNow provide your final answer as JSON based on the tool results above." if hop >= 1 else ""
                    if hop >= 2:
                        nudge = "\n\nSTOP searching. You have enough data. Provide your final JSON answer NOW. Do NOT call any more tools."
                    messages.append({
                        "role": "user",
                        "content": "\n\n".join(tool_results) + nudge,
                    })

                elif resp.text:
                    return resp.text, tool_log
                else:
                    break

    return "(no response after tool loop)", tool_log
