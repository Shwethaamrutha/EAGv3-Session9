"""Decision — selects the next action for one bounded goal.

Returns either a final answer (plain text) or a single tool call.
"""
from __future__ import annotations

import json
import re

from config import settings
from llm_gateway import gateway
from logger import get_logger
from schemas import DecisionOutput, Goal, MemoryItem, SYNTHESIS_KEYWORDS, ToolCall

log = get_logger("decision")

from datetime import date as _date

def _decision_system():
    today = _date.today()
    weekday = today.strftime("%A")
    return f"""You are the Decision module of an agentic system. You receive ONE goal and must take exactly ONE action.

Today is {weekday}, {today.isoformat()}.

You have two options:
1. ANSWER: If you have enough information to satisfy the goal, respond clearly.
   - Use markdown formatting (headers, bullet points, bold) for readability.
   - Be comprehensive but focused — cover the key points.
   - NEVER use emojis in your response.
   - For recommendations: pick ONE specific named option with a clear reason.
   - ONLY use information explicitly present in MEMORY HITS, ATTACHED ARTIFACTS, and RECENT HISTORY.
     NEVER add facts, claims, or details from your own training knowledge.
     If a paper or source is not quoted in the provided context, do NOT mention it.
   - If MEMORY HITS is empty or has no relevant content, and no ATTACHED ARTIFACTS exist,
     say "This information is not in the indexed knowledge base." Do NOT answer from your own knowledge.
   - Reference items from MEMORY HITS or RECENT HISTORY — never invent new options.

2. TOOL CALL: If you need external information or must perform an action, call exactly ONE tool.
   - Pick the most appropriate tool from the available tools.
   - For weather data: use fetch_url("https://wttr.in/CITY?format=3") — fast, reliable, text response.
   - For calendar reminders: create .ics files (iCalendar format) that can be imported into calendar apps.
   - NEVER pass artifact handles (strings starting with "art:") as file paths or URLs.
   - Use index_document when content must become FAISS-searchable for later queries.
     After fetching a URL, save it to the sandbox then index it for semantic search.
   - Use search_knowledge when Memory already contains indexed chunks for the topic.
     PREFER search_knowledge over web_search/fetch_url when MEMORY HITS show indexed chunks.
   - Use read_file for one-shot inspection of a file's contents (not indexing).
   - For research tasks: fetch → create_file (save to sandbox) → index_document → repeat.
     Once all sources are indexed, use search_knowledge for synthesis.

CRITICAL PRIORITY:
- Check MEMORY HITS first. If the answer is already there, answer immediately.
- If ATTACHED ARTIFACTS contains content, use it to form your answer.
- Only call a tool if memory hits AND attached artifacts do NOT contain the answer.
- If a previous fetch_url returned an error or very short content (verification page, 403, etc.),
  call web_search with a refined query to find an alternative source. Never retry the same failed URL.

Rules:
- Do exactly one thing: answer OR call one tool. Never both.
- NEVER narrate, explain your reasoning, or show your thought process.
- NEVER say "Wait", "Let me", "Based on the artifacts", "reviewing", or similar meta-commentary.
- NEVER mention chunks, memory hits, artifacts, sources, tool internals, or data limitations.
- Start your answer DIRECTLY with the factual content the user wants. Nothing else.
- Keep answers concise: max 150 words for simple questions, max 300 words for synthesis.
- Be efficient. One tool call should accomplish the goal if possible.
- Always give a concrete answer. Never ask for clarification.
- Keep answers under 5 sentences. Be direct and factual.
"""

DECISION_USER = """GOAL: {goal_text}

MEMORY HITS:
{hits_text}

ATTACHED ARTIFACTS:
{attached_text}

RECENT HISTORY:
{history_text}

{pending_urls_text}AVAILABLE TOOLS:
{tools_text}

Decide: respond with EITHER an answer OR a single tool call.
If there are PENDING URLs listed above, fetch the NEXT one (not one you've already fetched).
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for h in hits:
        lines.append(f"  ({h.kind}) {h.descriptor}")
        if h.kind == "fact" and h.value.get("chunk"):
            # Show only descriptor — Decision must call search_knowledge for full content
            pass
        elif h.value:
            val_str = json.dumps(h.value, default=str)[:300]
            lines.append(f"    value: {val_str}")
        if h.kind == "tool_outcome" and h.value.get("result_preview"):
            preview = h.value["result_preview"]
            lines.append(f"    preview: {preview}")
    return "\n".join(lines)


def _format_attached(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return "(none)"
    total_budget = settings.attachment_budget_bytes
    per_artifact = total_budget // max(len(attached), 1)
    parts = []
    for i, (art_id, blob) in enumerate(attached):
        text = blob.decode("utf-8", errors="replace")[:per_artifact]
        parts.append(f"--- SOURCE {i+1}: {art_id} ({len(blob)} bytes) ---\n{text}")
    return "\n".join(parts)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    fetched_urls = set()
    for event in history:
        if event.get("kind") == "action" and event.get("tool") == "fetch_url":
            url = event.get("arguments", {}).get("url", "")
            fetched_urls.add(url)

    failed_urls = set()
    for event in history[-8:]:
        if event.get("kind") == "action":
            result = event.get("result_descriptor", "")
            tag = ""
            if event.get("tool") == "fetch_url" and len(result) < 200:
                tag = " [FAILED - find alternative]"
                failed_urls.add(event.get("arguments", {}).get("url", ""))
            args_str = json.dumps(event.get("arguments", {}))
            lines.append(f"  TOOL {event['tool']}({args_str}) → {result}{tag}")
        elif event.get("kind") == "answer":
            lines.append(f"  ANSWER: {event.get('text', '')[:100]}")

    if fetched_urls:
        lines.append(f"\n  ALREADY FETCHED URLs (do NOT re-fetch): {list(fetched_urls)}")
    if failed_urls:
        lines.append(f"  FAILED URLs (search for alternatives): {list(failed_urls)}")
    return "\n".join(lines)


def _format_tools(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        params = ""
        if t.get("parameters", {}).get("properties"):
            params = ", ".join(t["parameters"]["properties"].keys())
        lines.append(f"  {t['name']}({params}): {t.get('description', '')[:80]}")
    return "\n".join(lines)


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    # Compute pending URLs: found in search results but not yet fetched
    fetched_urls = {
        e.get("arguments", {}).get("url", "")
        for e in history
        if e.get("kind") == "action" and e.get("tool") == "fetch_url"
    }
    search_urls = []
    for h in hits:
        if h.kind == "tool_outcome" and h.value.get("tool") == "web_search":
            preview = h.value.get("result_preview", "")
            urls = re.findall(r'URL:\s*(https?://[^\s]+)', preview)
            search_urls.extend(urls)
    unfetched = [u for u in search_urls if u not in fetched_urls]

    pending_urls_text = ""
    if unfetched:
        pending_urls_text = f"PENDING URLs TO FETCH (not yet read):\n  " + "\n  ".join(unfetched[:5]) + "\n\n"

    # Determine if this is a FINAL synthesis/recommendation goal (combines multiple sources)
    goal_is_synthesis = any(kw in goal.text.lower() for kw in SYNTHESIS_KEYWORDS)

    # Force answer (no tools) when:
    # 1. Artifacts attached and no pending URLs to fetch
    # 2. This is a FINAL synthesis goal (not intermediate) and sufficient data exists
    has_sufficient_data = sum(1 for h in hits if h.kind == "tool_outcome") >= 2

    if (attached and not unfetched) or (goal_is_synthesis and has_sufficient_data):
        use_tools = None
        tools_text = "(tools disabled — answer using MEMORY HITS and any ATTACHED ARTIFACTS)"
        pending_urls_text = ""
    else:
        use_tools = mcp_tools
        tools_text = _format_tools(mcp_tools)

    user_msg = DECISION_USER.format(
        goal_text=goal.text,
        hits_text=_format_hits(hits),
        attached_text=_format_attached(attached),
        history_text=_format_history(history),
        tools_text=tools_text,
        pending_urls_text=pending_urls_text,
    )

    resp = gateway.chat(
        messages=[
            {"role": "system", "content": _decision_system()},
            {"role": "user", "content": user_msg},
        ],
        tools=use_tools,
        tool_choice="auto" if use_tools else None,
        auto_route="decision",
        temperature=0.7,
    )

    # Error detection — never return gateway errors as valid answers
    if resp.is_error:
        log.warning("decision_gateway_error", goal=goal.text, transient=resp.error_transient)
        return DecisionOutput(is_error=True)

    if resp.tool_calls:
        tc = resp.tool_calls[0]
        return DecisionOutput(tool_call=ToolCall(name=tc["name"], arguments=tc["arguments"]))

    if resp.text:
        if resp.text.startswith("[gateway error"):
            log.warning("decision_error_in_text", text=resp.text[:100])
            return DecisionOutput(is_error=True)
        return DecisionOutput(answer=resp.text)

    return DecisionOutput(is_error=True)
