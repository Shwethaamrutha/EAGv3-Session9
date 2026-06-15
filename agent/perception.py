"""Perception — the orchestrator role.

Decomposes queries into goals, tracks completion, decides artifact attachments.
"""
from __future__ import annotations

import json
import uuid

from llm_gateway import gateway
from logger import get_logger
from schemas import Goal, MemoryItem, Observation, SYNTHESIS_KEYWORDS

log = get_logger("perception")

from datetime import date as _date

def _perception_system():
    today = _date.today()
    weekday = today.strftime("%A")
    return f"""You are the Perception module of an agentic system. Today is {weekday}, {today.isoformat()}.

Your responsibilities:
1. DECOMPOSE a user query into a sequence of concrete, actionable goals.
2. TRACK which goals are satisfied based on the run history.
3. DECIDE whether the next unfinished goal needs raw artifact bytes attached.

Decomposition rules:
- Prefer FEWER goals. Group related extractions into ONE goal.
  e.g. "tell me the summary, key points, and conclusion" = ONE goal, not three.
- Only separate goals when they require DIFFERENT actions (fetch vs create vs search).
- Resolve relative references into absolute values (e.g. dates, quantities).
- Goals should be ordered by dependency: gather information first, then synthesize.
- If prior_goals is provided, preserve the list — only update done flags.

Research queries:
- When the user asks to "research", "explore", "survey", or "deep dive" on a topic,
  AND memory hits do NOT contain indexed chunks, decompose into:
  (1) search from multiple angles, (2) fetch the most relevant sources,
  (3) index all fetched content for semantic retrieval, (4) synthesize findings.
- If MEMORY HITS already contain indexed chunks relevant to the query,
  do NOT search the web. Create a single synthesis goal instead.

Completion rules:
- A goal is done when the HISTORY contains a tool result or answer that satisfies it.
- Goals that require CREATING something (files, reminders, documents) are only done when
  history shows the creation tool was called successfully. Memory hits alone do NOT satisfy these.
- Goals that require RETRIEVING or ANSWERING can be satisfied by memory hits or history.
- Mark goals done based on what information is NOW available, not what is perfect.
- Once done, a goal remains done permanently.
- A web_search result that returns full article content (marked with char counts) counts as BOTH
  searching AND reading. No separate fetch is needed when content is already available.
- When MEMORY HITS already contain indexed chunks (descriptors showing "chunk N/M" or
  containing actual document content), the content is ALREADY indexed and searchable.
  Create a SINGLE goal to synthesize the answer directly. Do NOT create goals to
  "index", "retrieve", or "read" content that already appears in memory hits.

Artifact attachment:
- Set artifact_index to a valid MEMORY HITS index when the next goal needs raw content
  from a previously fetched resource (e.g. extraction from a web page).
- Set artifact_index to -1 when no attachment is needed.

Constraints:
- Preserve goal order. Do not reorder, insert, or drop goals.
- Respond in JSON matching the schema provided.
"""

PERCEPTION_USER = """QUERY: {query}

MEMORY HITS:
{hits_text}

HISTORY:
{history_text}

PRIOR GOALS:
{prior_goals_text}

Produce an Observation with the current goal list. For each goal:
- id: keep the same id if updating, or generate a short id for new goals
- text: short imperative description
- done: true/false
- artifact_index: integer index into MEMORY HITS (only for the first unfinished goal, -1 otherwise)
"""


def _format_hits(hits: list[MemoryItem]) -> str:
    if not hits:
        return "(none)"
    lines = []
    for i, h in enumerate(hits):
        art_tag = f" [artifact: {h.artifact_id}]" if h.artifact_id else ""
        lines.append(f"  [{i}] ({h.kind}) {h.descriptor}{art_tag}")
        if h.kind == "fact" and h.value.get("raw"):
            lines.append(f"       value: {h.value['raw'][:200]}")
        elif h.kind == "fact" and h.value.get("date"):
            lines.append(f"       value: {json.dumps(h.value, default=str)[:200]}")
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for event in history[-10:]:
        if event.get("kind") == "action":
            lines.append(f"  iter {event['iter']}: TOOL {event['tool']}({json.dumps(event.get('arguments', {}))[:80]}) → {event.get('result_descriptor', '')[:100]}")
        elif event.get("kind") == "answer":
            lines.append(f"  iter {event['iter']}: ANSWER for goal {event.get('goal_id', '?')}: {event.get('text', '')[:150]}")
    return "\n".join(lines) if lines else "(none)"


def _format_prior_goals(goals: list[Goal]) -> str:
    if not goals:
        return "(none — first iteration, decompose the query)"
    lines = []
    for g in goals:
        status = "DONE" if g.done else "OPEN"
        lines.append(f"  [{status}] {g.id}: {g.text}")
    return "\n".join(lines)


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    hits_text = _format_hits(hits)
    history_text = _format_history(history)
    prior_goals_text = _format_prior_goals(prior_goals)

    user_msg = PERCEPTION_USER.format(
        query=query,
        hits_text=hits_text,
        history_text=history_text,
        prior_goals_text=prior_goals_text,
    )

    response_schema = {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "done": {"type": "boolean"},
                        "artifact_index": {"type": "integer"},
                    },
                    "required": ["id", "text", "done"],
                },
            }
        },
        "required": ["goals"],
    }

    resp = gateway.chat(
        messages=[
            {"role": "system", "content": _perception_system()},
            {"role": "user", "content": user_msg},
        ],
        response_format={"schema": response_schema},
        auto_route="perception",
        temperature=0.7,
    )

    if resp.parsed and "goals" in resp.parsed and resp.parsed["goals"]:
        goals = []
        for g_data in resp.parsed["goals"]:
            art_id = None
            art_idx = g_data.get("artifact_index")
            # -1 or missing means no attachment; valid index means attach
            if art_idx is not None and isinstance(art_idx, int) and art_idx >= 0 and art_idx < len(hits):
                art_id = hits[art_idx].artifact_id

            goal_id = g_data.get("id", uuid.uuid4().hex[:8])
            goals.append(Goal(
                id=goal_id,
                text=g_data["text"],
                done=g_data.get("done", False),
                attach_artifact_id=art_id,
            ))

        # Enforce sticky-done: if a prior goal was done, keep it done
        if prior_goals:
            prior_done_ids = {g.id for g in prior_goals if g.done}
            for g in goals:
                if g.id in prior_done_ids:
                    g.done = True

        # Enforce multi-fetch goals: if a goal says "read/fetch top N",
        # don't mark done until N distinct SUCCESSFUL fetches exist in history.
        # Skip enforcement if web_search already returned rich content.
        import re
        has_rich_search = any(
            e.get("kind") == "action" and e.get("tool") == "web_search"
            for e in history
        )
        if not has_rich_search:
            for g in goals:
                if g.done:
                    match = re.search(r'(?:read|fetch|get|visit).*?(\d+)', g.text.lower())
                    if match:
                        required_count = int(match.group(1))
                        successful_fetches = [
                            e for e in history
                            if e.get("kind") == "action" and e.get("tool") == "fetch_url"
                            and len(e.get("result_descriptor", "")) > 200
                        ]
                        distinct_urls = len(set(e.get("arguments", {}).get("url", "") for e in successful_fetches))
                        if distinct_urls < required_count:
                            g.done = False

        # Force-attach for final synthesis/extraction goals only
        # NOT for search/find/check goals which just need tool calls
        if goals:
            next_unfinished = None
            for g in goals:
                if not g.done:
                    next_unfinished = g
                    break
            if next_unfinished and not next_unfinished.attach_artifact_id:
                # Only attach for goals that need to READ content (not find/search/check)
                goal_lower = next_unfinished.text.lower()
                skip_keywords = {"find", "search", "fetch", "check", "get weather"}
                if any(kw in goal_lower for kw in SYNTHESIS_KEYWORDS) and not any(kw in goal_lower for kw in skip_keywords):
                    for h in hits:
                        if h.artifact_id:
                            next_unfinished.attach_artifact_id = h.artifact_id
                            break

        return Observation(goals=goals)

    # Fallback: single goal from the query
    log.warning("perception_fallback", reason="LLM response unparseable")
    return Observation(goals=[Goal(id=uuid.uuid4().hex[:8], text=query, done=False)])
