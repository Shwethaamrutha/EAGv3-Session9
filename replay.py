"""Replay a persisted Session 8 run, one node at a time.

Usage:
    python replay.py <session_id>

Keys:
    enter   advance to next node
    p       expand the full rendered prompt
    o       expand the full output
    t       show trace/timing summary
    q       quit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from persistence import store


SESSIONS_DIR = Path("state/sessions")


def _print_block(i: int, n: int, state: dict) -> None:
    skill = state.get("skill", "?")
    status = state.get("status", "?")
    elapsed = state.get("elapsed_s", 0)
    node_id = state.get("node_id", "?")
    metadata = state.get("metadata", {})

    print()
    print(f"{'─' * 60}")
    print(f"  node {i}/{n}  │  {node_id}  │  {skill}")
    print(f"  status: {status}  │  elapsed: {elapsed:.1f}s")
    if metadata.get("question"):
        print(f"  question: {metadata['question'][:80]}")
    if state.get("error"):
        print(f"  error: {state['error'][:200]}")

    result = state.get("result")
    if result:
        if isinstance(result, dict):
            output = result.get("output", {})
            preview = json.dumps(output, ensure_ascii=False)[:300]
        else:
            preview = str(result)[:300]
        print(f"  output: {preview}")


def replay(session_id: str) -> int:
    session_path = SESSIONS_DIR / session_id
    if not session_path.exists():
        print(f"Session not found: {session_id}", file=sys.stderr)
        return 2

    query_path = session_path / "query.txt"
    query = query_path.read_text() if query_path.exists() else "(no query)"
    nodes_dir = session_path / "nodes"

    if not nodes_dir.exists():
        print(f"No nodes directory for session {session_id}", file=sys.stderr)
        return 2

    node_files = sorted(nodes_dir.glob("*.json"))
    if not node_files:
        print(f"No node files found", file=sys.stderr)
        return 2

    states = []
    for f in node_files:
        try:
            states.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] skipped {f.name}: {e}", file=sys.stderr)

    print(f"\n{'═' * 60}")
    print(f"  Session:  {session_id}")
    print(f"  Query:    {query[:120]}")
    print(f"  Nodes:    {len(states)}")
    print(f"{'═' * 60}")
    print("\n  [enter] next  [p] prompt  [o] full output  [t] trace  [s] screenshots  [q] quit\n")

    # Load trace if available
    trace_path = session_path / "traces.jsonl"
    traces = []
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    i = 0
    while i < len(states):
        st = states[i]
        _print_block(i + 1, len(states), st)
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if cmd == "q":
            return 0
        elif cmd == "p":
            prompt = st.get("prompt_sent", "(not captured)")
            print(f"\n{'─' * 60}\nPROMPT:\n{'─' * 60}")
            print(prompt[:3000])
            print(f"{'─' * 60}")
            continue
        elif cmd == "o":
            result = st.get("result", {})
            print(f"\n{'─' * 60}\nFULL OUTPUT:\n{'─' * 60}")
            print(json.dumps(result, indent=2, ensure_ascii=False)[:5000])
            print(f"{'─' * 60}")
            continue
        elif cmd == "t":
            if traces:
                print(f"\n{'─' * 60}\nTRACE SUMMARY:\n{'─' * 60}")
                total_tokens = sum(t.get("tokens_in", 0) + t.get("tokens_out", 0) for t in traces)
                print(f"  Total spans: {len(traces)}")
                print(f"  Total tokens: {total_tokens}")
                for t in traces:
                    print(f"    {t['node_id']:8s} {t['skill']:16s} {t['elapsed_s']:.1f}s  "
                          f"in={t.get('tokens_in', 0)} out={t.get('tokens_out', 0)}")
                print(f"{'─' * 60}")
            else:
                print("  (no trace data)")
            continue
        elif cmd == "s":
            result = st.get("result", {})
            output = result.get("output", {}) if isinstance(result, dict) else {}
            screenshots = output.get("screenshots", [])
            if screenshots:
                print(f"\n{'─' * 60}\nSCREENSHOTS ({len(screenshots)}):\n{'─' * 60}")
                for sp in screenshots:
                    print(f"  {sp}")
                layer = output.get("layer_used", "")
                actions = output.get("actions", [])
                if layer:
                    print(f"\n  Layer: {layer}")
                if actions:
                    print(f"  Actions ({len(actions)}):")
                    for a in actions:
                        print(f"    Turn {a.get('turn','?')}: {a.get('type','?')} → \"{a.get('target','')[:50]}\"")
                print(f"{'─' * 60}")
            else:
                print("  (no screenshots for this node)")
            continue
        else:
            i += 1

    print("\n  (end of session)\n")
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        sessions = [d.name for d in SESSIONS_DIR.iterdir() if d.is_dir()] if SESSIONS_DIR.exists() else []
        if sessions:
            print("Available sessions:")
            for s in sorted(sessions):
                print(f"  {s}")
        else:
            print("No sessions found.")
        print(f"\nUsage: python replay.py <session_id>")
        return 0
    return replay(args[0])


if __name__ == "__main__":
    sys.exit(main())
