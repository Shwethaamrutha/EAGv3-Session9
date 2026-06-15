"""Session 9 Replay Report Generator.

Produces a structured report with all 8 required items:
1. Original user goal
2. Planner DAG
3. Browser path chosen
4. Browser actions taken
5. Screenshots or page-state logs
6. Extracted data
7. Final comparison table
8. Turn count and cost summary

Usage:
    python report.py <session_id>
    python report.py <session_id> --html  # generates HTML report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SESSIONS_DIR = Path("state/sessions")


def generate_report(session_id: str, html: bool = False) -> str:
    session_path = SESSIONS_DIR / session_id
    if not session_path.exists():
        return f"Session not found: {session_id}"

    # Load data
    query = (session_path / "query.txt").read_text() if (session_path / "query.txt").exists() else "(no query)"
    graph_data = {}
    if (session_path / "graph.json").exists():
        graph_data = json.loads((session_path / "graph.json").read_text())

    nodes_dir = session_path / "nodes"
    node_states = {}
    if nodes_dir.exists():
        for f in sorted(nodes_dir.glob("*.json")):
            data = json.loads(f.read_text())
            node_states[data.get("node_id", f.stem)] = data

    traces = []
    trace_path = session_path / "traces.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    screenshots_dir = session_path / "screenshots"
    screenshots = sorted(screenshots_dir.glob("*.png")) if screenshots_dir.exists() else []

    # --- Build report sections ---
    sections = []

    # 1. Original user goal
    sections.append(("1. Original User Goal", query))

    # 2. Planner DAG
    dag_nodes = graph_data.get("nodes", [])
    dag_lines = []
    for node in dag_nodes:
        nid = node.get("id", "")
        skill = node.get("skill", "")
        status = node.get("status", "")
        meta = node.get("metadata", {})
        question = meta.get("question", "")[:80]
        dag_lines.append(f"  {nid}: {skill} ({status}) — {question}")
    sections.append(("2. Planner DAG", "\n".join(dag_lines) if dag_lines else "(no DAG data)"))

    # 3-6: Browser-specific data
    browser_nodes = {nid: state for nid, state in node_states.items()
                     if state.get("skill") == "browser"}

    browser_paths = []
    browser_actions = []
    browser_data = []

    for nid, state in browser_nodes.items():
        result = state.get("result", {})
        output = result.get("output", {}) if isinstance(result, dict) else {}

        # 3. Browser path chosen
        layer = output.get("layer_used", "unknown")
        url = output.get("final_url", "") or (state.get("metadata", {}).get("question", ""))[:100]
        browser_paths.append(f"  {nid}: path={layer} → {url}")

        # 4. Browser actions
        actions = output.get("actions", [])
        for a in actions:
            browser_actions.append(
                f"  {nid} Turn {a.get('turn', '?')}: {a.get('type', '?')} → \"{a.get('target', '')[:50]}\""
                + (f" [error: {a.get('error', '')[:40]}]" if a.get("error") else "")
            )

        # 6. Extracted data
        content = output.get("content", "") or output.get("text", "")
        if content:
            browser_data.append(f"  [{nid}] ({len(content)} chars):\n    {content[:400]}")

    sections.append(("3. Browser Path Chosen",
                     "\n".join(browser_paths) if browser_paths else "(no browser nodes)"))
    sections.append(("4. Browser Actions Taken",
                     f"  Total actions: {len(browser_actions)}\n" +
                     "\n".join(browser_actions) if browser_actions else "(no browser actions)"))

    # 5. Screenshots
    screenshot_lines = [f"  {s.name}" for s in screenshots]
    sections.append(("5. Screenshots",
                     f"  Total: {len(screenshots)} screenshots\n" +
                     "\n".join(screenshot_lines[:20]) if screenshots else "(no screenshots)"))

    sections.append(("6. Extracted Data",
                     "\n".join(browser_data) if browser_data else "(data extracted via researcher/other skills)"))

    # 7. Final comparison table (from formatter node)
    final_answer = ""
    for nid, state in node_states.items():
        if state.get("skill") == "formatter" and state.get("status") == "complete":
            result = state.get("result", {})
            output = result.get("output", {}) if isinstance(result, dict) else {}
            final_answer = output.get("text", "") or output.get("final_answer", "")
            if not final_answer:
                final_answer = json.dumps(output, indent=2)[:2000]
    sections.append(("7. Final Comparison Table", final_answer or "(no final answer)"))

    # 8. Turn count and cost summary
    total_tokens_in = sum(t.get("tokens_in", 0) for t in traces)
    total_tokens_out = sum(t.get("tokens_out", 0) for t in traces)
    total_elapsed = sum(t.get("elapsed_s", 0) for t in traces)
    total_browser_turns = sum(
        (node_states.get(nid, {}).get("result", {}) or {}).get("output", {}).get("turns", 0)
        for nid in browser_nodes
    )
    total_browser_actions = len(browser_actions)

    cost_lines = [
        f"  Nodes executed: {len(node_states)}",
        f"  Browser nodes: {len(browser_nodes)}",
        f"  Browser turns (total): {total_browser_turns}",
        f"  Browser actions (total): {total_browser_actions}",
        f"  Screenshots: {len(screenshots)}",
        f"  Tokens in: {total_tokens_in}",
        f"  Tokens out: {total_tokens_out}",
        f"  Wall-clock (serial sum): {total_elapsed:.1f}s",
        f"  Estimated cost: ${(total_tokens_in * 3 + total_tokens_out * 15) / 1_000_000:.4f}",
    ]

    # Per-skill breakdown
    skill_stats: dict[str, dict] = {}
    for t in traces:
        skill = t.get("skill", "?")
        if skill not in skill_stats:
            skill_stats[skill] = {"count": 0, "elapsed": 0.0, "tokens_in": 0, "tokens_out": 0}
        skill_stats[skill]["count"] += 1
        skill_stats[skill]["elapsed"] += t.get("elapsed_s", 0)
        skill_stats[skill]["tokens_in"] += t.get("tokens_in", 0)
        skill_stats[skill]["tokens_out"] += t.get("tokens_out", 0)

    cost_lines.append("\n  Per-skill breakdown:")
    for skill, stats in sorted(skill_stats.items()):
        cost_lines.append(f"    {skill:16s} {stats['count']}x  {stats['elapsed']:.1f}s  "
                          f"in={stats['tokens_in']} out={stats['tokens_out']}")

    sections.append(("8. Turn Count and Cost Summary", "\n".join(cost_lines)))

    # --- Format output ---
    if html:
        return _format_html(session_id, sections, screenshots, session_path)

    lines = [f"{'═' * 70}", f"  SESSION 9 REPLAY REPORT: {session_id}", f"{'═' * 70}", ""]
    for title, content in sections:
        lines.append(f"{'─' * 70}")
        lines.append(f"  {title}")
        lines.append(f"{'─' * 70}")
        lines.append(content)
        lines.append("")
    lines.append(f"{'═' * 70}")
    return "\n".join(lines)


def _format_html(session_id: str, sections: list, screenshots: list, session_path: Path) -> str:
    """Generate an HTML report with embedded screenshots."""
    import base64

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Session 9 Report: {session_id}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #0d1117; color: #e6edf3; }}
h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
h2 {{ color: #79c0ff; margin-top: 30px; }}
pre {{ background: #161b22; padding: 16px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; border: 1px solid #30363d; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
th {{ background: #161b22; color: #79c0ff; }}
.screenshots {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; }}
.screenshots img {{ width: 100%; border: 1px solid #30363d; border-radius: 4px; }}
.screenshot-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
.metric {{ display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 8px 12px; margin: 4px; }}
.metric-value {{ font-size: 20px; font-weight: bold; color: #58a6ff; }}
.metric-label {{ font-size: 11px; color: #8b949e; }}
</style></head><body>
<h1>Session 9 Replay Report</h1>
<p style="color:#8b949e;">Session: <code>{session_id}</code></p>
"""]

    for title, content in sections:
        html_parts.append(f"<h2>{title}</h2>")
        if "|" in content and "---" in content:
            html_parts.append(_markdown_table_to_html(content))
        else:
            html_parts.append(f"<pre>{_esc(content)}</pre>")

    # Embedded screenshots
    if screenshots:
        html_parts.append("<h2>Screenshots Gallery</h2><div class='screenshots'>")
        for sp in screenshots[:20]:
            try:
                img_data = base64.b64encode(sp.read_bytes()).decode()
                html_parts.append(
                    f"<div><img src='data:image/png;base64,{img_data}' alt='{sp.name}'/>"
                    f"<div class='screenshot-label'>{sp.name}</div></div>"
                )
            except Exception:
                pass
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def _markdown_table_to_html(text: str) -> str:
    """Convert markdown table to HTML table."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    table_lines = [l for l in lines if "|" in l and not all(c in "-| " for c in l)]
    if not table_lines:
        return f"<pre>{_esc(text)}</pre>"

    html = "<table>"
    for i, line in enumerate(table_lines):
        cells = [c.strip() for c in line.split("|")[1:-1]]
        tag = "th" if i == 0 else "td"
        html += "<tr>" + "".join(f"<{tag}>{_esc(c)}</{tag}>" for c in cells) + "</tr>"
    html += "</table>"

    non_table = [l for l in lines if "|" not in l]
    if non_table:
        html += f"<pre>{_esc(chr(10).join(non_table))}</pre>"
    return html


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    args = sys.argv[1:]
    if not args:
        sessions = sorted(d.name for d in SESSIONS_DIR.iterdir() if d.is_dir()) if SESSIONS_DIR.exists() else []
        if sessions:
            print("Available sessions:")
            for s in sessions[-10:]:
                print(f"  {s}")
        print(f"\nUsage: python report.py <session_id> [--html]")
        return 0

    session_id = args[0]
    html_mode = "--html" in args

    report = generate_report(session_id, html=html_mode)

    if html_mode:
        out_path = SESSIONS_DIR / session_id / "report.html"
        out_path.write_text(report)
        print(f"HTML report saved to: {out_path}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
