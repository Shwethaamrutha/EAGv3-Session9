"""Dashboard server — WebSocket-driven DAG visualization.

Runs flow.py with real-time event streaming to the dashboard.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "agent")

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_gateway import gateway
from memory import memory
from cache import cache
from persistence import store
from recovery import classify_failure
from sandbox import run_code
from schemas_v2 import AgentResult, NodeSpec, RunBudget
from skills import Skill, load_skills
from tracing import TraceLog

import networkx as nx
from pydantic import ValidationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="AXON DAG Dashboard", lifespan=lifespan)

DASHBOARD_HTML = Path(__file__).parent / "dashboard_s8.html"


@app.get("/")
async def index():
    return HTMLResponse(DASHBOARD_HTML.read_text())


@app.post("/api/clear")
async def clear_state():
    """Clear memory, FAISS index, and all sessions."""
    import shutil
    from pathlib import Path
    for f in ["state/memory.json", "state/index.faiss", "state/index_ids.json"]:
        p = Path(f)
        if p.exists():
            p.unlink()
    sessions_dir = Path("state/sessions")
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
    cache.clear()
    memory.clear()
    return JSONResponse({"status": "cleared"})


@app.get("/api/screenshot/{sid}/{filename:path}")
async def get_screenshot(sid: str, filename: str):
    """Serve a screenshot PNG from a session."""
    from fastapi.responses import FileResponse
    path = Path(f"state/sessions/{sid}/screenshots/{filename}")
    if path.exists() and path.suffix == ".png":
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/sessions")
async def list_sessions():
    sessions_dir = Path("state/sessions")
    if not sessions_dir.exists():
        return JSONResponse([])
    items = []
    dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        query = ""
        qf = d / "query.txt"
        if qf.exists():
            query = qf.read_text()[:100]
        items.append({"id": d.name, "query": query})
    return JSONResponse(items)


@app.get("/api/browser_io/{sid}")
async def get_browser_io(sid: str):
    """Return persisted browser I/O data for a session."""
    bio_path = Path(f"state/sessions/{sid}/browser_io.jsonl")
    if not bio_path.exists():
        return JSONResponse([])
    items = []
    for line in bio_path.read_text().splitlines():
        try:
            items.append(json.loads(line))
        except:
            pass
    return JSONResponse(items)


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    sessions_dir = Path("state/sessions") / sid
    if not sessions_dir.exists():
        return JSONResponse({"error": "not found"}, status_code=404)

    query = ""
    qf = sessions_dir / "query.txt"
    if qf.exists():
        query = qf.read_text()

    # Load graph
    dag = []
    gf = sessions_dir / "graph.json"
    if gf.exists():
        try:
            data = json.loads(gf.read_text())
            g = nx.node_link_graph(data)
            # Compute layers by topological sort
            try:
                topo = list(nx.topological_sort(g))
                layers = {}
                for nid in topo:
                    preds = list(g.predecessors(nid))
                    if not preds:
                        layers[nid] = 0
                    else:
                        layers[nid] = max(layers.get(p, 0) for p in preds) + 1
                for nid in g.nodes:
                    attrs = g.nodes[nid]
                    dag.append({
                        "id": nid,
                        "skill": attrs.get("skill", "?"),
                        "status": attrs.get("status", "pending"),
                        "layer": layers.get(nid, 0),
                        "question": (attrs.get("metadata") or {}).get("question", ""),
                    })
            except nx.NetworkXUnfeasible:
                pass
        except Exception:
            pass

    # Load traces
    logs = []
    tf = sessions_dir / "traces.jsonl"
    if tf.exists():
        for line in tf.read_text().splitlines():
            try:
                t = json.loads(line)
                logs.append({
                    "skill": t.get("skill", "system"),
                    "message": f"{t['node_id']} {t['status']} ({t.get('elapsed_s', 0):.1f}s)",
                    "extra": f"tokens={t.get('tokens_in',0)}+{t.get('tokens_out',0)}",
                })
            except Exception:
                pass

    # Load nodes for answer
    answer = ""
    nodes_dir = sessions_dir / "nodes"
    if nodes_dir.exists():
        for f in sorted(nodes_dir.glob("*.json"), reverse=True):
            try:
                nd = json.loads(f.read_text())
                if nd.get("skill") == "formatter" and nd.get("status") == "complete":
                    result = nd.get("result", {})
                    output = result.get("output", {}) if isinstance(result, dict) else {}
                    answer = output.get("final_answer", "") or output.get("text", "")
                    if answer:
                        break
            except Exception:
                pass

    # Compute metrics
    metrics = {"nodes": len(dag), "by_skill": {}}
    if tf.exists():
        traces = []
        for line in tf.read_text().splitlines():
            try:
                traces.append(json.loads(line))
            except:
                pass
        total_time = sum(t.get("elapsed_s", 0) for t in traces)
        metrics["serial"] = total_time
        metrics["tokens_in"] = sum(t.get("tokens_in", 0) for t in traces)
        metrics["tokens_out"] = sum(t.get("tokens_out", 0) for t in traces)
        for t in traces:
            sk = t.get("skill", "?")
            if sk not in metrics["by_skill"]:
                metrics["by_skill"][sk] = {"calls": 0, "total_s": 0.0, "tokens_in": 0, "tokens_out": 0}
            metrics["by_skill"][sk]["calls"] += 1
            metrics["by_skill"][sk]["total_s"] += t.get("elapsed_s", 0)
            metrics["by_skill"][sk]["tokens_in"] += t.get("tokens_in", 0)
            metrics["by_skill"][sk]["tokens_out"] += t.get("tokens_out", 0)

    # Extract edges from the graph
    edges = []
    if gf.exists():
        try:
            data = json.loads(gf.read_text())
            # NetworkX uses "links" with "source"/"target" keys
            for link in data.get("links", data.get("edges", [])):
                src = link.get("source") or link.get("from", "")
                tgt = link.get("target") or link.get("to", "")
                if src and tgt:
                    edges.append({"from": src, "to": tgt})
        except Exception:
            pass

    # Build node I/O from graph data
    node_io = []
    if gf.exists():
        try:
            data = json.loads(gf.read_text())
            g2 = nx.node_link_graph(data)
            for nid in g2.nodes:
                attrs = g2.nodes[nid]
                inputs_data = {}
                for inp in (attrs.get("inputs") or []):
                    if inp == "USER_QUERY":
                        inputs_data["USER_QUERY"] = query
                    elif inp.startswith("n:") and inp in g2.nodes:
                        up_result = g2.nodes[inp].get("result")
                        if isinstance(up_result, dict):
                            output = up_result.get("output", {})
                            inputs_data[inp] = json.dumps(output, default=str)
                        elif up_result is not None:
                            inputs_data[inp] = str(up_result)
                node_output = ""
                result = attrs.get("result")
                if isinstance(result, dict) and "output" in result:
                    node_output = json.dumps(result["output"], default=str)
                question = (attrs.get("metadata") or {}).get("question", "")
                node_io.append({
                    "node_id": nid,
                    "skill": attrs.get("skill", "?"),
                    "question": question,
                    "inputs": inputs_data,
                    "output": node_output,
                })
        except:
            pass

    return JSONResponse({"query": query, "dag": dag, "edges": edges, "logs": logs, "answer": answer, "metrics": metrics, "node_io": node_io})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cancel_event = asyncio.Event()
    executor_task: asyncio.Task | None = None

    async def listen_for_client():
        """Listen for client messages (stop command or disconnect)."""
        try:
            while True:
                data = await ws.receive_json()
                if data.get("type") == "stop":
                    cancel_event.set()
                    break
                elif data.get("type") == "query":
                    query = data.get("text", "").strip()
                    if query:
                        cancel_event.clear()
                        return ("query", query)
                elif data.get("type") == "resume":
                    sid = data.get("session_id", "")
                    if sid:
                        cancel_event.clear()
                        return ("resume", sid)
        except WebSocketDisconnect:
            cancel_event.set()
        return None

    try:
        while True:
            cmd = await listen_for_client()
            if cmd is None:
                break

            action, value = cmd
            if action == "query":
                executor_task = asyncio.create_task(
                    run_via_executor(value, ws, resume=False, cancel_event=cancel_event))
            elif action == "resume":
                executor_task = asyncio.create_task(
                    run_via_executor("", ws, resume=True, session_id=value, cancel_event=cancel_event))

            # Wait for either: executor finishes OR client sends stop/disconnects
            listener_task = asyncio.create_task(listen_for_client())
            done, pending = await asyncio.wait(
                [executor_task, listener_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if listener_task in done:
                # Client sent stop or disconnected — cancel the executor
                cancel_event.set()
                executor_task.cancel()
                try:
                    await executor_task
                except (asyncio.CancelledError, Exception):
                    pass
                # If listener returned a new command, loop continues
                result = listener_task.result()
                if result is None:
                    break
            else:
                # Executor finished normally — cancel the listener
                listener_task.cancel()
                try:
                    await listener_task
                except (asyncio.CancelledError, Exception):
                    pass

    except (WebSocketDisconnect, Exception):
        cancel_event.set()
        if executor_task and not executor_task.done():
            executor_task.cancel()


async def run_via_executor(query: str, ws: WebSocket, resume: bool = False,
                           session_id: str | None = None, cancel_event: asyncio.Event | None = None):
    """Run using flow.py's Executor directly — single code path for CLI and dashboard."""
    from flow import Executor
    import time as _time

    if session_id is None:
        session_id = f"s8_{uuid.uuid4().hex[:8]}"

    skills = load_skills()

    async def on_event(event_type, **data):
        # Check cancellation before every emit
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Client stopped")
        try:
            await ws.send_json({"type": event_type, **data})
        except:
            if cancel_event:
                cancel_event.set()
            raise asyncio.CancelledError("WebSocket closed")

    executor = Executor(session_id=session_id, skills_catalogue=skills, on_event=on_event)
    run_start = _time.time()

    try:
        answer = await executor.run(query, resume=resume)
    except asyncio.CancelledError:
        # Cancelled by client — graph already persisted per-node
        return
    except Exception as e:
        try:
            await on_event("error", text=str(e)[:200])
        except:
            pass
        return

    # Send final metrics
    wall = _time.time() - run_start
    summary = executor.trace.summary()
    serial = sum(s.get("total_s", 0) for s in summary.get("by_skill", {}).values())
    try:
        await on_event("metrics",
            nodes=executor.graph.node_count, wallclock=wall, serial=serial,
            speedup=serial/wall if wall > 0 else 1.0,
            tokens_in=executor.budget.used_input, tokens_out=executor.budget.used_output,
            cache_hits=cache.hits, cache_misses=cache.misses,
            recoveries=sum(executor._recovery_count.values()),
            by_skill=summary.get("by_skill", {}))
        await on_event("answer", text=answer)
    except:
        pass


async def run_dag_streaming(query: str, ws: WebSocket, resume_session: str | None = None):
    """Run the DAG orchestrator with WebSocket events."""
    import re
    from schemas import ToolCall
    import action

    if resume_session:
        session_id = resume_session
        query = store.load_query(session_id)
    else:
        session_id = f"s8_{uuid.uuid4().hex[:8]}"
        store.save_query(session_id, query)

    skills = load_skills()
    budget = RunBudget()
    trace = TraceLog(session_id)
    run_start = time.time()

    hits = memory.read(query, [])
    memory_hits_text = _format_memory_hits(hits)
    if not resume_session:
        try:
            memory.remember(query, source="user_query", run_id=session_id)
        except Exception:
            pass

    # Build graph (or load from disk for resume)
    graph = nx.DiGraph()
    counter = [0]
    label_to_id = {}

    if resume_session:
        gf = Path("state/sessions") / session_id / "graph.json"
        if gf.exists():
            data = json.loads(gf.read_text())
            graph = nx.node_link_graph(data)
            # Reset running nodes to pending
            for nid, d in graph.nodes(data=True):
                if d.get("status") == "running":
                    d["status"] = "pending"
                label = (d.get("metadata") or {}).get("label")
                if label:
                    label_to_id[label] = nid
            counter[0] = len(graph.nodes)
            await send("memory_event", node_id="", description=f"Resumed session {session_id} — {len(graph.nodes)} nodes loaded")

    def add_node(skill, inputs, metadata=None):
        counter[0] += 1
        nid = f"n:{counter[0]}"
        meta = dict(metadata or {})
        label = meta.get("label")
        if label:
            label_to_id[label] = nid
        graph.add_node(nid, skill=skill, inputs=list(inputs), metadata=meta, status="pending", result=None)
        for inp in inputs:
            if inp.startswith("n:") and inp in graph.nodes:
                graph.add_edge(inp, nid)
        return nid

    def get_dag_state():
        try:
            topo = list(nx.topological_sort(graph))
            layers = {}
            for nid in topo:
                preds = list(graph.predecessors(nid))
                layers[nid] = (max(layers.get(p, 0) for p in preds) + 1) if preds else 0
        except:
            layers = {nid: 0 for nid in graph.nodes}
        nodes_out = [{"id": nid, "skill": graph.nodes[nid].get("skill",""), "status": graph.nodes[nid].get("status","pending"), "layer": layers.get(nid,0), "question": (graph.nodes[nid].get("metadata") or {}).get("question","")} for nid in graph.nodes]
        edges_out = [{"from": u, "to": v} for u, v in graph.edges]
        return nodes_out, edges_out

    async def send(event_type, **data):
        try:
            await ws.send_json({"type": event_type, "session_id": session_id, **data})
        except:
            pass

    # Start with planner
    if not resume_session:
        planner_id = add_node("planner", ["USER_QUERY"], {"label": "plan"})

    recovery_count = {}
    executed = 0

    while True:
        ready = [nid for nid, d in graph.nodes(data=True) if d["status"] == "pending" and all(graph.nodes[p]["status"] in ("complete","skipped") for p in graph.predecessors(nid))]
        has_incomplete = any(d["status"] in ("pending","running") for _, d in graph.nodes(data=True))
        if not ready and not has_incomplete:
            break
        if not ready:
            break
        if executed > 60:
            break

        if len(ready) > 1:
            await send("parallel", count=len(ready))

        for nid in ready:
            graph.nodes[nid]["status"] = "running"

        # Send node_start for each ready node
        for nid in ready:
            _dag, _edges = get_dag_state()
            meta = graph.nodes[nid].get("metadata") or {}
            await send("node_start", dag=_dag, edges=_edges, node_id=nid, skill=graph.nodes[nid]["skill"], question=meta.get("question",""))

        async def run_one(nid):
            nonlocal executed
            attrs = graph.nodes[nid]
            skill_name = attrs["skill"]
            skill = skills.get(skill_name)
            span = trace.start_span(nid, skill_name)
            metadata = attrs.get("metadata", {})
            question = metadata.get("question", "")

            # Memory event — what context this node sees
            mem_desc = f"{len(hits)} FAISS hits" if hits else "no hits"
            # Check upstream inputs
            upstream_nodes = [i for i in (attrs.get("inputs") or []) if i.startswith("n:") and i in graph.nodes]
            if upstream_nodes:
                upstream_skills = [graph.nodes[u].get("skill","?") for u in upstream_nodes]
                mem_desc += f" + {len(upstream_nodes)} upstream ({', '.join(upstream_skills)})"
            await send("memory_event", node_id=nid, description=mem_desc)

            try:
                if skill_name == "planner":
                    result = await _run_planner(nid, query, skill, graph, skills, label_to_id, counter, add_node, memory_hits_text, metadata, send)
                elif skill_name == "sandbox_executor":
                    result = _run_sandbox(nid, graph)
                elif skill_name == "critic":
                    result = await _run_critic(nid, query, skill, graph, metadata, memory_hits_text, recovery_count, label_to_id, counter, add_node, send)
                elif skill_name == "researcher" and question and not any(kw in question.lower() for kw in ["create", "reminder", "write", "save", "generate file"]):
                    result = await _run_researcher(question, query, send, skill_name, node_id=nid)
                else:
                    result = await _run_generic(nid, query, skill, graph, memory_hits_text, question, send)

                graph.nodes[nid]["result"] = result
                graph.nodes[nid]["status"] = "complete" if result.success else "failed"
                span.tokens_in = result.tokens_in
                span.tokens_out = result.tokens_out
                trace.end_span(span, "complete" if result.success else "failed", error=result.error)
                budget.record(result.tokens_in, result.tokens_out)
                executed += 1

                elapsed = time.time() - span.start_time

                # Decision event — what was produced
                if result.success:
                    out = result.output or {}
                    if skill_name == "planner":
                        n_count = out.get('node_count', len(out.get('nodes', [])) or '?')
                        decision_desc = f"Emitted {n_count} nodes: {out.get('rationale', '')}"
                    elif skill_name == "researcher":
                        findings = (out.get("findings") or "")[:80]
                        decision_desc = f"Found: {findings[:200]}"
                    elif skill_name == "critic":
                        decision_desc = f"Verdict: {out.get('verdict','?').upper()} — {out.get('rationale','')}"
                    elif skill_name == "formatter":
                        decision_desc = f"Final answer rendered ({len(out.get('text','') or out.get('final_answer',''))} chars)"
                    elif skill_name == "coder":
                        decision_desc = f"Code generated ({len(out.get('code',''))} chars)"
                    elif skill_name == "sandbox_executor":
                        decision_desc = f"Exit code: {out.get('exit_code', '?')} | stdout: {(out.get('stdout',''))[:40]}"
                    elif skill_name == "comparator":
                        decision_desc = f"Comparison complete ({len(out.get('text',''))} chars)"
                    else:
                        decision_desc = f"Output ready ({len(json.dumps(out, default=str))} chars)"
                else:
                    decision_desc = f"FAILED: {(result.error or '')[:80]}"

                await send("decision_event", node_id=nid, description=decision_desc)

                # Persist graph after each node completes (for resumability)
                try:
                    sd = Path("state/sessions") / session_id
                    sd.mkdir(parents=True, exist_ok=True)
                    (sd / "graph.json").write_text(json.dumps(nx.node_link_data(graph), indent=2, default=str))
                except:
                    pass

                _dag, _edges = get_dag_state()
                await send("node_complete", dag=_dag, edges=_edges, node_id=nid, skill=skill_name, elapsed=elapsed)

                if not result.success:
                    _dag2, _edges2 = get_dag_state()
                    await send("node_failed", dag=_dag2, edges=_edges2, node_id=nid, error=result.error or "")
                    classification = classify_failure(result.error or "")
                    if classification == "upstream_failure" and skill_name != "planner":
                        if recovery_count.get(nid, 0) < 1:
                            recovery_count[nid] = 1
                            add_node("planner", ["USER_QUERY"], {"label": f"recovery_{counter[0]}", "failure_report": f"Node {nid} failed: {result.error}"})
                            await send("recovery", reason=result.error[:80])

            except Exception as e:
                graph.nodes[nid]["status"] = "failed"
                trace.end_span(span, "failed", error=str(e))
                _dag3, _edges3 = get_dag_state()
                await send("node_failed", dag=_dag3, edges=_edges3, node_id=nid, error=str(e)[:100])

        await asyncio.gather(*[run_one(nid) for nid in ready])

    # Extract answer
    answer = ""
    for nid in reversed(list(graph.nodes)):
        d = graph.nodes[nid]
        if d.get("skill") == "formatter" and d.get("status") == "complete":
            result = d.get("result")
            if isinstance(result, AgentResult):
                answer = result.output.get("final_answer", "") or result.output.get("text", "") or result.text
                break
    if not answer:
        for nid in reversed(list(graph.nodes)):
            d = graph.nodes[nid]
            if d.get("status") == "complete" and d.get("skill") not in ("planner","critic","sandbox_executor"):
                result = d.get("result")
                if isinstance(result, AgentResult) and result.text:
                    answer = result.text
                    break

    # Persist graph for replay
    try:
        session_dir = Path("state/sessions") / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        graph_data = nx.node_link_data(graph)
        (session_dir / "graph.json").write_text(json.dumps(graph_data, indent=2, default=str))
    except Exception:
        pass

    # Send metrics
    summary = trace.summary()
    wall = time.time() - run_start
    serial = sum(s.get("total_s", 0) for s in summary.get("by_skill", {}).values())
    await send("metrics",
        nodes=len(graph.nodes), wallclock=wall, serial=serial,
        speedup=serial/wall if wall > 0 else 1.0,
        tokens_in=budget.used_input, tokens_out=budget.used_output,
        cache_hits=cache.hits, cache_misses=cache.misses,
        critic="", recoveries=sum(recovery_count.values()),
        by_skill=summary.get("by_skill", {}))

    await send("answer", text=answer)


# ── Skill runners (simplified for dashboard) ──

async def _run_planner(nid, query, skill, graph, skills_cat, label_to_id, counter, add_node_fn, memory_hits_text, metadata, send):
    failure_report = metadata.get("failure_report", "")
    inputs = {"USER_QUERY": query}
    if failure_report:
        inputs["FAILURE"] = failure_report
    prompt = skill.render_prompt(inputs, memory_hits=memory_hits_text)
    user_msg = query if not failure_report else f"FAILURE: {failure_report}\nOriginal query: {query}"

    resp = gateway.chat(messages=[{"role":"system","content":prompt},{"role":"user","content":user_msg}], temperature=skill.temperature)
    if resp.is_error:
        return AgentResult(success=False, agent_name="planner", error=resp.text)

    text = resp.text or ""
    parsed = _parse_json(text)
    if not parsed or "nodes" not in parsed:
        return AgentResult(success=False, agent_name="planner", error="Invalid planner JSON")

    # Extend graph
    raw_nodes = parsed.get("nodes", [])
    new_label_map = {}
    new_ids = []
    for spec in raw_nodes:
        nid_new = add_node_fn(spec["skill"], [], spec.get("metadata", {}))
        new_ids.append(nid_new)
        label = spec.get("metadata", {}).get("label", "")
        if label:
            new_label_map[label] = nid_new

    # Resolve inputs and add edges
    for i, spec in enumerate(raw_nodes):
        new_nid = new_ids[i]
        resolved = []
        for inp in spec.get("inputs", []):
            if inp == "USER_QUERY":
                resolved.append(inp)
            elif inp.startswith("n:"):
                suffix = inp[2:]
                if suffix in new_label_map:
                    resolved.append(new_label_map[suffix])
                    graph.add_edge(new_label_map[suffix], new_nid)
                elif suffix in label_to_id:
                    resolved.append(label_to_id[suffix])
                    graph.add_edge(label_to_id[suffix], new_nid)
            elif inp in new_label_map:
                resolved.append(new_label_map[inp])
                graph.add_edge(new_label_map[inp], new_nid)
        graph.nodes[new_nid]["inputs"] = resolved
        # Add structural edge from parent if no node-reference inputs exist
        has_node_input = any(r.startswith("n:") for r in resolved)
        if not has_node_input:
            graph.add_edge(nid, new_nid)

    # Internal successors
    emitted_skills = {s.get("skill") for s in raw_nodes}
    for i, spec in enumerate(raw_nodes):
        sk = skills_cat.get(spec["skill"])
        if sk and sk.internal_successors:
            for succ in sk.internal_successors:
                if succ not in emitted_skills:
                    succ_id = add_node_fn(succ, [new_ids[i]], {"label": f"{succ}_{counter[0]}"})
                    graph.add_edge(new_ids[i], succ_id)

    rationale = parsed.get("rationale", "")
    await send("planner", node_id=nid, rationale=rationale, node_count=len(raw_nodes))
    return AgentResult(success=True, agent_name="planner", output={"rationale": rationale, "node_count": len(raw_nodes)}, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens, provider=resp.provider)


async def _run_researcher(question, query, send, skill_name, node_id=""):
    import re
    from schemas import ToolCall
    import action

    # Check cache for extracted answer
    cached = cache.tool_get("researcher_answer", {"query": question})
    if cached:
        await send("cache_hit", node_id=node_id, tool="web_search", key_preview=question[:40])
        return AgentResult(success=True, agent_name="researcher", output={"findings": cached, "question": question, "cached": True})

    url_match = re.search(r'https?://[^\s,)]+', question)
    # Weather detection: use wttr.in for forecast queries
    weather_keywords = ["weather", "forecast", "temperature", "rain"]
    is_weather = any(kw in question.lower() for kw in weather_keywords)
    city_match = None
    if is_weather:
        # Extract city name from question
        import re as _re
        city_match = _re.search(r'(?:in|for|at)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:this|on|today|tomorrow|saturday|sunday|monday|tuesday|wednesday|thursday|friday)|\?|$)', question, _re.IGNORECASE)

    server_params = StdioServerParameters(command=sys.executable, args=["mcp_server.py"], env={**os.environ, "MCP_LOG_LEVEL": "error"})
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if url_match:
                url = url_match.group(0).rstrip(".")
                await send("tool_call", node_id=node_id, skill="researcher", tool="fetch_url", args_preview=url[:60])
                result_text, _ = await action.execute(session, ToolCall(name="fetch_url", arguments={"url": url}))
            elif is_weather and city_match:
                city = city_match.group(1).strip().replace(" ", "+")
                weather_url = f"https://wttr.in/{city}?format=j1"
                await send("tool_call", node_id=node_id, skill="researcher", tool="fetch_url", args_preview=weather_url[:60])
                result_text, _ = await action.execute(session, ToolCall(name="fetch_url", arguments={"url": weather_url}))
            else:
                await send("tool_call", node_id=node_id, skill="researcher", tool="web_search", args_preview=question[:60])
                result_text, _ = await action.execute(session, ToolCall(name="web_search", arguments={"query": question}))

    # Tavily content field is already a concise summary — no extra LLM needed
    cache.tool_put("researcher_answer", {"query": question}, result_text[:1500])

    return AgentResult(
        success=True, agent_name="researcher",
        output={"findings": result_text[:1500], "question": question},
    )


async def _run_critic(nid, query, skill, graph, metadata, memory_hits_text, recovery_count, label_to_id, counter, add_node_fn, send):
    target_id = metadata.get("target_node")
    upstream_text = ""
    for inp in graph.nodes[nid].get("inputs", []):
        if inp.startswith("n:") and inp in graph.nodes:
            upstream = graph.nodes[inp].get("result")
            if isinstance(upstream, AgentResult):
                upstream_text = json.dumps(upstream.output, default=str)[:4000]
                if not target_id:
                    target_id = inp
                break

    constraint = metadata.get("question", "Verify correctness")
    prompt = skill.render_prompt({"upstream_output": upstream_text}, question=constraint)
    resp = gateway.chat(messages=[{"role":"system","content":prompt},{"role":"user","content":f"Upstream:\n{upstream_text}\n\nConstraint: {constraint}"}], temperature=0.0)

    if resp.is_error:
        return AgentResult(success=False, agent_name="critic", error=resp.text)

    parsed = _parse_json(resp.text or "")
    verdict = (parsed or {}).get("verdict", "pass")
    rationale = (parsed or {}).get("rationale", "")
    await send("critic_verdict", node_id=nid, verdict=verdict, rationale=rationale)

    if verdict == "fail":
        child_id = metadata.get("child_node")
        if child_id and child_id in graph.nodes:
            graph.nodes[child_id]["status"] = "skipped"
        if target_id and recovery_count.get(target_id, 0) < 1:
            recovery_count[target_id] = 1
            add_node_fn("planner", ["USER_QUERY"], {"label": f"recovery_{counter[0]}", "failure_report": f"Critic failed: {rationale}"})
            await send("recovery", reason=rationale)

    return AgentResult(success=True, agent_name="critic", output=parsed or {"verdict": verdict, "rationale": rationale}, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens)


def _run_sandbox(nid, graph):
    inputs = graph.nodes[nid].get("inputs", [])
    code = ""
    for inp in inputs:
        if inp.startswith("n:") and inp in graph.nodes:
            upstream = graph.nodes[inp].get("result")
            if isinstance(upstream, AgentResult):
                code = upstream.output.get("code", "")
    if not code:
        return AgentResult(success=False, agent_name="sandbox_executor", error="no code")
    result = run_code(code)
    return AgentResult(success=(result.exit_code == 0), agent_name="sandbox_executor", output={"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code})


async def _run_generic(nid, query, skill, graph, memory_hits_text, question, send):
    resolved_inputs = {}
    for inp in graph.nodes[nid].get("inputs", []):
        if inp == "USER_QUERY":
            resolved_inputs["USER_QUERY"] = query
        elif inp.startswith("n:") and inp in graph.nodes:
            upstream = graph.nodes[inp].get("result")
            if isinstance(upstream, AgentResult):
                resolved_inputs[f"node:{inp}"] = json.dumps(upstream.output, default=str)[:4000]

    prompt = skill.render_prompt(resolved_inputs, memory_hits=memory_hits_text, question=question)
    resp = gateway.chat(messages=[{"role":"system","content":prompt},{"role":"user","content":question or query}], temperature=skill.temperature)

    if resp.is_error:
        return AgentResult(success=False, agent_name=skill.name, error=resp.text)

    text = resp.text or ""
    parsed = _parse_json(text)
    output = parsed if parsed else {"text": text}
    return AgentResult(success=True, agent_name=skill.name, output=output, tokens_in=resp.input_tokens, tokens_out=resp.output_tokens, provider=resp.provider)


def _format_memory_hits(hits):
    if not hits:
        return ""
    lines = []
    for h in hits[:8]:
        chunk = h.value.get("chunk", "")
        preview = chunk[:2000] if chunk else h.descriptor
        lines.append(f"- [{h.kind}] {h.descriptor[:100]}: {preview}")
    return "\n".join(lines)


def _parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except:
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except:
                pass
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
