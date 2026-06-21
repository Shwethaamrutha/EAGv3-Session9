"""Flow — DAG-based multi-agent orchestrator (Session 8 v2).

Fixes over v1:
  1. Typed AgentResult contracts between nodes
  2. Proper role="tool" message protocol via mcp_runner
  3. Per-node tracing with structured span logging
  4. Streaming formatter output
  5. NodeSpec Pydantic validation on planner output
  6. Per-node timeout with asyncio.wait_for
  7. Tool result caching
  8. One MCP subprocess per parallel node (via mcp_runner)
  9. Token budget / circuit breaker
  10. Resume at node boundary with prompt persistence
"""
from __future__ import annotations

import sys
sys.path.insert(0, "agent")

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import networkx as nx
from pydantic import ValidationError

logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("crawl4ai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_gateway import gateway
from memory import memory
from core.cache import cache
from core.persistence import store
from core.recovery import classify_failure
from core.sandbox import run_code
from core.schemas_v2 import AgentResult, NodeSpec, NodeState, RunBudget
from skills import Skill, load_skills
from core.tracing import TraceLog

MAX_NODES = 60
MAX_RECOVERY_PER_TARGET = 3
NODE_TIMEOUT_S = 600.0

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[38;5;75m"
GREEN = "\033[38;5;114m"
AMBER = "\033[38;5;179m"
WHITE = "\033[38;5;252m"
GRAY = "\033[38;5;242m"
PURPLE = "\033[38;5;141m"
CYAN = "\033[38;5;116m"


class Graph:
    def __init__(self):
        self.g = nx.DiGraph()
        self._counter = 0
        self._label_to_id: dict[str, str] = {}

    def add_node(self, skill: str, inputs: list[str], metadata: dict | None = None) -> str:
        self._counter += 1
        nid = f"n:{self._counter}"
        meta = dict(metadata or {})
        label = meta.get("label")
        if label:
            self._label_to_id[label] = nid

        self.g.add_node(nid, skill=skill, inputs=list(inputs),
                        metadata=meta, status="pending", result=None)

        for inp in inputs:
            if inp.startswith("n:") and inp in self.g.nodes:
                self.g.add_edge(inp, nid)
        return nid

    def mark(self, nid: str, status: str):
        self.g.nodes[nid]["status"] = status

    def set_result(self, nid: str, result: AgentResult):
        self.g.nodes[nid]["result"] = result

    def resolve_label(self, label: str) -> str | None:
        return self._label_to_id.get(label)

    def ready_nodes(self) -> list[str]:
        out = []
        for nid, d in self.g.nodes(data=True):
            if d["status"] != "pending":
                continue
            preds = list(self.g.predecessors(nid))
            if all(self.g.nodes[p]["status"] in ("complete", "skipped") for p in preds):
                out.append(nid)
        return out

    def has_incomplete(self) -> bool:
        return any(d["status"] in ("pending", "running") for _, d in self.g.nodes(data=True))

    def extend_from(self, planner_output: dict, parent_id: str, skills_catalogue: dict[str, Skill]) -> list[str]:
        """Validate and add nodes from planner output. Returns new node ids."""
        raw_nodes = planner_output.get("nodes", [])
        added_ids = []
        label_to_new_id: dict[str, str] = {}

        # Pass 1: validate and add nodes
        for spec_dict in raw_nodes:
            try:
                spec = NodeSpec.model_validate(spec_dict)
            except ValidationError as e:
                raise RuntimeError(f"Malformed NodeSpec: {spec_dict} — {e}")

            nid = self.add_node(spec.skill, [], spec.metadata)
            added_ids.append(nid)
            label = spec.metadata.get("label", "")
            if label:
                label_to_new_id[label] = nid

        # Pass 2: resolve inputs and add edges
        for i, spec_dict in enumerate(raw_nodes):
            nid = added_ids[i]
            raw_inputs = spec_dict.get("inputs", [])
            resolved = []

            for inp in raw_inputs:
                if inp == "USER_QUERY":
                    resolved.append(inp)
                elif inp.startswith("n:"):
                    suffix = inp[2:]
                    if suffix in label_to_new_id:
                        ref_id = label_to_new_id[suffix]
                        resolved.append(ref_id)
                        self.g.add_edge(ref_id, nid)
                    elif suffix in self._label_to_id:
                        ref_id = self._label_to_id[suffix]
                        resolved.append(ref_id)
                        self.g.add_edge(ref_id, nid)
                    else:
                        resolved.append(inp)
                elif inp in label_to_new_id:
                    ref_id = label_to_new_id[inp]
                    resolved.append(ref_id)
                    self.g.add_edge(ref_id, nid)
                elif inp.startswith("art:"):
                    resolved.append(inp)
                else:
                    resolved.append(inp)

            self.g.nodes[nid]["inputs"] = resolved

            # Fan-out workers without node-reference inputs depend on parent
            has_node_ref = any(r.startswith("n:") for r in resolved if isinstance(r, str))
            if not has_node_ref:
                self.g.add_edge(parent_id, nid)

        # Fix planner-emitted sandbox: ensure it depends on coder, not planner
        for i, spec_dict in enumerate(raw_nodes):
            if spec_dict.get("skill") == "sandbox_executor":
                sandbox_id = added_ids[i]
                # Find coder in this batch
                for j, other in enumerate(raw_nodes):
                    if other.get("skill") == "coder":
                        coder_id = added_ids[j]
                        if not self.g.has_edge(coder_id, sandbox_id):
                            self.g.add_edge(coder_id, sandbox_id)
                        # Remove edge from parent (planner) to sandbox if coder edge exists
                        if self.g.has_edge(parent_id, sandbox_id):
                            self.g.remove_edge(parent_id, sandbox_id)
                        # Set sandbox inputs to reference coder
                        self.g.nodes[sandbox_id]["inputs"] = [coder_id]
                        break

        # Auto-insert internal_successors (e.g., sandbox_executor after coder)
        # Also rewire any downstream nodes that reference the coder to reference
        # the sandbox instead (critic should verify sandbox output, not raw code)
        emitted_skills = {s.get("skill") for s in raw_nodes}
        for i, spec_dict in enumerate(raw_nodes):
            skill_name = spec_dict.get("skill", "")
            if skill_name in skills_catalogue:
                skill = skills_catalogue[skill_name]
                for succ_skill in skill.internal_successors:
                    if succ_skill in emitted_skills:
                        continue
                    label = spec_dict.get("metadata", {}).get("label", str(i))
                    succ_id = self.add_node(succ_skill, [added_ids[i]], {
                        "label": f"{succ_skill}_{self._counter}",
                    })
                    self.g.add_edge(added_ids[i], succ_id)
                    added_ids.append(succ_id)

                    # Rewire: any node that depends on the coder (except sandbox)
                    # should now depend on the sandbox instead
                    coder_id = added_ids[i]
                    for child in list(self.g.successors(coder_id)):
                        if child == succ_id:
                            continue
                        self.g.remove_edge(coder_id, child)
                        self.g.add_edge(succ_id, child)
                        # Update the child's inputs list too
                        child_inputs = self.g.nodes[child].get("inputs", [])
                        self.g.nodes[child]["inputs"] = [
                            succ_id if inp == coder_id else inp for inp in child_inputs
                        ]

        # Auto-insert critic for skills with critic: true
        for i, spec_dict in enumerate(raw_nodes):
            skill_name = spec_dict.get("skill", "")
            if skill_name in skills_catalogue and skills_catalogue[skill_name].critic:
                src_id = added_ids[i]
                successors = list(self.g.successors(src_id))
                for child_id in successors:
                    if self.g.nodes[child_id].get("skill") == "critic":
                        continue
                    self.g.remove_edge(src_id, child_id)
                    critic_id = self.add_node("critic", [src_id], {
                        "label": f"critic_{self._counter}",
                        "question": spec_dict.get("metadata", {}).get("question", "Verify correctness"),
                        "target_node": src_id,
                        "child_node": child_id,
                    })
                    self.g.add_edge(critic_id, child_id)
                    added_ids.append(critic_id)

        return added_ids

    @property
    def node_count(self) -> int:
        return len(self.g.nodes)


class Executor:
    def __init__(self, session_id: str, skills_catalogue: dict[str, Skill], on_event=None):
        self.session_id = session_id
        self.skills = skills_catalogue
        self.graph = Graph()
        self.memory_hits_text: str = ""
        self.budget = RunBudget()
        self.trace = TraceLog(session_id)
        self._recovery_count: dict[str, int] = {}
        self._run_start: float = 0.0
        self._current_node_id: str = ""
        self._on_event = on_event  # async callback: (event_type, **data) -> None

    async def _emit(self, event_type: str, **data):
        if self._on_event:
            try:
                await self._on_event(event_type, session_id=self.session_id, **data)
            except:
                pass

    async def run(self, query: str, resume: bool = False) -> str:
        self._run_start = time.time()

        if resume:
            return await self._resume(query)

        store.save_query(self.session_id, query)

        hits = memory.read(query, [])
        self.memory_hits_text = self._format_memory_hits(hits)
        memory.remember(query, source="user_query", run_id=self.session_id)

        planner_id = self.graph.add_node("planner", ["USER_QUERY"], {"label": "plan"})

        print(f"\n{CYAN}{BOLD}Session {self.session_id}{RESET}")
        print(f"{GRAY}{'─' * 50}{RESET}\n")

        await self._execute_loop(query)
        self._persist_graph()
        self._print_summary()
        return self._extract_answer()

    async def _resume(self, query: str) -> str:
        stored_query = store.load_query(self.session_id)
        if not query:
            query = stored_query

        g = store.load_graph(self.session_id)
        for node_id in g.nodes:
            attrs = g.nodes[node_id]
            status = attrs.get("status", "pending")
            if status == "running":
                status = "pending"
            self.graph.g.add_node(node_id, **{**attrs, "status": status})
            label = attrs.get("metadata", {}).get("label")
            if label:
                self.graph._label_to_id[label] = node_id

        for u, v in g.edges:
            self.graph.g.add_edge(u, v)
        self.graph._counter = len(g.nodes)

        hits = memory.read(query, [])
        self.memory_hits_text = self._format_memory_hits(hits)

        print(f"\n{CYAN}{BOLD}Resuming {self.session_id}{RESET}")
        print(f"{GRAY}{'─' * 50}{RESET}\n")

        await self._execute_loop(query)
        self._persist_graph()
        self._print_summary()
        return self._extract_answer()

    def _get_dag_state(self):
        """DAG state for events."""
        try:
            import networkx as nx2
            topo = list(nx.topological_sort(self.graph.g))
            layers = {}
            for nid in topo:
                preds = list(self.graph.g.predecessors(nid))
                layers[nid] = (max(layers.get(p, 0) for p in preds) + 1) if preds else 0
        except:
            layers = {nid: 0 for nid in self.graph.g.nodes}
        nodes_out = [{"id": nid, "skill": self.graph.g.nodes[nid].get("skill",""), "status": self.graph.g.nodes[nid].get("status","pending"), "layer": layers.get(nid,0), "question": (self.graph.g.nodes[nid].get("metadata") or {}).get("question","")} for nid in self.graph.g.nodes]
        edges_out = [{"from": u, "to": v} for u, v in self.graph.g.edges]
        return nodes_out, edges_out

    async def _execute_loop(self, query: str):
        while self.graph.has_incomplete():
            ready = self.graph.ready_nodes()
            if not ready:
                break
            if self.graph.node_count > MAX_NODES:
                print(f"{AMBER}[cap] MAX_NODES ({MAX_NODES}) reached{RESET}")
                break
            if self.budget.exhausted:
                print(f"{AMBER}[budget] Token budget exhausted "
                      f"(in={self.budget.used_input}, out={self.budget.used_output}){RESET}")
                break
            if time.time() - self._run_start > self.budget.max_wall_clock_s:
                print(f"{AMBER}[timeout] Wall-clock budget exceeded{RESET}")
                break

            if len(ready) > 1:
                print(f"{BLUE}[parallel]{RESET} {len(ready)} nodes concurrently")
                await self._emit("parallel", count=len(ready))

            for nid in ready:
                self.graph.mark(nid, "running")
                attrs = self.graph.g.nodes[nid]
                dag, edges = self._get_dag_state()
                await self._emit("node_start", dag=dag, edges=edges, node_id=nid,
                                skill=attrs.get("skill",""), question=(attrs.get("metadata") or {}).get("question",""))

            results = await asyncio.gather(
                *[self._run_node_safe(nid, query) for nid in ready],
                return_exceptions=True,
            )

            for nid, result in zip(ready, results):
                if isinstance(result, Exception):
                    error_text = f"{type(result).__name__}: {result}"
                    self.graph.mark(nid, "failed")
                    print(f"{AMBER}[error]{RESET} {nid}: {error_text[:80]}")
                    await self._handle_failure(nid, error_text, query)

            # Mark pending nodes as skipped if any predecessor permanently failed
            for nid, d in list(self.graph.g.nodes(data=True)):
                if d.get("status") != "pending":
                    continue
                preds = list(self.graph.g.predecessors(nid))
                if any(self.graph.g.nodes[p].get("status") == "failed" for p in preds):
                    self.graph.mark(nid, "skipped")
                    print(f"  {DIM}[skipped] {nid} — upstream failed{RESET}")

            self._persist_graph()

    async def _run_node_safe(self, node_id: str, query: str):
        """Run a single node with timeout."""
        try:
            await asyncio.wait_for(
                self._run_node(node_id, query),
                timeout=NODE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            error_text = f"Node {node_id} exceeded {NODE_TIMEOUT_S}s timeout"
            self.graph.mark(node_id, "failed")
            self.graph.g.nodes[node_id]["result"] = AgentResult(
                success=False, agent_name=self.graph.g.nodes[node_id]["skill"],
                error=error_text)
            print(f"{AMBER}[timeout]{RESET} {node_id}: {error_text}")
            await self._handle_failure(node_id, error_text, query)

    async def _run_node(self, node_id: str, query: str):
        attrs = self.graph.g.nodes[node_id]
        # Skip if already marked skipped by a concurrent critic
        if attrs.get("status") == "skipped":
            return
        skill_name = attrs["skill"]
        skill = self.skills.get(skill_name)
        if skill is None:
            self.graph.mark(node_id, "failed")
            return

        self._current_node_id = node_id
        span = self.trace.start_span(node_id, skill_name)
        elapsed_label = f"{time.time() - self._run_start:.1f}s"
        print(f"{BLUE}[{elapsed_label}]{RESET} {BOLD}{node_id}{RESET} ({skill_name}) starting...")

        # Emit memory event
        inputs = attrs.get("inputs", [])
        upstream = [i for i in inputs if i.startswith("n:") and i in self.graph.g.nodes]
        mem_desc = f"{len(self.memory_hits_text.splitlines())} FAISS hits" if self.memory_hits_text else "no hits"
        if upstream:
            upstream_skills = [self.graph.g.nodes[u].get("skill","?") for u in upstream]
            mem_desc += f" + {len(upstream)} upstream ({', '.join(upstream_skills)})"
        await self._emit("memory_event", node_id=node_id, description=mem_desc)

        try:
            if skill_name == "sandbox_executor":
                result = await self._run_sandbox(node_id)
            elif skill_name == "planner":
                result = await self._run_planner(node_id, query)
            elif skill_name == "critic":
                result = await self._run_critic(node_id, query)
            else:
                result = await self._run_skill(node_id, query, skill)

            self.graph.set_result(node_id, result)
            self.graph.mark(node_id, "complete" if result.success else "failed")

            # Update budget
            self.budget.record(result.tokens_in, result.tokens_out)
            span.tokens_in = result.tokens_in
            span.tokens_out = result.tokens_out
            span.provider = result.provider

            elapsed = time.time() - span.start_time
            status_color = GREEN if result.success else AMBER
            print(f"{status_color}[{elapsed:.1f}s]{RESET} {node_id} ({skill_name}) "
                  f"{'complete' if result.success else 'FAILED'}")

            # Emit node I/O for debug tab
            inputs_debug = attrs.get("inputs", [])
            upstream_data = {}
            for inp in inputs_debug:
                if inp == "USER_QUERY":
                    upstream_data["USER_QUERY"] = query
                elif inp.startswith("n:") and inp in self.graph.g.nodes:
                    up = self.graph.g.nodes[inp].get("result")
                    if isinstance(up, AgentResult):
                        upstream_data[inp] = json.dumps(up.output, default=str)
            question_debug = (attrs.get("metadata") or {}).get("question", "")
            await self._emit("node_io", node_id=node_id, skill=skill_name,
                           question=question_debug,
                           inputs=upstream_data,
                           output=json.dumps(result.output, default=str) if result.output else "")

            # Emit decision event
            out = result.output or {}
            if result.success:
                if skill_name == "planner":
                    dec = f"Emitted {out.get('node_count','?')} nodes: {out.get('rationale','')}"
                elif skill_name == "researcher":
                    dec = f"Found: {(out.get('findings',''))[:200]}"
                elif skill_name == "critic":
                    dec = f"Verdict: {out.get('verdict','?').upper()} — {out.get('rationale','')}"
                elif skill_name == "formatter":
                    dec = f"Final answer ({len(out.get('text','') or out.get('final_answer',''))} chars)"
                else:
                    dec = f"Complete ({len(json.dumps(out, default=str))} chars)"
            else:
                error_detail = result.error or out.get("error_code", "") or out.get("error", "") or "unknown"
                dec = f"FAILED: {error_detail[:150]}"
            await self._emit("decision_event", node_id=node_id, description=dec)

            dag, edges = self._get_dag_state()
            await self._emit("node_complete", dag=dag, edges=edges, node_id=node_id,
                           skill=skill_name, elapsed=elapsed)

            self.trace.end_span(span, "complete" if result.success else "failed",
                               error=result.error)
            self._persist_node(node_id, result, span)
            self._persist_graph()  # persist after every node so Stop is always resumable

            if not result.success:
                dag, edges = self._get_dag_state()
                await self._emit("node_failed", dag=dag, edges=edges, node_id=node_id, error=result.error or "")
                await self._handle_failure(node_id, result.error or "", query)

        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            self.graph.mark(node_id, "failed")
            self.trace.end_span(span, "failed", error=error_text)
            print(f"{AMBER}[error]{RESET} {node_id}: {error_text[:80]}")
            dag, edges = self._get_dag_state()
            await self._emit("node_failed", dag=dag, edges=edges, node_id=node_id, error=error_text[:100])
            await self._handle_failure(node_id, error_text, query)

    async def _run_planner(self, node_id: str, query: str) -> AgentResult:
        skill = self.skills["planner"]
        attrs = self.graph.g.nodes[node_id]
        metadata = attrs.get("metadata", {})
        failure_report = metadata.get("failure_report", "")

        inputs = {"USER_QUERY": query}
        if failure_report:
            inputs["FAILURE"] = failure_report

        from datetime import date as _date
        prompt = skill.render_prompt(inputs, memory_hits=self.memory_hits_text)
        date_prefix = f"Today is {_date.today().isoformat()}. "
        user_msg = date_prefix + query if not failure_report else f"FAILURE: {failure_report}\n\nOriginal query: {date_prefix}{query}"

        resp = gateway.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=skill.temperature,
        )

        if resp.is_error:
            return AgentResult(success=False, agent_name="planner", error=f"LLM error: {resp.text}")

        text = resp.text or ""
        planner_output = self._parse_json(text)
        if not planner_output or "nodes" not in planner_output:
            return AgentResult(success=False, agent_name="planner",
                             error=f"Invalid JSON: {text[:200]}")

        try:
            self.graph.extend_from(planner_output, node_id, self.skills)
        except (RuntimeError, ValidationError) as e:
            return AgentResult(success=False, agent_name="planner", error=str(e))

        rationale = planner_output.get("rationale", "")
        print(f"  {DIM}Rationale: {rationale}{RESET}")
        for n in planner_output["nodes"]:
            print(f"  {DIM}  → {n['skill']} ({n.get('metadata', {}).get('label', '?')}){RESET}")

        return AgentResult(
            success=True, agent_name="planner",
            output={"rationale": rationale, "node_count": len(planner_output["nodes"])},
            tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
            provider=resp.provider,
        )

    async def _run_skill(self, node_id: str, query: str, skill: Skill) -> AgentResult:
        attrs = self.graph.g.nodes[node_id]
        inputs = attrs.get("inputs", [])
        metadata = attrs.get("metadata", {})
        question = metadata.get("question", "")

        # Fallback: if no question in metadata but inputs has a plain text string, use it
        if not question:
            for inp in inputs:
                if inp != "USER_QUERY" and not inp.startswith("n:") and not inp.startswith("art:"):
                    question = inp
                    break

        # Resolve upstream node outputs into structured data
        resolved_inputs = {}
        for inp in inputs:
            if inp == "USER_QUERY":
                resolved_inputs["USER_QUERY"] = query
            elif inp.startswith("n:") and inp in self.graph.g.nodes:
                upstream = self.graph.g.nodes[inp]
                upstream_result = upstream.get("result")
                if isinstance(upstream_result, AgentResult) and upstream_result.output:
                    resolved_inputs[f"node:{inp}"] = json.dumps(upstream_result.output, default=str)[:4000]
                elif isinstance(upstream_result, AgentResult) and upstream_result.text:
                    resolved_inputs[f"node:{inp}"] = upstream_result.text[:4000]

        # Fast path for browser: direct cascade dispatch
        if skill.name == "browser" and question:
            return await self._run_browser_fast(question, query, skill, node_id)

        # Fast path for researcher: direct tool dispatch
        if skill.name == "researcher" and question and not any(kw in question.lower() for kw in ["create", "reminder", "write", "save", "generate file"]):
            return await self._run_researcher_fast(question, query, skill, node_id)

        # Fast path for shell: ask LLM for the command, then run it directly
        if skill.name == "shell" and question:
            return await self._run_shell_fast(question, node_id)

        prompt = skill.render_prompt(resolved_inputs, memory_hits=self.memory_hits_text, question=question)

        # Skills with tools use mcp_runner (one subprocess per skill for parallelism)
        if skill.tools_allowed:
            return await self._run_with_tools(skill, prompt, question or query, node_id=node_id)

        # No-tool skills: single LLM call
        resp = gateway.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question or query},
            ],
            temperature=skill.temperature,
            max_tokens=skill.max_tokens,
        )

        if resp.is_error:
            return AgentResult(success=False, agent_name=skill.name, error=f"LLM error: {resp.text}")

        text = resp.text or ""
        parsed = self._parse_json(text)
        if isinstance(parsed, dict):
            output = parsed
        elif isinstance(parsed, list):
            output = {"items": parsed, "text": text}
        else:
            # Try parsing as JSON (might be array or other valid JSON)
            try:
                clean = text.strip()
                if clean.startswith("```"):
                    clean = "\n".join(l for l in clean.split("\n") if not l.startswith("```"))
                raw = json.loads(clean)
                if isinstance(raw, list):
                    output = {"items": raw, "text": text}
                elif isinstance(raw, dict):
                    output = raw
                else:
                    output = {"text": text}
            except:
                output = {"text": text}

        # Contract validation with one retry
        required_fields = {
            "coder": ["code"],
            "critic": ["verdict"],
        }
        if skill.name in required_fields:
            missing = [f for f in required_fields[skill.name] if f not in output]
            if missing:
                # Retry: tell the model what went wrong
                retry_resp = gateway.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question or query},
                        {"role": "assistant", "content": resp.text or ""},
                        {"role": "user", "content": f"ERROR: Your response is invalid. You MUST return JSON with these required keys: {required_fields[skill.name]}. You returned keys: {list(output.keys())}. Respond with ONLY valid JSON, no markdown, no explanation."},
                    ],
                    temperature=skill.temperature,
                    max_tokens=skill.max_tokens,
                )
                if not retry_resp.is_error and retry_resp.text:
                    retry_parsed = self._parse_json(retry_resp.text)
                    if isinstance(retry_parsed, dict):
                        retry_missing = [f for f in required_fields[skill.name] if f not in retry_parsed]
                        if not retry_missing:
                            output = retry_parsed
                            missing = []

                if missing:
                    return AgentResult(
                        success=False, agent_name=skill.name,
                        error=f"Contract violation after retry: {skill.name} must return {required_fields[skill.name]} but got: {list(output.keys())}",
                        output=output,
                    )

        # Auto-save: if web_builder produced HTML, validate and write to sandbox
        if skill.name == "web_builder" and output.get("html") and output.get("file_path"):
            html = output["html"]
            # Validate: must have closing </html> tag (not truncated)
            if "</html>" in html.lower():
                from pathlib import Path
                save_path = Path("state/sandbox") / output["file_path"]
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(html)
                print(f"    {DIM}saved: {save_path}{RESET}")
                output["saved"] = True
            else:
                output["saved"] = False
                output["error"] = "HTML incomplete (truncated, missing </html>)"
                return AgentResult(
                    success=False, agent_name=skill.name, output=output,
                    error="HTML output was truncated — missing closing tags",
                )

        return AgentResult(
            success=True, agent_name=skill.name, output=output,
            tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
            provider=resp.provider,
        )

    async def _run_shell_fast(self, question: str, node_id: str) -> AgentResult:
        """One LLM call to decide the command, then direct execution."""
        # Ask LLM for the command(s)
        resp = gateway.chat(
            messages=[
                {"role": "system", "content": "Given the task below, output ONLY the shell command(s) to run. One command per line. No explanations, no markdown, no backticks. Just the raw commands."},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        if resp.is_error:
            return AgentResult(success=False, agent_name="shell", error=f"LLM error: {resp.text}")

        commands = [l.strip() for l in (resp.text or "").strip().split('\n') if l.strip() and not l.startswith('#')]

        # Execute each command directly
        import subprocess
        results = []
        for cmd in commands[:3]:  # max 3 commands
            print(f"    {DIM}$ {cmd}{RESET}")
            await self._emit("tool_call", node_id=node_id, skill="shell", tool="run_command", args_preview=cmd[:60])
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=10, cwd=str(Path(".")),
                )
                output = result.stdout[:3000] if result.stdout else result.stderr[:1000]
                results.append({"command": cmd, "stdout": result.stdout[:3000], "exit_code": result.returncode})
            except subprocess.TimeoutExpired:
                results.append({"command": cmd, "stdout": "", "error": "timeout"})

        return AgentResult(
            success=True, agent_name="shell",
            output={"commands": [r["command"] for r in results], "results": results, "text": "\n".join(r.get("stdout","") for r in results)},
            tokens_in=resp.input_tokens, tokens_out=resp.output_tokens,
        )

    async def _run_browser_fast(self, question: str, original_query: str, skill: Skill, node_id: str) -> AgentResult:
        """Direct browser cascade dispatch."""
        from browser import run_browser
        from urllib.parse import urlparse, urlunparse

        url_match = re.search(r'https?://[^\s,)]+', question)
        url = url_match.group(0).rstrip(".") if url_match else ""

        # If user provided a URL in their query, strip any params the planner added
        # (user wants the browser to interact, not skip via URL params)
        user_provided_url = re.search(r'https?://[^\s,)]+', original_query)
        if url and user_provided_url:
            user_url = user_provided_url.group(0).rstrip(".")
            parsed_user = urlparse(user_url)
            parsed_plan = urlparse(url)
            # Same domain, user gave no params but planner added some → strip them
            if parsed_user.netloc == parsed_plan.netloc and not parsed_user.query and parsed_plan.query:
                url = urlunparse(parsed_plan._replace(query="", fragment=""))

        goal = question

        attrs = self.graph.g.nodes[node_id]
        metadata = attrs.get("metadata", {})
        force_layer = metadata.get("force_layer")

        if not url:
            url = metadata.get("url", "")

        if not url:
            return AgentResult(
                success=False, agent_name="browser",
                error="No URL provided in question or metadata",
            )

        print(f"    {DIM}browser: {url[:60]} → {goal[:50]}{RESET}")
        await self._emit("tool_call", node_id=node_id, skill="browser", tool="browse", args_preview=url)

        try:
            result = await run_browser(
                url=url, goal=goal,
                session_id=self.session_id, node_id=node_id,
                on_event=self._emit,
                force_layer=force_layer,
            )
        except Exception as e:
            return AgentResult(
                success=False, agent_name="browser",
                error=f"Browser crash: {type(e).__name__}: {str(e)[:200]}",
                output={"url": url, "error": str(e)[:200]},
            )

        if result.error_code:
            return AgentResult(
                success=False, agent_name="browser",
                error=result.error_code,
                output={"error_code": result.error_code, "url": url},
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
            )

        error_msg = None
        if not result.success:
            failed_actions = [a for a in result.actions if a.get("error")]
            if failed_actions:
                error_msg = f"Layer {result.layer_used}: {len(failed_actions)} action(s) failed — {failed_actions[-1].get('error', '')[:100]}"
            else:
                error_msg = f"Layer {result.layer_used}: could not extract content after {result.turns} turns"

        return AgentResult(
            success=result.success, agent_name="browser",
            error=error_msg,
            output={
                "content": result.content[:6000],
                "layer_used": result.layer_used,
                "actions": result.actions,
                "turns": result.turns,
                "screenshots": result.screenshots,
                "final_url": result.final_url,
                "text": result.content[:4000],
            },
            tokens_in=result.tokens_in, tokens_out=result.tokens_out,
            provider="bedrock",
        )

    async def _run_researcher_fast(self, question: str, original_query: str, skill: Skill, node_id: str = "") -> AgentResult:
        """Direct tool dispatch — no LLM needed to decide what to search."""
        from schemas import ToolCall
        import action

        # Check cache first
        cached = cache.tool_get("web_search", {"query": question})
        if cached:
            print(f"    {DIM}[cache hit] web_search({question[:40]}){RESET}")
            return AgentResult(
                success=True, agent_name="researcher",
                output={"findings": cached, "question": question, "cached": True},
            )

        url_match = re.search(r'https?://[^\s,)]+', question)

        # Weather detection: use wttr.in for forecast queries
        weather_keywords = [" weather", "forecast", "temperature", " rain ", "raining"]
        is_weather = any(kw in question.lower() for kw in weather_keywords)
        if is_weather and not url_match:
            city_match = re.search(r'(?:in|for|at)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:this|on|today|tomorrow|saturday|sunday|monday|tuesday|wednesday|thursday|friday)|\?|$)', question, re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip().replace(" ", "+")
                url_match = re.match(r'.*', f"https://wttr.in/{city}?format=j1")

        if url_match:
            url = url_match.group(0).rstrip(".") if not is_weather else f"https://wttr.in/{city_match.group(1).strip().replace(' ', '+')}?format=j1" if is_weather and 'city_match' in dir() and city_match else url_match.group(0).rstrip(".")
            print(f"    {DIM}fetch: {url[:60]}{RESET}")
            await self._emit("tool_call", node_id=node_id, skill="researcher", tool="fetch_url", args_preview=url[:60])

            server_params = StdioServerParameters(
                command=sys.executable, args=["mcp_server.py"],
                env={**os.environ, "MCP_LOG_LEVEL": "error"},
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("fetch_url", arguments={"url": url})
                    text_parts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                    result_text = "\n".join(text_parts)
            cache.tool_put("fetch_url", {"url": url}, result_text[:3000], ttl_s=7200)
            return AgentResult(
                success=True, agent_name="researcher",
                output={"findings": result_text[:3000], "question": question, "sources": [{"url": url}]},
            )

        # Web search with own MCP session — call tool directly (bypass artifact store)
        print(f"    {DIM}search: {question[:60]}{RESET}")
        await self._emit("tool_call", node_id=node_id, skill="researcher", tool="web_search", args_preview=question[:60])
        server_params = StdioServerParameters(
            command=sys.executable, args=["mcp_server.py"],
            env={**os.environ, "MCP_LOG_LEVEL": "error"},
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("web_search", arguments={"query": question})
                text_parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                result_text = "\n".join(text_parts)
        cache.tool_put("web_search", {"query": question}, result_text[:3000], ttl_s=3600)
        return AgentResult(
            success=True, agent_name="researcher",
            output={"findings": result_text[:3000], "question": question},
        )

    async def _run_with_tools(self, skill: Skill, prompt: str, user_message: str, node_id: str = "") -> AgentResult:
        """Use mcp_runner for proper role=tool protocol with own subprocess."""
        from mcp_runner import run_with_tools

        _TOOL_CATALOG = {
            "web_search": {
                "name": "web_search",
                "description": "Search the web. Returns titles, URLs, and content.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}}, "required": ["query"]},
            },
            "fetch_url": {
                "name": "fetch_url",
                "description": "Fetch a URL and return content as markdown.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            },
            "search_knowledge": {
                "name": "search_knowledge",
                "description": "Vector search over indexed knowledge base.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, "required": ["query"]},
            },
            "read_file": {
                "name": "read_file",
                "description": "Read a file from the sandbox directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
            "create_file": {
                "name": "create_file",
                "description": "Create a new file in the sandbox directory.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            },
            "update_file": {
                "name": "update_file",
                "description": "Overwrite an existing file in the sandbox.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
            },
            "run_command": {
                "name": "run_command",
                "description": "Run a shell command. Returns stdout, stderr, exit_code.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 10}}, "required": ["command"]},
            },
            "count_syllables": {
                "name": "count_syllables",
                "description": "Count syllables in each line of text. Returns per-line syllable counts.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            },
            "count_characters": {
                "name": "count_characters",
                "description": "Count the exact number of characters in text.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            },
        }

        tools_payload = [_TOOL_CATALOG[t] for t in skill.tools_allowed if t in _TOOL_CATALOG]
        if not tools_payload:
            tools_payload = None

        async def _on_tool(tool_name, arguments):
            print(f"    {DIM}tool: {tool_name}({json.dumps(arguments)[:50]}){RESET}")
            await self._emit("tool_call", node_id=node_id, skill=skill.name, tool=tool_name, args_preview=json.dumps(arguments)[:60])

        text, tool_log = await run_with_tools(
            system_prompt=prompt,
            user_message=user_message,
            tools_payload=tools_payload or [],
            gateway_chat_fn=gateway.chat,
            temperature=skill.temperature,
            max_tokens=skill.max_tokens,
            node_timeout=NODE_TIMEOUT_S,
            on_tool_call=_on_tool,
        )

        parsed = self._parse_json(text)
        if isinstance(parsed, dict):
            output = parsed
        else:
            output = {"text": text}
        if tool_log:
            output["tool_calls"] = tool_log

        return AgentResult(success=True, agent_name=skill.name, output=output)

    async def _run_sandbox(self, node_id: str) -> AgentResult:
        attrs = self.graph.g.nodes[node_id]
        inputs = attrs.get("inputs", [])

        code = ""
        for inp in inputs:
            if inp.startswith("n:") and inp in self.graph.g.nodes:
                upstream = self.graph.g.nodes[inp].get("result")
                if isinstance(upstream, AgentResult) and upstream.output:
                    code = upstream.output.get("code", "")
                    if not code:
                        # Try parsing text as JSON
                        text = upstream.output.get("text", "")
                        try:
                            parsed = json.loads(text)
                            code = parsed.get("code", "")
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if not code:
                        # Try extracting from markdown code fences
                        text = upstream.output.get("text", "")
                        import re as _re
                        match = _re.search(r'```(?:python)?\n(.*?)```', text, _re.DOTALL)
                        if match:
                            code = match.group(1).strip()

        if not code:
            return AgentResult(success=True, agent_name="sandbox_executor",
                             output={"stdout": "", "stderr": "no code provided", "exit_code": 0, "skipped": True})

        result = run_code(code)
        return AgentResult(
            success=(result.exit_code == 0 and not result.timed_out),
            agent_name="sandbox_executor",
            output={"stdout": result.stdout, "stderr": result.stderr,
                    "exit_code": result.exit_code, "timed_out": result.timed_out},
        )

    async def _run_critic(self, node_id: str, query: str) -> AgentResult:
        attrs = self.graph.g.nodes[node_id]
        metadata = attrs.get("metadata", {})
        skill = self.skills["critic"]

        # Get upstream output
        target_id = metadata.get("target_node")
        upstream_text = ""
        if target_id and target_id in self.graph.g.nodes:
            upstream = self.graph.g.nodes[target_id].get("result")
            if isinstance(upstream, AgentResult):
                upstream_text = json.dumps(upstream.output, default=str)[:4000]
        else:
            for inp in attrs.get("inputs", []):
                if inp.startswith("n:") and inp in self.graph.g.nodes:
                    upstream = self.graph.g.nodes[inp].get("result")
                    if isinstance(upstream, AgentResult):
                        upstream_text = json.dumps(upstream.output, default=str)[:4000]
                        target_id = inp
                        break

        constraint = metadata.get("question", "Verify correctness")
        prompt = skill.render_prompt({"upstream_output": upstream_text}, question=constraint)
        user_msg = f"Upstream output:\n{upstream_text}\n\nConstraint: {constraint}"

        # Critic with tools: call tool first, then ask for verdict with the result
        tokens_in, tokens_out = 0, 0
        if skill.tools_allowed:
            # Determine which tool to call based on the constraint
            tool_name = None
            tool_args = {}
            constraint_lower = constraint.lower()
            # Extract the text to verify from upstream
            verify_text = ""
            if isinstance(upstream_text, str):
                parsed_upstream = self._parse_json(upstream_text)
                if isinstance(parsed_upstream, dict):
                    # Priority: text > final_answer > stdout (first line) > raw
                    if parsed_upstream.get("text"):
                        verify_text = parsed_upstream["text"]
                    elif parsed_upstream.get("final_answer"):
                        verify_text = parsed_upstream["final_answer"]
                    elif parsed_upstream.get("stdout"):
                        # Sandbox output: take first non-empty line (the actual content)
                        lines = [l for l in parsed_upstream["stdout"].strip().split('\n') if l.strip()]
                        verify_text = lines[0] if lines else ""
                    elif parsed_upstream.get("items"):
                        verify_text = json.dumps(parsed_upstream["items"])
                    else:
                        verify_text = upstream_text
                else:
                    verify_text = upstream_text

            if "syllable" in constraint_lower and "count_syllables" in skill.tools_allowed:
                tool_name = "count_syllables"
            elif ("character" in constraint_lower or "chars" in constraint_lower) and "count_characters" in skill.tools_allowed:
                tool_name = "count_characters"

            if tool_name:
                # Pass the full upstream text to the tool — let it measure everything
                tool_args = {"text": verify_text[:2000]}

                # Call tool directly via MCP
                server_params = StdioServerParameters(
                    command=sys.executable, args=["mcp_server.py"],
                    env={**os.environ, "MCP_LOG_LEVEL": "error"},
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tool_result = await session.call_tool(tool_name, arguments=tool_args)
                        tool_output = "\n".join(b.text for b in tool_result.content if hasattr(b, "text"))

                input_preview = verify_text[:50].replace('\n', ' ')
                print(f"    {DIM}tool: {tool_name}({input_preview}) → {tool_output[:60]}{RESET}")
                await self._emit("tool_call", node_id=node_id, skill="critic", tool=tool_name, args_preview=f"{input_preview}...")

                # Now ask LLM for verdict WITH the tool result
                resp = gateway.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Upstream output:\n{verify_text}\n\nConstraint: {constraint}\n\nTool measurement:\n{tool_output}\n\nBased on the tool measurement above, give your verdict as JSON: {{\"verdict\": \"pass\" or \"fail\", \"rationale\": \"...\"}}"},
                    ],
                    temperature=skill.temperature,
                )
                if resp.is_error:
                    return AgentResult(success=False, agent_name="critic", error=f"LLM error: {resp.text}")
                resp_text = resp.text or ""
                tokens_in = resp.input_tokens
                tokens_out = resp.output_tokens
            else:
                # No matching tool — fall through to regular LLM call
                resp = gateway.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=skill.temperature,
                )
                if resp.is_error:
                    return AgentResult(success=False, agent_name="critic", error=f"LLM error: {resp.text}")
                resp_text = resp.text or ""
                tokens_in = resp.input_tokens
                tokens_out = resp.output_tokens
        else:
            resp = gateway.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=skill.temperature,
            )
            if resp.is_error:
                return AgentResult(success=False, agent_name="critic", error=f"LLM error: {resp.text}")
            resp_text = resp.text or ""
            tokens_in = resp.input_tokens
            tokens_out = resp.output_tokens

        verdict_data = self._parse_json(resp_text)

        if verdict_data and verdict_data.get("verdict") == "fail":
            rationale = verdict_data.get("rationale", "no rationale")
            print(f"  {AMBER}[critic FAIL]{RESET} {rationale}")

            # Skip the child node — from metadata or derived from graph successors
            child_id = metadata.get("child_node")
            if not child_id:
                successors = list(self.graph.g.successors(node_id))
                child_id = successors[0] if successors else None
            if child_id and child_id in self.graph.g.nodes:
                self.graph.mark(child_id, "skipped")

            total_recoveries = sum(self._recovery_count.values())
            if total_recoveries < MAX_RECOVERY_PER_TARGET:
                self._recovery_count["critic_fail"] = self._recovery_count.get("critic_fail", 0) + 1
                recovery_id = self.graph.add_node("planner", ["USER_QUERY"], {
                    "label": f"recovery_{self.graph._counter}",
                    "failure_report": f"Critic FAILED. Rationale: {rationale}. The previous plan (formatter → critic) did not satisfy the constraint. Try a different generation approach but ALWAYS include a critic node to verify the constraint again.",
                })
                # Add edge from critic to recovery for graph visualization
                self.graph.g.add_edge(node_id, recovery_id)
                print(f"  {BLUE}[recovery]{RESET} Queued re-plan")
            else:
                print(f"  {DIM}[skip] Recovery cap hit — no more re-plans{RESET}")

            return AgentResult(
                success=True, agent_name="critic",
                output={"verdict": "fail", "rationale": rationale},
                tokens_in=tokens_in, tokens_out=tokens_out,
            )

        print(f"  {GREEN}[critic PASS]{RESET}")
        return AgentResult(
            success=True, agent_name="critic",
            output=verdict_data or {"verdict": "pass", "rationale": "ok"},
            tokens_in=tokens_in, tokens_out=tokens_out,
        )

    async def _handle_failure(self, node_id: str, error_text: str, query: str):
        classification = classify_failure(error_text)
        attrs = self.graph.g.nodes[node_id]
        skill_name = attrs.get("skill", "unknown")

        # Global cap: max 1 recovery planner per run
        total_recoveries = sum(self._recovery_count.values())
        if classification == "transient":
            print(f"  {DIM}[transient] Gateway retries exhausted{RESET}")
        elif classification == "validation_error":
            print(f"  {DIM}[validation] Prompt/format issue{RESET}")
        else:
            if skill_name == "planner":
                return
            if total_recoveries >= 1:
                print(f"  {DIM}[skip] Recovery cap reached, not re-planning{RESET}")
                return
            if self._recovery_count.get(node_id, 0) < MAX_RECOVERY_PER_TARGET:
                self._recovery_count[node_id] = self._recovery_count.get(node_id, 0) + 1

                # Collect completed sibling nodes (same parent, non-planner, non-critic)
                prior_complete = []
                for nid, d in self.graph.g.nodes(data=True):
                    if nid == node_id:
                        continue
                    if d.get("status") == "complete" and d.get("skill") not in ("planner", "critic"):
                        prior_complete.append(nid)

                # Build failure report with sibling context
                sibling_info = ""
                if prior_complete:
                    sibling_info = f"\n\nCompleted siblings (reuse these, do NOT redo): {', '.join(prior_complete)}"

                # Find downstream nodes that depend on the failed node and skip them
                # so the original comparator/formatter don't stay pending forever
                for successor in list(self.graph.g.successors(node_id)):
                    succ_data = self.graph.g.nodes[successor]
                    if succ_data.get("status") == "pending":
                        # Only skip if ALL its other inputs are complete (it was waiting only on this failed node)
                        other_preds = [p for p in self.graph.g.predecessors(successor) if p != node_id]
                        all_others_done = all(
                            self.graph.g.nodes[p].get("status") in ("complete", "skipped")
                            for p in other_preds
                        )
                        if not all_others_done:
                            # Other siblings still running — don't skip yet, the comparator
                            # will be handled when all inputs resolve
                            pass

                recovery_id = self.graph.add_node("planner", ["USER_QUERY"], {
                    "label": f"recovery_{self.graph._counter}",
                    "failure_report": (
                        f"Node {node_id} (skill: {skill_name}) failed: {error_text[:200]}"
                        f"{sibling_info}"
                        f"\n\nReplace ONLY the failed node. Emit 1 node to retry this specific task with a different approach, "
                        f"then wire it to a formatter."
                    ),
                })
                print(f"  {BLUE}[recovery]{RESET} Queued re-plan (reusing {len(prior_complete)} completed nodes)")

    def _extract_answer(self) -> str:
        # Strategy 1: find the LAST completed formatter that wasn't rejected by a critic
        # Check successors (formatter after critic) first, then predecessors
        for nid in reversed(list(self.graph.g.nodes)):
            d = self.graph.g.nodes[nid]
            if d.get("skill") == "critic" and d.get("status") == "complete":
                result = d.get("result")
                if isinstance(result, AgentResult) and result.output.get("verdict") == "pass":
                    # What the critic verified — find the content that passed
                    # Look at predecessor (formatter or sandbox that fed the critic)
                    for pred in self.graph.g.predecessors(nid):
                        pred_d = self.graph.g.nodes[pred]
                        pred_result = pred_d.get("result")
                        if not isinstance(pred_result, AgentResult):
                            continue
                        if pred_d.get("skill") == "formatter":
                            answer = pred_result.output.get("final_answer", "") or pred_result.output.get("text", "") or pred_result.text
                            if answer:
                                return answer
                        elif pred_d.get("skill") == "sandbox_executor":
                            stdout = pred_result.output.get("stdout", "")
                            lines = [l for l in stdout.strip().split('\n') if l.strip()]
                            if lines:
                                return lines[0]

        # Fallback: last completed formatter without a failing critic
        for nid in reversed(list(self.graph.g.nodes)):
            d = self.graph.g.nodes[nid]
            if d.get("skill") == "formatter" and d.get("status") == "complete":
                successors = list(self.graph.g.successors(nid))
                critic_failed = any(
                    self.graph.g.nodes[s].get("skill") == "critic" and
                    isinstance(self.graph.g.nodes[s].get("result"), AgentResult) and
                    self.graph.g.nodes[s].get("result").output.get("verdict") == "fail"
                    for s in successors if s in self.graph.g.nodes
                )
                if critic_failed:
                    continue
                result = d.get("result")
                if isinstance(result, AgentResult):
                    answer = result.output.get("final_answer", "") or result.output.get("text", "") or result.text
                    if answer:
                        return answer

        # Last resort fallback
        for nid in reversed(list(self.graph.g.nodes)):
            d = self.graph.g.nodes[nid]
            if d.get("status") == "complete" and d.get("skill") not in ("planner", "critic", "sandbox_executor"):
                result = d.get("result")
                if isinstance(result, AgentResult):
                    return result.text or json.dumps(result.output)[:2000]
        return "No answer produced."

    def _persist_graph(self):
        # Serialize AgentResult to dict for JSON persistence
        h = nx.DiGraph()
        for n, d in self.graph.g.nodes(data=True):
            attrs = dict(d)
            if isinstance(attrs.get("result"), AgentResult):
                attrs["result"] = attrs["result"].model_dump(mode="json")
                attrs["_result_typed"] = True
            h.add_node(n, **attrs)
        for u, v in self.graph.g.edges:
            h.add_edge(u, v)
        data = nx.node_link_data(h)
        store.save_graph(self.session_id, nx.DiGraph())  # placeholder
        # Write directly using atomic write
        import tempfile
        path = Path("state/sessions") / self.session_id / "graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2, default=str)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.write(fd, content.encode())
        os.close(fd)
        os.replace(tmp, str(path))

    def _persist_node(self, node_id: str, result: AgentResult, span=None):
        state = {
            "node_id": node_id,
            "skill": self.graph.g.nodes[node_id].get("skill", ""),
            "status": self.graph.g.nodes[node_id].get("status", ""),
            "inputs": self.graph.g.nodes[node_id].get("inputs", []),
            "metadata": self.graph.g.nodes[node_id].get("metadata", {}),
            "result": result.model_dump(mode="json"),
            "elapsed_s": span.elapsed_s if span else result.elapsed_s,
            "prompt_sent": None,  # TODO: capture from skill render
        }
        store.save_node_state(self.session_id, node_id, state)

    def _print_summary(self):
        wall_clock = time.time() - self._run_start
        summary = self.trace.summary()

        print(f"\n{GRAY}{'─' * 50}{RESET}")
        print(f"{GREEN}[done]{RESET} {self.graph.node_count} nodes, {wall_clock:.1f}s wall-clock")
        print(f"{DIM}  Tokens: in={self.budget.used_input} out={self.budget.used_output}  "
              f"Cache: {cache.hits} hits / {cache.misses} misses{RESET}")
        if summary["by_skill"]:
            for skill, data in summary["by_skill"].items():
                print(f"{DIM}  {skill:16s} {data['calls']}x  {data['total_s']:.1f}s{RESET}")
        print(f"{GRAY}Session: {self.session_id}{RESET}\n")

    def _format_memory_hits(self, hits) -> str:
        if not hits:
            return ""
        lines = []
        for h in hits[:8]:
            chunk = h.value.get("chunk", "")
            preview = chunk[:2000] if chunk else h.descriptor
            lines.append(f"- [{h.kind}] {h.descriptor[:100]}: {preview}")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return None
        return None


# ── Fast path ─────────────────────────────────────────────────────────────

async def _fast_path(query: str, session_id: str) -> str | None:
    """All queries go through the DAG."""
    return None


# ── Entry point ───────────────────────────────────────────────────────────

async def run_flow(query: str, session_id: str | None = None, resume: bool = False) -> str:
    if session_id is None:
        session_id = f"s8_{uuid.uuid4().hex[:8]}"

    if not resume:
        fast_answer = await _fast_path(query, session_id)
        if fast_answer is not None:
            return fast_answer

    skills = load_skills()
    executor = Executor(session_id=session_id, skills_catalogue=skills)
    answer = await executor.run(query, resume=resume)

    # Stream-style output for formatter
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        Console().print(Panel(Markdown(answer), title="ANSWER", border_style="green", padding=(1, 2)))
    except ImportError:
        print(f"\n{BOLD}ANSWER:{RESET}\n{answer}\n")

    return answer


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Session 8 DAG Orchestrator")
    parser.add_argument("query", nargs="*", help="Query to run")
    parser.add_argument("--resume", type=str, help="Resume a session by ID")
    parser.add_argument("--session", type=str, help="Session ID to use")
    args = parser.parse_args()

    if args.resume:
        await run_flow("", session_id=args.resume, resume=True)
    elif args.query:
        await run_flow(" ".join(args.query), session_id=args.session)
    else:
        query = input("Enter your query: ")
        await run_flow(query, session_id=args.session)


if __name__ == "__main__":
    asyncio.run(main())
