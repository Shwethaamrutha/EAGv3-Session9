"""Persistence layer — atomic graph and node state writes to disk."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import networkx as nx


class SessionLoadError(Exception):
    pass


class SessionStore:
    def __init__(self, base_dir: str = "state/sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def save_query(self, session_id: str, query: str):
        path = self.session_path(session_id)
        path.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path / "query.txt", query)

    def load_query(self, session_id: str) -> str:
        path = self.session_path(session_id) / "query.txt"
        if not path.exists():
            raise SessionLoadError(f"query.txt not found for session {session_id}")
        return path.read_text()

    def save_graph(self, session_id: str, graph: nx.DiGraph):
        path = self.session_path(session_id)
        path.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(graph)
        for node in data.get("nodes", []):
            if "result" in node and node["result"] is not None:
                node["_result_typed"] = True
        self._atomic_write(path / "graph.json", json.dumps(data, indent=2, default=str))

    def load_graph(self, session_id: str) -> nx.DiGraph:
        path = self.session_path(session_id) / "graph.json"
        if not path.exists():
            raise SessionLoadError(f"graph.json not found for session {session_id}")
        try:
            data = json.loads(path.read_text())
            graph = nx.node_link_graph(data)
            return graph
        except Exception as e:
            raise SessionLoadError(f"Failed to load graph for {session_id}: {e}")

    def save_node_state(self, session_id: str, node_id: str, state: dict):
        path = self.session_path(session_id) / "nodes"
        path.mkdir(parents=True, exist_ok=True)
        safe_id = node_id.replace(":", "_").replace("/", "_")
        self._atomic_write(path / f"{safe_id}.json", json.dumps(state, indent=2, default=str))

    def load_node_state(self, session_id: str, node_id: str) -> dict | None:
        path = self.session_path(session_id) / "nodes"
        safe_id = node_id.replace(":", "_").replace("/", "_")
        file_path = path / f"{safe_id}.json"
        if not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text())
        except Exception:
            return None

    def list_sessions(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def _atomic_write(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp_path, str(path))
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


store = SessionStore()
