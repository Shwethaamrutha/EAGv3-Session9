"""Result caching — tool results and planner outputs.

TTL-based in-memory cache keyed by (tool_name, args_hash) or (query_hash).
Saves redundant API calls for identical searches within a session.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    value: str
    created_at: float
    ttl_s: float

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_s


class ResultCache:
    def __init__(self, default_ttl_s: float = 3600.0):
        self._store: dict[str, CacheEntry] = {}
        self.default_ttl_s = default_ttl_s
        self.hits = 0
        self.misses = 0

    def _key(self, namespace: str, *args) -> str:
        raw = json.dumps([namespace, *args], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, namespace: str, *args) -> str | None:
        key = self._key(namespace, *args)
        entry = self._store.get(key)
        if entry is None or entry.expired:
            self.misses += 1
            if entry and entry.expired:
                del self._store[key]
            return None
        self.hits += 1
        return entry.value

    def put(self, value: str, namespace: str, *args, ttl_s: float | None = None):
        key = self._key(namespace, *args)
        self._store[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_s=ttl_s or self.default_ttl_s,
        )

    def tool_get(self, tool_name: str, arguments: dict) -> str | None:
        return self.get("tool", tool_name, arguments)

    def tool_put(self, tool_name: str, arguments: dict, result: str, ttl_s: float | None = None):
        self.put(result, "tool", tool_name, arguments, ttl_s=ttl_s)

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


cache = ResultCache()
