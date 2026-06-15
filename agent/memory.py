"""Memory service — typed store with FAISS vector search, keyword fallback, file locking, dedup, and eviction."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from filelock import FileLock

from config import settings
from llm_gateway import gateway
from logger import get_logger
from schemas import MemoryItem

log = get_logger("memory")

MEMORY_FILE = Path(settings.memory_file)
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
MEMORY_LOCK = FileLock(str(MEMORY_FILE) + ".lock", timeout=10)

FAISS_INDEX_FILE = Path("state/index.faiss")
FAISS_IDS_FILE = Path("state/index_ids.json")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "if", "or", "and", "but", "that", "this", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "these", "those", "am", "about", "up", "also", "tell",
    "give", "find", "get", "check", "make", "let", "know",
}

CLASSIFY_PROMPT = """You are a memory classifier. Given the user's text, extract a structured memory item.

Respond in JSON with exactly these fields:
- "kind": one of "fact", "preference", "tool_outcome", "scratchpad"
- "keywords": list of 3-8 relevant keywords (lowercase, no stopwords)
- "descriptor": one short human-readable line summarizing the item
- "value": a structured dict capturing the key information

Rules:
- "fact" = a durable observed truth (dates, names, relationships, locations)
- "preference" = a user-stated or inferred preference
- "tool_outcome" = record of a tool dispatch (only used internally)
- "scratchpad" = a temporary working note

Text to classify:
{text}
"""


def _try_embed(text: str, *, task_type: str = "retrieval_document") -> list[float] | None:
    try:
        return gateway.embed(text, task_type=task_type)
    except Exception as e:
        log.debug("embed_failed", error=str(e)[:80])
        return None


def _new_id(prefix: str = "mem") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class MemoryService:
    def __init__(self):
        self._items: list[MemoryItem] = []
        self._load()

    @property
    def item_count(self) -> int:
        """Public read-only access to the number of stored memory items."""
        return len(self._items)

    def clear(self) -> None:
        """Remove all items from memory and FAISS index."""
        self._items.clear()
        self._save()
        if FAISS_INDEX_FILE.exists():
            FAISS_INDEX_FILE.unlink()
        if FAISS_IDS_FILE.exists():
            FAISS_IDS_FILE.unlink()

    def _load(self):
        with MEMORY_LOCK:
            if MEMORY_FILE.exists() and MEMORY_FILE.stat().st_size > 0:
                try:
                    data = json.loads(MEMORY_FILE.read_text())
                    self._items = [MemoryItem.model_validate(d) for d in data]
                except (json.JSONDecodeError, OSError) as e:
                    log.error("memory_load_failed", error=str(e))
                    self._items = []

    def _save(self):
        self._evict_if_needed()
        with MEMORY_LOCK:
            MEMORY_FILE.write_text(
                json.dumps([item.model_dump(mode="json") for item in self._items], indent=2)
            )

    def _tokenize(self, text: str) -> set[str]:
        tokens = set()
        for word in text.lower().split():
            word = word.strip(".,!?;:'\"()-[]{}/@#$%^&*")
            if word and word not in STOPWORDS and len(word) > 1:
                tokens.add(word)
        return tokens

    def _is_duplicate(self, new_item: MemoryItem) -> bool:
        new_tokens = set(new_item.keywords)
        if not new_tokens:
            return False
        for existing in self._items:
            if existing.kind != new_item.kind:
                continue
            existing_tokens = set(existing.keywords)
            if not existing_tokens:
                continue
            union = new_tokens | existing_tokens
            overlap = len(new_tokens & existing_tokens) / max(len(union), 1)
            if overlap >= settings.memory_dedup_threshold:
                return True
        return False

    def _evict_if_needed(self):
        if len(self._items) <= settings.memory_max_items:
            return
        evictable = [i for i in self._items if i.kind in ("scratchpad", "tool_outcome")]
        evictable.sort(key=lambda x: x.created_at)
        to_remove = len(self._items) - settings.memory_max_items
        removed = 0
        for item in evictable:
            if removed >= to_remove:
                break
            self._items.remove(item)
            removed += 1
        if removed:
            log.info("memory_evicted", count=removed)

    def _load_faiss_index(self):
        """Load FAISS index and parallel ID list from disk."""
        import faiss
        if FAISS_INDEX_FILE.exists() and FAISS_IDS_FILE.exists():
            try:
                index = faiss.read_index(str(FAISS_INDEX_FILE))
                ids = json.loads(FAISS_IDS_FILE.read_text())
                return index, ids
            except Exception as e:
                log.debug("faiss_load_failed", error=str(e)[:80])
        return None, []

    def _save_faiss_index(self, index, ids: list[str]):
        """Persist FAISS index and parallel ID list to disk."""
        import faiss
        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_FILE))
        FAISS_IDS_FILE.write_text(json.dumps(ids))

    def _append_to_faiss(self, item_id: str, embedding: list[float]):
        """Append a single vector to the FAISS index."""
        import faiss
        index, ids = self._load_faiss_index()
        if index is None:
            from llm_gateway.gateway import EMBED_DIMENSION
            index = faiss.IndexFlatIP(EMBED_DIMENSION)
            ids = []
        vec = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vec)
        index.add(vec)
        ids.append(item_id)
        self._save_faiss_index(index, ids)

    _query_embed_cache: dict = {}

    def _vector_search(self, query: str, k: int, kinds: list[str] | None = None) -> list[MemoryItem]:
        """Run vector search over FAISS index, return matching MemoryItems."""
        import faiss
        index, ids = self._load_faiss_index()
        if index is None or index.ntotal == 0:
            return []
        # Cache query embeddings — same query doesn't need re-embedding
        if query in self._query_embed_cache:
            embedding = self._query_embed_cache[query]
        else:
            embedding = _try_embed(query, task_type="retrieval_query")
            if embedding is not None:
                self._query_embed_cache[query] = embedding
        if embedding is None:
            return []
        vec = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vec)
        actual_k = min(k, index.ntotal)
        scores, positions = index.search(vec, actual_k)
        # Reload items from disk for cross-process consistency
        self._load()
        id_to_item = {item.id: item for item in self._items}
        query_tokens = self._tokenize(query)

        scored_results = []
        for i in range(actual_k):
            pos = int(positions[0][i])
            score = float(scores[0][i])
            if pos < 0 or score < 0.3:
                continue
            if pos < len(ids):
                item_id = ids[pos]
                item = id_to_item.get(item_id)
                if item and (kinds is None or item.kind in kinds):
                    # Hybrid boost: add keyword overlap signal to vector score
                    chunk_text = item.value.get("chunk", item.descriptor)
                    chunk_tokens = self._tokenize(chunk_text)
                    overlap = len(query_tokens & chunk_tokens)
                    boosted_score = score + (overlap * 0.02)
                    scored_results.append((boosted_score, item))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results]

    _last_read_source: str = ""

    def read(self, query: str, history: list[dict], kinds: list[str] | None = None, top_k: int = 5) -> list[MemoryItem]:
        # Vector-first path — pure cosine similarity ranking
        vector_results = self._vector_search(query, top_k * 2, kinds=kinds)
        if vector_results:
            self._last_read_source = "faiss"
            # If indexed fact chunks exist, prioritize them over tool_outcomes
            facts = [r for r in vector_results if r.kind == "fact"]
            others = [r for r in vector_results if r.kind != "fact"]
            if facts:
                return (facts + others)[:top_k]
            return vector_results[:top_k]

        # Keyword fallback
        self._last_read_source = "keyword"
        self._load()
        query_tokens = self._tokenize(query)
        for event in history[-5:]:
            if "result_descriptor" in event:
                query_tokens |= self._tokenize(event["result_descriptor"])
            if "tool" in event:
                query_tokens |= self._tokenize(event["tool"])

        scored: list[tuple[float, MemoryItem]] = []
        for item in self._items:
            if kinds and item.kind not in kinds:
                continue
            item_tokens = set(item.keywords) | self._tokenize(item.descriptor)
            overlap = len(query_tokens & item_tokens)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def filter(self, kinds: list[str] | None = None, goal_id: str | None = None, recent: int | None = None) -> list[MemoryItem]:
        results = self._items
        if kinds:
            results = [i for i in results if i.kind in kinds]
        if goal_id:
            results = [i for i in results if i.goal_id == goal_id]
        if recent:
            results = results[-recent:]
        return results

    def remember(self, raw_text: str, source: str, run_id: str, goal_id: str | None = None) -> MemoryItem | None:
        # Reload from disk to pick up cross-process writes
        self._load()
        # Skip queries that don't contain personal facts worth remembering
        lower = raw_text.lower()
        stripped = raw_text.strip().rstrip(".")

        # Questions without personal info
        if stripped.endswith("?") and not any(kw in lower for kw in ["remember", "my ", "i ", "our "]):
            return None

        # Instructional commands (fetch/search/find/tell me) with no personal facts
        action_verbs = ["fetch", "search", "find", "get", "look up", "show me", "tell me", "list", "compare"]
        has_personal = any(kw in lower for kw in ["my ", "i am", "i'm", "our ", "remember", "birthday", "prefer"])
        if any(lower.startswith(v) for v in action_verbs) and not has_personal:
            return None

        prompt = CLASSIFY_PROMPT.format(text=raw_text)
        response_schema = {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fact", "preference", "tool_outcome", "scratchpad"]},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "descriptor": {"type": "string"},
                "value": {"type": "object"},
            },
            "required": ["kind", "keywords", "descriptor", "value"],
        }

        resp = gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"schema": response_schema},
            auto_route="memory",
            temperature=0.3,
        )

        if resp.is_error:
            log.warning("memory_classify_failed", source=source)
            return None

        if resp.parsed:
            data = resp.parsed
            kind = data.get("kind", "scratchpad")
            descriptor = data.get("descriptor", raw_text[:100])
            embedding = None
            if kind in ("fact", "preference", "tool_outcome"):
                embedding = _try_embed(descriptor, task_type="retrieval_document")
            item = MemoryItem(
                id=_new_id("mem"),
                kind=kind,
                keywords=[k.lower() for k in data.get("keywords", [])],
                descriptor=descriptor,
                value=data.get("value", {"raw": raw_text}),
                embedding=embedding,
                source=source,
                run_id=run_id,
                goal_id=goal_id,
                confidence=0.9,
            )
            if self._is_duplicate(item):
                log.debug("memory_dedup_skipped", descriptor=item.descriptor)
                return None
            self._items.append(item)
            self._save()
            # Only add_fact writes to FAISS — remember() items stay in keyword search only
            log.info("memory_stored", kind=item.kind, descriptor=item.descriptor[:60])
            return item
        return None

    def record_outcome(
        self,
        tool_call,
        result_text: str,
        artifact_id: str | None,
        run_id: str,
        goal_id: str | None,
    ) -> MemoryItem:
        # Reload from disk to pick up cross-process writes (e.g. from MCP subprocess)
        self._load()

        keywords = [tool_call.name.lower()]
        for k, v in tool_call.arguments.items():
            keywords.extend(self._tokenize(str(v)))
        keywords = list(set(keywords))[:10]

        descriptor = f"{tool_call.name}({json.dumps(tool_call.arguments)[:80]}) → {result_text[:150]}"
        embedding = _try_embed(descriptor, task_type="retrieval_document")

        item = MemoryItem(
            id=_new_id("mem"),
            kind="tool_outcome",
            keywords=keywords,
            descriptor=descriptor,
            value={
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
                "result_preview": result_text[:8000],
            },
            embedding=embedding,
            artifact_id=artifact_id,
            source=f"action:{tool_call.name}",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
        )
        if self._is_duplicate(item):
            log.debug("memory_dedup_outcome", tool=tool_call.name)
            return item
        self._items.append(item)
        self._save()
        # Don't add tool_outcomes to FAISS — keeps vector index clean for fact chunks only
        return item

    def add_fact(
        self,
        descriptor: str,
        *,
        value: dict,
        keywords: list[str],
        source: str,
        run_id: str,
        goal_id: str | None = None,
    ) -> MemoryItem:
        """Write a fact item directly (used by index_document for chunks)."""
        # Embed the full chunk text (not just the descriptor) for better semantic retrieval
        embed_text = value.get("chunk", descriptor)
        embedding = _try_embed(embed_text, task_type="retrieval_document")
        item = MemoryItem(
            id=_new_id("mem"),
            kind="fact",
            keywords=[k.lower() for k in keywords],
            descriptor=descriptor,
            value=value,
            embedding=embedding,
            source=source,
            run_id=run_id,
            goal_id=goal_id,
        )
        self._items.append(item)
        self._save()
        if embedding:
            self._append_to_faiss(item.id, embedding)
        log.info("fact_stored", descriptor=descriptor[:60])
        return item


memory = MemoryService()
