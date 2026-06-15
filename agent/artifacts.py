"""Content-addressable artifact store — in-memory with optional disk persistence."""
from __future__ import annotations

import hashlib
from pathlib import Path

from config import settings
from logger import get_logger
from schemas import Artifact

log = get_logger("artifacts")


class ArtifactStore:
    """Stores artifacts in memory. Optionally persists to disk for cross-run access."""

    def __init__(self):
        self._blobs: dict[str, bytes] = {}
        self._meta: dict[str, Artifact] = {}

    def put(self, blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
        sha = hashlib.sha256(blob).hexdigest()[:16]
        art_id = f"art:{sha}"

        if art_id not in self._blobs:
            self._blobs[art_id] = blob
            self._meta[art_id] = Artifact(
                id=art_id,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor[:200],
            )

        return art_id

    def get_bytes(self, artifact_id: str) -> bytes:
        return self._blobs[artifact_id]

    def get_meta(self, artifact_id: str) -> Artifact:
        return self._meta[artifact_id]

    def exists(self, artifact_id: str) -> bool:
        return artifact_id in self._blobs

    def clear(self):
        """Clear all artifacts from memory."""
        self._blobs.clear()
        self._meta.clear()

    def cleanup(self, max_age_hours: int = 72):
        """No-op for in-memory store. Artifacts are cleared on process exit."""
        pass


artifact_store = ArtifactStore()
