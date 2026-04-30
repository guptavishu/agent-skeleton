"""VectorMemory: embeddings-based RAG memory using Ollama + numpy.

Implements the nerve Memory protocol. Stores document chunks with vector
embeddings for cosine-similarity retrieval.

Usage:
    from vector_memory import VectorMemory

    mem = VectorMemory()
    mem.ingest_file("docs/architecture.md")
    mem.ingest_file("docs/api.md")

    results = mem.retrieve("how does authentication work?", limit=5)
    for entry in results:
        print(entry.key, entry.content[:100])

Requirements:
    pip install httpx numpy
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from nerve import MemoryEntry

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_STORE_DIR = Path.home() / ".nerve" / "vector_memory"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count, breaking at newlines when possible."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start + chunk_size // 2:
                end = nl + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


class VectorMemory:
    """Embedding-backed memory with cosine-similarity retrieval.

    Persists chunks + embeddings to disk as JSON so you don't re-embed on restart.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.path = Path(path or DEFAULT_STORE_DIR)
        self.path.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._entries: list[MemoryEntry] = []
        self._embeddings: list[np.ndarray] = []
        self._load_index()

    def _embed(self, text: str) -> np.ndarray:
        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(t) for t in texts]

    def store(self, key: str, content: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(
            key=key,
            content=content,
            metadata=metadata or {},
        )
        embedding = self._embed(content)
        self._entries.append(entry)
        self._embeddings.append(embedding)
        self._save_entry(entry, embedding)
        return entry

    def ingest_file(self, file_path: str, metadata: dict | None = None) -> list[MemoryEntry]:
        """Read a file, chunk it, embed each chunk, and store."""
        p = Path(file_path)
        text = p.read_text()
        return self.ingest_text(text, source=str(p), metadata=metadata)

    def ingest_directory(self, dir_path: str, glob: str = "**/*.md", metadata: dict | None = None) -> list[MemoryEntry]:
        """Recursively ingest files matching a glob pattern."""
        entries = []
        for f in sorted(Path(dir_path).glob(glob)):
            if f.is_file():
                entries.extend(self.ingest_file(str(f), metadata=metadata))
        return entries

    def ingest_text(self, text: str, source: str = "", metadata: dict | None = None) -> list[MemoryEntry]:
        """Chunk text, embed, and store all chunks."""
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        embeddings = self._embed_batch(chunks)
        meta = metadata or {}
        entries = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            entry = MemoryEntry(
                key=f"{source}:chunk_{i}" if source else f"chunk_{i}",
                content=chunk,
                metadata={**meta, "source": source, "chunk_index": i, "total_chunks": len(chunks)},
            )
            self._entries.append(entry)
            self._embeddings.append(emb)
            self._save_entry(entry, emb)
            entries.append(entry)

        return entries

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []

        query_emb = self._embed(query)
        matrix = np.stack(self._embeddings)
        # cosine similarity
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1, norms)
        scores = matrix @ query_emb / norms

        top_indices = np.argsort(scores)[::-1][:limit]
        return [self._entries[i] for i in top_indices]

    def forget(self, key: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.key == key:
                self._entries.pop(i)
                self._embeddings.pop(i)
                entry_path = self.path / f"{entry.id}.json"
                entry_path.unlink(missing_ok=True)
                return True
        return False

    def list_all(self) -> list[MemoryEntry]:
        return list(self._entries)

    def _save_entry(self, entry: MemoryEntry, embedding: np.ndarray) -> None:
        data = {
            "id": entry.id,
            "key": entry.key,
            "content": entry.content,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp,
            "embedding": embedding.tolist(),
        }
        (self.path / f"{entry.id}.json").write_text(json.dumps(data))

    def _load_index(self) -> None:
        for f in sorted(self.path.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                entry = MemoryEntry(
                    key=data["key"],
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp", 0),
                    id=data["id"],
                )
                embedding = np.array(data["embedding"], dtype=np.float32)
                self._entries.append(entry)
                self._embeddings.append(embedding)
            except (json.JSONDecodeError, KeyError):
                pass
