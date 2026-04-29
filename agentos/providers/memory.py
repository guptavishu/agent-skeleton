from __future__ import annotations

import json
from pathlib import Path

from ..types import MemoryEntry


class FileMemory:
    """JSON-file-backed memory. Each entry is a separate .json file.

    Default location: ~/.agentos/memory/
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".agentos" / "memory")
        self.path.mkdir(parents=True, exist_ok=True)

    def _entry_path(self, entry_id: str) -> Path:
        return self.path / f"{entry_id}.json"

    def store(self, key: str, content: str, metadata: dict | None = None) -> MemoryEntry:
        for existing in self.list_all():
            if existing.key == key:
                existing.content = content
                existing.metadata = metadata or {}
                self._write(existing)
                return existing

        entry = MemoryEntry(key=key, content=content, metadata=metadata or {})
        self._write(entry)
        return entry

    def _write(self, entry: MemoryEntry) -> None:
        data = {
            "id": entry.id,
            "key": entry.key,
            "content": entry.content,
            "metadata": entry.metadata,
            "timestamp": entry.timestamp,
        }
        self._entry_path(entry.id).write_text(json.dumps(data, indent=2))

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        query_lower = query.lower()
        results = []
        for entry in self.list_all():
            score = 0
            if query_lower in entry.key.lower():
                score += 2
            if query_lower in entry.content.lower():
                score += 1
            if any(query_lower in str(v).lower() for v in entry.metadata.values()):
                score += 1
            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [entry for _, entry in results[:limit]]

    def forget(self, key: str) -> bool:
        for entry in self.list_all():
            if entry.key == key:
                path = self._entry_path(entry.id)
                if path.exists():
                    path.unlink()
                    return True
        return False

    def list_all(self) -> list[MemoryEntry]:
        entries = []
        for f in self.path.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                entries.append(
                    MemoryEntry(
                        key=data["key"],
                        content=data["content"],
                        metadata=data.get("metadata", {}),
                        timestamp=data.get("timestamp", 0),
                        id=data["id"],
                    )
                )
            except (json.JSONDecodeError, KeyError):
                pass
        entries.sort(key=lambda e: -e.timestamp)
        return entries
