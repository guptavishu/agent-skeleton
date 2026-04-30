"""Tests for VectorMemory chunking and retrieval logic (no network calls)."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from vector_memory import VectorMemory, chunk_text

import numpy as np


class TestChunking:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world", chunk_size=100) == ["hello world"]

    def test_splits_at_newlines(self):
        text = "line1\nline2\nline3\nline4\nline5\nline6"
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) >= 2
        assert all(len(c) <= 20 for c in chunks)

    def test_overlap(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 3
        # verify overlap exists
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i][-10:]
            assert end_of_current in chunks[i + 1] or chunks[i + 1][:10] in chunks[i]

    def test_empty_chunks_filtered(self):
        text = "hello\n\n\n\nworld"
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        assert all(c.strip() for c in chunks)


class TestVectorMemory:
    def _make_memory(self, tmp_path):
        """Create a VectorMemory with mocked embedding calls."""
        dim = 8

        def fake_embed(text):
            np.random.seed(hash(text) % 2**31)
            vec = np.random.randn(dim).astype(np.float32)
            return vec / np.linalg.norm(vec)

        mem = VectorMemory.__new__(VectorMemory)
        mem.path = Path(tmp_path)
        mem.path.mkdir(parents=True, exist_ok=True)
        mem.model = "test"
        mem.base_url = "http://localhost:11434"
        mem.chunk_size = 512
        mem.chunk_overlap = 64
        mem._entries = []
        mem._embeddings = []
        mem._embed = fake_embed
        return mem

    def test_store_and_retrieve(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.store("greeting", "hello world")
        mem.store("farewell", "goodbye world")
        results = mem.retrieve("hello", limit=1)
        assert len(results) == 1

    def test_retrieve_returns_limit(self, tmp_path):
        mem = self._make_memory(tmp_path)
        for i in range(10):
            mem.store(f"doc_{i}", f"document number {i} with content")
        results = mem.retrieve("document", limit=3)
        assert len(results) == 3

    def test_forget(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.store("temp", "temporary data")
        assert mem.forget("temp")
        assert len(mem.list_all()) == 0
        assert not mem.forget("nonexistent")

    def test_ingest_text(self, tmp_path):
        mem = self._make_memory(tmp_path)
        text = "paragraph one\n" * 50 + "paragraph two\n" * 50
        entries = mem.ingest_text(text, source="test.md")
        assert len(entries) >= 2
        assert all("test.md" in e.key for e in entries)
        assert all(e.metadata["source"] == "test.md" for e in entries)

    def test_ingest_file(self, tmp_path):
        mem = self._make_memory(tmp_path)
        doc = tmp_path / "doc.md"
        doc.write_text("This is a test document with some content.\n" * 20)
        entries = mem.ingest_file(str(doc))
        assert len(entries) >= 1

    def test_persistence(self, tmp_path):
        mem = self._make_memory(tmp_path)
        mem.store("key1", "value1")
        mem.store("key2", "value2")

        # create a new instance pointing to same dir
        mem2 = self._make_memory(tmp_path)
        mem2._load_index()
        assert len(mem2.list_all()) == 2

    def test_empty_retrieve(self, tmp_path):
        mem = self._make_memory(tmp_path)
        results = mem.retrieve("anything")
        assert results == []
