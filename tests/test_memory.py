"""Tests for FileMemory."""

import tempfile

from agentos.providers.memory import FileMemory


def _make_memory():
    d = tempfile.mkdtemp()
    return FileMemory(path=d), d


def test_store_and_retrieve():
    mem, _ = _make_memory()
    mem.store("api_key", "use env vars for secrets")
    results = mem.retrieve("api_key")
    assert len(results) == 1
    assert results[0].key == "api_key"
    assert results[0].content == "use env vars for secrets"


def test_store_updates_existing_key():
    mem, _ = _make_memory()
    mem.store("k", "v1")
    mem.store("k", "v2")
    entries = mem.list_all()
    assert len(entries) == 1
    assert entries[0].content == "v2"


def test_retrieve_keyword_search():
    mem, _ = _make_memory()
    mem.store("db", "postgres connection on port 5432")
    mem.store("cache", "redis on port 6379")
    results = mem.retrieve("postgres")
    assert len(results) == 1
    assert results[0].key == "db"


def test_retrieve_limit():
    mem, _ = _make_memory()
    for i in range(10):
        mem.store(f"key_{i}", f"content {i}")
    results = mem.retrieve("content", limit=3)
    assert len(results) == 3


def test_forget():
    mem, _ = _make_memory()
    mem.store("temp", "delete me")
    assert mem.forget("temp") is True
    assert mem.list_all() == []


def test_forget_nonexistent():
    mem, _ = _make_memory()
    assert mem.forget("nope") is False


def test_list_all_empty():
    mem, _ = _make_memory()
    assert mem.list_all() == []


def test_retrieve_no_match():
    mem, _ = _make_memory()
    mem.store("a", "b")
    assert mem.retrieve("zzz") == []


def test_metadata_search():
    mem, _ = _make_memory()
    mem.store("item", "some content", metadata={"tag": "important"})
    results = mem.retrieve("important")
    assert len(results) == 1
