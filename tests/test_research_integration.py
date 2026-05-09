"""Integration tests for the research agent.

These hit real Ollama endpoints and require:
    - Ollama running locally
    - An embedding model pulled (e.g. ollama pull nomic-embed-text)
    - An LLM model pulled (e.g. qwen2.5-coder:32b)

Run manually:
    python -m pytest tests/test_research_integration.py -v -x

Skip reason: these are slow (~10-60s each) and need local infra.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

try:
    import httpx
    import numpy as np
    _resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
    _models = [m["name"] for m in _resp.json().get("models", [])]
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False
    _models = []

HAS_EMBED_MODEL = any("nomic-embed-text" in m for m in _models)
HAS_LLM = any("qwen" in m or "llama" in m or "mistral" in m for m in _models)

requires_ollama = pytest.mark.skipif(not HAS_OLLAMA, reason="Ollama not running")
requires_embeddings = pytest.mark.skipif(not HAS_EMBED_MODEL, reason="No embedding model (ollama pull nomic-embed-text)")
requires_llm = pytest.mark.skipif(not HAS_LLM, reason="No LLM model available")


SAMPLE_DOC = """\
# Nerve Architecture

Nerve is a thin, extensible agent framework. It has several key components:

## Provider
The Provider protocol defines how the agent talks to an LLM backend.
The built-in OllamaProvider talks to a local Ollama instance via REST.

## Tools
Tools are typed Python functions that the agent can call. The framework
auto-generates JSON schemas from type hints and docstrings.

## Skills
Skills bundle a prompt, description, and optional tools into a reusable
capability that can be loaded from files or registered programmatically.

## Memory
The Memory protocol supports store, retrieve, forget, and list_all.
FileMemory persists to JSON files. VectorMemory uses embeddings for
semantic retrieval.

## Sandbox
Code execution runs in a sandbox. LocalSandbox uses subprocess,
RestrictedSandbox limits builtins and imports.
"""

SAMPLE_DOC_2 = """\
# API Reference

## Agent

The Agent class is the main entry point. Constructor parameters:
- name: agent identifier
- provider: LLM backend (defaults to OllamaProvider)
- tools: list of Tool objects
- skills: list of Skill objects
- memory: Memory implementation
- sandbox: Sandbox implementation
- orchestration: list of prompt modes like "planning", "repair"

### Methods
- run(task): execute a task with the agent loop
- complete(prompt): single LLM call, no looping
- session(): create a multi-turn Session
- run_stream(task): streaming version of run()
- delegate(agent, task): hand off to another agent
"""


@requires_ollama
@requires_embeddings
class TestVectorMemoryIntegration:
    """Tests that actually call Ollama for embeddings."""

    def test_embed_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory
            mem = VectorMemory(path=tmp)

            mem.store("python", "Python is a programming language created by Guido van Rossum")
            mem.store("rust", "Rust is a systems programming language focused on safety and performance")
            mem.store("cooking", "A good risotto requires patience and constant stirring")

            results = mem.retrieve("programming language", limit=2)
            assert len(results) == 2
            keys = {r.key for r in results}
            assert "cooking" not in keys

    def test_ingest_and_retrieve_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory
            mem = VectorMemory(path=tmp)

            doc_path = Path(tmp) / "arch.md"
            doc_path.write_text(SAMPLE_DOC)

            entries = mem.ingest_file(str(doc_path))
            assert len(entries) >= 1

            results = mem.retrieve("how does code execution work?", limit=3)
            assert len(results) > 0
            # should find the sandbox section
            assert any("sandbox" in r.content.lower() or "sandbox" in r.key.lower() for r in results)

    def test_persistence_with_real_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory

            mem1 = VectorMemory(path=tmp)
            mem1.store("fact", "The speed of light is approximately 299,792,458 meters per second")

            mem2 = VectorMemory(path=tmp)
            assert len(mem2.list_all()) == 1
            results = mem2.retrieve("light speed", limit=1)
            assert len(results) == 1
            assert "speed of light" in results[0].content


@requires_ollama
@requires_embeddings
@requires_llm
class TestResearchAgentIntegration:
    """End-to-end tests that run the research agent with real LLM calls."""

    def test_agent_answers_from_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory
            from nerve import Agent, Skill, Tool

            mem = VectorMemory(path=tmp)
            doc_path = Path(tmp) / "arch.md"
            doc_path.write_text(SAMPLE_DOC)
            mem.ingest_file(str(doc_path))

            agent = Agent(
                name="test-researcher",
                memory=mem,
                system="Answer questions using your memory. Be concise.",
                orchestration=["repair"],
                builtins=False,
            )

            result = agent.run(
                "What sandbox options does Nerve provide?",
                tools_only=True,
                max_rounds=3,
            )

            content = result.content.lower()
            assert "localsandbox" in content or "restrictedsandbox" in content or "sandbox" in content

    def test_multi_doc_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory
            from nerve import Agent

            mem = VectorMemory(path=tmp)
            for name, text in [("arch.md", SAMPLE_DOC), ("api.md", SAMPLE_DOC_2)]:
                doc_path = Path(tmp) / name
                doc_path.write_text(text)
                mem.ingest_file(str(doc_path))

            agent = Agent(
                name="test-researcher",
                memory=mem,
                system="Answer questions using your memory. Be concise — one paragraph max.",
                orchestration=["repair"],
                builtins=False,
            )

            result = agent.run(
                "What methods does the Agent class have?",
                tools_only=True,
                max_rounds=3,
            )

            content = result.content.lower()
            assert any(m in content for m in ["run", "complete", "session"])

    def test_session_retains_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            from vector_memory import VectorMemory
            from nerve import Agent

            mem = VectorMemory(path=tmp)
            mem.store("project", "The project is called Nerve and it is an agent framework")

            agent = Agent(
                name="test-researcher",
                memory=mem,
                system="Answer questions using your memory. Be concise.",
                builtins=False,
            )

            session = agent.session(tools_only=True, max_rounds=2)
            r1 = session.send("What is the project called?")
            assert "nerve" in r1.content.lower()

            r2 = session.send("What kind of project is it?")
            assert "agent" in r2.content.lower() or "framework" in r2.content.lower()
