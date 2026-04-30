"""Example: a research agent with RAG over local documents.

Ingests documents into VectorMemory, then answers questions using
retrieved context + web search.

Usage:
    # First, pull an embedding model:
    ollama pull nomic-embed-text

    # Ingest docs and ask questions interactively:
    python research_agent.py --ingest ./docs/

    # One-shot question:
    python research_agent.py --ingest ./docs/ "How does the auth system work?"

Requirements:
    pip install nerve[ollama] numpy
"""

from __future__ import annotations

import argparse
import sys

from nerve import Agent, HITLPolicy, Skill, Tool
from vector_memory import VectorMemory


# --- Tools ---

def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results."""
    import subprocess
    result = subprocess.run(
        ["curl", "-s", f"https://html.duckduckgo.com/html/?q={query}"],
        capture_output=True, text=True, timeout=15,
    )
    from html.parser import HTMLParser

    results = []

    class DDGParser(HTMLParser):
        in_result = False
        current = ""

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                self.in_result = True
                self.current = attrs_dict.get("href", "")

        def handle_data(self, data):
            if self.in_result:
                results.append(f"{data.strip()} — {self.current}")
                self.in_result = False

    DDGParser().feed(result.stdout)
    return "\n".join(results[:5]) if results else "No results found."


def fetch_url(url: str) -> str:
    """Fetch a URL and return its text content (first 3000 chars)."""
    import subprocess
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "10", url],
        capture_output=True, text=True, timeout=15,
    )
    text = result.stdout

    # strip HTML tags for readability
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]


def save_note(key: str, content: str) -> str:
    """Save a research note to memory for later retrieval."""
    _memory.store(key, content, metadata={"type": "note"})
    return f"Saved note: {key}"


# --- Global memory (set in main) ---
_memory: VectorMemory = None


# --- Skill ---

research_skill = Skill(
    name="research",
    description="Research a topic using documents and web sources",
    prompt="""You are a research assistant. When answering a question:

1. First check your memory for relevant documents (they'll appear in the system prompt)
2. If the documents don't fully answer the question, search the web
3. Synthesize information from multiple sources
4. Cite your sources — mention which document or URL the information came from
5. If you learn something important, save it as a note for future reference

Be thorough but concise. Distinguish between what the documents say and what you found online.""",
    tools=[
        Tool.from_function(web_search),
        Tool.from_function(fetch_url),
        Tool.from_function(save_note),
    ],
)


def main():
    global _memory

    parser = argparse.ArgumentParser(description="Research agent with RAG")
    parser.add_argument("question", nargs="?", help="Question to research (omit for interactive)")
    parser.add_argument("--ingest", action="append", default=[], help="File or directory to ingest (can repeat)")
    parser.add_argument("--glob", default="**/*.md", help="Glob pattern for directory ingestion")
    parser.add_argument("--model", default="", help="LLM model override")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Embedding model")
    parser.add_argument("--memory-dir", default=None, help="Directory for vector memory storage")
    args = parser.parse_args()

    _memory = VectorMemory(path=args.memory_dir, model=args.embed_model)

    # Ingest documents
    from pathlib import Path
    for source in args.ingest:
        p = Path(source)
        if p.is_dir():
            entries = _memory.ingest_directory(str(p), glob=args.glob)
            print(f"Ingested {len(entries)} chunks from {p}/")
        elif p.is_file():
            entries = _memory.ingest_file(str(p))
            print(f"Ingested {len(entries)} chunks from {p}")
        else:
            print(f"Warning: {source} not found, skipping", file=sys.stderr)

    total = len(_memory.list_all())
    print(f"Memory: {total} chunks loaded\n")

    agent = Agent(
        name="researcher",
        model=args.model,
        skills=[research_skill],
        memory=_memory,
        orchestration=["planning", "repair"],
        system="You are a research assistant with access to a document library and web search.",
    )

    if args.question:
        result = agent.run(args.question)
        print(result.content)
    else:
        session = agent.session()
        print("Research agent — ask questions about your documents (or anything).")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                question = input("research> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            response = session.send(question)
            print(f"\n{response.content}\n")


if __name__ == "__main__":
    main()
