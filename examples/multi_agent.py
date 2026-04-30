"""Example: multi-agent delegation — a lead agent coordinates researcher + writer."""

from nerve import Agent, Skill, Tool


def web_search(query: str) -> str:
    """Search the web for information."""
    return f"(stub) Search results for: {query}"


def write_document(title: str, content: str) -> str:
    """Write a document to disk."""
    from pathlib import Path
    path = Path(f"{title.lower().replace(' ', '_')}.md")
    path.write_text(content)
    return f"Wrote {path}"


researcher = Agent(
    name="researcher",
    skills=[
        Skill(
            name="research",
            prompt="You research topics thoroughly. Search multiple sources and synthesize findings.",
            tools=[Tool.from_function(web_search)],
        )
    ],
    discover_skills=False,
)

writer = Agent(
    name="writer",
    skills=[
        Skill(
            name="writing",
            prompt="You write clear, concise documents. Use the research provided to create well-structured output.",
            tools=[Tool.from_function(write_document)],
        )
    ],
    discover_skills=False,
)

lead = Agent(
    name="lead",
    system="You coordinate research and writing tasks. Delegate research to the researcher, then writing to the writer.",
    delegates=[researcher, writer],
    discover_skills=False,
)


if __name__ == "__main__":
    # the lead orchestrates — research first, then writing
    research = lead.delegate(researcher, "Research the history of agent frameworks in AI")
    print(f"Research done: {research.content[:200]}...")

    result = lead.delegate(writer, f"Write a summary document based on this research:\n{research.content}")
    print(f"Writing done: {result.content[:200]}...")
