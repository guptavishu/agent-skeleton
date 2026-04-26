"""Example: a minimal skill file that can be auto-discovered.

Drop this in ~/.agentos/skills/ and it will be loaded automatically.
"""

from agentos import Skill, Tool


def count_words(text: str) -> str:
    """Count words in the given text."""
    words = text.split()
    return f"{len(words)} words"


def summarize_file(path: str) -> str:
    """Read a file and return a brief summary of its structure."""
    from pathlib import Path
    content = Path(path).read_text()
    lines = content.split("\n")
    return f"{len(lines)} lines, {len(content)} chars, first line: {lines[0][:80]}"


# Module-level `skill` variable — this is what SkillRegistry.discover() looks for
skill = Skill(
    name="text_analysis",
    description="Basic text analysis tools",
    prompt="You can analyze text files — count words, summarize structure.",
    tools=[
        Tool.from_function(count_words),
        Tool.from_function(summarize_file),
    ],
)
