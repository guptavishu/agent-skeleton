"""Example: a PR review agent with skills, memory, and HITL gates."""

from agentos import Agent, FileMemory, HITLPolicy, Skill, Tool


# --- Define tools ---

def fetch_diff(repo: str, pr_number: int) -> str:
    """Fetch the diff for a pull request."""
    import subprocess
    result = subprocess.run(
        ["gh", "pr", "diff", str(pr_number), "--repo", repo],
        capture_output=True, text=True,
    )
    return result.stdout or result.stderr


def post_comment(repo: str, pr_number: int, body: str) -> str:
    """Post a review comment on a pull request."""
    import subprocess
    result = subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--repo", repo, "--body", body],
        capture_output=True, text=True,
    )
    return result.stdout or result.stderr


# --- Define skill ---

pr_review_skill = Skill(
    name="pr_review",
    description="Review pull requests for bugs, style, and security",
    prompt="""You are a code reviewer. When reviewing a PR:
1. Fetch the diff
2. Check for bugs, security issues, and style problems
3. Post constructive comments with specific line references
4. Be concise — flag real issues, skip nitpicks""",
    tools=[
        Tool.from_function(fetch_diff),
        Tool.from_function(post_comment),
    ],
)


# --- Build the agent ---

reviewer = Agent(
    name="pr-reviewer",
    skills=[pr_review_skill],
    memory=FileMemory("~/.agentos/reviewer-memory/"),
    hitl=HITLPolicy(
        approve_before=["post_comment"],  # human approves before commenting
        auto_approve=["fetch_diff", "read_file"],
    ),
    orchestration=["planning", "repair"],
)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pr_reviewer.py <repo> <pr_number>")
        sys.exit(1)
    repo = sys.argv[1]
    pr = sys.argv[2]
    result = reviewer.run(f"Review PR #{pr} in {repo}")
    print(result.content)
