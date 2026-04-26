# agent-skeleton

Thin, extensible agent framework with reasonable defaults. Build agents that use tools, execute code, remember things, and coordinate with each other — in under 20 lines.

## Quick Start

```bash
pip install -e .
```

### Simplest Agent

```python
from agentos import Agent

agent = Agent("my-agent")
result = agent.run("What files are in the current directory?")
print(result.content)
```

The agent uses hybrid mode by default — it has tools (read_file, write_file, shell_exec, list_directory) and can write Python code. It picks whichever fits each step.

### With Skills, Memory, and Guardrails

```python
from agentos import Agent, FileMemory, HITLPolicy, Skill, Tool

def fetch_logs(service: str, hours: int = 1) -> str:
    """Fetch recent logs for a service."""
    ...

oncall_skill = Skill(
    name="oncall",
    prompt="You triage production incidents. Check logs, identify root cause, suggest fixes.",
    tools=[Tool.from_function(fetch_logs)],
)

agent = Agent(
    name="oncall-helper",
    skills=[oncall_skill],
    memory=FileMemory(),
    hitl=HITLPolicy(
        approve_before=["shell_exec"],  # human approves dangerous tools
        auto_approve=["read_file", "fetch_logs"],
    ),
    orchestration=["planning", "repair"],
)

result = agent.run("API latency spiked to 2s. Investigate.")
```

## CLI

```bash
# batch mode
agentos "List all Python files in this directory"

# interactive mode
agentos

# options
agentos --tools-only "Read config.yaml"       # no code execution
agentos --exec "Calculate fibonacci(30)"       # code-only mode
agentos --plan "Refactor the auth module"      # enable planning
agentos --skill ./my_skill.py "Do the thing"   # load a skill file
```

Interactive commands: `/skills`, `/tools`, `/memory <query>`, `/remember key=value`, `/help`.

## Architecture

```
Agent(name, provider?, skills[], memory?, hitl?, delegates?)
  .run(task)        → execute with tools + code (hybrid loop)
  .complete(task)   → single LLM call, no looping
  .delegate(agent)  → hand off to another agent
  .__call__(task)   → shorthand for .complete()
```

### Components

| Component | What it does | Default | Extensible via |
|-----------|-------------|---------|----------------|
| **Provider** | Talks to LLMs | Ollama (local) | `Provider` protocol — implement 4 methods |
| **Tools** | Structured actions the agent can take | read_file, write_file, shell_exec, list_directory | `Tool.from_function(fn)` or `Tool(...)` |
| **Skills** | Reusable prompt+tool bundles | Auto-discovered from `~/.agentos/skills/` | `Skill(name, prompt, tools)` |
| **Memory** | Persistent knowledge across runs | File-backed JSON in `~/.agentos/memory/` | `Memory` protocol — implement 4 methods |
| **HITL** | Human approval gates | Everything auto-approved | `HITLPolicy(approve_before=[...])` |
| **Coordinator** | Multi-agent delegation | Sequential (one at a time) | `Coordinator` protocol |
| **Telemetry** | Structured event logging | JSON lines to `~/.agentos/agentos.log` | `Telemetry` class |

### Execution Modes

| Mode | Flag | What happens |
|------|------|-------------|
| **Hybrid** (default) | — | LLM has both tools and code execution available, picks per step |
| **Tools only** | `tools_only=True` | LLM can only use structured tool calls |
| **Code only** | `exec_code=True` | LLM can only write and execute Python code blocks |

### Orchestration

Stack prompt fragments to control agent behavior:

```python
agent = Agent(orchestration=["planning", "repair"])

# or per-call:
agent.run("complex task", orchestration=["planning", "repair"])
agent.run("simple task", plan=True)  # shorthand for adding "planning"
```

Built-in orchestration modes: `planning`, `repair`, `hybrid`, `code_exec`.

## Skills

A skill is a Python file with a module-level `skill` variable:

```python
# ~/.agentos/skills/my_skill.py
from agentos import Skill, Tool

def my_tool(arg: str) -> str:
    """Does something useful."""
    ...

skill = Skill(
    name="my_skill",
    description="What this skill does",
    prompt="Instructions for the LLM when this skill is active",
    tools=[Tool.from_function(my_tool)],
)
```

Skills in `~/.agentos/skills/` or `./skills/` are auto-discovered. Load others with `--skill path` or `skills=[Skill.load("path")]`.

## Multi-Agent

```python
researcher = Agent("researcher", skills=[research_skill])
writer = Agent("writer", skills=[writing_skill])

lead = Agent("lead", delegates=[researcher, writer])

# delegate sub-tasks
research = lead.delegate(researcher, "Research topic X")
doc = lead.delegate(writer, f"Write about: {research.content}")
```

## Custom Provider

Implement the `Provider` protocol to plug in any LLM:

```python
from agentos import Provider, Message, Response, Tool

class MyProvider:
    def complete(self, messages, *, system="", model="", temperature=0.7,
                 max_tokens=4096, tools=None) -> Response:
        # call your LLM, return a Response
        ...

    def stream(self, messages, **kwargs):
        # yield string chunks
        ...

    async def acomplete(self, messages, **kwargs) -> Response:
        ...

    async def astream(self, messages, **kwargs):
        ...

agent = Agent("my-agent", provider=MyProvider())
```

## Custom Memory Backend

```python
from agentos import Memory, MemoryEntry

class RedisMemory:
    def store(self, key, content, metadata=None) -> MemoryEntry: ...
    def retrieve(self, query, limit=5) -> list[MemoryEntry]: ...
    def forget(self, key) -> bool: ...
    def list_all(self) -> list[MemoryEntry]: ...

agent = Agent("my-agent", memory=RedisMemory())
```

## Project Structure

```
agent-skeleton/
├── pyproject.toml
├── agentos/
│   ├── __init__.py       # re-exports everything
│   ├── agent.py          # Agent class — the main entry point
│   ├── provider.py       # Provider protocol + OllamaProvider
│   ├── tools.py          # Tool, ToolRegistry, built-in tools
│   ├── skills.py         # Skill, SkillRegistry, filesystem discovery
│   ├── memory.py         # Memory protocol + FileMemory
│   ├── executor.py       # Code execution in subprocess
│   ├── hitl.py           # Human-in-the-loop approval gates
│   ├── coordinator.py    # Multi-agent coordination
│   ├── telemetry.py      # Structured JSON logging
│   ├── types.py          # Message, Response, ToolCall, etc.
│   └── cli.py            # CLI entry point
└── examples/
    ├── pr_reviewer.py    # PR review agent with HITL
    ├── multi_agent.py    # Lead → researcher → writer delegation
    └── simple_skill.py   # Minimal auto-discoverable skill
```

## Design Principles

- **Protocol-based extensibility** — Provider, Memory, Coordinator are all typing.Protocol. Structural subtyping, no inheritance required.
- **Reasonable defaults** — works out of the box with Ollama, file-backed memory, auto-discovered skills, and permissive HITL. Override only what you need.
- **Thin by design** — ~11 files, no unnecessary abstractions. The framework gets out of your way.
- **Hybrid by default** — the agent has both tools and code execution available and picks the right approach per step.

## Requirements

- Python >= 3.11
- Ollama running locally (if using default provider)
- `pip install -e .` to install

## License

MIT
