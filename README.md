# nerve

Thin, extensible agent framework with zero required dependencies. Build agents that use tools, execute code, remember things, and coordinate with each other — in under 20 lines.

## Install

```bash
pip install nerve                  # core — zero dependencies, bring your own provider
pip install nerve[ollama]          # adds Ollama support (local LLMs)
pip install nerve[web]             # adds web UI
pip install nerve[all]             # everything
```

For development:

```bash
git clone https://github.com/guptavishu/nerve.git
cd nerve
pip install -e ".[ollama,web,dev]"
```

## Quick Start

### Bring Your Own Provider

```python
from nerve import Agent, Provider, Message, Response, Usage

class MyProvider:
    def complete(self, messages, *, system="", model="", temperature=0.7,
                 max_tokens=4096, tools=None) -> Response:
        # call your LLM, return a Response
        ...

    def stream(self, messages, **kwargs):
        yield "chunk"

    async def acomplete(self, messages, **kwargs) -> Response: ...
    async def astream(self, messages, **kwargs): ...

agent = Agent("my-agent", provider=MyProvider())
result = agent.run("What files are in the current directory?")
print(result.content)
```

### With Ollama (Local LLMs)

```bash
pip install nerve[ollama]
ollama pull qwen2.5-coder:32b
```

```python
from nerve import Agent

agent = Agent("my-agent")  # defaults to OllamaProvider
result = agent.run("What files are in the current directory?")
print(result.content)
```

The agent uses hybrid mode by default — it has tools (read_file, write_file, shell_exec, list_directory) and can write Python code. It picks whichever fits each step.

### With Skills, Memory, and Guardrails

```python
from nerve import Agent, FileMemory, HITLPolicy, Skill, Tool

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
nerve "List all Python files in this directory"

# interactive mode
nerve

# web UI
nerve --web
nerve --web --port 9000

# options
nerve --tools-only "Read config.yaml"       # no code execution
nerve --exec "Calculate fibonacci(30)"       # code-only mode
nerve --plan "Refactor the auth module"      # enable planning
nerve --skill ./my_skill.py "Do the thing"   # load a skill file
```

Interactive commands: `/skills`, `/tools`, `/memory <query>`, `/remember key=value`, `/defer`, `/deferred`, `/resume <id>`, `/help`.

## Architecture

```
Agent(name, provider, skills[], memory?, hitl?, delegates?)
  .run(task)              → execute with tools + code (hybrid loop)
  .run_background(task)   → run in background, returns RunHandle
  .complete(task)         → single LLM call, no looping
  .delegate(agent)        → hand off to another agent
  .stop()                 → interrupt after current round
  .defer()                → save state and stop
  .resume(state)          → continue from saved state
  .__call__(task)         → shorthand for .complete()
```

### Components

| Component | What it does | Default | Extensible via |
|-----------|-------------|---------|----------------|
| **Provider** | Talks to LLMs | Ollama (if installed) | `Provider` protocol — implement 4 methods |
| **Tools** | Structured actions the agent can take | read_file, write_file, shell_exec, list_directory | `Tool.from_function(fn)` or `Tool(...)` |
| **Skills** | Reusable prompt+tool bundles | Auto-discovered from `~/.nerve/skills/` | `Skill(name, prompt, tools)` |
| **Memory** | Persistent knowledge across runs | File-backed JSON in `~/.nerve/memory/` | `Memory` protocol — implement 4 methods |
| **HITL** | Human approval gates | Everything auto-approved | `HITLPolicy(approve_before=[...])` |
| **Coordinator** | Multi-agent delegation | Sequential (one at a time) | `Coordinator` protocol |
| **Telemetry** | Structured event logging | JSON lines to `~/.nerve/nerve.log` | `Telemetry` class |

### Execution Modes

| Mode | Flag | What happens |
|------|------|-------------|
| **Hybrid** (default) | — | LLM has both tools and code execution available, picks per step |
| **Tools only** | `tools_only=True` | LLM can only use structured tool calls |
| **Code only** | `exec_code=True` | LLM can only write and execute Python code blocks |

### Stop Reasons

Every response has a `stop_reason`: `done`, `max_rounds`, `interrupted`, or `deferred`.

### Orchestration

Stack prompt fragments to control agent behavior:

```python
agent = Agent(orchestration=["planning", "repair"])

# or per-call:
agent.run("complex task", orchestration=["planning", "repair"])
agent.run("simple task", plan=True)  # shorthand for adding "planning"
```

Built-in orchestration modes: `planning`, `repair`, `hybrid`, `code_exec`.

## Background Execution

```python
# Run in background with notification
handle = agent.run_background("complex task", on_complete=lambda r: print(r.content))

# Check status
handle.done       # non-blocking bool
handle.wait(30)   # block up to 30s
handle.result     # block until done

# Interrupt or defer
handle.stop()
handle.defer()
```

## Interruption & Deferral

```python
# Interrupt from a callback
def on_tool(tc):
    if tc.name == "dangerous_tool":
        agent.stop()

# Defer and resume later
agent.defer()
result = agent.run("task")  # stop_reason="deferred"
state_id = agent.last_state.state_id

# Resume (even in a new process)
result = agent.resume(state_id)
```

## Skills

A skill is a Python file with a module-level `skill` variable:

```python
# ~/.nerve/skills/my_skill.py
from nerve import Skill, Tool

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

Skills in `~/.nerve/skills/` or `./skills/` are auto-discovered. Load others with `--skill path` or `skills=[Skill.load("path")]`.

## Multi-Agent

```python
researcher = Agent("researcher", provider=my_provider, skills=[research_skill])
writer = Agent("writer", provider=my_provider, skills=[writing_skill])

lead = Agent("lead", provider=my_provider, delegates=[researcher, writer])

# delegate sub-tasks
research = lead.delegate(researcher, "Research topic X")
doc = lead.delegate(writer, f"Write about: {research.content}")
```

## Custom Provider

Implement the `Provider` protocol to plug in any LLM:

```python
from nerve import Provider, Message, Response, Tool

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
from nerve import Memory, MemoryEntry

class RedisMemory:
    def store(self, key, content, metadata=None) -> MemoryEntry: ...
    def retrieve(self, query, limit=5) -> list[MemoryEntry]: ...
    def forget(self, key) -> bool: ...
    def list_all(self) -> list[MemoryEntry]: ...

agent = Agent("my-agent", provider=my_provider, memory=RedisMemory())
```

## Project Structure

```
nerve/
├── pyproject.toml
├── nerve/
│   ├── __init__.py       # re-exports everything
│   ├── agent.py          # Agent class, RunHandle, execution loops
│   ├── provider.py       # Provider protocol + OllamaProvider
│   ├── tools.py          # Tool, ToolRegistry, built-in tools
│   ├── skills.py         # Skill, SkillRegistry, filesystem discovery
│   ├── memory.py         # Memory protocol + FileMemory
│   ├── executor.py       # Code execution in subprocess
│   ├── hitl.py           # Human-in-the-loop approval gates
│   ├── coordinator.py    # Multi-agent coordination
│   ├── telemetry.py      # Structured JSON logging
│   ├── types.py          # Message, Response, ToolCall, RunState, etc.
│   ├── cli.py            # CLI entry point
│   └── web.py            # Web UI (FastAPI)
├── tests/                # 109 unit tests
├── evals/                # Eval harness + 9 cases
└── examples/
    ├── pr_reviewer.py    # PR review agent with HITL
    ├── multi_agent.py    # Lead → researcher → writer delegation
    ├── simple_skill.py   # Minimal auto-discoverable skill
    ├── background_basic.py
    ├── background_interrupt.py
    └── background_defer_resume.py
```

## Design Principles

- **Zero required dependencies** — the core has no deps. Providers, web UI, etc. are optional extras.
- **Protocol-based extensibility** — Provider, Memory, Coordinator are all typing.Protocol. Structural subtyping, no inheritance required.
- **Thin by design** — ~1,400 lines, no unnecessary abstractions. The framework gets out of your way.
- **Hybrid by default** — the agent has both tools and code execution available and picks the right approach per step.

## Requirements

- Python >= 3.11
- A provider (Ollama, or bring your own)

## License

MIT
