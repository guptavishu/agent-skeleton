"""Tests for Agent — uses a fake provider to avoid network calls."""

from nerve.agent import Agent, BUILTIN_PROMPTS
from nerve.tools import Tool
from nerve.types import Message, Response, ToolCall, Usage


class FakeProvider:
    """Returns canned responses. Cycles through them on successive calls."""

    def __init__(self, responses: list[Response]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls = []

    def complete(self, messages, *, system="", model="", temperature=0.7,
                 max_tokens=4096, tools=None):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def stream(self, messages, **kw):
        yield "chunk"

    async def acomplete(self, messages, **kw):
        return self._responses[0]

    async def astream(self, messages, **kw):
        yield "chunk"


def test_complete_single_call():
    provider = FakeProvider([Response(content="42")])
    agent = Agent("test", provider=provider)
    result = agent.complete("What is 6*7?")
    assert result.content == "42"


def test_callable_shorthand():
    provider = FakeProvider([Response(content="hi")])
    agent = Agent("test", provider=provider)
    result = agent("hello")
    assert result.content == "hi"


def test_run_no_tool_calls_returns_immediately():
    provider = FakeProvider([Response(content="plain answer")])
    agent = Agent("test", provider=provider)
    result = agent.run("question")
    assert result.content == "plain answer"
    assert len(provider.calls) == 1


def test_run_tool_loop():
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    tool = Tool.from_function(add)
    tc = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    responses = [
        Response(content="", tool_calls=[tc]),
        Response(content="The answer is 5"),
    ]
    provider = FakeProvider(responses)
    agent = Agent("test", provider=provider, tools=[tool], builtins=False)
    result = agent.run("2+3", tools_only=True)
    assert result.content == "The answer is 5"
    assert len(provider.calls) == 2


def test_run_code_loop():
    responses = [
        Response(content="```python\nprint(42)\n```"),
        Response(content="The answer is 42"),
    ]
    provider = FakeProvider(responses)
    agent = Agent("test", provider=provider, builtins=False)
    result = agent.run("compute", exec_code=True)
    assert result.content == "The answer is 42"


def test_run_hybrid_tool_and_text():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tool = Tool.from_function(echo)
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    responses = [
        Response(content="", tool_calls=[tc]),
        Response(content="Done"),
    ]
    provider = FakeProvider(responses)
    agent = Agent("test", provider=provider, tools=[tool], builtins=False)
    result = agent.run("echo hi")
    assert result.content == "Done"


def test_max_rounds_prevents_infinite_loop():
    tc = ToolCall(id="c1", name="noop", arguments={})
    provider = FakeProvider([Response(content="", tool_calls=[tc])])

    def noop() -> str:
        """No-op."""
        return "ok"

    agent = Agent("test", provider=provider, tools=[Tool.from_function(noop)], builtins=False)
    result = agent.run("loop", tools_only=True, max_rounds=3)
    assert len(provider.calls) == 3


def test_system_prompt_includes_orchestration():
    provider = FakeProvider([Response(content="ok")])
    agent = Agent("test", provider=provider, orchestration=["planning", "repair"])
    agent.run("task")
    system = provider.calls[0]["system"]
    assert "steps" in system.lower() or "plan" in system.lower()
    assert "review" in system.lower()


def test_callbacks_fire():
    tool_calls_seen = []
    tool_results_seen = []

    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tool = Tool.from_function(echo)
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    responses = [
        Response(content="", tool_calls=[tc]),
        Response(content="done"),
    ]
    provider = FakeProvider(responses)
    agent = Agent(
        "test",
        provider=provider,
        tools=[tool],
        builtins=False,
        on_tool_call=lambda tc: tool_calls_seen.append(tc.name),
        on_tool_result=lambda tr: tool_results_seen.append(tr.output),
    )
    agent.run("go", tools_only=True)
    assert tool_calls_seen == ["echo"]
    assert tool_results_seen == ["hi"]


def test_agent_with_no_builtins():
    provider = FakeProvider([Response(content="ok")])
    agent = Agent("test", provider=provider, builtins=False)
    assert agent.tool_registry.list() == []


def test_builtin_prompts_exist():
    assert "hybrid" in BUILTIN_PROMPTS
    assert "planning" in BUILTIN_PROMPTS
    assert "repair" in BUILTIN_PROMPTS
    assert "code_exec" in BUILTIN_PROMPTS


def test_run_tracks_elapsed_and_usage():
    provider = FakeProvider([
        Response(content="", tool_calls=[ToolCall(id="c1", name="echo", arguments={"msg": "hi"})],
                 usage=Usage(100, 20, 120)),
        Response(content="done", usage=Usage(200, 30, 230)),
    ])

    def echo(msg: str) -> str:
        """Echo."""
        return msg

    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)
    result = agent.run("go", tools_only=True)
    assert result.elapsed > 0
    assert result.usage.prompt_tokens == 300
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 350


def test_run_single_round_has_elapsed():
    provider = FakeProvider([Response(content="ok", usage=Usage(50, 10, 60))])
    agent = Agent("test", provider=provider, builtins=False)
    result = agent.run("question")
    assert result.elapsed > 0
    assert result.usage.total_tokens == 60
