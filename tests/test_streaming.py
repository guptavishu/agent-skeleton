"""Tests for streaming agent execution."""

from agentos.agent import Agent
from agentos.tools import Tool
from agentos.types import Response, StopReason, StreamEvent, ToolCall, Usage


class FakeStreamProvider:
    """Provider that streams text chunks and supports tool calls via text."""

    def __init__(self, rounds):
        """rounds: list of (chunks, has_more) tuples.
        chunks: list of strings to yield.
        has_more: if True, the agent should loop (text contains a tool call).
        """
        self._rounds = list(rounds)
        self._call_count = 0

    def complete(self, messages, **kwargs):
        chunks = self._rounds[min(self._call_count, len(self._rounds) - 1)][0]
        self._call_count += 1
        text = "".join(chunks)
        return Response(content=text)

    def stream(self, messages, **kwargs):
        round_data = self._rounds[min(self._call_count, len(self._rounds) - 1)]
        self._call_count += 1
        for chunk in round_data[0]:
            yield chunk

    async def acomplete(self, messages, **kw):
        return Response(content="")

    async def astream(self, messages, **kw):
        yield ""


def test_stream_simple_text():
    provider = FakeStreamProvider([
        (["Hello", " world", "!"], False),
    ])
    agent = Agent("test", provider=provider, builtins=False)

    events = list(agent.run_stream("greet"))
    text_events = [e for e in events if e.type == "text"]
    done_events = [e for e in events if e.type == "done"]

    assert len(text_events) == 3
    assert text_events[0].data == "Hello"
    assert text_events[1].data == " world"
    assert text_events[2].data == "!"
    assert len(done_events) == 1
    assert done_events[0].data.content == "Hello world!"
    assert done_events[0].data.stop_reason == StopReason.DONE.value


def test_stream_with_tool_call():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    provider = FakeStreamProvider([
        (['{"name": "echo", "arguments": {"msg": "hi"}}'], True),
        (["Done: hi"], False),
    ])
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    events = list(agent.run_stream("echo hi"))
    types = [e.type for e in events]

    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types

    tc_event = next(e for e in events if e.type == "tool_call")
    assert tc_event.data.name == "echo"

    tr_event = next(e for e in events if e.type == "tool_result")
    assert tr_event.data.output == "hi"

    done_event = next(e for e in events if e.type == "done")
    assert "Done" in done_event.data.content


def test_stream_with_embedded_tool_call():
    def greet(name: str) -> str:
        """Greet."""
        return f"Hello {name}"

    provider = FakeStreamProvider([
        (["Let me check.\n\n", '{"name": "greet", "arguments": {"name": "World"}}'], True),
        (["Greeting sent!"], False),
    ])
    agent = Agent("test", provider=provider, tools=[Tool.from_function(greet)], builtins=False)

    events = list(agent.run_stream("greet World"))
    text_chunks = [e.data for e in events if e.type == "text"]
    assert "Let me check.\n\n" in text_chunks

    tc = next(e for e in events if e.type == "tool_call")
    assert tc.data.name == "greet"

    tr = next(e for e in events if e.type == "tool_result")
    assert tr.data.output == "Hello World"


def test_stream_code_execution():
    provider = FakeStreamProvider([
        (["```python\nprint(42)\n```"], True),
        (["The answer is 42"], False),
    ])
    agent = Agent("test", provider=provider, builtins=False)

    events = list(agent.run_stream("compute"))
    types = [e.type for e in events]

    assert "code_exec" in types
    assert "code_result" in types
    assert "done" in types


def test_stream_tools_only_ignores_code():
    provider = FakeStreamProvider([
        (["```python\nprint(42)\n```"], False),
    ])
    agent = Agent("test", provider=provider, builtins=False)

    events = list(agent.run_stream("compute", tools_only=True))
    types = [e.type for e in events]

    assert "code_exec" not in types
    assert "done" in types


def test_stream_max_rounds():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    provider = FakeStreamProvider([
        (['{"name": "echo", "arguments": {"msg": "hi"}}'], True),
    ] * 10)
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    events = list(agent.run_stream("loop", max_rounds=3))
    done = next(e for e in events if e.type == "done")
    assert done.data.stop_reason == StopReason.MAX_ROUNDS.value


def test_stream_interrupted():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    provider = FakeStreamProvider([
        (['{"name": "echo", "arguments": {"msg": "hi"}}'], True),
    ] * 10)
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    events = []
    for event in agent.run_stream("loop", max_rounds=20):
        events.append(event)
        if event.type == "tool_result":
            agent.stop()

    done = next(e for e in events if e.type == "done")
    assert done.data.stop_reason == StopReason.INTERRUPTED.value


def test_stream_collect_full_text():
    """Verify you can reconstruct the full response from text events."""
    provider = FakeStreamProvider([
        (["The ", "capital ", "of ", "France ", "is ", "Paris."], False),
    ])
    agent = Agent("test", provider=provider, builtins=False)

    full = "".join(e.data for e in agent.run_stream("question") if e.type == "text")
    assert full == "The capital of France is Paris."


def test_stream_multi_round():
    def add(a: int, b: int) -> str:
        """Add."""
        return str(a + b)

    provider = FakeStreamProvider([
        (['{"name": "add", "arguments": {"a": 2, "b": 3}}'], True),
        (['{"name": "add", "arguments": {"a": 5, "b": 10}}'], True),
        (["Results: 5 and 15"], False),
    ])
    agent = Agent("test", provider=provider, tools=[Tool.from_function(add)], builtins=False)

    events = list(agent.run_stream("add stuff"))
    tool_calls = [e for e in events if e.type == "tool_call"]
    assert len(tool_calls) == 2
    assert tool_calls[0].data.arguments == {"a": 2, "b": 3}
    assert tool_calls[1].data.arguments == {"a": 5, "b": 10}
