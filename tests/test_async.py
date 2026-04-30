"""Tests for async agent methods."""

import asyncio

import pytest

from nerve.agent import Agent, Session
from nerve.tools import Tool
from nerve.types import Message, Response, StopReason, StreamEvent, ToolCall


class FakeAsyncProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.calls = []

    def complete(self, messages, **kwargs):
        return self._do_complete(messages)

    async def acomplete(self, messages, **kwargs):
        return self._do_complete(messages)

    def _do_complete(self, messages):
        self.calls.append(list(messages))
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def stream(self, messages, **kw):
        resp = self._do_complete(messages)
        yield resp.content

    async def astream(self, messages, **kw):
        resp = self._do_complete(messages)
        yield resp.content


# --- acomplete ---

@pytest.mark.asyncio
async def test_acomplete():
    provider = FakeAsyncProvider([Response(content="42")])
    agent = Agent("test", provider=provider, builtins=False)
    result = await agent.acomplete("What is 6*7?")
    assert result.content == "42"


# --- arun ---

@pytest.mark.asyncio
async def test_arun_no_tool_calls():
    provider = FakeAsyncProvider([Response(content="plain answer")])
    agent = Agent("test", provider=provider, builtins=False)
    result = await agent.arun("question")
    assert result.content == "plain answer"
    assert result.stop_reason == StopReason.DONE.value


@pytest.mark.asyncio
async def test_arun_with_tool_calls():
    def add(a: int, b: int) -> str:
        """Add."""
        return str(a + b)

    tool = Tool.from_function(add)
    tc = ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
    provider = FakeAsyncProvider([
        Response(content="", tool_calls=[tc]),
        Response(content="The answer is 5"),
    ])
    agent = Agent("test", provider=provider, tools=[tool], builtins=False)
    result = await agent.arun("2+3", tools_only=True)
    assert result.content == "The answer is 5"
    assert result.stop_reason == StopReason.DONE.value


@pytest.mark.asyncio
async def test_arun_max_rounds():
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeAsyncProvider([Response(content="", tool_calls=[tc])])

    def echo(msg: str) -> str:
        """Echo."""
        return msg

    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)
    result = await agent.arun("loop", tools_only=True, max_rounds=3)
    assert result.stop_reason == StopReason.MAX_ROUNDS.value


@pytest.mark.asyncio
async def test_arun_code_execution():
    provider = FakeAsyncProvider([
        Response(content="```python\nprint(42)\n```"),
        Response(content="The answer is 42"),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    result = await agent.arun("compute", exec_code=True)
    assert result.content == "The answer is 42"


@pytest.mark.asyncio
async def test_arun_interrupted():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeAsyncProvider([Response(content="", tool_calls=[tc])] * 10)
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    call_count = [0]
    def on_tool(tc):
        call_count[0] += 1
        if call_count[0] >= 2:
            agent.stop()

    agent.on_tool_call = on_tool
    result = await agent.arun("go", tools_only=True, max_rounds=20)
    assert result.stop_reason == StopReason.INTERRUPTED.value


# --- arun_stream ---

@pytest.mark.asyncio
async def test_arun_stream_text():
    provider = FakeAsyncProvider([Response(content="Hello world")])
    agent = Agent("test", provider=provider, builtins=False)

    events = []
    async for event in agent.arun_stream("greet"):
        events.append(event)

    text = "".join(e.data for e in events if e.type == "text")
    assert text == "Hello world"
    done = next(e for e in events if e.type == "done")
    assert done.data.stop_reason == StopReason.DONE.value


@pytest.mark.asyncio
async def test_arun_stream_with_tools():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tc_json = '{"name": "echo", "arguments": {"msg": "hi"}}'
    provider = FakeAsyncProvider([
        Response(content=tc_json),
        Response(content="Done"),
    ])
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    events = []
    async for event in agent.arun_stream("echo"):
        events.append(event)

    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types


# --- async session ---

@pytest.mark.asyncio
async def test_session_asend():
    provider = FakeAsyncProvider([
        Response(content="Hi Vishu"),
        Response(content="Your name is Vishu"),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    r1 = await session.asend("My name is Vishu")
    assert r1.content == "Hi Vishu"

    r2 = await session.asend("What's my name?")
    assert r2.content == "Your name is Vishu"

    # Second call should see full history
    second_call = provider.calls[-1]
    roles = [m.role for m in second_call]
    assert roles == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_session_asend_independent():
    provider = FakeAsyncProvider([
        Response(content="session"),
        Response(content="standalone"),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    await session.asend("session msg")
    await agent.arun("standalone msg")

    # arun should only see its own message
    run_call = provider.calls[1]
    assert len(run_call) == 1
