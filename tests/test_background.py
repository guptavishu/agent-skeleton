"""Tests for background execution."""

import time
import threading
from unittest.mock import patch

from nerve.agent import Agent, RunHandle
from nerve.tools import Tool
from nerve.types import Response, StopReason, ToolCall


class FakeProvider:
    def __init__(self, responses, delay=0):
        self._responses = list(responses)
        self._call_count = 0
        self._delay = delay

    def complete(self, messages, **kwargs):
        if self._delay:
            time.sleep(self._delay)
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def stream(self, messages, **kw):
        yield "chunk"

    async def acomplete(self, messages, **kw):
        return self._responses[0]

    async def astream(self, messages, **kw):
        yield "chunk"


def test_run_background_returns_handle():
    provider = FakeProvider([Response(content="done")])
    agent = Agent("test", provider=provider, builtins=False)
    handle = agent.run_background("task")
    assert isinstance(handle, RunHandle)
    result = handle.result
    assert result.content == "done"


def test_run_background_done_property():
    provider = FakeProvider([Response(content="ok")], delay=0.1)
    agent = Agent("test", provider=provider, builtins=False)
    handle = agent.run_background("task")
    # Should not be done immediately (provider has 100ms delay)
    handle.wait(timeout=1)
    assert handle.done is True
    assert handle.result.content == "ok"


def test_run_background_on_complete_callback():
    provider = FakeProvider([Response(content="result")])
    agent = Agent("test", provider=provider, builtins=False)

    received = []
    handle = agent.run_background("task", on_complete=lambda r: received.append(r.content))
    handle.wait(timeout=5)
    assert received == ["result"]


def test_run_background_on_complete_after_done():
    provider = FakeProvider([Response(content="result")])
    agent = Agent("test", provider=provider, builtins=False)
    handle = agent.run_background("task")
    handle.wait(timeout=5)

    received = []
    handle.on_complete(lambda r: received.append(r.content))
    assert received == ["result"]


def test_run_background_stop():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeProvider([Response(content="", tool_calls=[tc])], delay=0.05)
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    handle = agent.run_background("go", tools_only=True, max_rounds=50)
    time.sleep(0.15)
    handle.stop()
    result = handle.result
    assert result.stop_reason == StopReason.INTERRUPTED.value


def test_run_background_defer(tmp_path):
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeProvider([Response(content="", tool_calls=[tc])], delay=0.05)
    agent = Agent("test", provider=provider, tools=[Tool.from_function(echo)], builtins=False)

    with patch("nerve.agent.DEFERRED_DIR", tmp_path):
        handle = agent.run_background("go", tools_only=True, max_rounds=50)
        time.sleep(0.15)
        handle.defer()
        result = handle.result
        assert result.stop_reason == StopReason.DEFERRED.value
        assert agent.last_state is not None


def test_run_background_wait_timeout():
    provider = FakeProvider([Response(content="ok")], delay=0.5)
    agent = Agent("test", provider=provider, builtins=False)
    handle = agent.run_background("task")
    assert handle.wait(timeout=0.01) is False
    assert handle.wait(timeout=2) is True


def test_run_background_error_handling():
    class BrokenProvider:
        def complete(self, messages, **kw):
            raise RuntimeError("provider crashed")
        def stream(self, messages, **kw):
            yield ""
        async def acomplete(self, messages, **kw):
            pass
        async def astream(self, messages, **kw):
            yield ""

    agent = Agent("test", provider=BrokenProvider(), builtins=False)
    handle = agent.run_background("task")
    handle.wait(timeout=5)
    assert handle.done

    raised = False
    try:
        _ = handle.result
    except RuntimeError as e:
        raised = True
        assert "provider crashed" in str(e)
    assert raised


def test_run_background_multiple_concurrent():
    provider1 = FakeProvider([Response(content="a")], delay=0.1)
    provider2 = FakeProvider([Response(content="b")], delay=0.1)
    agent1 = Agent("a", provider=provider1, builtins=False)
    agent2 = Agent("b", provider=provider2, builtins=False)

    h1 = agent1.run_background("task1")
    h2 = agent2.run_background("task2")

    h1.wait(timeout=5)
    h2.wait(timeout=5)

    assert h1.result.content == "a"
    assert h2.result.content == "b"
