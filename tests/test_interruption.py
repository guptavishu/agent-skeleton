"""Tests for interruption and deferral."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from nerve.agent import Agent, DEFERRED_DIR
from nerve.tools import Tool
from nerve.types import Message, Response, RunState, StopReason, ToolCall, Usage


class FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0

    def complete(self, messages, **kwargs):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def stream(self, messages, **kw):
        yield "chunk"

    async def acomplete(self, messages, **kw):
        return self._responses[0]

    async def astream(self, messages, **kw):
        yield "chunk"


def _echo_tool():
    def echo(msg: str) -> str:
        """Echo."""
        return msg
    return Tool.from_function(echo)


def _agent_with_loop(stop_after=1):
    """Agent that loops: tool call on every response, stops after N rounds."""
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    responses = [Response(content="", tool_calls=[tc])] * 10 + [Response(content="done")]
    provider = FakeProvider(responses)
    agent = Agent("test", provider=provider, tools=[_echo_tool()], builtins=False)
    return agent


# --- Interruption ---

def test_stop_interrupts_loop():
    agent = _agent_with_loop()

    # Stop after the first tool call callback
    call_count = [0]
    def on_tool(tc):
        call_count[0] += 1
        if call_count[0] >= 2:
            agent.stop()

    agent.on_tool_call = on_tool
    result = agent.run("go", tools_only=True, max_rounds=20)
    assert result.stop_reason == StopReason.INTERRUPTED.value
    assert call_count[0] == 2


def test_stop_resets_between_runs():
    agent = _agent_with_loop()
    agent.stop()
    result = agent.run("go", tools_only=True, max_rounds=5)
    # stop flag is reset at start of run, so it should NOT be interrupted
    assert result.stop_reason != StopReason.INTERRUPTED.value


def test_stop_reason_done():
    provider = FakeProvider([Response(content="answer")])
    agent = Agent("test", provider=provider, builtins=False)
    result = agent.run("question")
    assert result.stop_reason == StopReason.DONE.value


def test_stop_reason_max_rounds():
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeProvider([Response(content="", tool_calls=[tc])])
    agent = Agent("test", provider=provider, tools=[_echo_tool()], builtins=False)
    result = agent.run("go", tools_only=True, max_rounds=3)
    assert result.stop_reason == StopReason.MAX_ROUNDS.value


# --- Deferral ---

def test_defer_saves_state(tmp_path):
    with patch("nerve.agent.DEFERRED_DIR", tmp_path):
        agent = _agent_with_loop()

        call_count = [0]
        def on_tool(tc):
            call_count[0] += 1
            if call_count[0] >= 2:
                agent.defer()

        agent.on_tool_call = on_tool
        result = agent.run("do stuff", tools_only=True, max_rounds=20)

        assert result.stop_reason == StopReason.DEFERRED.value
        assert agent.last_state is not None
        assert agent.last_state.task == "do stuff"
        assert agent.last_state.round >= 2

        # Check file was written
        state_files = list(tmp_path.glob("*.json"))
        assert len(state_files) == 1

        data = json.loads(state_files[0].read_text())
        assert data["task"] == "do stuff"
        assert data["mode"] == "tools_only"
        assert len(data["messages"]) > 1


def test_resume_from_state(tmp_path):
    with patch("nerve.agent.DEFERRED_DIR", tmp_path):
        # Create a state with 2 messages already in history
        tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
        responses = [Response(content="final answer")]
        provider = FakeProvider(responses)
        agent = Agent("test", provider=provider, tools=[_echo_tool()], builtins=False)

        state = RunState(
            state_id="test123",
            agent_name="test",
            task="original task",
            messages=[
                Message(role="user", content="original task"),
                Message(role="assistant", content="", tool_calls=[tc]),
                Message(role="tool", content="hi", tool_call_id="c1"),
            ],
            system="test system",
            mode="tools_only",
            round=1,
            max_rounds=10,
            session_id="sess1",
        )

        # Save and resume
        agent._save_state(state)
        result = agent.resume("test123")

        assert result.content == "final answer"
        assert result.stop_reason == StopReason.DONE.value


def test_resume_from_state_object():
    responses = [Response(content="resumed answer")]
    provider = FakeProvider(responses)
    agent = Agent("test", provider=provider, builtins=False)

    state = RunState(
        state_id="s1",
        agent_name="test",
        task="task",
        messages=[Message(role="user", content="task")],
        system="",
        mode="hybrid",
        round=0,
        max_rounds=10,
        session_id="sess1",
    )

    result = agent.resume(state)
    assert result.content == "resumed answer"


def test_list_deferred(tmp_path):
    with patch("nerve.agent.DEFERRED_DIR", tmp_path):
        assert Agent.list_deferred() == []

        provider = FakeProvider([Response(content="")])
        agent = Agent("test", provider=provider, builtins=False)

        state = RunState(
            state_id="abc",
            agent_name="test",
            task="my task",
            messages=[Message(role="user", content="my task")],
            system="",
            mode="hybrid",
            round=3,
            max_rounds=10,
            session_id="s1",
        )
        agent._save_state(state)

        deferred = Agent.list_deferred()
        assert len(deferred) == 1
        assert deferred[0].state_id == "abc"
        assert deferred[0].task == "my task"


def test_defer_and_resume_roundtrip(tmp_path):
    with patch("nerve.agent.DEFERRED_DIR", tmp_path):
        tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
        responses = [
            Response(content="", tool_calls=[tc]),
            Response(content="", tool_calls=[tc]),
            Response(content="", tool_calls=[tc]),
            Response(content="final"),
        ]
        provider = FakeProvider(responses)
        agent = Agent("test", provider=provider, tools=[_echo_tool()], builtins=False)

        call_count = [0]
        def on_tool(tc):
            call_count[0] += 1
            if call_count[0] == 2:
                agent.defer()

        agent.on_tool_call = on_tool
        result = agent.run("go", tools_only=True, max_rounds=20)
        assert result.stop_reason == StopReason.DEFERRED.value

        # Now resume — provider picks up from call_count 2 with remaining responses
        provider2 = FakeProvider([
            Response(content="", tool_calls=[tc]),
            Response(content="done after resume"),
        ])
        agent2 = Agent("test", provider=provider2, tools=[_echo_tool()], builtins=False)
        result2 = agent2.resume(agent.last_state.state_id)
        assert result2.content == "done after resume"
        assert result2.stop_reason == StopReason.DONE.value


# --- StopReason enum ---

def test_stop_reason_values():
    assert StopReason.DONE.value == "done"
    assert StopReason.MAX_ROUNDS.value == "max_rounds"
    assert StopReason.INTERRUPTED.value == "interrupted"
    assert StopReason.DEFERRED.value == "deferred"
