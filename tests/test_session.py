"""Tests for multi-turn conversation sessions."""

from agentos.agent import Agent, Session
from agentos.tools import Tool
from agentos.types import Message, Response, StopReason, ToolCall


class FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._call_count = 0
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(list(messages))
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def stream(self, messages, **kw):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        yield resp.content

    async def acomplete(self, messages, **kw):
        return self._responses[0]

    async def astream(self, messages, **kw):
        yield ""


def test_session_preserves_history():
    provider = FakeProvider([
        Response(content="Hi! Nice to meet you."),
        Response(content="Your name is Vishu."),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    r1 = session.send("My name is Vishu")
    assert r1.content == "Hi! Nice to meet you."

    r2 = session.send("What's my name?")
    assert r2.content == "Your name is Vishu."

    # Second call should see all prior messages
    second_call_messages = provider.calls[1]
    roles = [m.role for m in second_call_messages]
    assert roles == ["user", "assistant", "user"]
    assert second_call_messages[0].content == "My name is Vishu"
    assert second_call_messages[1].content == "Hi! Nice to meet you."
    assert second_call_messages[2].content == "What's my name?"


def test_session_messages_property():
    provider = FakeProvider([Response(content="ok")])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    assert session.messages == []
    session.send("hello")
    msgs = session.messages
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


def test_session_clear():
    provider = FakeProvider([Response(content="ok"), Response(content="fresh")])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    session.send("first")
    assert len(session.messages) == 2

    session.clear()
    assert session.messages == []

    session.send("second")
    # After clear, provider should only see the new message
    assert len(provider.calls[1]) == 1
    assert provider.calls[1][0].content == "second"


def test_session_with_tool_calls():
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    tool = Tool.from_function(echo)
    tc = ToolCall(id="c1", name="echo", arguments={"msg": "hi"})
    provider = FakeProvider([
        Response(content="", tool_calls=[tc]),
        Response(content="Echo said: hi"),
        Response(content="Yes, I used echo before."),
    ])
    agent = Agent("test", provider=provider, tools=[tool], builtins=False)
    session = agent.session(tools_only=True)

    r1 = session.send("echo hi")
    assert r1.content == "Echo said: hi"

    r2 = session.send("Did you use the echo tool?")
    assert r2.content == "Yes, I used echo before."

    # Third provider call should have full history including tool messages
    third_call = provider.calls[2]
    roles = [m.role for m in third_call]
    assert "tool" in roles


def test_session_id_is_stable():
    provider = FakeProvider([Response(content="a"), Response(content="b")])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    session.send("1")
    session.send("2")
    assert len(session.session_id) == 12


def test_session_stream():
    provider = FakeProvider([
        Response(content="Hello there"),
        Response(content="I remember you"),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    events = list(session.send_stream("hi"))
    text = "".join(e.data for e in events if e.type == "text")
    assert text == "Hello there"
    done = next(e for e in events if e.type == "done")
    assert done.data.stop_reason == StopReason.DONE.value

    # Second call should have history
    events2 = list(session.send_stream("remember me?"))
    text2 = "".join(e.data for e in events2 if e.type == "text")
    assert text2 == "I remember you"

    assert len(session.messages) == 4  # user, assistant, user, assistant


def test_session_independent_from_run():
    """session() and run() don't share state."""
    provider = FakeProvider([
        Response(content="session reply"),
        Response(content="run reply"),
        Response(content="session still has context"),
    ])
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    session.send("session msg")
    agent.run("standalone msg")  # should NOT see session history

    # run() should have only its own message
    run_call = provider.calls[1]
    assert len(run_call) == 1
    assert run_call[0].content == "standalone msg"

    # session should still have its history
    session.send("follow up")
    session_call = provider.calls[2]
    assert len(session_call) == 3  # user, assistant, user


def test_multiple_sessions_independent():
    provider = FakeProvider([
        Response(content="s1 reply"),
        Response(content="s2 reply"),
    ])
    agent = Agent("test", provider=provider, builtins=False)

    s1 = agent.session()
    s2 = agent.session()

    s1.send("hello from s1")
    s2.send("hello from s2")

    assert s1.session_id != s2.session_id
    assert len(s1.messages) == 2
    assert len(s2.messages) == 2
    assert s1.messages[0].content == "hello from s1"
    assert s2.messages[0].content == "hello from s2"
