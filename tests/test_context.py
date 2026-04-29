"""Tests for context management policies."""

from agentos.protocols import ContextPolicy
from agentos.providers.context import (
    SummarizeContext,
    TokenWindowContext,
    estimate_tokens,
    message_tokens,
)
from agentos.types import Message, Response


def _msgs(n: int, content: str = "hello world") -> list[Message]:
    return [Message(role="user" if i % 2 == 0 else "assistant", content=content) for i in range(n)]


# --- TokenWindowContext ---

def test_token_window_under_limit():
    ctx = TokenWindowContext()
    msgs = _msgs(3, "short")
    result = ctx.manage(msgs, 100000)
    assert result == msgs


def test_token_window_trims_oldest():
    ctx = TokenWindowContext()
    msgs = _msgs(10, "hello world")
    result = ctx.manage(msgs, 30)
    assert len(result) < 10
    assert result[0] is msgs[0]
    assert result[-1] is msgs[-1]


def test_token_window_keeps_first_message():
    ctx = TokenWindowContext()
    msgs = _msgs(20, "some content here")
    result = ctx.manage(msgs, 50)
    assert result[0] is msgs[0]


def test_token_window_is_context_policy():
    assert isinstance(TokenWindowContext(), ContextPolicy)


# --- SummarizeContext ---

class FakeSummaryProvider:
    context_window = 100000

    def complete(self, messages, **kw):
        return Response(content="Summary: stuff happened.")

    def stream(self, messages, **kw):
        yield ""

    async def acomplete(self, messages, **kw):
        return Response(content="")

    async def astream(self, messages, **kw):
        yield ""


def test_summarize_under_limit():
    ctx = SummarizeContext(provider=FakeSummaryProvider(), keep_recent=4)
    msgs = _msgs(3, "short")
    result = ctx.manage(msgs, 100000)
    assert result == msgs


def test_summarize_replaces_old_with_summary():
    ctx = SummarizeContext(provider=FakeSummaryProvider(), keep_recent=3)
    msgs = _msgs(10, "x" * 200)
    result = ctx.manage(msgs, 50)
    assert len(result) == 5
    assert result[0] is msgs[0]
    assert "[Summary" in result[1].content
    assert "stuff happened" in result[1].content
    assert result[2] is msgs[7]


def test_summarize_caches():
    provider = FakeSummaryProvider()
    call_count = [0]
    orig = provider.complete

    def counting_complete(*a, **kw):
        call_count[0] += 1
        return orig(*a, **kw)

    provider.complete = counting_complete
    ctx = SummarizeContext(provider=provider, keep_recent=3)

    msgs = _msgs(10, "x" * 200)
    ctx.manage(msgs, 50)
    ctx.manage(msgs, 50)
    assert call_count[0] == 1

    msgs.extend(_msgs(2, "x" * 200))
    ctx.manage(msgs, 50)
    assert call_count[0] == 2


def test_summarize_is_context_policy():
    assert isinstance(SummarizeContext(provider=FakeSummaryProvider()), ContextPolicy)


# --- Token estimation ---

def test_estimate_tokens():
    assert estimate_tokens("hello") > 0
    assert estimate_tokens("a" * 400) > estimate_tokens("hi")


def test_message_tokens():
    msg = Message(role="user", content="hello world")
    tokens = message_tokens(msg)
    assert tokens > 0


# --- Provider context_window ---

def test_provider_context_window_used():
    from agentos.agent import Agent

    class FakeProvider:
        context_window = 50

        def __init__(self):
            self.calls = []

        def complete(self, messages, **kw):
            self.calls.append(list(messages))
            return Response(content="ok")

        def stream(self, messages, **kw):
            yield "ok"

        async def acomplete(self, messages, **kw):
            return Response(content="")

        async def astream(self, messages, **kw):
            yield ""

    provider = FakeProvider()
    agent = Agent("test", provider=provider, builtins=False)
    session = agent.session()

    for i in range(20):
        session.send(f"message {i} with some extra content to fill tokens")

    # With context_window=50, the provider should see trimmed messages
    last_call = provider.calls[-1]
    assert len(last_call) < 40  # 20 user + 20 assistant = 40 without trimming


def test_provider_without_context_window_defaults():
    """Providers that don't set context_window get 128k default."""
    from agentos.agent import Agent

    class BareProvider:
        def complete(self, messages, **kw):
            return Response(content="ok")

        def stream(self, messages, **kw):
            yield "ok"

        async def acomplete(self, messages, **kw):
            return Response(content="")

        async def astream(self, messages, **kw):
            yield ""

    agent = Agent("test", provider=BareProvider(), builtins=False)
    session = agent.session()

    for i in range(5):
        session.send(f"msg {i}")

    # Should not trim — 128k is plenty for 10 messages
    assert len(session.messages) == 10
