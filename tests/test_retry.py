"""Tests for RetryProvider."""

import time

import pytest

from agentos.providers.retry import RetryProvider
from agentos.types import Response


class FailThenSucceedProvider:
    context_window = 128000

    def __init__(self, fail_count: int):
        self._fail_count = fail_count
        self._call_count = 0

    def complete(self, messages, **kw):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError("server down")
        return Response(content="ok")

    def stream(self, messages, **kw):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError("server down")
        yield "ok"

    async def acomplete(self, messages, **kw):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError("server down")
        return Response(content="ok")

    async def astream(self, messages, **kw):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ConnectionError("server down")
        yield "ok"


class AlwaysFailProvider:
    context_window = 128000

    def complete(self, messages, **kw):
        raise ConnectionError("permanent failure")

    def stream(self, messages, **kw):
        raise ConnectionError("permanent failure")

    async def acomplete(self, messages, **kw):
        raise ConnectionError("permanent failure")

    async def astream(self, messages, **kw):
        raise ConnectionError("permanent failure")


def test_retry_succeeds_after_failures():
    inner = FailThenSucceedProvider(fail_count=2)
    provider = RetryProvider(inner, max_retries=3, base_delay=0.01)
    result = provider.complete([])
    assert result.content == "ok"
    assert inner._call_count == 3


def test_retry_gives_up():
    inner = AlwaysFailProvider()
    provider = RetryProvider(inner, max_retries=2, base_delay=0.01)
    with pytest.raises(ConnectionError):
        provider.complete([])


def test_retry_no_retries_needed():
    inner = FailThenSucceedProvider(fail_count=0)
    provider = RetryProvider(inner, max_retries=3, base_delay=0.01)
    result = provider.complete([])
    assert result.content == "ok"
    assert inner._call_count == 1


def test_retry_stream():
    inner = FailThenSucceedProvider(fail_count=1)
    provider = RetryProvider(inner, max_retries=2, base_delay=0.01)
    chunks = list(provider.stream([]))
    assert chunks == ["ok"]


def test_retry_respects_retry_on():
    inner = AlwaysFailProvider()
    provider = RetryProvider(inner, max_retries=2, base_delay=0.01, retry_on=(ValueError,))
    with pytest.raises(ConnectionError):
        provider.complete([])


def test_retry_context_window_passthrough():
    inner = FailThenSucceedProvider(fail_count=0)
    inner.context_window = 42000
    provider = RetryProvider(inner, max_retries=1)
    assert provider.context_window == 42000


def test_retry_exponential_backoff():
    inner = FailThenSucceedProvider(fail_count=2)
    provider = RetryProvider(inner, max_retries=3, base_delay=0.05, max_delay=1.0)
    start = time.time()
    provider.complete([])
    elapsed = time.time() - start
    # 0.05 + 0.10 = 0.15 minimum
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_retry_async():
    inner = FailThenSucceedProvider(fail_count=1)
    provider = RetryProvider(inner, max_retries=2, base_delay=0.01)
    result = await provider.acomplete([])
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_retry_async_stream():
    inner = FailThenSucceedProvider(fail_count=1)
    provider = RetryProvider(inner, max_retries=2, base_delay=0.01)
    chunks = []
    async for chunk in provider.astream([]):
        chunks.append(chunk)
    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_retry_async_gives_up():
    inner = AlwaysFailProvider()
    provider = RetryProvider(inner, max_retries=1, base_delay=0.01)
    with pytest.raises(ConnectionError):
        await provider.acomplete([])
