from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..types import Message, Response


class RetryProvider:
    """Wraps any provider with retry and exponential backoff on errors."""

    def __init__(
        self,
        provider,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ):
        self._provider = provider
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on

    @property
    def context_window(self) -> int:
        return getattr(self._provider, 'context_window', 128000)

    def _delay(self, attempt: int) -> float:
        return min(self.base_delay * (2 ** attempt), self.max_delay)

    def complete(self, messages, **kwargs) -> Response:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._provider.complete(messages, **kwargs)
            except self.retry_on as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self._delay(attempt))
        raise last_error

    def stream(self, messages, **kwargs) -> Iterator[str]:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                yield from self._provider.stream(messages, **kwargs)
                return
            except self.retry_on as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self._delay(attempt))
        raise last_error

    async def acomplete(self, messages, **kwargs) -> Response:
        import asyncio
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._provider.acomplete(messages, **kwargs)
            except self.retry_on as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._delay(attempt))
        raise last_error

    async def astream(self, messages, **kwargs) -> AsyncIterator[str]:
        import asyncio
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in self._provider.astream(messages, **kwargs):
                    yield chunk
                return
            except self.retry_on as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self._delay(attempt))
        raise last_error
