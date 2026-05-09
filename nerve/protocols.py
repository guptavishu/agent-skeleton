from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import ExecResult, MemoryEntry, Message, Response

if TYPE_CHECKING:
    from .tools import Tool


@runtime_checkable
class Provider(Protocol):
    """Implement these methods to plug in any LLM backend."""

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response: ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Iterator[str]: ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response: ...

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class Sandbox(Protocol):
    """Implement this to control where and how LLM-generated code runs."""

    def execute(self, code: str, timeout: int = 30) -> ExecResult: ...


@runtime_checkable
class Memory(Protocol):
    """Implement these methods to plug in any storage backend."""

    def store(self, key: str, content: str, metadata: dict | None = None) -> MemoryEntry: ...
    def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]: ...
    def forget(self, key: str) -> bool: ...
    def list_all(self) -> list[MemoryEntry]: ...


@runtime_checkable
class ContextPolicy(Protocol):
    """Implement this to control how messages are trimmed to fit the context window."""

    def manage(self, messages: list[Message], max_tokens: int) -> list[Message]: ...


@runtime_checkable
class Coordinator(Protocol):
    """Implement this to control multi-agent delegation strategies."""

    def delegate(self, agent, task: str) -> Response: ...
    def broadcast(self, agents: list, message: str) -> list[Response]: ...


@runtime_checkable
class UX(Protocol):
    """Implement this to build a user-facing interface for an agent.

    CLI, web, Slack, TUI, etc. — each is a UX implementation.
    """

    def start(self, agent, **config) -> None: ...
