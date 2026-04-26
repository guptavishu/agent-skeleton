"""Multi-agent coordination — delegate tasks between agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import Response

if TYPE_CHECKING:
    from .agent import Agent


@runtime_checkable
class Coordinator(Protocol):
    """Interface for multi-agent coordination strategies."""

    def delegate(self, agent: Agent, task: str) -> Response: ...
    def broadcast(self, agents: list[Agent], message: str) -> list[Response]: ...


class SequentialCoordinator:
    """Runs delegated tasks one at a time. The simplest coordinator."""

    def delegate(self, agent: Agent, task: str) -> Response:
        return agent.run(task)

    def broadcast(self, agents: list[Agent], message: str) -> list[Response]:
        return [agent.run(message) for agent in agents]
