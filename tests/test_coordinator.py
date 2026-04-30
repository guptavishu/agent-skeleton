"""Tests for coordinator."""

from nerve.providers.coordinator import SequentialCoordinator
from nerve.types import Response


class FakeAgent:
    def __init__(self, name, response_text):
        self.name = name
        self._response = Response(content=response_text)

    def run(self, task):
        return self._response


def test_delegate():
    coord = SequentialCoordinator()
    agent = FakeAgent("worker", "done")
    result = coord.delegate(agent, "do something")
    assert result.content == "done"


def test_broadcast():
    coord = SequentialCoordinator()
    agents = [FakeAgent("a", "r1"), FakeAgent("b", "r2")]
    results = coord.broadcast(agents, "hello")
    assert len(results) == 2
    assert results[0].content == "r1"
    assert results[1].content == "r2"
