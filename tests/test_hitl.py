"""Tests for HITL policy."""

from agentos.hitl import HITLPolicy, PERMISSIVE
from agentos.types import ToolCall


def _call(name: str) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments={})


def test_permissive_approves_everything():
    assert PERMISSIVE.should_approve(_call("anything")) is True


def test_auto_approve_bypasses_gate():
    policy = HITLPolicy(
        approve_before=["shell_exec"],
        auto_approve=["shell_exec"],
    )
    assert policy.should_approve(_call("shell_exec")) is True


def test_approve_before_calls_confirm():
    called = []

    def fake_confirm(tc: ToolCall) -> bool:
        called.append(tc.name)
        return False

    policy = HITLPolicy(approve_before=["shell_exec"], confirm_fn=fake_confirm)
    assert policy.should_approve(_call("shell_exec")) is False
    assert called == ["shell_exec"]


def test_approve_before_confirm_allows():
    policy = HITLPolicy(
        approve_before=["shell_exec"],
        confirm_fn=lambda tc: True,
    )
    assert policy.should_approve(_call("shell_exec")) is True


def test_unlisted_tool_auto_approved():
    policy = HITLPolicy(approve_before=["shell_exec"])
    assert policy.should_approve(_call("read_file")) is True


def test_empty_approve_before_approves_all():
    policy = HITLPolicy()
    assert policy.should_approve(_call("shell_exec")) is True
    assert policy.should_approve(_call("read_file")) is True
