"""Tests for core data types."""

from agentos.types import (
    ExecResult,
    MemoryEntry,
    Message,
    Response,
    ToolCall,
    ToolResult,
    Usage,
)


def test_message_defaults():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls == []
    assert msg.tool_call_id is None


def test_message_with_tool_calls():
    tc = ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
    msg = Message(role="assistant", content="", tool_calls=[tc])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "read_file"


def test_tool_result_success():
    tr = ToolResult(tool_call_id="c1", output="file contents")
    assert tr.output == "file contents"
    assert tr.error is None


def test_tool_result_error():
    tr = ToolResult(tool_call_id="c1", output="", error="not found")
    assert tr.error == "not found"


def test_exec_result():
    er = ExecResult(stdout="42\n", stderr="", returncode=0)
    assert er.returncode == 0


def test_usage_defaults():
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.total_tokens == 0


def test_response_defaults():
    r = Response(content="hello")
    assert r.model == ""
    assert r.tool_calls == []
    assert r.usage.total_tokens == 0


def test_memory_entry_has_id_and_timestamp():
    e = MemoryEntry(key="k", content="v")
    assert len(e.id) == 12
    assert e.timestamp > 0
