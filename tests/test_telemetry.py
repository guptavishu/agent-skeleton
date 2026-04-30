"""Tests for Telemetry logging."""

import json
import tempfile
from pathlib import Path

from nerve.telemetry import Telemetry


def _make_telemetry():
    d = tempfile.mkdtemp()
    path = Path(d) / "test.log"
    return Telemetry(path=path), path


def _read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().split("\n") if line]


def test_log_writes_jsonl():
    t, path = _make_telemetry()
    t.log("test_event", foo="bar")
    events = _read_events(path)
    assert len(events) == 1
    assert events[0]["event"] == "test_event"
    assert events[0]["foo"] == "bar"
    assert "ts" in events[0]


def test_session_id_included():
    t, path = _make_telemetry()
    t.set_session("abc123")
    t.log("e")
    events = _read_events(path)
    assert events[0]["session"] == "abc123"


def test_session_id_absent_when_not_set():
    t, path = _make_telemetry()
    t.log("e")
    events = _read_events(path)
    assert "session" not in events[0]


def test_tool_call_event():
    t, path = _make_telemetry()
    t.tool_call("read_file", {"path": "a.py"})
    events = _read_events(path)
    assert events[0]["event"] == "tool_call"
    assert events[0]["tool"] == "read_file"


def test_tool_result_truncates():
    t, path = _make_telemetry()
    t.tool_result("x", output="a" * 1000)
    events = _read_events(path)
    assert len(events[0]["output"]) == 500


def test_llm_call_event():
    t, path = _make_telemetry()
    t.llm_call("model-x", 100, 50)
    events = _read_events(path)
    assert events[0]["model"] == "model-x"
    assert events[0]["prompt_tokens"] == 100


def test_agent_lifecycle():
    t, path = _make_telemetry()
    t.agent_start("bot", "do stuff")
    t.agent_finish("bot", 3)
    events = _read_events(path)
    assert events[0]["event"] == "agent_start"
    assert events[1]["event"] == "agent_finish"
    assert events[1]["rounds"] == 3
