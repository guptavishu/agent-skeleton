"""Tests for OllamaProvider parsing logic (no network calls)."""

from agentos.provider import OllamaProvider, parse_tool_calls_from_text, _try_extract_json
from agentos.types import ToolCall


def _provider():
    return OllamaProvider()


# --- _parse_response ---

def test_parse_response_plain_text():
    p = _provider()
    data = {
        "message": {"role": "assistant", "content": "Hello!"},
        "model": "test",
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    resp = p._parse_response(data)
    assert resp.content == "Hello!"
    assert resp.tool_calls == []
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5


def test_parse_response_structured_tool_calls():
    p = _provider()
    data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    },
                }
            ],
        },
        "model": "test",
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    resp = p._parse_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "a.py"}


def test_parse_response_json_in_content():
    p = _provider()
    data = {
        "message": {
            "role": "assistant",
            "content": '{"name": "list_directory", "arguments": {"path": "."}}',
        },
        "model": "test",
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    resp = p._parse_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "list_directory"
    assert resp.content == ""


def test_parse_response_json_embedded_in_text():
    p = _provider()
    data = {
        "message": {
            "role": "assistant",
            "content": 'Let me check.\n\n{"name": "shell_exec", "arguments": {"command": "ls"}}',
        },
        "model": "test",
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    resp = p._parse_response(data)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "shell_exec"
    assert "Let me check." in resp.content


def test_parse_response_no_false_positive():
    p = _provider()
    data = {
        "message": {
            "role": "assistant",
            "content": '{"key": "value"}',
        },
        "model": "test",
        "done_reason": "stop",
        "prompt_eval_count": 0,
        "eval_count": 0,
    }
    resp = p._parse_response(data)
    assert resp.tool_calls == []
    assert resp.content == '{"key": "value"}'


# --- _try_parse_tool_call ---

def test_try_parse_pure_json():
    calls, text = parse_tool_calls_from_text('{"name": "x", "arguments": {"a": 1}}')
    assert len(calls) == 1
    assert calls[0].name == "x"
    assert text == ""


def test_try_parse_json_array():
    calls, text = parse_tool_calls_from_text(
        '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
    )
    assert len(calls) == 2


def test_try_parse_embedded():
    calls, text = parse_tool_calls_from_text(
        'Some text\n{"name": "x", "arguments": {"k": "v"}}\nMore text'
    )
    assert len(calls) == 1
    assert "Some text" in text
    assert "More text" in text


def test_try_parse_no_match():
    calls, text = parse_tool_calls_from_text("just regular text")
    assert calls == []
    assert text == "just regular text"


def test_try_parse_json_without_name_ignored():
    calls, text = parse_tool_calls_from_text('{"foo": "bar"}')
    assert calls == []


def test_try_parse_string_arguments():
    calls, _ = parse_tool_calls_from_text(
        '{"name": "x", "arguments": "{\\"a\\": 1}"}'
    )
    assert len(calls) == 1
    assert calls[0].arguments == {"a": 1}


# --- _try_extract_json ---

def test_extract_json_simple():
    obj, end = _try_extract_json('{"a": 1}', 0)
    assert obj == {"a": 1}
    assert end == 8


def test_extract_json_with_offset():
    text = 'prefix {"b": 2} suffix'
    obj, end = _try_extract_json(text, 7)
    assert obj == {"b": 2}


def test_extract_json_nested():
    text = '{"a": {"b": 1}}'
    obj, end = _try_extract_json(text, 0)
    assert obj == {"a": {"b": 1}}
    assert end == len(text)


def test_extract_json_with_strings_containing_braces():
    text = '{"a": "{}}"}'
    obj, end = _try_extract_json(text, 0)
    assert obj is not None


def test_extract_json_invalid():
    obj, end = _try_extract_json("{not json}", 0)
    assert obj is None
