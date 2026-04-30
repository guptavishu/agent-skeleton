"""Tests for Tool, ToolRegistry, and built-in tools."""

import os
import tempfile
from pathlib import Path

from nerve.tools import BUILTIN_TOOLS, Tool, ToolRegistry
from nerve.types import ToolCall


# --- Tool.from_function ---

def greet(name: str, excited: bool = False) -> str:
    """Say hello."""
    return f"Hello, {name}{'!' if excited else '.'}"


def test_tool_from_function_schema():
    tool = Tool.from_function(greet)
    assert tool.name == "greet"
    assert tool.description == "Say hello."
    assert tool.parameters["properties"]["name"]["type"] == "string"
    assert tool.parameters["properties"]["excited"]["type"] == "boolean"
    assert "name" in tool.parameters["required"]
    assert "excited" not in tool.parameters["required"]


def test_tool_from_function_call():
    tool = Tool.from_function(greet)
    assert tool.function(name="World") == "Hello, World."
    assert tool.function(name="World", excited=True) == "Hello, World!"


def test_tool_ollama_schema():
    tool = Tool.from_function(greet)
    schema = tool.to_ollama_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "greet"
    assert "parameters" in schema["function"]


# --- ToolRegistry ---

def test_registry_register_and_list():
    reg = ToolRegistry()
    tool = Tool.from_function(greet)
    reg.register(tool)
    assert len(reg.list()) == 1
    assert reg.get("greet") is tool


def test_registry_get_unknown():
    reg = ToolRegistry()
    assert reg.get("nope") is None


def test_registry_execute_success():
    reg = ToolRegistry()
    reg.register(Tool.from_function(greet))
    call = ToolCall(id="c1", name="greet", arguments={"name": "Test"})
    result = reg.execute(call)
    assert result.output == "Hello, Test."
    assert result.error is None


def test_registry_execute_unknown_tool():
    reg = ToolRegistry()
    call = ToolCall(id="c1", name="nope", arguments={})
    result = reg.execute(call)
    assert "Unknown tool" in result.error


def test_registry_execute_exception():
    def fail() -> str:
        """Always fails."""
        raise ValueError("boom")

    reg = ToolRegistry()
    reg.register(Tool.from_function(fail))
    call = ToolCall(id="c1", name="fail", arguments={})
    result = reg.execute(call)
    assert "ValueError" in result.error
    assert "boom" in result.error


def test_register_function_shorthand():
    reg = ToolRegistry()
    reg.register_function(greet)
    assert reg.get("greet") is not None


# --- Built-in tools ---

def test_builtin_tools_exist():
    names = {t.name for t in BUILTIN_TOOLS}
    assert names == {"read_file", "write_file", "shell_exec", "list_directory"}


def test_read_file():
    tool = next(t for t in BUILTIN_TOOLS if t.name == "read_file")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        f.flush()
        path = f.name
    try:
        result = tool.function(path=path)
        assert result == "hello world"
    finally:
        os.unlink(path)


def test_write_file():
    tool = next(t for t in BUILTIN_TOOLS if t.name == "write_file")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "out.txt")
        result = tool.function(path=path, content="test data")
        assert "Wrote" in result
        assert Path(path).read_text() == "test data"


def test_shell_exec():
    tool = next(t for t in BUILTIN_TOOLS if t.name == "shell_exec")
    result = tool.function(command="echo hi")
    assert "hi" in result


def test_list_directory():
    tool = next(t for t in BUILTIN_TOOLS if t.name == "list_directory")
    with tempfile.TemporaryDirectory() as d:
        Path(d, "a.txt").touch()
        Path(d, "subdir").mkdir()
        result = tool.function(path=d)
        assert "a.txt" in result
        assert "subdir" in result
        assert "file" in result
        assert "dir" in result
