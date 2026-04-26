"""Tool registry with auto-schema generation and built-in tools."""

from __future__ import annotations

import inspect
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints

from .types import ToolCall, ToolResult

# Python type → JSON Schema type
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    function: Callable[..., str]

    @classmethod
    def from_function(cls, fn: Callable[..., str]) -> Tool:
        """Build a Tool from a typed function. Introspects type hints and docstring."""
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            ptype = hints.get(param_name, str)
            json_type = _TYPE_MAP.get(ptype, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return cls(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0],
            parameters=schema,
            function=fn,
        )

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Manages tool registration and execution."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_function(self, fn: Callable[..., str]) -> None:
        self.register(Tool.from_function(fn))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(call.id, "", error=f"Unknown tool: {call.name}")
        try:
            result = tool.function(**call.arguments)
            return ToolResult(call.id, str(result))
        except Exception as e:
            return ToolResult(call.id, "", error=f"{type(e).__name__}: {e}")


# --- Built-in tools ---

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    return Path(path).read_text()


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def shell_exec(command: str, timeout: int = 30) -> str:
    """Run a shell command and return stdout + stderr."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    entries = sorted(Path(path).iterdir())
    lines = []
    for e in entries:
        kind = "dir" if e.is_dir() else "file"
        lines.append(f"  {kind}  {e.name}")
    return "\n".join(lines) if lines else "(empty directory)"


BUILTIN_TOOLS = [
    Tool.from_function(read_file),
    Tool.from_function(write_file),
    Tool.from_function(shell_exec),
    Tool.from_function(list_directory),
]
