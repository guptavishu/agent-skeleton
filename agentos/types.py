from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    DONE = "done"
    MAX_ROUNDS = "max_rounds"
    INTERRUPTED = "interrupted"
    DEFERRED = "deferred"


@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    output: str
    error: str | None = None


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Response:
    content: str
    model: str = ""
    stop_reason: str = ""
    elapsed: float = 0.0
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class StreamEvent:
    type: str  # "text", "tool_call", "tool_result", "code_exec", "code_result", "done", "error"
    data: Any = None


@dataclass
class RunState:
    state_id: str
    agent_name: str
    task: str
    messages: list[Message]
    system: str
    mode: str  # "hybrid", "tools_only", "code_only"
    round: int
    max_rounds: int
    session_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryEntry:
    key: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
