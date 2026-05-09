"""agent-skeleton — thin, extensible agent framework with reasonable defaults."""

__version__ = "0.1.0"

from .agent import Agent, RunHandle, Session
from .executor import execute_code, extract_code_blocks
from .hitl import HITLPolicy
from .protocols import ContextPolicy, Coordinator, Memory, Provider, Sandbox, UX
from .providers import (
    FileMemory,
    LocalSandbox,
    OllamaProvider,
    RestrictedSandbox,
    RetryProvider,
    SequentialCoordinator,
    SummarizeContext,
    TokenWindowContext,
    parse_tool_calls_from_text,
)
from .skills import Skill, SkillRegistry
from .telemetry import Telemetry
from .tools import BUILTIN_TOOLS, Tool, ToolRegistry
from .types import ExecResult, MemoryEntry, Message, Response, RunState, StopReason, StreamEvent, ToolCall, ToolResult, Usage
from .ux import CliUX, WebUX

__all__ = [
    "Agent",
    "BUILTIN_TOOLS",
    "CliUX",
    "ContextPolicy",
    "Coordinator",
    "ExecResult",
    "FileMemory",
    "HITLPolicy",
    "LocalSandbox",
    "Memory",
    "MemoryEntry",
    "Message",
    "OllamaProvider",
    "Provider",
    "RestrictedSandbox",
    "RetryProvider",
    "Response",
    "RunHandle",
    "RunState",
    "Sandbox",
    "SequentialCoordinator",
    "Session",
    "Skill",
    "SkillRegistry",
    "StopReason",
    "StreamEvent",
    "SummarizeContext",
    "Telemetry",
    "TokenWindowContext",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "UX",
    "Usage",
    "WebUX",
    "execute_code",
    "extract_code_blocks",
    "parse_tool_calls_from_text",
]
