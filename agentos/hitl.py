"""Human-in-the-loop controls — approval gates and supervision hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import ToolCall


def _terminal_confirm(call: ToolCall) -> bool:
    """Ask the user in the terminal whether to allow a tool call."""
    print(f"\n[HITL] Agent wants to call: {call.name}")
    print(f"  Arguments: {call.arguments}")
    answer = input("  Allow? [y/N] ").strip().lower()
    return answer in ("y", "yes")


@dataclass
class HITLPolicy:
    """Controls which tool calls need human approval before execution.

    approve_before: tool names that require human confirmation
    auto_approve: tool names that run without asking (default: everything)
    on_stuck: callback when the agent hits max rounds without finishing
    confirm_fn: function that asks for human approval (default: terminal input)
    """

    approve_before: list[str] = field(default_factory=list)
    auto_approve: list[str] = field(default_factory=list)
    on_stuck: Callable[[dict], None] | None = None
    confirm_fn: Callable[[ToolCall], bool] = field(default=None)

    def __post_init__(self):
        if self.confirm_fn is None:
            self.confirm_fn = _terminal_confirm

    def should_approve(self, call: ToolCall) -> bool:
        """Check whether a tool call is allowed to proceed."""
        if call.name in self.auto_approve:
            return True
        if call.name in self.approve_before:
            return self.confirm_fn(call)
        # if neither list mentions the tool, auto-approve by default
        if not self.approve_before:
            return True
        # if approve_before is set but doesn't include this tool, auto-approve
        return True


# Default: no gates, everything auto-approved
PERMISSIVE = HITLPolicy()
