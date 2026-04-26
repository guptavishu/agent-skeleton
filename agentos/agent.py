"""Agent — the main entry point. Wraps a provider with skills, memory, tools, and HITL."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .coordinator import Coordinator, SequentialCoordinator
from .executor import execute_code, extract_code_blocks
from .hitl import PERMISSIVE, HITLPolicy
from .memory import FileMemory, Memory
from .provider import OllamaProvider, Provider
from .skills import Skill, SkillRegistry
from .telemetry import Telemetry
from .tools import BUILTIN_TOOLS, Tool, ToolRegistry
from .types import Message, Response, ToolCall, ToolResult

MAX_ROUNDS = 20

# --- System prompts ---

HYBRID_PROMPT = """You have two ways to act:

1. **Tool calls** — use the provided tools for discrete operations (reading files, running commands, etc.)
2. **Code blocks** — write Python in fenced ```python blocks for computation, data processing, or anything the tools don't cover.

Pick whichever fits each step. You can mix both in a single response.
When you have the final answer, respond with plain text (no tool calls, no code blocks)."""

CODE_EXEC_PROMPT = """You can execute Python code by writing it in fenced ```python blocks.
Always print() your results so they appear in stdout.
When you have the final answer, respond with plain text (no code blocks)."""

PLANNING_PROMPT = """Before acting, break the task into numbered steps.
Execute each step, then check your progress before moving to the next.
If a step fails, revise your plan."""

REPAIR_PROMPT = """Before giving your final answer, review your work:
- Did you actually accomplish what was asked?
- Are there any errors in your output?
- Is anything missing?
If you find issues, fix them before responding."""


BUILTIN_PROMPTS = {
    "hybrid": HYBRID_PROMPT,
    "code_exec": CODE_EXEC_PROMPT,
    "planning": PLANNING_PROMPT,
    "repair": REPAIR_PROMPT,
}


class Agent:
    """An agent that can use tools, execute code, and delegate to other agents.

    Usage:
        agent = Agent("my-agent")
        result = agent.run("What files are in the current directory?")

        # With skills and memory:
        agent = Agent("coder", skills=[my_skill], memory=FileMemory())
        result = agent.run("Refactor the auth module")

        # Callable shorthand:
        result = agent("What is 2+2?")
    """

    def __init__(
        self,
        name: str = "agent",
        *,
        provider: Provider | None = None,
        model: str = "",
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        skills: list[Skill] | None = None,
        memory: Memory | None = None,
        hitl: HITLPolicy | None = None,
        coordinator: Coordinator | None = None,
        delegates: list[Agent] | None = None,
        orchestration: list[str] | None = None,
        prompts: dict[str, str] | None = None,
        discover_skills: bool = True,
        builtins: bool = True,
        telemetry: Telemetry | None = None,
        # callbacks for visibility
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_code_exec: Callable[[str], None] | None = None,
        on_code_result: Callable[[str], None] | None = None,
    ):
        self.name = name
        self.provider = provider or OllamaProvider()
        self.model = model
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hitl = hitl or PERMISSIVE
        self.coordinator = coordinator or SequentialCoordinator()
        self.delegates = delegates or []
        self.orchestration = orchestration or ["repair"]
        self.telemetry = telemetry or Telemetry()
        self.memory = memory

        # prompts: builtins + user overrides
        self.prompts = dict(BUILTIN_PROMPTS)
        if prompts:
            self.prompts.update(prompts)

        # tools: builtins + user-provided
        self.tool_registry = ToolRegistry()
        if builtins:
            for t in BUILTIN_TOOLS:
                self.tool_registry.register(t)
        for t in (tools or []):
            self.tool_registry.register(t)

        # skills: explicit + discovered
        self.skill_registry = SkillRegistry()
        for s in (skills or []):
            self.skill_registry.register(s)
        if discover_skills:
            self.skill_registry.discover()

        # register skill tools
        for t in self.skill_registry.get_tools():
            self.tool_registry.register(t)

        # callbacks
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_code_exec = on_code_exec
        self.on_code_result = on_code_result

    def __call__(self, prompt: str, **kwargs) -> Response:
        return self.complete(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs) -> Response:
        """Single LLM call, no looping."""
        messages = [Message(role="user", content=prompt)]
        return self.provider.complete(
            messages,
            system=self.system,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    def run(
        self,
        task: str,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> Response:
        """Run the agent loop on a task.

        Default: hybrid mode (tools + code).
        tools_only=True: only tool calls, no code execution.
        exec_code=True: only code execution, no tool calls.
        plan=True: adds planning prompt to orchestration.
        """
        session_id = uuid.uuid4().hex[:12]
        self.telemetry.set_session(session_id)
        self.telemetry.agent_start(self.name, task)

        # build orchestration list
        orch = list(orchestration or self.orchestration)
        if plan and "planning" not in orch:
            orch.insert(0, "planning")

        # build system prompt
        system = self._build_system(orch, tools_only=tools_only, exec_code=exec_code)

        # inject memory context if available
        if self.memory:
            memories = self.memory.retrieve(task, limit=5)
            if memories:
                memory_text = "\n".join(f"- [{m.key}]: {m.content}" for m in memories)
                system += f"\n\n## Relevant Memory\n{memory_text}"

        # inject skill prompts
        skill_prompts = self.skill_registry.get_prompts()
        if skill_prompts:
            system += f"\n\n{skill_prompts}"

        messages = [Message(role="user", content=task)]

        if exec_code:
            response = self._run_code_loop(messages, system, max_rounds)
        elif tools_only:
            response = self._run_tool_loop(messages, system, max_rounds)
        else:
            response = self._run_hybrid_loop(messages, system, max_rounds)

        self.telemetry.agent_finish(self.name, len(messages))
        return response

    def delegate(self, agent: Agent, task: str) -> Response:
        """Delegate a sub-task to another agent."""
        return self.coordinator.delegate(agent, task)

    # --- Execution loops ---

    def _run_tool_loop(self, messages: list[Message], system: str, max_rounds: int) -> Response:
        tools = self.tool_registry.list()
        for _ in range(max_rounds):
            response = self.provider.complete(
                messages, system=system, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                tools=tools,
            )
            self._log_llm(response)

            if not response.tool_calls:
                return response

            # execute tool calls
            messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))
            for tc in response.tool_calls:
                result = self._execute_tool_call(tc)
                messages.append(Message(role="tool", content=result.output or result.error or "", tool_call_id=tc.id))

        return response

    def _run_code_loop(self, messages: list[Message], system: str, max_rounds: int) -> Response:
        for _ in range(max_rounds):
            response = self.provider.complete(
                messages, system=system, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            self._log_llm(response)

            blocks = extract_code_blocks(response.content)
            if not blocks:
                return response

            messages.append(Message(role="assistant", content=response.content))
            for code in blocks:
                result = self._execute_code(code)
                output = result if isinstance(result, str) else f"{result}"
                messages.append(Message(role="user", content=f"[Code output]\n{output}"))

        return response

    def _run_hybrid_loop(self, messages: list[Message], system: str, max_rounds: int) -> Response:
        tools = self.tool_registry.list()
        for _ in range(max_rounds):
            response = self.provider.complete(
                messages, system=system, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                tools=tools,
            )
            self._log_llm(response)

            has_tool_calls = bool(response.tool_calls)
            has_code = bool(extract_code_blocks(response.content))

            if not has_tool_calls and not has_code:
                return response

            messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))

            # handle tool calls
            if has_tool_calls:
                for tc in response.tool_calls:
                    result = self._execute_tool_call(tc)
                    messages.append(
                        Message(role="tool", content=result.output or result.error or "", tool_call_id=tc.id)
                    )

            # handle code blocks
            if has_code:
                for code in extract_code_blocks(response.content):
                    output = self._execute_code(code)
                    messages.append(Message(role="user", content=f"[Code output]\n{output}"))

        return response

    # --- Helpers ---

    def _build_system(self, orchestration: list[str], tools_only: bool, exec_code: bool) -> str:
        parts = []
        if self.system:
            parts.append(self.system)

        # mode prompt
        if not tools_only and not exec_code:
            parts.append(self.prompts.get("hybrid", ""))
        elif exec_code:
            parts.append(self.prompts.get("code_exec", ""))

        # orchestration prompts
        for key in orchestration:
            if key in self.prompts:
                parts.append(self.prompts[key])

        # delegate awareness
        if self.delegates:
            names = ", ".join(d.name for d in self.delegates)
            parts.append(f"You can delegate sub-tasks to these agents: {names}")

        return "\n\n".join(p for p in parts if p)

    def _execute_tool_call(self, tc: ToolCall) -> ToolResult:
        if self.on_tool_call:
            self.on_tool_call(tc)
        self.telemetry.tool_call(tc.name, tc.arguments)

        if not self.hitl.should_approve(tc):
            result = ToolResult(tc.id, "", error="Denied by HITL policy")
        else:
            result = self.tool_registry.execute(tc)

        if self.on_tool_result:
            self.on_tool_result(result)
        self.telemetry.tool_result(tc.name, result.output, result.error)
        return result

    def _execute_code(self, code: str) -> str:
        if self.on_code_exec:
            self.on_code_exec(code)
        self.telemetry.code_exec(code)

        from .executor import execute_code as exec_fn
        result = exec_fn(code)

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        if self.on_code_result:
            self.on_code_result(output)
        self.telemetry.code_result(result.stdout, result.stderr, result.returncode)
        return output

    def _log_llm(self, response: Response) -> None:
        self.telemetry.llm_call(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
