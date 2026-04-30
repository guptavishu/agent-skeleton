from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .executor import extract_code_blocks
from .hitl import PERMISSIVE, HITLPolicy
from .protocols import ContextPolicy, Coordinator, Memory, Provider, Sandbox
from .providers import FileMemory, LocalSandbox, OllamaProvider, SequentialCoordinator, TokenWindowContext, parse_tool_calls_from_text
from .skills import Skill, SkillRegistry
from .telemetry import Telemetry
from .tools import BUILTIN_TOOLS, Tool, ToolRegistry
from .types import Message, Response, RunState, StopReason, StreamEvent, ToolCall, ToolResult, Usage

DEFERRED_DIR = Path.home() / ".nerve" / "deferred"

MAX_ROUNDS = 20

HYBRID_PROMPT = """You have two ways to act:

1. **Tool calls** — use the provided tools for discrete operations (reading files, running commands, etc.)
2. **Code blocks** — write Python in fenced ```python blocks for computation, data processing, or anything the tools don't cover.

Pick whichever fits each step. You can mix both in a single response.
When you have the final answer, respond with plain text (no tool calls, no code blocks)."""

CODE_EXEC_PROMPT = """You can execute Python code by writing it in fenced ```python blocks.
The code runs locally on the user's machine in a subprocess with full filesystem and network access.
You can read/write files, run shell commands via subprocess, install packages, and do anything Python can do.
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


class RunHandle:
    """Handle for a background agent run. Returned by Agent.run_background()."""

    def __init__(self, agent: Agent, thread: threading.Thread):
        self._agent = agent
        self._thread = thread
        self._result: Response | None = None
        self._error: Exception | None = None
        self._event = threading.Event()
        self._callbacks: list[Callable[[Response], None]] = []

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def result(self) -> Response:
        """Block until the run completes and return the response."""
        self._event.wait()
        if self._error:
            raise self._error
        return self._result

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to `timeout` seconds. Returns True if done."""
        return self._event.wait(timeout)

    def stop(self) -> None:
        self._agent.stop()

    def defer(self) -> None:
        self._agent.defer()

    def on_complete(self, callback: Callable[[Response], None]) -> None:
        """Register a callback fired when the run finishes."""
        if self._event.is_set():
            callback(self._result)
        else:
            self._callbacks.append(callback)

    def _finish(self, response: Response) -> None:
        self._result = response
        self._event.set()
        for cb in self._callbacks:
            try:
                cb(response)
            except Exception:
                pass

    def _fail(self, error: Exception) -> None:
        self._error = error
        self._event.set()
        for cb in self._callbacks:
            try:
                cb(Response(content="", stop_reason="error"))
            except Exception:
                pass


class Session:
    """Multi-turn conversation session. Maintains message history across calls."""

    def __init__(
        self,
        agent: Agent,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ):
        self._agent = agent
        self._tools_only = tools_only
        self._exec_code = exec_code
        self._plan = plan
        self._orchestration = orchestration
        self._max_rounds = max_rounds

        self._session_id = uuid.uuid4().hex[:12]
        self._messages: list[Message] = []
        self._system = self._build_system()
        self._total_usage = Usage()

    @property
    def _use_tools(self) -> bool:
        return not self._exec_code

    @property
    def _use_code(self) -> bool:
        return not self._tools_only

    def _build_system(self) -> str:
        orch = list(self._orchestration or self._agent.orchestration)
        if self._plan and "planning" not in orch:
            orch.insert(0, "planning")
        system = self._agent._build_system(orch, tools_only=self._tools_only, exec_code=self._exec_code)
        skill_prompts = self._agent.skill_registry.get_prompts()
        if skill_prompts:
            system += f"\n\n{skill_prompts}"
        return system

    def _prepare_send(self, message: str) -> None:
        self._agent._stopped = False
        self._agent._defer_requested = False
        self._agent.telemetry.set_session(self._session_id)

    def _system_with_memory(self, message: str) -> str:
        system = self._system
        if self._agent.memory:
            memories = self._agent.memory.retrieve(message, limit=5)
            if memories:
                memory_text = "\n".join(f"- [{m.key}]: {m.content}" for m in memories)
                system += f"\n\n## Relevant Memory\n{memory_text}"
        return system

    def send(self, message: str, **kwargs) -> Response:
        """Send a message and get a response. Conversation history is preserved."""
        self._prepare_send(message)
        system = self._system_with_memory(message)
        self._messages.append(Message(role="user", content=message))

        mode = "code_only" if self._exec_code else ("tools_only" if self._tools_only else "hybrid")
        response = self._agent._run_loop(
            self._messages, system, mode,
            self._max_rounds, self._session_id, message,
        )

        self._messages.append(Message(role="assistant", content=response.content))
        self._accumulate_usage(response.usage)
        return response

    def send_stream(self, message: str, **kwargs) -> Iterator[StreamEvent]:
        """Send a message and stream the response. Conversation history is preserved."""
        self._prepare_send(message)
        system = self._system_with_memory(message)
        self._messages.append(Message(role="user", content=message))
        yield from self._agent._stream_loop(self._messages, system, self._use_tools, self._use_code, self._max_rounds)

    async def asend_stream(self, message: str, **kwargs) -> AsyncIterator[StreamEvent]:
        """Async version of send_stream()."""
        self._prepare_send(message)
        system = self._system_with_memory(message)
        self._messages.append(Message(role="user", content=message))
        async for event in self._agent._astream_loop(self._messages, system, self._use_tools, self._use_code, self._max_rounds):
            yield event

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def session_id(self) -> str:
        return self._session_id

    async def asend(self, message: str, **kwargs) -> Response:
        """Async version of send()."""
        self._prepare_send(message)
        system = self._system_with_memory(message)
        self._messages.append(Message(role="user", content=message))

        mode = "code_only" if self._exec_code else ("tools_only" if self._tools_only else "hybrid")
        response = await self._agent._arun_loop(
            self._messages, system, mode,
            self._max_rounds, self._session_id, message,
        )

        self._messages.append(Message(role="assistant", content=response.content))
        self._accumulate_usage(response.usage)
        return response

    def _accumulate_usage(self, usage: Usage) -> None:
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        self._total_usage.total_tokens += usage.total_tokens

    @property
    def usage(self) -> Usage:
        return Usage(
            self._total_usage.prompt_tokens,
            self._total_usage.completion_tokens,
            self._total_usage.total_tokens,
        )

    def clear(self) -> None:
        """Reset conversation history."""
        self._messages.clear()


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
        sandbox: Sandbox | None = None,
        context: ContextPolicy | None = None,
        telemetry: Telemetry | None = None,
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_code_exec: Callable[[str], None] | None = None,
        on_code_result: Callable[[str], None] | None = None,
    ):
        self.name = name
        if provider is None:
            try:
                provider = OllamaProvider()
            except ImportError:
                raise TypeError(
                    "No provider specified and OllamaProvider is not available. "
                    "Either pass a provider: Agent(provider=MyProvider()) "
                    "or install Ollama support: pip install nerve[ollama]"
                ) from None
        self.provider = provider
        self.model = model
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hitl = hitl or PERMISSIVE
        self.coordinator = coordinator or SequentialCoordinator()
        self.delegates = delegates or []
        self.orchestration = orchestration or ["repair"]
        self.sandbox = sandbox or LocalSandbox()
        self.context = context or TokenWindowContext()
        self.telemetry = telemetry or Telemetry()
        self.memory = memory

        self.prompts = dict(BUILTIN_PROMPTS)
        if prompts:
            self.prompts.update(prompts)

        self.tool_registry = ToolRegistry()
        if builtins:
            for t in BUILTIN_TOOLS:
                self.tool_registry.register(t)
        for t in (tools or []):
            self.tool_registry.register(t)

        self.skill_registry = SkillRegistry()
        for s in (skills or []):
            self.skill_registry.register(s)
        if discover_skills:
            self.skill_registry.discover()

        for t in self.skill_registry.get_tools():
            self.tool_registry.register(t)

        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_code_exec = on_code_exec
        self.on_code_result = on_code_result

        self._stopped = False
        self._defer_requested = False
        self._last_state: RunState | None = None

    def stop(self) -> None:
        """Signal the agent to stop after the current round."""
        self._stopped = True

    def defer(self) -> None:
        """Signal the agent to save state and stop after the current round."""
        self._defer_requested = True
        self._stopped = True

    @property
    def last_state(self) -> RunState | None:
        """The saved state from the most recent deferred run."""
        return self._last_state

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
        self._stopped = False
        self._defer_requested = False
        session_id = uuid.uuid4().hex[:12]
        self.telemetry.set_session(session_id)
        self.telemetry.agent_start(self.name, task)

        system, mode = self._build_run_context(task, tools_only, exec_code, plan, orchestration)
        messages = [Message(role="user", content=task)]
        response = self._run_loop(messages, system, mode, max_rounds, session_id, task)

        self.telemetry.agent_finish(self.name, len(messages))
        return response

    def resume(self, state: RunState | str) -> Response:
        """Resume a deferred run from saved state."""
        if isinstance(state, str):
            state = self._load_state(state)

        self._stopped = False
        self._defer_requested = False
        self.telemetry.set_session(state.session_id)

        response = self._run_loop(
            state.messages, state.system, state.mode,
            state.max_rounds, state.session_id, state.task,
            start_round=state.round,
        )

        self.telemetry.agent_finish(self.name, len(state.messages))
        return response

    def run_stream(
        self,
        task: str,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> Iterator[StreamEvent]:
        """Run the agent loop, yielding StreamEvents as they happen."""
        self._stopped = False
        self._defer_requested = False
        session_id = uuid.uuid4().hex[:12]
        self.telemetry.set_session(session_id)
        self.telemetry.agent_start(self.name, task)

        system, mode = self._build_run_context(task, tools_only, exec_code, plan, orchestration)
        use_tools = mode in ("tools_only", "hybrid")
        use_code = mode in ("code_only", "hybrid")
        messages = [Message(role="user", content=task)]

        yield from self._stream_loop(messages, system, use_tools, use_code, max_rounds)
        self.telemetry.agent_finish(self.name, len(messages))

    async def acomplete(self, prompt: str, **kwargs) -> Response:
        """Async single LLM call, no looping."""
        messages = [Message(role="user", content=prompt)]
        return await self.provider.acomplete(
            messages,
            system=self.system,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

    async def arun(
        self,
        task: str,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> Response:
        """Async version of run()."""
        self._stopped = False
        self._defer_requested = False
        session_id = uuid.uuid4().hex[:12]
        self.telemetry.set_session(session_id)
        self.telemetry.agent_start(self.name, task)

        system, mode = self._build_run_context(task, tools_only, exec_code, plan, orchestration)
        messages = [Message(role="user", content=task)]
        response = await self._arun_loop(messages, system, mode, max_rounds, session_id, task)

        self.telemetry.agent_finish(self.name, len(messages))
        return response

    async def arun_stream(
        self,
        task: str,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> AsyncIterator[StreamEvent]:
        """Async version of run_stream()."""
        self._stopped = False
        self._defer_requested = False
        session_id = uuid.uuid4().hex[:12]
        self.telemetry.set_session(session_id)
        self.telemetry.agent_start(self.name, task)

        system, mode = self._build_run_context(task, tools_only, exec_code, plan, orchestration)
        use_tools = mode in ("tools_only", "hybrid")
        use_code = mode in ("code_only", "hybrid")
        messages = [Message(role="user", content=task)]

        async for event in self._astream_loop(messages, system, use_tools, use_code, max_rounds):
            yield event
        self.telemetry.agent_finish(self.name, len(messages))

    def run_background(
        self,
        task: str,
        *,
        on_complete: Callable[[Response], None] | None = None,
        **kwargs,
    ) -> RunHandle:
        """Run the agent in a background thread. Returns a RunHandle immediately."""
        handle = RunHandle.__new__(RunHandle)
        thread = threading.Thread(target=self._bg_worker, args=(handle, task, kwargs), daemon=True)
        handle.__init__(self, thread)
        if on_complete:
            handle.on_complete(on_complete)
        thread.start()
        return handle

    def _bg_worker(self, handle: RunHandle, task: str, kwargs: dict) -> None:
        try:
            response = self.run(task, **kwargs)
            handle._finish(response)
        except Exception as e:
            handle._fail(e)

    def session(
        self,
        *,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        orchestration: list[str] | None = None,
        max_rounds: int = MAX_ROUNDS,
    ) -> Session:
        """Create a multi-turn conversation session."""
        return Session(
            self,
            tools_only=tools_only,
            exec_code=exec_code,
            plan=plan,
            orchestration=orchestration,
            max_rounds=max_rounds,
        )

    def delegate(self, agent: Agent, task: str) -> Response:
        """Delegate a sub-task to another agent."""
        return self.coordinator.delegate(agent, task)

    def _check_stopped(self, messages, system, mode, round_num, max_rounds, session_id, task):
        """Check if the agent was stopped/deferred. Returns a StopReason or None."""
        if not self._stopped:
            return None
        reason = StopReason.DEFERRED if self._defer_requested else StopReason.INTERRUPTED
        if self._defer_requested:
            self._save_state(RunState(
                state_id=uuid.uuid4().hex[:12],
                agent_name=self.name,
                task=task,
                messages=list(messages),
                system=system,
                mode=mode,
                round=round_num,
                max_rounds=max_rounds,
                session_id=session_id,
            ))
        return reason

    def _process_response(self, response, messages, use_tools, use_code):
        """Execute tool calls and code blocks from a response. Returns True if the loop should continue."""
        has_tool_calls = use_tools and bool(response.tool_calls)
        has_code = use_code and bool(extract_code_blocks(response.content))

        if not has_tool_calls and not has_code:
            return False

        messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))

        if has_tool_calls:
            for tc in response.tool_calls:
                result = self._execute_tool_call(tc)
                messages.append(
                    Message(role="tool", content=result.output or result.error or "", tool_call_id=tc.id)
                )

        if has_code:
            for code in extract_code_blocks(response.content):
                output = self._execute_code(code)
                messages.append(Message(role="user", content=f"[Code output]\n{output}"))

        return True

    def _run_loop(
        self,
        messages: list[Message],
        system: str,
        mode: str,
        max_rounds: int,
        session_id: str,
        task: str,
        start_round: int = 0,
    ) -> Response:
        use_tools = mode in ("tools_only", "hybrid")
        use_code = mode in ("code_only", "hybrid")
        tools = self.tool_registry.list() if use_tools else None

        start_time = time.time()
        total_usage = Usage()
        response = Response(content="")

        for round_num in range(start_round, max_rounds):
            reason = self._check_stopped(messages, system, mode, round_num, max_rounds, session_id, task)
            if reason:
                response.stop_reason = reason.value
                response.elapsed = time.time() - start_time
                response.usage = total_usage
                return response

            managed = self.context.manage(messages, getattr(self.provider, 'context_window', 128000))
            response = self.provider.complete(
                managed, system=system, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                tools=tools,
            )
            self._log_llm(response)
            total_usage.prompt_tokens += response.usage.prompt_tokens
            total_usage.completion_tokens += response.usage.completion_tokens
            total_usage.total_tokens += response.usage.total_tokens

            if not self._process_response(response, messages, use_tools, use_code):
                response.stop_reason = StopReason.DONE.value
                response.elapsed = time.time() - start_time
                response.usage = total_usage
                return response

        response.stop_reason = StopReason.MAX_ROUNDS.value
        response.elapsed = time.time() - start_time
        response.usage = total_usage
        return response

    async def _arun_loop(
        self,
        messages: list[Message],
        system: str,
        mode: str,
        max_rounds: int,
        session_id: str,
        task: str,
        start_round: int = 0,
    ) -> Response:
        use_tools = mode in ("tools_only", "hybrid")
        use_code = mode in ("code_only", "hybrid")
        tools = self.tool_registry.list() if use_tools else None

        start_time = time.time()
        total_usage = Usage()
        response = Response(content="")

        for round_num in range(start_round, max_rounds):
            reason = self._check_stopped(messages, system, mode, round_num, max_rounds, session_id, task)
            if reason:
                response.stop_reason = reason.value
                response.elapsed = time.time() - start_time
                response.usage = total_usage
                return response

            managed = self.context.manage(messages, getattr(self.provider, 'context_window', 128000))
            response = await self.provider.acomplete(
                managed, system=system, model=self.model,
                temperature=self.temperature, max_tokens=self.max_tokens,
                tools=tools,
            )
            self._log_llm(response)
            total_usage.prompt_tokens += response.usage.prompt_tokens
            total_usage.completion_tokens += response.usage.completion_tokens
            total_usage.total_tokens += response.usage.total_tokens

            if not self._process_response(response, messages, use_tools, use_code):
                response.stop_reason = StopReason.DONE.value
                response.elapsed = time.time() - start_time
                response.usage = total_usage
                return response

        response.stop_reason = StopReason.MAX_ROUNDS.value
        response.elapsed = time.time() - start_time
        response.usage = total_usage
        return response

    def _process_stream_text(self, full_text, messages, use_tools, use_code):
        """Process streamed text for tool calls and code. Returns (events, should_continue)."""
        tool_calls, remaining_text = parse_tool_calls_from_text(full_text)
        has_tool_calls = use_tools and bool(tool_calls)
        has_code = use_code and bool(extract_code_blocks(full_text))

        if not has_tool_calls and not has_code:
            messages.append(Message(role="assistant", content=full_text))
            return [], False

        events = []
        messages.append(Message(role="assistant", content=remaining_text, tool_calls=tool_calls))

        if has_tool_calls:
            for tc in tool_calls:
                result = self._execute_tool_call(tc)
                events.append(StreamEvent("tool_call", tc))
                events.append(StreamEvent("tool_result", result))
                messages.append(Message(
                    role="tool", content=result.output or result.error or "", tool_call_id=tc.id,
                ))

        if has_code:
            for code in extract_code_blocks(full_text):
                output = self._execute_code(code)
                events.append(StreamEvent("code_exec", code))
                events.append(StreamEvent("code_result", output))
                messages.append(Message(role="user", content=f"[Code output]\n{output}"))

        return events, True

    def _stream_loop(self, messages, system, use_tools, use_code, max_rounds):
        """Core sync streaming loop shared by Agent.run_stream and Session.send_stream."""
        tools = self.tool_registry.list() if use_tools else None
        full_text = ""

        for round_num in range(max_rounds):
            if self._stopped:
                yield StreamEvent("done", Response(content="", stop_reason=StopReason.INTERRUPTED.value))
                return

            chunks: list[str] = []
            managed = self.context.manage(messages, getattr(self.provider, 'context_window', 128000))
            try:
                for chunk in self.provider.stream(
                    managed, system=system, model=self.model,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    tools=tools,
                ):
                    chunks.append(chunk)
                    yield StreamEvent("text", chunk)
            except Exception as e:
                yield StreamEvent("error", str(e))
                return

            full_text = "".join(chunks)
            events, should_continue = self._process_stream_text(full_text, messages, use_tools, use_code)

            for event in events:
                yield event

            if not should_continue:
                yield StreamEvent("done", Response(content=full_text, stop_reason=StopReason.DONE.value))
                return

        yield StreamEvent("done", Response(content=full_text, stop_reason=StopReason.MAX_ROUNDS.value))

    async def _astream_loop(self, messages, system, use_tools, use_code, max_rounds):
        """Core async streaming loop shared by Agent.arun_stream and Session.asend_stream."""
        tools = self.tool_registry.list() if use_tools else None
        full_text = ""

        for round_num in range(max_rounds):
            if self._stopped:
                yield StreamEvent("done", Response(content="", stop_reason=StopReason.INTERRUPTED.value))
                return

            chunks: list[str] = []
            managed = self.context.manage(messages, getattr(self.provider, 'context_window', 128000))
            try:
                async for chunk in self.provider.astream(
                    managed, system=system, model=self.model,
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    tools=tools,
                ):
                    chunks.append(chunk)
                    yield StreamEvent("text", chunk)
            except Exception as e:
                yield StreamEvent("error", str(e))
                return

            full_text = "".join(chunks)
            events, should_continue = self._process_stream_text(full_text, messages, use_tools, use_code)

            for event in events:
                yield event

            if not should_continue:
                yield StreamEvent("done", Response(content=full_text, stop_reason=StopReason.DONE.value))
                return

        yield StreamEvent("done", Response(content=full_text, stop_reason=StopReason.MAX_ROUNDS.value))

    def _build_run_context(
        self,
        task: str,
        tools_only: bool,
        exec_code: bool,
        plan: bool,
        orchestration: list[str] | None,
    ) -> tuple[str, str]:
        """Build system prompt and mode string. Shared by sync and async run methods."""
        orch = list(orchestration or self.orchestration)
        if plan and "planning" not in orch:
            orch.insert(0, "planning")

        system = self._build_system(orch, tools_only=tools_only, exec_code=exec_code)

        if self.memory:
            memories = self.memory.retrieve(task, limit=5)
            if memories:
                memory_text = "\n".join(f"- [{m.key}]: {m.content}" for m in memories)
                system += f"\n\n## Relevant Memory\n{memory_text}"

        skill_prompts = self.skill_registry.get_prompts()
        if skill_prompts:
            system += f"\n\n{skill_prompts}"

        mode = "code_only" if exec_code else ("tools_only" if tools_only else "hybrid")
        return system, mode

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

        result = self.sandbox.execute(code)

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        if self.on_code_result:
            self.on_code_result(output)
        self.telemetry.code_result(result.stdout, result.stderr, result.returncode)
        return output

    def _save_state(self, state: RunState) -> None:
        DEFERRED_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "state_id": state.state_id,
            "agent_name": state.agent_name,
            "task": state.task,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in m.tool_calls
                    ],
                    "tool_call_id": m.tool_call_id,
                }
                for m in state.messages
            ],
            "system": state.system,
            "mode": state.mode,
            "round": state.round,
            "max_rounds": state.max_rounds,
            "session_id": state.session_id,
            "timestamp": state.timestamp,
        }
        path = DEFERRED_DIR / f"{state.state_id}.json"
        path.write_text(json.dumps(data, indent=2))
        self._last_state = state
        self.telemetry.log("deferred", state_id=state.state_id, round=state.round)

    @staticmethod
    def _load_state(state_id: str) -> RunState:
        path = DEFERRED_DIR / f"{state_id}.json"
        data = json.loads(path.read_text())
        messages = []
        for m in data["messages"]:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in m.get("tool_calls", [])
            ]
            messages.append(Message(
                role=m["role"],
                content=m["content"],
                tool_calls=tool_calls,
                tool_call_id=m.get("tool_call_id"),
            ))
        return RunState(
            state_id=data["state_id"],
            agent_name=data["agent_name"],
            task=data["task"],
            messages=messages,
            system=data["system"],
            mode=data["mode"],
            round=data["round"],
            max_rounds=data["max_rounds"],
            session_id=data["session_id"],
            timestamp=data.get("timestamp", 0),
        )

    @staticmethod
    def list_deferred() -> list[RunState]:
        if not DEFERRED_DIR.exists():
            return []
        states = []
        for f in sorted(DEFERRED_DIR.glob("*.json")):
            try:
                states.append(Agent._load_state(f.stem))
            except (json.JSONDecodeError, KeyError):
                pass
        return states

    def _log_llm(self, response: Response) -> None:
        self.telemetry.llm_call(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
