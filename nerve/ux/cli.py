"""CLI UX — interactive terminal interface for a Nerve agent."""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

from ..types import StopReason, ToolCall, ToolResult

if TYPE_CHECKING:
    from ..agent import Agent


class CliUX:
    """Interactive terminal UX. Supports one-shot and REPL modes."""

    def start(
        self,
        agent: Agent,
        *,
        prompt: str | None = None,
        tools_only: bool = False,
        exec_code: bool = False,
        plan: bool = False,
        max_rounds: int = 20,
    ) -> None:
        agent.on_tool_call = agent.on_tool_call or _print_tool_call
        agent.on_tool_result = agent.on_tool_result or _print_tool_result
        agent.on_code_exec = agent.on_code_exec or _print_code
        agent.on_code_result = agent.on_code_result or _print_code_result

        run_kwargs = dict(tools_only=tools_only, exec_code=exec_code, plan=plan, max_rounds=max_rounds)

        if prompt:
            response = _run_interruptible(agent, prompt, run_kwargs)
            print(response.content)
        else:
            _interactive(agent, run_kwargs)

    def __repr__(self) -> str:
        return "CliUX()"


def _print_tool_call(tc: ToolCall) -> None:
    print(f"  [tool] {tc.name}({tc.arguments})")


def _print_tool_result(tr: ToolResult) -> None:
    output = tr.output[:200] if tr.output else ""
    if tr.error:
        print(f"  [error] {tr.error}")
    elif output:
        preview = output.replace("\n", "\\n")
        print(f"  [result] {preview}")


def _print_code(code: str) -> None:
    lines = code.strip().split("\n")
    preview = "\n".join(lines[:5])
    if len(lines) > 5:
        preview += f"\n  ... ({len(lines) - 5} more lines)"
    print(f"  [code]\n{preview}")


def _print_code_result(output: str) -> None:
    preview = output[:200].replace("\n", "\\n")
    print(f"  [output] {preview}")


def _print_stats(response) -> None:
    u = response.usage
    elapsed = response.elapsed
    if u.total_tokens and elapsed > 0:
        tok_s = u.completion_tokens / elapsed
        print(f"  [{u.prompt_tokens} prompt + {u.completion_tokens} completion = {u.total_tokens} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)]")


def _run_interruptible(agent, prompt, run_kwargs):
    original_handler = signal.getsignal(signal.SIGINT)

    def on_interrupt(sig, frame):
        print("\n  [interrupting...]")
        agent.stop()

    signal.signal(signal.SIGINT, on_interrupt)
    try:
        response = agent.run(prompt, **run_kwargs)
        if response.stop_reason == StopReason.INTERRUPTED.value:
            print("\n  [interrupted]")
        elif response.stop_reason == StopReason.DEFERRED.value:
            state = agent.last_state
            print(f"\n  [deferred: {state.state_id}]")
        elif response.stop_reason == StopReason.MAX_ROUNDS.value:
            print(f"\n{response.content}\n  [max rounds reached]")
        else:
            print(f"\n{response.content}\n")
        _print_stats(response)
        return response
    finally:
        signal.signal(signal.SIGINT, original_handler)


def _interactive(agent, run_kwargs):
    from ..agent import Agent

    print(f"nerve ({agent.name}) — interactive mode")
    print("Type your task, or 'quit' to exit.\n")

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break

        if prompt.startswith("/"):
            _handle_command(agent, prompt)
            continue

        _run_interruptible(agent, prompt, run_kwargs)


def _handle_command(agent, cmd: str) -> None:
    from ..agent import Agent

    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()

    if command == "/skills":
        skills = agent.skill_registry.list()
        if skills:
            for s in skills:
                print(f"  {s.name}: {s.description or '(no description)'}")
        else:
            print("  (no skills loaded)")

    elif command == "/tools":
        tools = agent.tool_registry.list()
        for t in tools:
            print(f"  {t.name}: {t.description}")

    elif command == "/memory":
        if not agent.memory:
            print("  (memory disabled)")
            return
        if len(parts) > 1:
            results = agent.memory.retrieve(parts[1])
            for m in results:
                print(f"  [{m.key}] {m.content[:100]}")
            if not results:
                print("  (no matches)")
        else:
            entries = agent.memory.list_all()
            for m in entries[:10]:
                print(f"  [{m.key}] {m.content[:80]}")
            if not entries:
                print("  (empty)")

    elif command == "/remember":
        if not agent.memory:
            print("  (memory disabled)")
            return
        if len(parts) < 2:
            print("  Usage: /remember key=value")
            return
        text = parts[1]
        if "=" in text:
            key, content = text.split("=", 1)
            agent.memory.store(key.strip(), content.strip())
            print(f"  Stored: {key.strip()}")
        else:
            print("  Usage: /remember key=value")

    elif command == "/defer":
        agent.defer()
        print("  Agent will defer after the current round.")

    elif command == "/deferred":
        states = Agent.list_deferred()
        if states:
            for s in states:
                print(f"  [{s.state_id}] round {s.round}/{s.max_rounds} — {s.task[:60]}")
        else:
            print("  (no deferred runs)")

    elif command == "/resume":
        if len(parts) < 2:
            print("  Usage: /resume <state_id>")
            return
        state_id = parts[1].strip()
        try:
            response = agent.resume(state_id)
            if response.stop_reason == StopReason.DEFERRED.value:
                print(f"  [deferred again: {agent.last_state.state_id}]")
            else:
                print(f"\n{response.content}\n")
        except FileNotFoundError:
            print(f"  No deferred state found: {state_id}")

    elif command == "/help":
        print("  /skills    — list loaded skills")
        print("  /tools     — list available tools")
        print("  /memory    — list or search memory (/memory <query>)")
        print("  /remember  — store a memory (/remember key=value)")
        print("  /defer     — defer the current run (use during execution)")
        print("  /deferred  — list deferred runs")
        print("  /resume    — resume a deferred run (/resume <state_id>)")
        print("  /help      — this message")

    else:
        print(f"  Unknown command: {command}. Type /help for options.")
