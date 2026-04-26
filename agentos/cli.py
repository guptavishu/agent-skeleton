"""CLI entry point for agent-skeleton."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .memory import FileMemory
from .skills import Skill
from .tools import Tool
from .types import ToolCall, ToolResult


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


def main():
    parser = argparse.ArgumentParser(
        prog="agentos",
        description="Thin, extensible agent framework",
    )
    parser.add_argument("prompt", nargs="?", help="Task to run (omit for interactive mode)")
    parser.add_argument("--model", "-m", default="", help="Model override")
    parser.add_argument("--system", "-s", default="", help="System prompt")
    parser.add_argument("--tools-only", action="store_true", help="Tool-only mode (no code execution)")
    parser.add_argument("--exec", dest="exec_code", action="store_true", help="Code-only mode (no tools)")
    parser.add_argument("--plan", action="store_true", help="Enable planning")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    parser.add_argument("--no-builtins", action="store_true", help="Disable built-in tools")
    parser.add_argument("--skill", action="append", default=[], help="Load a skill file (can repeat)")
    parser.add_argument("--orch", action="append", default=[], help="Add orchestration mode (can repeat)")
    parser.add_argument("--max-rounds", type=int, default=20, help="Max loop iterations")
    parser.add_argument("--name", default="agentos", help="Agent name")
    args = parser.parse_args()

    # load skills from files
    skills = []
    for path in args.skill:
        try:
            skills.append(Skill.load(path))
        except Exception as e:
            print(f"Warning: could not load skill {path}: {e}", file=sys.stderr)

    # build agent
    memory = None if args.no_memory else FileMemory()
    agent = Agent(
        name=args.name,
        model=args.model,
        system=args.system,
        skills=skills,
        memory=memory,
        builtins=not args.no_builtins,
        orchestration=args.orch or ["repair"],
        on_tool_call=_print_tool_call,
        on_tool_result=_print_tool_result,
        on_code_exec=_print_code,
        on_code_result=_print_code_result,
    )

    if args.prompt:
        # batch mode
        response = agent.run(
            args.prompt,
            tools_only=args.tools_only,
            exec_code=args.exec_code,
            plan=args.plan,
            max_rounds=args.max_rounds,
        )
        print(response.content)
    else:
        # interactive mode
        _interactive(agent, args)


def _interactive(agent: Agent, args) -> None:
    print(f"agent-skeleton ({agent.name}) — interactive mode")
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

        # slash commands
        if prompt.startswith("/"):
            _handle_command(agent, prompt)
            continue

        response = agent.run(
            prompt,
            tools_only=args.tools_only,
            exec_code=args.exec_code,
            plan=args.plan,
            max_rounds=args.max_rounds,
        )
        print(f"\n{response.content}\n")


def _handle_command(agent: Agent, cmd: str) -> None:
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

    elif command == "/help":
        print("  /skills    — list loaded skills")
        print("  /tools     — list available tools")
        print("  /memory    — list or search memory (/memory <query>)")
        print("  /remember  — store a memory (/remember key=value)")
        print("  /help      — this message")

    else:
        print(f"  Unknown command: {command}. Type /help for options.")


if __name__ == "__main__":
    main()
