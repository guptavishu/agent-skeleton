"""CLI entry point for nerve."""

from __future__ import annotations

import argparse
import sys

from .providers import FileMemory
from .skills import Skill


def main():
    parser = argparse.ArgumentParser(
        prog="nerve",
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
    parser.add_argument("--name", default="nerve", help="Agent name")
    parser.add_argument("--ux", default="cli", choices=["cli", "web"], help="UX mode (default: cli)")
    parser.add_argument("--host", default="127.0.0.1", help="Web UX host")
    parser.add_argument("--port", type=int, default=8420, help="Web UX port")
    args = parser.parse_args()

    # load skills from files
    skills = []
    for path in args.skill:
        try:
            skills.append(Skill.load(path))
        except Exception as e:
            print(f"Warning: could not load skill {path}: {e}", file=sys.stderr)

    # build agent
    from .agent import Agent
    memory = None if args.no_memory else FileMemory()
    agent = Agent(
        name=args.name,
        model=args.model,
        system=args.system,
        skills=skills,
        memory=memory,
        builtins=not args.no_builtins,
        orchestration=args.orch or ["repair"],
    )

    # pick UX
    if args.ux == "web":
        from .ux import WebUX
        ux = WebUX(host=args.host, port=args.port)
    else:
        from .ux import CliUX
        ux = CliUX()

    ux.start(
        agent,
        prompt=args.prompt,
        tools_only=args.tools_only,
        exec_code=args.exec_code,
        plan=args.plan,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
