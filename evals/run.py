"""Run evals and compare tool-only vs hybrid modes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from agentos.agent import Agent

from .cases import build_cases
from .harness import print_comparison, run_suite


def main():
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["tools_only", "hybrid"]
    cases = build_cases()

    print(f"AgentOS Evals — {len(cases)} cases x {len(modes)} modes")
    print(f"Modes: {', '.join(modes)}")
    print()

    agent = Agent("eval-agent", orchestration=["repair"])

    results = run_suite(cases, modes, agent=agent, max_rounds=10)

    print_comparison(results, modes)

    # save raw results
    out_path = Path("evals") / "results.json"
    out_data = []
    for r in results:
        out_data.append({
            "name": r.name,
            "mode": r.mode,
            "passed": r.passed,
            "content": r.content,
            "rounds": r.rounds,
            "tool_calls": r.tool_calls,
            "elapsed": round(r.elapsed, 2),
            "tokens": {
                "prompt": r.tokens.prompt_tokens,
                "completion": r.tokens.completion_tokens,
                "total": r.tokens.total_tokens,
                "llm_calls": r.tokens.llm_calls,
            },
            "error": r.error,
        })
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
