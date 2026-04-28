"""Eval harness — runs tasks against the agent and scores results."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agentos.agent import Agent
from agentos.types import Response


@dataclass
class EvalCase:
    name: str
    task: str
    check: Callable[[Response, dict[str, Any]], bool]
    setup: Callable[[], dict[str, Any]] | None = None
    teardown: Callable[[dict[str, Any]], None] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


@dataclass
class EvalResult:
    name: str
    mode: str
    passed: bool
    content: str
    rounds: int
    tool_calls: list[str]
    elapsed: float
    tokens: TokenUsage = field(default_factory=TokenUsage)
    error: str = ""


class _TokenTracker:
    """Wraps a provider to accumulate token usage across calls."""

    def __init__(self, provider):
        self._provider = provider
        self.usage = TokenUsage()

    def complete(self, messages, **kwargs):
        resp = self._provider.complete(messages, **kwargs)
        self.usage.prompt_tokens += resp.usage.prompt_tokens
        self.usage.completion_tokens += resp.usage.completion_tokens
        self.usage.total_tokens += resp.usage.total_tokens
        self.usage.llm_calls += 1
        return resp

    def stream(self, messages, **kwargs):
        return self._provider.stream(messages, **kwargs)

    async def acomplete(self, messages, **kwargs):
        return await self._provider.acomplete(messages, **kwargs)

    async def astream(self, messages, **kwargs):
        async for chunk in self._provider.astream(messages, **kwargs):
            yield chunk


def run_eval(
    agent: Agent,
    case: EvalCase,
    mode: str,
    max_rounds: int = 10,
) -> EvalResult:
    tool_log: list[str] = []

    orig_on_tool_call = agent.on_tool_call
    orig_on_tool_result = agent.on_tool_result
    orig_provider = agent.provider

    tracker = _TokenTracker(orig_provider)
    agent.provider = tracker

    def track_tool(tc):
        tool_log.append(tc.name)
        if orig_on_tool_call:
            orig_on_tool_call(tc)

    agent.on_tool_call = track_tool

    ctx: dict[str, Any] = {}
    if case.setup:
        ctx = case.setup() or {}

    task = ctx.pop("_task", None) or case.task

    start = time.time()
    try:
        if mode == "tools_only":
            response = agent.run(task, tools_only=True, max_rounds=max_rounds)
        elif mode == "hybrid":
            response = agent.run(task, max_rounds=max_rounds)
        elif mode == "code_only":
            response = agent.run(task, exec_code=True, max_rounds=max_rounds)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        elapsed = time.time() - start
        passed = case.check(response, ctx)

        return EvalResult(
            name=case.name,
            mode=mode,
            passed=passed,
            content=response.content[:500],
            rounds=len(tool_log),
            tool_calls=tool_log,
            elapsed=elapsed,
            tokens=tracker.usage,
        )
    except Exception as e:
        return EvalResult(
            name=case.name,
            mode=mode,
            passed=False,
            content="",
            rounds=len(tool_log),
            tool_calls=tool_log,
            elapsed=time.time() - start,
            tokens=tracker.usage,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        agent.on_tool_call = orig_on_tool_call
        agent.on_tool_result = orig_on_tool_result
        agent.provider = orig_provider
        if case.teardown:
            case.teardown(ctx)


def run_suite(
    cases: list[EvalCase],
    modes: list[str],
    agent: Agent | None = None,
    max_rounds: int = 10,
) -> list[EvalResult]:
    if agent is None:
        agent = Agent("eval-agent", orchestration=["repair"])

    results = []
    for case in cases:
        for mode in modes:
            print(f"  [{mode:>10}] {case.name} ...", end=" ", flush=True)
            result = run_eval(agent, case, mode, max_rounds=max_rounds)
            status = "PASS" if result.passed else "FAIL"
            t = result.tokens
            print(
                f"{status} ({result.elapsed:.1f}s, {result.rounds} tools, "
                f"{t.llm_calls} llm calls, {t.prompt_tokens}+{t.completion_tokens}={t.total_tokens} tok)"
            )
            if result.error:
                print(f"             error: {result.error}")
            results.append(result)

    return results


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def print_comparison(results: list[EvalResult], modes: list[str]) -> None:
    by_case: dict[str, dict[str, EvalResult]] = {}
    for r in results:
        by_case.setdefault(r.name, {})[r.mode] = r

    # Per-mode column: pass | time | tools | llm | prompt | compl | total
    col_header = "pass  time  tools  llm  prompt  compl  total"
    header = f"{'Case':<30}"
    for m in modes:
        header += f" | {m}: {col_header}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    totals: dict[str, dict] = {
        m: {"pass": 0, "total": 0, "time": 0.0, "prompt": 0, "compl": 0, "tok": 0, "llm": 0}
        for m in modes
    }

    for case_name, mode_results in by_case.items():
        row = f"{case_name:<30}"
        for m in modes:
            r = mode_results.get(m)
            if r:
                mark = "Y" if r.passed else "N"
                t = r.tokens
                row += (
                    f" |    {mark}  {r.elapsed:4.1f}s  {r.rounds:>3}    "
                    f"{t.llm_calls:>2}   {_fmt_tokens(t.prompt_tokens):>6}  "
                    f"{_fmt_tokens(t.completion_tokens):>5}  {_fmt_tokens(t.total_tokens):>5}"
                )
                totals[m]["total"] += 1
                totals[m]["time"] += r.elapsed
                totals[m]["prompt"] += t.prompt_tokens
                totals[m]["compl"] += t.completion_tokens
                totals[m]["tok"] += t.total_tokens
                totals[m]["llm"] += t.llm_calls
                if r.passed:
                    totals[m]["pass"] += 1
            else:
                row += f" |    -     -     -     -      -      -      -"
        print(row)

    print("-" * len(header))
    summary = f"{'TOTALS':<30}"
    for m in modes:
        t = totals[m]
        pct = (t["pass"] / t["total"] * 100) if t["total"] else 0
        summary += (
            f" | {t['pass']}/{t['total']} ({pct:3.0f}%) {t['time']:5.0f}s       "
            f"{t['llm']:>3}   {_fmt_tokens(t['prompt']):>6}  "
            f"{_fmt_tokens(t['compl']):>5}  {_fmt_tokens(t['tok']):>5}"
        )
    print(summary)
    print("=" * len(header))
