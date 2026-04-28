"""Eval cases for tool-only, hybrid, and code-only modes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from agentos.types import Response
from .harness import EvalCase


def _contains_any(text: str, needles: list[str]) -> bool:
    text_lower = text.lower()
    return any(n.lower() in text_lower for n in needles)


# ---------------------------------------------------------------------------
# Each builder returns (task_str, ctx_dict) from setup, and a check function.
# The harness calls setup() to get (task, ctx), runs agent.run(task), then
# calls check(response, ctx).
# ---------------------------------------------------------------------------

def build_cases() -> list[EvalCase]:
    return [
        _case_read_file(),
        _case_write_file(),
        _case_list_dir(),
        _case_shell_exec(),
        _case_multi_step(),
        _case_read_and_compute(),
        _case_math(),
        _case_read_transform(),
        _case_plain_question(),
    ]


def _case_read_file() -> EvalCase:
    def setup():
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="eval_")
        f.write("The secret password is: hunter2")
        f.flush()
        f.close()
        task = f"Read the file at {f.name} and tell me what the secret password is."
        return {"path": f.name, "_task": task}

    def teardown(ctx):
        os.unlink(ctx["path"])

    return EvalCase(
        name="read_file_content",
        task="(set by setup)",
        check=lambda r, ctx: "hunter2" in r.content,
        setup=setup,
        teardown=teardown,
        tags=["tool", "read_file"],
    )


def _case_write_file() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        path = os.path.join(d, "output.txt")
        task = f"Write a haiku about coding and save it to the file {path}. The file must exist on disk when you're done."
        return {"dir": d, "path": path, "_task": task}

    def check(r, ctx):
        return Path(ctx["path"]).exists() and len(Path(ctx["path"]).read_text().strip()) > 0

    def teardown(ctx):
        import shutil
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    return EvalCase(
        name="write_file_creates_file",
        task="(set by setup)",
        check=check,
        setup=setup,
        teardown=teardown,
        tags=["tool", "write_file"],
    )


def _case_list_dir() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "alpha.py").touch()
        Path(d, "beta.txt").touch()
        Path(d, "subdir").mkdir()
        task = f"List all files and directories in {d} and tell me what you find."
        return {"dir": d, "_task": task}

    def teardown(ctx):
        import shutil
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    return EvalCase(
        name="list_directory_contents",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["alpha.py", "beta.txt"]),
        setup=setup,
        teardown=teardown,
        tags=["tool", "list_directory"],
    )


def _case_shell_exec() -> EvalCase:
    return EvalCase(
        name="shell_exec_command",
        task="Use the shell_exec tool to run 'echo hello_from_eval' and tell me the output.",
        check=lambda r, ctx: "hello_from_eval" in r.content,
        tags=["tool", "shell_exec"],
    )


def _case_multi_step() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "data.txt").write_text("42")
        task = f"List the files in {d}, then read whichever file you find, and tell me its contents."
        return {"dir": d, "_task": task}

    def teardown(ctx):
        import shutil
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    return EvalCase(
        name="multi_step_tool_chain",
        task="(set by setup)",
        check=lambda r, ctx: "42" in r.content,
        setup=setup,
        teardown=teardown,
        tags=["tool", "multi_step"],
    )


def _case_read_and_compute() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "numbers.txt").write_text("10\n20\n30\n40\n50")
        task = f"Read the file {d}/numbers.txt, compute the average of the numbers in it, and tell me the result."
        return {"dir": d, "_task": task}

    def teardown(ctx):
        import shutil
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    return EvalCase(
        name="hybrid_read_and_compute",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["30", "150"]),
        setup=setup,
        teardown=teardown,
        tags=["hybrid", "multi_step"],
    )


def _case_math() -> EvalCase:
    return EvalCase(
        name="hybrid_math_calculation",
        task="Calculate the factorial of 10. Show the exact number in your response.",
        check=lambda r, ctx: "3628800" in r.content,
        tags=["hybrid", "math"],
    )


def _case_read_transform() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "input.csv").write_text("name,score\nalice,85\nbob,92\ncharlie,78\n")
        task = f"Read {d}/input.csv and tell me who has the highest score."
        return {"dir": d, "_task": task}

    def teardown(ctx):
        import shutil
        shutil.rmtree(ctx["dir"], ignore_errors=True)

    return EvalCase(
        name="hybrid_read_transform_report",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["bob", "92"]),
        setup=setup,
        teardown=teardown,
        tags=["hybrid", "transform"],
    )


def _case_plain_question() -> EvalCase:
    return EvalCase(
        name="hybrid_plain_question",
        task="What is the capital of France? Answer in one word.",
        check=lambda r, ctx: "paris" in r.content.lower(),
        tags=["hybrid", "no_tools"],
    )
