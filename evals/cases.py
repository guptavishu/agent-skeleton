"""Eval cases for tool-only, hybrid, and code-only modes.

Categories:
  - tool_natural: tasks where structured tools are the natural fit
  - code_natural: tasks where code execution is the natural fit
  - ambiguous: tasks where either approach works
  - multi_step: tasks requiring multiple rounds
  - no_tools: tasks the LLM should answer directly
"""

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


def _file_exists(ctx: dict, filename: str) -> bool:
    return Path(ctx["dir"], filename).exists()


def _file_has_content(ctx: dict, filename: str) -> bool:
    p = Path(ctx["dir"], filename)
    return p.exists() and len(p.read_text().strip()) > 0


def build_cases() -> list[EvalCase]:
    return [
        # --- tool_natural ---
        _case_read_file(),
        _case_write_file(),
        _case_list_dir(),
        _case_shell_exec(),
        _case_read_and_summarize(),
        _case_copy_file(),
        _case_read_python_describe(),

        # --- code_natural ---
        _case_math(),
        _case_fibonacci(),
        _case_primes(),
        _case_string_manipulation(),
        _case_sort_numbers(),
        _case_math_expression(),

        # --- ambiguous (tools + code) ---
        _case_multi_step(),
        _case_read_and_compute(),
        _case_read_transform(),
        _case_csv_statistics(),
        _case_json_transform(),
        _case_generate_and_save(),

        # --- no_tools ---
        _case_plain_question(),
        _case_reasoning(),
    ]


# ---------------------------------------------------------------------------
# Tool-natural cases
# ---------------------------------------------------------------------------

def _case_read_file() -> EvalCase:
    def setup():
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="eval_")
        f.write("The secret password is: hunter2")
        f.flush()
        f.close()
        return {"path": f.name, "_task": f"Read the file at {f.name} and tell me what the secret password is."}

    return EvalCase(
        name="read_file_content",
        task="(set by setup)",
        check=lambda r, ctx: "hunter2" in r.content,
        setup=setup,
        teardown=lambda ctx: os.unlink(ctx["path"]),
        tags=["tool_natural", "file_io"],
    )


def _case_write_file() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        path = os.path.join(d, "output.txt")
        return {"dir": d, "path": path, "_task": f"Write a haiku about coding and save it to the file {path}. The file must exist on disk when you're done."}

    return EvalCase(
        name="write_file_creates_file",
        task="(set by setup)",
        check=lambda r, ctx: Path(ctx["path"]).exists() and len(Path(ctx["path"]).read_text().strip()) > 0,
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["tool_natural", "file_io"],
    )


def _case_list_dir() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "alpha.py").touch()
        Path(d, "beta.txt").touch()
        Path(d, "subdir").mkdir()
        return {"dir": d, "_task": f"List all files and directories in {d} and tell me what you find."}

    return EvalCase(
        name="list_directory_contents",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["alpha.py", "beta.txt"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["tool_natural", "file_io"],
    )


def _case_shell_exec() -> EvalCase:
    return EvalCase(
        name="shell_exec_command",
        task="Use the shell_exec tool to run 'echo hello_from_eval' and tell me the output.",
        check=lambda r, ctx: "hello_from_eval" in r.content,
        tags=["tool_natural", "shell"],
    )


def _case_read_and_summarize() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        config = '{"database": {"host": "localhost", "port": 5432}, "features": ["auth", "logging", "cache"]}'
        Path(d, "config.json").write_text(config)
        return {"dir": d, "_task": f"Read {d}/config.json and list the configured features."}

    return EvalCase(
        name="read_and_summarize_json",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["auth", "logging", "cache"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["tool_natural", "file_io"],
    )


def _case_copy_file() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "source.txt").write_text("copy this content exactly")
        return {"dir": d, "_task": f"Read {d}/source.txt and write its content to {d}/copy.txt"}

    return EvalCase(
        name="copy_file_content",
        task="(set by setup)",
        check=lambda r, ctx: _file_has_content(ctx, "copy.txt"),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["tool_natural", "file_io", "multi_step"],
    )


def _case_read_python_describe() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
'''
        Path(d, "example.py").write_text(code)
        return {"dir": d, "_task": f"Read {d}/example.py and list the function names defined in it."}

    return EvalCase(
        name="read_python_describe",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["fibonacci", "factorial"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["tool_natural", "file_io", "code_understanding"],
    )


# ---------------------------------------------------------------------------
# Code-natural cases
# ---------------------------------------------------------------------------

def _case_math() -> EvalCase:
    return EvalCase(
        name="factorial_calculation",
        task="Calculate the factorial of 10. Show the exact number in your response.",
        check=lambda r, ctx: "3628800" in r.content,
        tags=["code_natural", "computation"],
    )


def _case_fibonacci() -> EvalCase:
    return EvalCase(
        name="fibonacci_20th",
        task="Calculate the 20th Fibonacci number. Show the exact number.",
        check=lambda r, ctx: "6765" in r.content,
        tags=["code_natural", "computation"],
    )


def _case_primes() -> EvalCase:
    return EvalCase(
        name="generate_primes",
        task="Generate all prime numbers below 50. List them all.",
        check=lambda r, ctx: "47" in r.content and "2" in r.content,
        tags=["code_natural", "computation"],
    )


def _case_string_manipulation() -> EvalCase:
    return EvalCase(
        name="string_manipulation",
        task='Reverse the string "hello world", uppercase it, and count the vowels in the original. Show all three results.',
        check=lambda r, ctx: _contains_any(r.content, ["dlrow olleh", "DLROW OLLEH"]),
        tags=["code_natural", "computation"],
    )


def _case_sort_numbers() -> EvalCase:
    return EvalCase(
        name="sort_descending",
        task="Sort these numbers in descending order: 42, 17, 93, 8, 55, 31, 76. Show the sorted list.",
        check=lambda r, ctx: "93" in r.content and r.content.index("93") < r.content.index("8"),
        tags=["code_natural", "computation"],
    )


def _case_math_expression() -> EvalCase:
    return EvalCase(
        name="math_expression",
        task="Evaluate this expression exactly: (17 * 23) + (45 / 9) - (12 ** 2). Show the result.",
        check=lambda r, ctx: _contains_any(r.content, ["252", "252.0"]),
        tags=["code_natural", "computation"],
    )


# ---------------------------------------------------------------------------
# Ambiguous cases (tools + code)
# ---------------------------------------------------------------------------

def _case_multi_step() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "data.txt").write_text("42")
        return {"dir": d, "_task": f"List the files in {d}, then read whichever file you find, and tell me its contents."}

    return EvalCase(
        name="multi_step_tool_chain",
        task="(set by setup)",
        check=lambda r, ctx: "42" in r.content,
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "multi_step"],
    )


def _case_read_and_compute() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "numbers.txt").write_text("10\n20\n30\n40\n50")
        return {"dir": d, "_task": f"Read the file {d}/numbers.txt, compute the average of the numbers in it, and tell me the result."}

    return EvalCase(
        name="read_and_compute_average",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["30", "150"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "file_io", "computation"],
    )


def _case_read_transform() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        Path(d, "input.csv").write_text("name,score\nalice,85\nbob,92\ncharlie,78\n")
        return {"dir": d, "_task": f"Read {d}/input.csv and tell me who has the highest score."}

    return EvalCase(
        name="read_transform_report",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["bob", "92"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "file_io", "computation"],
    )


def _case_csv_statistics() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        csv = "name,age,score\nalice,25,90\nbob,30,85\ncharlie,35,88\ndiana,28,92\neve,32,80\n"
        Path(d, "data.csv").write_text(csv)
        return {"dir": d, "_task": f"Read {d}/data.csv and compute the average age and average score. Show both numbers."}

    return EvalCase(
        name="csv_statistics",
        task="(set by setup)",
        check=lambda r, ctx: _contains_any(r.content, ["30", "87"]),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "file_io", "computation"],
    )


def _case_json_transform() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        config = '{"features": ["auth", "logging"]}'
        Path(d, "config.json").write_text(config)
        return {"dir": d, "_task": f'Read {d}/config.json, add "notifications" to the features list, and write the result to {d}/config_updated.json'}

    return EvalCase(
        name="json_transform_and_write",
        task="(set by setup)",
        check=lambda r, ctx: _file_has_content(ctx, "config_updated.json"),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "file_io", "computation"],
    )


def _case_generate_and_save() -> EvalCase:
    def setup():
        d = tempfile.mkdtemp(prefix="eval_")
        return {"dir": d, "_task": f"Generate a multiplication table for 1 through 5 and save it to {d}/table.txt"}

    return EvalCase(
        name="generate_and_save",
        task="(set by setup)",
        check=lambda r, ctx: _file_has_content(ctx, "table.txt"),
        setup=setup,
        teardown=lambda ctx: __import__("shutil").rmtree(ctx["dir"], ignore_errors=True),
        tags=["ambiguous", "computation", "file_io"],
    )


# ---------------------------------------------------------------------------
# No-tools cases
# ---------------------------------------------------------------------------

def _case_plain_question() -> EvalCase:
    return EvalCase(
        name="plain_question",
        task="What is the capital of France? Answer in one word.",
        check=lambda r, ctx: "paris" in r.content.lower(),
        tags=["no_tools"],
    )


def _case_reasoning() -> EvalCase:
    return EvalCase(
        name="reasoning",
        task="If a train travels 120km in 2 hours, what is its average speed in km/h? Answer with just the number.",
        check=lambda r, ctx: "60" in r.content,
        tags=["no_tools", "computation"],
    )
