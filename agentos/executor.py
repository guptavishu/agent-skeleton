"""Code execution engine — runs LLM-emitted Python in a subprocess."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .types import ExecResult

EXEC_TIMEOUT = 30
CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """Pull all fenced code blocks from LLM output."""
    return CODE_BLOCK_RE.findall(text)


def execute_code(
    code: str,
    timeout: int = EXEC_TIMEOUT,
    cwd: str | None = None,
) -> ExecResult:
    """Run Python code in a subprocess and return the result."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            returncode=-1,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
