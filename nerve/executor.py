"""Backwards compatibility + utility functions."""
from .protocols import Sandbox
from .providers.sandbox import LocalSandbox, RestrictedSandbox, EXEC_TIMEOUT

import re
from .types import ExecResult

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    return CODE_BLOCK_RE.findall(text)


def execute_code(
    code: str,
    timeout: int = EXEC_TIMEOUT,
    cwd: str | None = None,
) -> ExecResult:
    return LocalSandbox(cwd=cwd).execute(code, timeout=timeout)
