from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..types import ExecResult

EXEC_TIMEOUT = 30


class LocalSandbox:
    """Runs code in a local subprocess. No isolation — full system access."""

    def __init__(self, cwd: str | None = None, allowed_dirs: list[str] | None = None):
        self.cwd = cwd
        self.allowed_dirs = allowed_dirs

    def execute(self, code: str, timeout: int = EXEC_TIMEOUT) -> ExecResult:
        if self.allowed_dirs:
            guard = _build_path_guard(self.allowed_dirs)
            code = guard + "\n" + code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
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


class RestrictedSandbox:
    """Runs code with restricted builtins — blocks imports, file access, and subprocess."""

    def __init__(self, allowed_modules: list[str] | None = None):
        self.allowed_modules = set(allowed_modules or ["math", "json", "re", "datetime", "collections", "itertools", "functools"])

    def execute(self, code: str, timeout: int = EXEC_TIMEOUT) -> ExecResult:
        import builtins
        import io
        import contextlib

        allowed = self.allowed_modules
        real_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            if name.split(".")[0] not in allowed:
                raise ImportError(f"Import of '{name}' is not allowed in sandbox")
            return real_import(name, *args, **kwargs)

        blocked = {"open", "exec", "eval", "compile", "__import__", "exit", "quit"}
        safe_builtins = {k: v for k, v in vars(builtins).items() if k not in blocked}
        safe_builtins["__import__"] = restricted_import

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(code, {"__builtins__": safe_builtins})
            return ExecResult(
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                returncode=0,
            )
        except Exception as e:
            return ExecResult(
                stdout=stdout.getvalue(),
                stderr=f"{type(e).__name__}: {e}",
                returncode=1,
            )


def _build_path_guard(allowed_dirs: list[str]) -> str:
    dirs_repr = repr(allowed_dirs)
    return f"""\
import builtins as _b
_orig_open = _b.open
def _guarded_open(path, *a, **kw):
    import os
    resolved = os.path.realpath(str(path))
    allowed = {dirs_repr}
    if not any(resolved.startswith(os.path.realpath(d)) for d in allowed):
        raise PermissionError(f"Access denied: {{path}} (allowed: {{allowed}})")
    return _orig_open(path, *a, **kw)
_b.open = _guarded_open
"""
