"""Tests for Sandbox protocol, LocalSandbox, and RestrictedSandbox."""

import os
import tempfile
from pathlib import Path

from agentos.executor import LocalSandbox, RestrictedSandbox, Sandbox


# --- LocalSandbox ---

def test_local_sandbox_executes_code():
    sb = LocalSandbox()
    result = sb.execute("print(2 + 2)")
    assert result.stdout.strip() == "4"
    assert result.returncode == 0


def test_local_sandbox_captures_error():
    sb = LocalSandbox()
    result = sb.execute("raise ValueError('boom')")
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_local_sandbox_timeout():
    sb = LocalSandbox()
    result = sb.execute("import time; time.sleep(10)", timeout=1)
    assert result.returncode == -1
    assert "timed out" in result.stderr.lower()


def test_local_sandbox_cwd():
    with tempfile.TemporaryDirectory() as d:
        sb = LocalSandbox(cwd=d)
        result = sb.execute("import os; print(os.getcwd())")
        assert d in result.stdout


def test_local_sandbox_allowed_dirs():
    with tempfile.TemporaryDirectory() as allowed, tempfile.TemporaryDirectory() as blocked:
        test_file = Path(allowed) / "ok.txt"
        test_file.write_text("hello")

        blocked_file = Path(blocked) / "secret.txt"
        blocked_file.write_text("secret")

        sb = LocalSandbox(allowed_dirs=[allowed])

        # Can read from allowed dir
        result = sb.execute(f"print(open('{test_file}').read())")
        assert result.returncode == 0
        assert "hello" in result.stdout

        # Blocked from other dirs
        result = sb.execute(f"print(open('{blocked_file}').read())")
        assert result.returncode != 0
        assert "PermissionError" in result.stderr


def test_local_sandbox_is_sandbox_protocol():
    assert isinstance(LocalSandbox(), Sandbox)


# --- RestrictedSandbox ---

def test_restricted_executes_safe_code():
    sb = RestrictedSandbox()
    result = sb.execute("print(3 * 7)")
    assert result.stdout.strip() == "21"
    assert result.returncode == 0


def test_restricted_allows_safe_modules():
    sb = RestrictedSandbox()
    result = sb.execute("import math; print(math.sqrt(16))")
    assert "4.0" in result.stdout
    assert result.returncode == 0


def test_restricted_blocks_os_import():
    sb = RestrictedSandbox()
    result = sb.execute("import os; os.system('echo pwned')")
    assert result.returncode == 1
    assert "not allowed" in result.stderr.lower()


def test_restricted_blocks_subprocess():
    sb = RestrictedSandbox()
    result = sb.execute("import subprocess; subprocess.run(['ls'])")
    assert result.returncode == 1
    assert "not allowed" in result.stderr.lower()


def test_restricted_blocks_open():
    sb = RestrictedSandbox()
    result = sb.execute("open('/etc/passwd').read()")
    assert result.returncode == 1


def test_restricted_blocks_eval():
    sb = RestrictedSandbox()
    result = sb.execute("eval('1+1')")
    assert result.returncode == 1


def test_restricted_custom_allowed_modules():
    sb = RestrictedSandbox(allowed_modules=["os"])
    result = sb.execute("import os; print(os.getcwd())")
    assert result.returncode == 0
    assert len(result.stdout.strip()) > 0


def test_restricted_blocks_unlisted_module():
    sb = RestrictedSandbox(allowed_modules=["math"])
    result = sb.execute("import json")
    assert result.returncode == 1
    assert "not allowed" in result.stderr.lower()


def test_restricted_is_sandbox_protocol():
    assert isinstance(RestrictedSandbox(), Sandbox)


# --- Agent integration ---

def test_agent_uses_sandbox():
    """Verify Agent passes code execution through the sandbox."""
    from agentos.agent import Agent
    from agentos.types import Response, ToolCall

    class FakeProvider:
        def __init__(self):
            self._call_count = 0
        def complete(self, messages, **kw):
            self._call_count += 1
            if self._call_count == 1:
                return Response(content="```python\nprint('hello from sandbox')\n```")
            return Response(content="Done")
        def stream(self, messages, **kw):
            yield "Done"
        async def acomplete(self, messages, **kw):
            return Response(content="")
        async def astream(self, messages, **kw):
            yield ""

    executed = []

    class TrackingSandbox:
        def execute(self, code, timeout=30):
            executed.append(code)
            return __import__("agentos.types", fromlist=["ExecResult"]).ExecResult(
                stdout="hello from sandbox\n", stderr="", returncode=0,
            )

    agent = Agent("test", provider=FakeProvider(), sandbox=TrackingSandbox(), builtins=False)
    result = agent.run("run code", exec_code=True)

    assert len(executed) == 1
    assert "hello from sandbox" in executed[0]
