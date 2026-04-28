from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path.home() / ".agentos" / "agentos.log"


class Telemetry:
    """Appends structured JSON-line events to a log file."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or DEFAULT_LOG_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "ts": time.time(),
            "event": event,
            **payload,
        }
        if self._session_id:
            record["session"] = self._session_id
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def tool_call(self, name: str, arguments: dict) -> None:
        self.log("tool_call", tool=name, arguments=arguments)

    def tool_result(self, name: str, output: str, error: str | None = None) -> None:
        self.log("tool_result", tool=name, output=output[:500], error=error)

    def code_exec(self, code: str) -> None:
        self.log("code_exec", code=code[:1000])

    def code_result(self, stdout: str, stderr: str, returncode: int) -> None:
        self.log("code_result", stdout=stdout[:500], stderr=stderr[:500], returncode=returncode)

    def llm_call(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.log("llm_call", model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def agent_start(self, agent_name: str, task: str) -> None:
        self.log("agent_start", agent=agent_name, task=task[:500])

    def agent_finish(self, agent_name: str, rounds: int) -> None:
        self.log("agent_finish", agent=agent_name, rounds=rounds)
