from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from .types import Message, Response, ToolCall, Usage

if TYPE_CHECKING:
    from .tools import Tool


@runtime_checkable
class Provider(Protocol):
    """Implement these methods to plug in any LLM backend."""

    def complete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response: ...

    def stream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Iterator[str]: ...

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> Response: ...

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str = "",
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator[str]: ...


DEFAULT_MODEL = "qwen2.5-coder:32b"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """Talks to a local Ollama instance via REST API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _build_payload(
        self,
        messages: list[Message],
        system: str,
        model: str,
        temperature: float,
        max_tokens: int,
        tools: list[Tool] | None,
        stream: bool,
    ) -> dict[str, Any]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for m in messages:
            msg: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            msgs.append(msg)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": msgs,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [t.to_ollama_schema() for t in tools]
        return payload

    def _parse_response(self, data: dict[str, Any]) -> Response:
        msg = data.get("message", {})
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", f"call_{len(tool_calls)}"),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        content = msg.get("content", "")

        # Fallback: some models emit tool calls as JSON text in content
        if not tool_calls and content:
            parsed, remaining = self._try_parse_tool_call(content)
            if parsed:
                tool_calls = parsed
                content = remaining

        tokens = data.get("prompt_eval_count", 0)
        completion = data.get("eval_count", 0)
        return Response(
            content=content,
            model=data.get("model", ""),
            stop_reason=data.get("done_reason", ""),
            usage=Usage(tokens, completion, tokens + completion),
            tool_calls=tool_calls,
        )

    def _try_parse_tool_call(self, content: str) -> tuple[list[ToolCall], str]:
        """Extract tool calls from content when the model outputs them as JSON text."""
        result = []
        remaining = content

        # Find JSON objects that look like tool calls: {"name": ..., "arguments": ...}
        i = 0
        spans_to_remove = []
        while i < len(content):
            if content[i] == '{':
                obj, end = self._try_extract_json(content, i)
                if obj and isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    result.append(ToolCall(
                        id=f"call_{len(result)}",
                        name=obj["name"],
                        arguments=args,
                    ))
                    spans_to_remove.append((i, end))
                    i = end
                    continue
            i += 1

        if result:
            for start, end in reversed(spans_to_remove):
                remaining = remaining[:start] + remaining[end:]
            remaining = remaining.strip()

        return result, remaining

    @staticmethod
    def _try_extract_json(text: str, start: int) -> tuple[Any, int]:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj, i + 1
                    except (json.JSONDecodeError, ValueError):
                        return None, start
        return None, start

    def complete(self, messages, *, system="", model="", temperature=0.7, max_tokens=4096, tools=None):
        payload = self._build_payload(messages, system, model, temperature, max_tokens, tools, stream=False)
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def stream(self, messages, *, system="", model="", temperature=0.7, max_tokens=4096, tools=None):
        payload = self._build_payload(messages, system, model, temperature, max_tokens, tools, stream=True)
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    async def acomplete(self, messages, *, system="", model="", temperature=0.7, max_tokens=4096, tools=None):
        payload = self._build_payload(messages, system, model, temperature, max_tokens, tools, stream=False)
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def astream(self, messages, *, system="", model="", temperature=0.7, max_tokens=4096, tools=None):
        payload = self._build_payload(messages, system, model, temperature, max_tokens, tools, stream=True)
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=120) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
