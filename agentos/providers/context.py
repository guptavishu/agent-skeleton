from __future__ import annotations

from ..types import Message


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English text."""
    return len(text) // 4 + 1


def message_tokens(msg: Message) -> int:
    tokens = estimate_tokens(msg.content)
    for tc in msg.tool_calls:
        tokens += estimate_tokens(str(tc.arguments)) + 10
    return tokens + 4  # per-message overhead


class TokenWindowContext:
    """Drop oldest messages (except the first) until total tokens fit."""

    def manage(self, messages: list[Message], max_tokens: int) -> list[Message]:
        total = sum(message_tokens(m) for m in messages)
        if total <= max_tokens:
            return messages

        first = messages[0]
        rest = list(messages[1:])
        total = sum(message_tokens(m) for m in [first] + rest)

        while rest and total > max_tokens:
            dropped = rest.pop(0)
            total -= message_tokens(dropped)

        return [first] + rest


class SummarizeContext:
    """Summarize older messages when context grows too large.

    Keeps recent messages intact and replaces older ones with a summary.
    """

    def __init__(self, provider=None, keep_recent: int = 6, summary_prompt: str = ""):
        self._provider = provider
        self._keep_recent = keep_recent
        self._summary_prompt = summary_prompt or (
            "Summarize the conversation so far in 2-3 sentences. "
            "Focus on what was asked, what was done, and any important results."
        )
        self._cached_summary: str | None = None
        self._summarized_count: int = 0

    def manage(self, messages: list[Message], max_tokens: int) -> list[Message]:
        total = sum(message_tokens(m) for m in messages)
        if total <= max_tokens:
            return messages

        if len(messages) <= self._keep_recent + 1:
            return messages

        first = messages[0]
        old = messages[1:-self._keep_recent]
        recent = messages[-self._keep_recent:]

        if len(old) > self._summarized_count and self._provider:
            self._cached_summary = self._summarize(old)
            self._summarized_count = len(old)

        if self._cached_summary:
            summary_msg = Message(
                role="user",
                content=f"[Summary of earlier conversation]\n{self._cached_summary}",
            )
            return [first, summary_msg] + recent

        return [first] + recent

    def _summarize(self, messages: list[Message]) -> str:
        convo = "\n".join(f"{m.role}: {m.content[:200]}" for m in messages if m.content)
        prompt = f"{self._summary_prompt}\n\nConversation:\n{convo}"
        response = self._provider.complete(
            [Message(role="user", content=prompt)],
            max_tokens=256,
        )
        return response.content
