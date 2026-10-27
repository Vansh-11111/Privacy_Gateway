from __future__ import annotations

from typing import Callable

LLMBackend = Callable[[list[dict], str], str]


class LLMError(Exception):
    """The upstream model failed. The message must never contain user data."""


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


class MockLLM:
    """Echoes the last user message back. Tokens like <AADHAAR_1> come back
    unchanged, which is exactly what lets the gateway restore them."""

    def __init__(self):
        self.calls: list[list[dict]] = []   # what the model was shown (tests inspect this)

    def __call__(self, messages: list[dict], model: str) -> str:
        self.calls.append(messages)
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last_user is None:
            return "Hello! How can I help?"
        return f"You said: {_text_of(last_user['content'])}"
