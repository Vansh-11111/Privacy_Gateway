from __future__ import annotations

import logging
import secrets
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .llm import LLMBackend, LLMError, MockLLM
from .policy import Policy
from .protect import protect, restore
from .vault import TokenVault

log = logging.getLogger("pii_gateway")

MAX_MESSAGES = 100
MAX_CHARS = 20_000


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = "mock-1"
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    stream: bool = False
    user: str | None = None


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": "gateway_error", "code": code}})


def _session_id(header: str | None, body_user: str | None) -> tuple[str, bool]:
    """Returns (session_id, caller_supplied). A caller who supplies nothing gets a
    random one-shot session: safe, but tokens can't be reused across requests."""
    for candidate in (header, body_user):
        if candidate and candidate.strip():
            return candidate.strip()[:128], True
    return "anon-" + secrets.token_hex(8), False


def create_app(llm: LLMBackend | None = None, vault: TokenVault | None = None,
               policy: Policy | None = None) -> FastAPI:
    llm = llm or MockLLM()
    vault = vault or TokenVault()
    policy = policy or Policy.default()
    app = FastAPI(title="India PII Gateway", version="0.3")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "vault_key": "ephemeral" if vault.ephemeral else "configured"}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, x_session_id: str | None = Header(default=None)):
        if req.stream:
            return _error(400, "stream_unsupported", "stream=true is not supported yet.")
        if sum(len(_flat(m.content)) for m in req.messages) > MAX_CHARS:
            return _error(413, "too_large", f"Conversation exceeds {MAX_CHARS} characters.")

        sid, supplied = _session_id(x_session_id, req.user)
        headers = {"X-Session-Id": sid}

        # --- 2. protect everything that will leave the building
        safe_messages: list[dict] = []
        summary: dict[str, dict[str, int]] = {}
        for m in req.messages:
            result_content, blocked_msg = _protect_content(m.content, sid, vault, policy, summary)
            if blocked_msg:
                log.info("blocked request session=%s", _log_sid(sid))
                return _error(400, "pii_blocked", blocked_msg, headers)
            safe = m.model_dump(exclude_none=True)
            safe["content"] = result_content
            safe_messages.append(safe)

        # --- 3. call the model with protected messages only
        try:
            reply = llm(safe_messages, req.model)
        except LLMError:
            return _error(502, "upstream_error", "The language model is unavailable.", headers)
        except Exception:  # never echo exception text: it could carry prompt contents
            log.exception("llm backend crashed")
            return _error(502, "upstream_error", "The language model is unavailable.", headers)

        # --- 4. restore, 5. respond
        reply = restore(reply, sid, vault)
        log.info("ok session=%s summary=%s", _log_sid(sid), summary)
        return JSONResponse(headers=headers, content={
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_pii_gateway": {"session_id": sid, "session_supplied": supplied, "detected": summary},
        })

    return app


# ----------------------------------------------------------------- helpers
def _flat(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text", ""), str))


def _log_sid(sid: str) -> str:
    return sid[:4] + "..." if len(sid) > 4 else "***"   # session ids can be user ids: don't log them whole


def _merge(into: dict, add: dict) -> None:
    for t, acts in add.items():
        for a, n in acts.items():
            into.setdefault(t, {})
            into[t][a] = into[t].get(a, 0) + n


def _protect_content(content, sid, vault, policy, summary):
    """Returns (protected_content, blocked_user_message_or_None)."""
    if content is None:
        return None, None
    if isinstance(content, str):
        r = protect(content, sid, vault, policy)
        if r.blocked:
            return None, r.user_message
        _merge(summary, r.summary())
        return r.text, None
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            r = protect(part["text"], sid, vault, policy)
            if r.blocked:
                return None, r.user_message
            _merge(summary, r.summary())
            out.append({**part, "text": r.text})
        else:
            out.append(part)   # non-text parts (images...) are not inspected yet: documented limit
    return out, None


app = create_app()