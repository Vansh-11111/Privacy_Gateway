from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from typing import Any, Iterable, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditLog
from .llm import LLMBackend, LLMError, MockLLM
from .output_scan import OutputPolicy, scan_and_restore
from .policy import Policy
from .protect import protect
from .ratelimit import RateLimiter
from .vault import TokenVault

log = logging.getLogger("pii_gateway")

MAX_MESSAGES = 100
MAX_CHARS = 20_000
MAX_BODY_BYTES = 1_000_000     # refuse oversized bodies before parsing them
FAILED_AUTH_PER_MIN = 20       # wrong-key attempts per client address before it is locked out


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


class GatewayHTTPError(Exception):
    """Raised by request guards; turned into the gateway's normal error shape."""

    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


def client_ip(request: Request, trust_proxy: bool = False) -> str:
    """The caller's address. Behind a trusted proxy the proxy appends the real address
    at the END of X-Forwarded-For; anything to its left can be forged by the client."""
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for", "")
        last = fwd.split(",")[-1].strip()
        if last:
            return last[:64]
    return request.client.host if request.client else "unknown"


def _session_id(header: str | None, body_user: str | None) -> tuple[str, bool]:
    """Returns (session_id, caller_supplied). A caller who supplies nothing gets a
    random one-shot session: safe, but tokens can't be reused across requests."""
    for candidate in (header, body_user):
        if candidate and candidate.strip():
            return candidate.strip()[:128], True
    return "anon-" + secrets.token_hex(8), False


def create_app(llm: LLMBackend | None = None, vault: TokenVault | None = None,
               policy: Policy | None = None, output_policy: OutputPolicy | None = None,
               audit: AuditLog | None = None, admin_key: str | None = None,
               api_keys: Iterable[str] | None = None, rate_limit_per_min: int | None = None,
               trust_proxy: bool = False, llm_kind: str = "mock") -> FastAPI:
    # `is None`, never `or`: TokenVault has __len__, so an EMPTY vault is falsy and `vault or ...`
    # would silently swap a configured vault (with its real key) for a random throwaway one.
    llm = llm if llm is not None else MockLLM()
    vault = vault if vault is not None else TokenVault()
    policy = policy if policy is not None else Policy.default()
    output_policy = output_policy if output_policy is not None else OutputPolicy()
    audit = audit if audit is not None else AuditLog(":memory:")
    admin_key = admin_key if admin_key is not None else os.environ.get("PII_ADMIN_KEY")
    app = FastAPI(title="India PII Gateway", version="0.7")

    # client keys are kept only as SHA-256 digests; the originals are not retained
    key_digests = [hashlib.sha256(k.encode()).digest() for k in (api_keys or ()) if k]
    limiter = RateLimiter(rate_limit_per_min) if rate_limit_per_min else None
    auth_failures = RateLimiter(FAILED_AUTH_PER_MIN)

    @app.exception_handler(GatewayHTTPError)
    async def _guard_error(_request: Request, exc: GatewayHTTPError):
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        """Runs before the framework reads the body, so an oversized upload is never buffered.
        POSTs must declare their length (every normal HTTP client does)."""
        if request.method == "POST":
            raw = request.headers.get("content-length")
            if raw is None:
                return _error(411, "length_required", "Content-Length header is required.")
            if not raw.isdigit() or int(raw) > MAX_BODY_BYTES:
                return _error(413, "too_large", "Request body is too large.")
        return await call_next(request)

    def guard(request: Request, authorization: str | None = Header(default=None)) -> None:
        """Runs after the size check: API key, then rate limit."""
        ip = client_ip(request, trust_proxy)
        if not key_digests:
            if limiter and not limiter.allow("ip:" + ip):
                raise GatewayHTTPError(429, "rate_limited", "Too many requests. Slow down.",
                                       {"Retry-After": str(limiter.retry_after("ip:" + ip))})
            return
        if auth_failures.over(ip):   # locked out: don't even check the key, so guessing gets nowhere
            raise GatewayHTTPError(429, "rate_limited", "Too many failed attempts.",
                                   {"Retry-After": str(auth_failures.retry_after(ip))})
        presented = ""
        if authorization and authorization[:7].lower() == "bearer ":
            presented = authorization[7:].strip()
        digest = hashlib.sha256(presented.encode()).digest()
        matched = False
        for known in key_digests:                       # no early exit: same work for every guess
            matched |= hmac.compare_digest(known, digest)
        if not matched:
            auth_failures.hit(ip)
            raise GatewayHTTPError(401, "invalid_api_key", "Missing or invalid API key.",
                                   {"WWW-Authenticate": "Bearer"})
        who = "key:" + digest.hex()[:16]
        if limiter and not limiter.allow(who):
            raise GatewayHTTPError(429, "rate_limited", "Too many requests. Slow down.",
                                   {"Retry-After": str(limiter.retry_after(who))})

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "vault_key": "ephemeral" if vault.ephemeral else "configured",
                "llm": llm_kind, "auth": "api_key" if key_digests else "open"}

    def _check_admin(key: str | None):
        if not admin_key:
            return _error(404, "not_found", "Not found.")
        if not key or not hmac.compare_digest(key.encode(), admin_key.encode()):
            return _error(401, "unauthorized", "Missing or wrong admin key.")
        return None

    @app.get("/audit/events")
    def audit_events(limit: int = 50, x_admin_key: str | None = Header(default=None)):
        denied = _check_admin(x_admin_key)
        return denied or {"events": audit.recent(limit)}

    @app.get("/audit/stats")
    def audit_stats(x_admin_key: str | None = Header(default=None)):
        denied = _check_admin(x_admin_key)
        return denied or audit.stats()

    @app.post("/v1/chat/completions", dependencies=[Depends(guard)])
    def chat(req: ChatRequest, x_session_id: str | None = Header(default=None)):
        started = time.perf_counter()
        sid, supplied = _session_id(x_session_id, req.user)
        headers = {"X-Session-Id": sid}

        def ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        if req.stream:
            return _error(400, "stream_unsupported", "stream=true is not supported yet.")
        if sum(len(_flat(m.content)) for m in req.messages) > MAX_CHARS:
            return _error(413, "too_large", f"Conversation exceeds {MAX_CHARS} characters.")

        # --- 2. protect everything that will leave the building
        safe_messages: list[dict] = []
        summary: dict[str, dict[str, int]] = {}
        for m in req.messages:
            result_content, blocked = _protect_content(m.content, sid, vault, policy, summary)
            if blocked:
                log.info("blocked request session=%s", _log_sid(sid))
                audit.record(session_id=sid, model=req.model, outcome="blocked",
                             blocked_types=blocked.blocked_types, latency_ms=ms())
                return _error(400, "pii_blocked", blocked.user_message, headers)
            safe = m.model_dump(exclude_none=True)
            safe["content"] = result_content
            safe_messages.append(safe)

        # --- 3. call the model with protected messages only
        try:
            reply = llm(safe_messages, req.model)
        except LLMError as exc:
            # LLMError text is written by our adapters and holds only a status code or a fixed phrase, never prompt data
            log.warning("upstream error session=%s reason=%s", _log_sid(sid), exc)
            audit.record(session_id=sid, model=req.model, outcome="upstream_error",
                         input_detected=summary, latency_ms=ms())
            return _error(502, "upstream_error", "The language model is unavailable.", headers)
        except Exception:  # never echo exception text: it could carry prompt contents
            log.exception("llm backend crashed")
            audit.record(session_id=sid, model=req.model, outcome="upstream_error",
                         input_detected=summary, latency_ms=ms())
            return _error(502, "upstream_error", "The language model is unavailable.", headers)

        # --- 4. scan the reply, then restore tokens; 5. audit and respond
        scan = scan_and_restore(reply, sid, vault, output_policy)
        audit.record(session_id=sid, model=req.model, outcome="ok", input_detected=summary,
                     output_redacted=scan.redacted, images_removed=scan.images_removed,
                     foreign_tokens=scan.foreign_tokens, latency_ms=ms())
        log.info("ok session=%s summary=%s output_flagged=%s", _log_sid(sid), summary, scan.flagged)
        return JSONResponse(headers=headers, content={
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": scan.text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_pii_gateway": {
                "session_id": sid, "session_supplied": supplied, "detected": summary,
                "output": {"redacted": scan.redacted, "images_removed": scan.images_removed,
                           "foreign_tokens": scan.foreign_tokens},
            },
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
    """Returns (protected_content, blocking ProtectResult or None)."""
    if content is None:
        return None, None
    if isinstance(content, str):
        r = protect(content, sid, vault, policy)
        if r.blocked:
            return None, r
        _merge(summary, r.summary())
        return r.text, None
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            r = protect(part["text"], sid, vault, policy)
            if r.blocked:
                return None, r
            _merge(summary, r.summary())
            out.append({**part, "text": r.text})
        else:
            out.append(part)   # non-text parts (images...) are not inspected yet: documented limit
    return out, None


app = create_app(audit=AuditLog(os.environ.get("PII_AUDIT_DB", "audit.db")))
