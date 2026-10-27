from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from cryptography.fernet import Fernet

from .detectors import normalize

_TYPE_RX = re.compile(r"^[A-Z][A-Z_]*$")
_TOKEN_RX = re.compile(r"<([A-Z][A-Z_]*)_(\d+)>")
_DIGIT_TYPES = frozenset({"AADHAAR", "CARD", "BANK_ACCOUNT", "OTP", "CVV"})
_ALNUM_UPPER_TYPES = frozenset({"PAN", "IFSC", "GSTIN"})


def _canonical(entity_type: str, value: str) -> str:
    """Formatting-independent form of a value, used only to spot repeats."""
    v = normalize(value)
    if entity_type in _DIGIT_TYPES:
        return "".join(ch for ch in v if ch.isdigit())
    if entity_type == "MOBILE":
        return "".join(ch for ch in v if ch.isdigit())[-10:]
    if entity_type in _ALNUM_UPPER_TYPES:
        return "".join(ch for ch in v if ch.isalnum()).upper()
    return "".join(v.split()).lower()


@dataclass
class _Session:
    last_used: float
    tokens: dict = field(default_factory=dict)     # token -> ciphertext
    by_digest: dict = field(default_factory=dict)  # HMAC(value) -> token
    counters: dict = field(default_factory=dict)   # entity type -> last number used


class TokenVault:
    def __init__(self, key: bytes | str | None = None, ttl_seconds: float | None = 3600,
                 clock: Callable[[], float] = time.monotonic):
        raw = key if key is not None else os.environ.get("PII_VAULT_KEY")
        self.ephemeral = raw is None
        if raw is None:
            raw = Fernet.generate_key()
        if isinstance(raw, str):
            raw = raw.encode()
        try:
            self._fernet = Fernet(raw)
        except ValueError as exc:
            raise ValueError("invalid vault key: it must be a Fernet key (Fernet.generate_key())") from exc
        self._dedupe_key = hmac.new(raw, b"pii-vault-dedupe-v1", hashlib.sha256).digest()
        self._ttl = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def __repr__(self) -> str:  # never show contents
        return f"TokenVault(sessions={len(self._sessions)}, ephemeral={self.ephemeral})"

    def __len__(self) -> int:
        return len(self._sessions)

    def _digest(self, entity_type: str, value: str) -> str:
        msg = f"{entity_type}:{_canonical(entity_type, value)}".encode("utf-8")
        return hmac.new(self._dedupe_key, msg, hashlib.sha256).hexdigest()

    def _purge_locked(self, now: float) -> None:
        if self._ttl is None:
            return
        for sid in [s for s, sess in self._sessions.items() if now - sess.last_used > self._ttl]:
            del self._sessions[sid]

    # --------------------------------------------------------------------- API
    def tokenize(self, session_id: str, entity_type: str, value: str) -> str:
        if not session_id:
            raise ValueError("session_id is required")
        if not _TYPE_RX.match(entity_type):
            raise ValueError("entity_type must be UPPER_CASE letters and underscores")
        digest = self._digest(entity_type, value)
        with self._lock:
            now = self._clock()
            self._purge_locked(now)
            sess = self._sessions.setdefault(session_id, _Session(last_used=now))
            sess.last_used = now
            existing = sess.by_digest.get(digest)
            if existing:
                return existing
            n = sess.counters.get(entity_type, 0) + 1
            sess.counters[entity_type] = n
            token = f"<{entity_type}_{n}>"
            sess.tokens[token] = self._fernet.encrypt(value.encode("utf-8"))
            sess.by_digest[digest] = token
            return token

    def resolve(self, session_id: str, token: str) -> str | None:
        with self._lock:
            now = self._clock()
            self._purge_locked(now)
            sess = self._sessions.get(session_id)
            if sess is None:
                return None
            ciphertext = sess.tokens.get(token)
            if ciphertext is None:
                return None
            sess.last_used = now
        return self._fernet.decrypt(ciphertext).decode("utf-8")

    def restore(self, session_id: str, text: str) -> str:
        """Replace every token this session issued with its real value.
        Unknown tokens (or tokens from other sessions) are left as they are."""
        def sub(m: re.Match) -> str:
            value = self.resolve(session_id, m.group(0))
            return m.group(0) if value is None else value
        return _TOKEN_RX.sub(sub, text)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)