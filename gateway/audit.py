from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session TEXT NOT NULL,
    model TEXT NOT NULL,
    outcome TEXT NOT NULL,
    blocked_types TEXT NOT NULL DEFAULT '[]',
    input_detected TEXT NOT NULL DEFAULT '{}',
    output_redacted TEXT NOT NULL DEFAULT '{}',
    images_removed INTEGER NOT NULL DEFAULT 0,
    foreign_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0
)
"""


class AuditLog:
    def __init__(self, path: str = ":memory:", salt: bytes | str | None = None):
        raw = salt if salt is not None else os.environ.get("PII_AUDIT_SALT")
        if raw is None:
            raw = secrets.token_bytes(16)
        self._salt = raw.encode() if isinstance(raw, str) else raw
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()

    def __repr__(self) -> str:
        return "AuditLog()"

    def session_hash(self, session_id: str) -> str:
        return hmac.new(self._salt, session_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

    def record(self, *, session_id: str, model: str, outcome: str, blocked_types=(),
               input_detected=None, output_redacted=None, images_removed: int = 0,
               foreign_tokens: int = 0, latency_ms: int = 0) -> None:
        row = (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            self.session_hash(session_id),
            model[:64],
            outcome,
            json.dumps(sorted(blocked_types)),
            json.dumps(input_detected or {}, sort_keys=True),
            json.dumps(output_redacted or {}, sort_keys=True),
            int(images_removed), int(foreign_tokens), int(latency_ms),
        )
        with self._lock:
            self._db.execute(
                "INSERT INTO events (ts, session, model, outcome, blocked_types, input_detected,"
                " output_redacted, images_removed, foreign_tokens, latency_ms)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)", row)
            self._db.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in ("blocked_types", "input_detected", "output_redacted"):
            d[k] = json.loads(d[k])
        return d

    def recent(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        with self._lock:
            rows = self._db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def stats(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT * FROM events").fetchall()
        by_outcome: dict[str, int] = {}
        found: dict[str, int] = {}
        redacted: dict[str, int] = {}
        blocked: dict[str, int] = {}
        images = foreign = 0
        for r in rows:
            d = self._row(r)
            by_outcome[d["outcome"]] = by_outcome.get(d["outcome"], 0) + 1
            for t, acts in d["input_detected"].items():
                found[t] = found.get(t, 0) + sum(acts.values())
            for t, n in d["output_redacted"].items():
                redacted[t] = redacted.get(t, 0) + n
            for t in d["blocked_types"]:
                blocked[t] = blocked.get(t, 0) + 1
            images += d["images_removed"]
            foreign += d["foreign_tokens"]
        return {"requests": len(rows), "by_outcome": by_outcome, "input_found_by_type": found,
                "blocked_by_type": blocked, "output_redacted_by_type": redacted,
                "images_removed": images, "foreign_tokens": foreign}

    def clear(self) -> None:
        """Delete all events (used by the demo's reset button)."""
        with self._lock:
            self._db.execute("DELETE FROM events")
            self._db.commit()
            
    def close(self) -> None:
        with self._lock:
            self._db.close()
