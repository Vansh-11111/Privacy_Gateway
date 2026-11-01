from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window limiter: at most `limit` hits per `window` seconds per key.

    In memory and per process, which matches the single-worker deployment. Keys are
    client addresses or hashed API keys, never message content. The key table is capped
    so a flood of fake addresses cannot grow memory without bound."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic, max_keys: int = 10_000):
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit, self.window, self._clock, self._max_keys = limit, window, clock, max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _live(self, key: str, now: float) -> deque[float]:
        q = self._hits.get(key)
        if q is None:
            return deque()
        while q and now - q[0] >= self.window:
            q.popleft()
        return q

    def over(self, key: str) -> bool:
        """True if `key` has used up its allowance. Does not record a hit."""
        with self._lock:
            return len(self._live(key, self._clock())) >= self.limit

    def hit(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            q.append(now)
            self._hits[key] = q
            if len(self._hits) > self._max_keys:
                self._prune(now)

    def allow(self, key: str) -> bool:
        """Record a hit unless the key is already over its limit."""
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if len(q) >= self.limit:
                self._hits[key] = q
                return False
            q.append(now)
            self._hits[key] = q
            if len(self._hits) > self._max_keys:
                self._prune(now)
            return True

    def retry_after(self, key: str) -> int:
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if len(q) < self.limit:
                return 0
            return max(1, int(self.window - (now - q[0])) + 1)

    def _prune(self, now: float) -> None:   # caller holds the lock
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]
        if len(self._hits) > self._max_keys:   # still full: drop the oldest half
            for k in sorted(self._hits, key=lambda k: self._hits[k][-1])[: len(self._hits) // 2]:
                del self._hits[k]
