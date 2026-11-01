from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from .llm import LLMError, _text_of

log = logging.getLogger("pii_gateway")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}   # the provider is busy or hiccuping: try again / next model
_sleep = time.sleep


class _Failed(Exception):
    """One attempt failed. `reason` is safe to log: a status code or a fixed phrase, never data."""

    def __init__(self, reason: str, retry: bool, next_model: bool):
        super().__init__(reason)
        self.reason, self.retry, self.next_model = reason, retry, next_model


class OpenAICompatLLM:
    """Forwards the already-protected messages to any OpenAI-compatible
    `/chat/completions` endpoint (OpenAI, Azure OpenAI, Groq, Gemini's compatibility
    endpoint, OpenRouter, a local Ollama or vLLM, ...).

    Rules this class keeps:
    * the API key lives only in memory and in the Authorization header; it is never
      logged, never put in an error, and never sent anywhere but `base_url`;
    * redirects are not followed, so a redirect cannot carry the key to another host;
    * errors and logs carry only a status code, never the response body or the prompt,
      because either could contain personal data;
    * plain http is accepted only for a model running on the same machine.

    Reliability: a busy provider (429/5xx/timeout) is retried `retries` times, then the next
    model in `fallback_models` is tried. A bad key or bad request (400/401/403) fails at once,
    because another attempt cannot fix it. `total_budget` caps how long one request may keep trying.
    """

    def __init__(self, base_url: str, api_key: str | None = None, model: str | None = None,
                 timeout: float = 30.0, client: httpx.Client | None = None,
                 fallback_models: tuple[str, ...] | list[str] = (), retries: int = 1,
                 backoff: float = 0.5, total_budget: float = 60.0, clock=time.monotonic):
        parts = urlparse(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("LLM_BASE_URL must be a full URL such as https://api.openai.com/v1")
        if parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS:
            raise ValueError("LLM_BASE_URL must use https unless the model runs on this machine")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._model = model          # when set, always tried first: the caller cannot pick a pricier model
        self._fallbacks = tuple(m for m in fallback_models if m)
        self._retries, self._backoff, self._budget, self._clock = max(0, retries), backoff, total_budget, clock
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=False)

    def __repr__(self) -> str:   # never show the key
        return f"OpenAICompatLLM(url={self._url!r}, model={self._model!r}, fallbacks={self._fallbacks!r})"

    def _once(self, messages: list[dict], model: str) -> str:
        try:
            resp = self._client.post(self._url, json={"model": model, "messages": messages}, headers=self._headers)
        except httpx.HTTPError:
            raise _Failed("upstream request failed", retry=True, next_model=True) from None
        if resp.status_code != 200:
            reason = f"upstream returned HTTP {resp.status_code}"
            if resp.status_code in _RETRYABLE:
                raise _Failed(reason, retry=True, next_model=True)
            if resp.status_code == 404:                   # that model name is not available: try the next one
                raise _Failed(reason, retry=False, next_model=True)
            raise _Failed(reason, retry=False, next_model=False)   # key or request problem: stop now
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Failed("upstream returned an unexpected response", retry=True, next_model=True) from None
        if isinstance(content, list):
            content = _text_of(content)
        if not isinstance(content, str):
            raise _Failed("upstream returned an unexpected response", retry=True, next_model=True)
        return content

    def __call__(self, messages: list[dict], model: str) -> str:
        models = list(dict.fromkeys([self._model or model, *self._fallbacks]))
        started, attempts, last = self._clock(), 0, "upstream request failed"
        for mi, m in enumerate(models):
            for attempt in range(self._retries + 1):
                if attempts and self._clock() - started >= self._budget:
                    raise LLMError(f"{last} (gave up after {attempts} attempts)") from None
                try:
                    return self._once(messages, m)
                except _Failed as f:
                    attempts += 1
                    last = f.reason
                    if not f.next_model:
                        raise LLMError(f.reason) from None
                    log.warning("model call failed model=%s reason=%s attempt=%d", m, f.reason, attempts)
                    if not f.retry:
                        break
                    if attempt < self._retries:
                        _sleep(self._backoff)
        raise LLMError(f"{last} (after {attempts} attempts)") from None
