from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .detectors import detect
from .vault import TokenVault

_TOKEN = r"<[A-Z][A-Z_]*_\d+>"
_TOKEN_RX = re.compile(_TOKEN)
# a URL may contain tokens such as <AADHAAR_1>, which hold characters '<' and '>'
_URL_RX = re.compile(rf"(?:https?://|www\.)(?:{_TOKEN}|[^\s<>\"')\]])+", re.I)
_MD_IMAGE_RX = re.compile(r"!\[[^\]]*\]\(([^)]*)\)")
_HTML_IMAGE_RX = re.compile(rf"<img\b(?:[^<>]|{_TOKEN})*>", re.I)
_SRC_RX = re.compile(r"""src\s*=\s*["']?([^\s"'>]+)""", re.I)

IMAGE_REMOVED = "[image removed by gateway]"


@dataclass(frozen=True)
class OutputPolicy:
    allow_types: frozenset = frozenset({"IFSC"})     # may appear in replies (identifies a branch)
    allow_values: frozenset = frozenset()            # exact strings, e.g. the bank's own helpline/email
    allowed_image_hosts: frozenset = frozenset()     # empty = no external images at all

    def value_allowed(self, value: str) -> bool:
        return value.strip().lower() in {v.lower() for v in self.allow_values}


@dataclass(frozen=True)
class OutputScan:
    text: str
    redacted: dict = field(default_factory=dict)     # type -> count
    images_removed: int = 0
    foreign_tokens: int = 0                          # tokens this session never issued

    @property
    def flagged(self) -> bool:
        return bool(self.redacted or self.images_removed or self.foreign_tokens)


def _host(url: str) -> str:
    url = url.strip()
    if url.lower().startswith("www."):
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def _strip_images(text: str, policy: OutputPolicy) -> tuple[str, int]:
    removed = 0

    def md(m: re.Match) -> str:
        nonlocal removed
        target = m.group(1).strip()
        if _host(target) in policy.allowed_image_hosts:
            return m.group(0)
        removed += 1
        return IMAGE_REMOVED

    def html(m: re.Match) -> str:
        nonlocal removed
        src = _SRC_RX.search(m.group(0))
        if src and _host(src.group(1)) in policy.allowed_image_hosts:
            return m.group(0)
        removed += 1
        return IMAGE_REMOVED

    text = _MD_IMAGE_RX.sub(md, text)
    text = _HTML_IMAGE_RX.sub(html, text)
    return text, removed


def _redact_entities(text: str, policy: OutputPolicy) -> tuple[str, dict]:
    out, counts, cursor = [], {}, 0
    for e in detect(text):
        original = e.slice(text)
        if e.type in policy.allow_types or policy.value_allowed(original):
            continue
        out.append(text[cursor:e.start])
        out.append(f"[REDACTED_{e.type}]")
        counts[e.type] = counts.get(e.type, 0) + 1
        cursor = e.end
    out.append(text[cursor:])
    return "".join(out), counts


def _restore_outside_urls(text: str, session_id: str, vault: TokenVault) -> str:
    out, cursor = [], 0
    for m in _URL_RX.finditer(text):
        out.append(vault.restore(session_id, text[cursor:m.start()]))
        out.append(m.group(0))                      # URL text stays exactly as the model wrote it
        cursor = m.end()
    out.append(vault.restore(session_id, text[cursor:]))
    return "".join(out)


def scan_and_restore(reply: str, session_id: str, vault: TokenVault,
                     policy: OutputPolicy | None = None) -> OutputScan:
    policy = policy or OutputPolicy()
    text, images_removed = _strip_images(reply, policy)
    text, redacted = _redact_entities(text, policy)
    foreign = sum(1 for t in _TOKEN_RX.findall(text) if vault.resolve(session_id, t) is None)
    text = _restore_outside_urls(text, session_id, vault)
    return OutputScan(text=text, redacted=redacted, images_removed=images_removed, foreign_tokens=foreign)
