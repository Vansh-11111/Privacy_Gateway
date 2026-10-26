from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import verhoeff

DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class Entity:
    type: str
    start: int
    end: int
    confidence: float
    meta: dict = field(default_factory=dict, repr=False, compare=False)

    def slice(self, text: str) -> str:
        return text[self.start:self.end]

#Normalization
_ZERO_WIDTH = frozenset("​‌‍⁠﻿")


def normalize(text: str) -> str:
    """Map Indic/Arabic/full-width digits to ASCII and zero-width chars to a space.

    Each character maps to exactly one character, so len(normalize(t)) == len(t).
    """
    out = []
    for ch in text:
        if not ch.isascii() and ch.isdecimal():
            out.append(str(unicodedata.decimal(ch)))
        elif ch in _ZERO_WIDTH:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)

#Checksum
def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_check_char(first14: str) -> str:
    """Check character (15th) of a GSTIN, from its first 14 characters."""
    total = 0
    for i, ch in enumerate(first14.upper()):
        p = _GSTIN_CHARS.index(ch) * (1 if i % 2 == 0 else 2)
        total += p // 36 + p % 36
    return _GSTIN_CHARS[(36 - total % 36) % 36]

def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


CTX_AADHAAR = _rx(r"a+d+h+a+r+|uidai|आधार")
CTX_PAN = _rx(r"\bpan\b|permanent account|पैन|पेन कार्ड")
CTX_MOBILE = _rx(r"mobile|phone|\bph\b|\bmob\b|contact|call|whatsapp|number|no\.|मोबाइल|फोन|फ़ोन|नंबर|नम्बर")
CTX_IFSC = _rx(r"ifsc|branch")
CTX_UPI = _rx(r"\bupi\b|\bvpa\b|gpay|google pay|phonepe|paytm|bhim")
CTX_ACCOUNT = _rx(r"a/c|\bacct?\b|account|खाता|खाते|\bkhata\b|\bkhate\b")
CTX_GST = _rx(r"\bgst|gstin")
CTX_CARD = _rx(r"card|visa|mastercard|rupay|credit|debit|कार्ड")
NEG_CTX = _rx(r"order|invoice|\btxn\b|transaction|\bref\b|reference|tracking|\bawb\b|consignment|ticket|booking|\bpnr\b")


def _near(rx: re.Pattern, text: str, start: int, end: int, before: int = 40, after: int = 20) -> bool:
    return rx.search(text, max(0, start - before), min(len(text), end + after)) is not None


def _before_only(rx: re.Pattern, text: str, start: int, window: int = 30) -> bool:
    return rx.search(text, max(0, start - window), start) is not None


_PREV_RUN = re.compile(r"\d{3,}[ .\-]?$")
_NEXT_RUN = re.compile(r"^[ .\-]?\d{3,}")


def _inside_longer_number(text: str, start: int, end: int) -> bool:
    """True if the candidate is just a piece of a longer separated digit run
    (e.g. the first 12 digits of a 16-digit card number)."""
    return bool(_PREV_RUN.search(text[max(0, start - 20):start]) or _NEXT_RUN.search(text[end:end + 20]))


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _clamp(x: float) -> float:
    return max(0.0, min(0.99, x))

_AADHAAR_RX = re.compile(r"(?<!\d)[2-9](?:[ .\-]?\d){11}(?!\d)")


def _find_aadhaar(text: str) -> Iterable[Entity]:
    for m in _AADHAAR_RX.finditer(text):
        if _inside_longer_number(text, m.start(), m.end()):
            continue
        digits = _digits(m.group())
        ok = verhoeff.is_valid(digits)
        ctx = _near(CTX_AADHAAR, text, m.start(), m.end())
        if ok:
            conf = 0.97 if ctx else 0.70
            if not ctx and _near(NEG_CTX, text, m.start(), m.end()):
                conf -= 0.30
        elif ctx:
            conf = 0.50  # keyword present but checksum fails: likely a typo
        else:
            continue
        yield Entity("AADHAAR", m.start(), m.end(), _clamp(conf), {"checksum_ok": ok})


_PAN_RX = re.compile(r"(?<![A-Za-z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z](?![A-Za-z0-9])", re.IGNORECASE)


def _find_pan(text: str) -> Iterable[Entity]:
    for m in _PAN_RX.finditer(text):
        ctx = _near(CTX_PAN, text, m.start(), m.end())
        conf = (0.90 if m.group().isupper() else 0.70) + (0.07 if ctx else 0.0)
        yield Entity("PAN", m.start(), m.end(), _clamp(conf))


_MOBILE_RX = re.compile(r"(?<![\d+])(?:(?:\+|00)?91[ .\-]?|0)?[6-9](?:[ .\-]?\d){9}(?!\d)")


def _find_mobile(text: str) -> Iterable[Entity]:
    for m in _MOBILE_RX.finditer(text):
        if _inside_longer_number(text, m.start(), m.end()):
            continue
        digits = _digits(m.group())
        prefixed = len(digits) > 10
        ctx = _near(CTX_MOBILE, text, m.start(), m.end())
        if prefixed:
            conf = 0.95 if ctx else 0.85
        else:
            conf = 0.90 if ctx else 0.55
        if not ctx and _near(NEG_CTX, text, m.start(), m.end()):
            conf -= 0.30
        yield Entity("MOBILE", m.start(), m.end(), _clamp(conf))


_IFSC_RX = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{4}0[A-Za-z0-9]{6}(?![A-Za-z0-9])")


def _find_ifsc(text: str) -> Iterable[Entity]:
    for m in _IFSC_RX.finditer(text):
        ctx = _near(CTX_IFSC, text, m.start(), m.end())
        conf = (0.80 if m.group().isupper() else 0.55) + (0.15 if ctx else 0.0)
        yield Entity("IFSC", m.start(), m.end(), _clamp(conf))


_EMAIL_RX = re.compile(r"(?<![\w.%+\-])[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}(?!\w)")


def _find_email(text: str) -> Iterable[Entity]:
    for m in _EMAIL_RX.finditer(text):
        yield Entity("EMAIL", m.start(), m.end(), 0.95)


# Common UPI handles (not exhaustive; unknown handles score lower).
_UPI_HANDLES = frozenset(
    "okhdfcbank okicici oksbi okaxis ybl ibl axl paytm apl upi fbl ikwik airtel jio "
    "yesbank sbi hdfcbank icici axisbank kotak pnb barodampay postbank cnrb".split()
)
_UPI_RX = re.compile(r"(?<![\w.%+\-@])[A-Za-z0-9._\-]{2,50}@([A-Za-z][A-Za-z0-9]{1,63})(?!\w)(?!\.[A-Za-z])")


def _find_upi(text: str) -> Iterable[Entity]:
    for m in _UPI_RX.finditer(text):
        handle = m.group(1).lower()
        ctx = _near(CTX_UPI, text, m.start(), m.end())
        conf = 0.95 if handle in _UPI_HANDLES else 0.60
        if ctx:
            conf += 0.30
        yield Entity("UPI", m.start(), m.end(), _clamp(conf))


_GSTIN_RX = re.compile(r"(?<![A-Za-z0-9])\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z](?![A-Za-z0-9])", re.IGNORECASE)
_GST_STATE_CODES = {f"{i:02d}" for i in range(1, 39)} | {"97", "99"}


def _find_gstin(text: str) -> Iterable[Entity]:
    for m in _GSTIN_RX.finditer(text):
        g = m.group().upper()
        if g[:2] not in _GST_STATE_CODES:
            continue
        ok = gstin_check_char(g[:14]) == g[14]
        ctx = _near(CTX_GST, text, m.start(), m.end())
        if ok:
            conf = 0.97
        elif ctx:
            conf = 0.50
        else:
            continue
        yield Entity("GSTIN", m.start(), m.end(), conf, {"checksum_ok": ok})


_CARD_RX = re.compile(r"(?<!\d)\d(?:[ .\-]?\d){12,18}(?!\d)")


def _find_card(text: str) -> Iterable[Entity]:
    for m in _CARD_RX.finditer(text):
        digits = _digits(m.group())
        if not 13 <= len(digits) <= 19 or not luhn_valid(digits):
            continue
        if _inside_longer_number(text, m.start(), m.end()):
            continue
        ctx = _near(CTX_CARD, text, m.start(), m.end())
        yield Entity("CARD", m.start(), m.end(), 0.97 if ctx else 0.88)


_ACCOUNT_RX = re.compile(r"(?<!\d)\d{9,18}(?!\d)")


def _find_bank_account(text: str) -> Iterable[Entity]:
    # Needs an account keyword just before it: a bare long number is too ambiguous.
    for m in _ACCOUNT_RX.finditer(text):
        if _before_only(CTX_ACCOUNT, text, m.start()):
            yield Entity("BANK_ACCOUNT", m.start(), m.end(), 0.85)


_OTP_AFTER = re.compile(
    r"(?:\botp\b|ओटीपी|one[\s-]?time[\s-]?(?:password|pin|code)|verification code)[^\d\n]{0,25}?(?<!\d)(\d{4,8})(?!\d)",
    re.IGNORECASE,
)
_OTP_BEFORE = re.compile(r"(?<!\d)(\d{4,8})(?!\d)[^\d\n]{0,20}?(?:\botp\b|ओटीपी)", re.IGNORECASE)
_CVV_AFTER = re.compile(r"(?:\bcvv2?\b|\bcvc\b|security code)[^\d\n]{0,15}?(?<!\d)(\d{3,4})(?!\d)", re.IGNORECASE)


def _find_secrets(text: str) -> Iterable[Entity]:
    """OTP and CVV: the digits only (group 1), never sent to an LLM under any policy."""
    for rx in (_OTP_AFTER, _OTP_BEFORE):
        for m in rx.finditer(text):
            yield Entity("OTP", m.start(1), m.end(1), 0.90)
    for m in _CVV_AFTER.finditer(text):
        yield Entity("CVV", m.start(1), m.end(1), 0.90)


_FINDERS: tuple[Callable[[str], Iterable[Entity]], ...] = (
    _find_aadhaar, _find_pan, _find_mobile, _find_ifsc, _find_email, _find_upi,
    _find_gstin, _find_card, _find_bank_account, _find_secrets,
)
#PublicApi
def _resolve_overlaps(cands: list[Entity]) -> list[Entity]:
    chosen: list[Entity] = []
    for e in sorted(cands, key=lambda e: (-e.confidence, -(e.end - e.start), e.start)):
        if all(e.end <= c.start or e.start >= c.end for c in chosen):
            chosen.append(e)
    return sorted(chosen, key=lambda e: e.start)


def detect(text: str, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[Entity]:
    """Return non-overlapping entities with confidence >= min_confidence,
    ordered by position. Offsets index into the original `text`."""
    norm = normalize(text)
    cands = [e for finder in _FINDERS for e in finder(norm) if e.confidence >= min_confidence]
    return _resolve_overlaps(cands)


def _main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--show"]
    if not args:
        print('usage: python -m gateway.detectors [--show] "text to scan"')
        return 2
    text = " ".join(args)
    hits = detect(text)
    if not hits:
        print("no entities found")
    for e in hits:
        shown = f"  {e.slice(text)!r}" if "--show" in argv else ""
        print(f"{e.type:<13} {e.start:>3}-{e.end:<3} conf={e.confidence:.2f}{shown}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
