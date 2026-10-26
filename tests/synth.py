"""Synthetic Indian identifiers for tests and evaluation.

Everything here is randomly generated with valid *format and checksum* but is
not looked up against, or issued by, any real registry. Never put real
Aadhaar, PAN, card or bank numbers in tests or demo data.
"""
from __future__ import annotations

import random
import string

from gateway import verhoeff
from gateway.detectors import gstin_check_char

_UPPER = string.ascii_uppercase
_ALNUM = string.ascii_uppercase + string.digits
UPI_HANDLES = ("okhdfcbank", "okicici", "oksbi", "okaxis", "ybl", "paytm", "ibl", "axl")
GST_STATES = [f"{i:02d}" for i in range(1, 38)]


def random_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def aadhaar(rng: random.Random) -> str:
    body = rng.choice("23456789") + random_digits(rng, 10)
    return body + verhoeff.check_digit(body)


def random_12_digit(rng: random.Random) -> str:
    """12 random digits, first digit 2-9 (shape of an Aadhaar, checksum not enforced)."""
    return rng.choice("23456789") + random_digits(rng, 11)


def pan(rng: random.Random) -> str:
    return (
        "".join(rng.choice(_UPPER) for _ in range(3))
        + rng.choice("ABCFGHLJPT")
        + rng.choice(_UPPER)
        + random_digits(rng, 4)
        + rng.choice(_UPPER)
    )


def mobile(rng: random.Random) -> str:
    return rng.choice("6789") + random_digits(rng, 9)


def ifsc(rng: random.Random) -> str:
    return "".join(rng.choice(_UPPER) for _ in range(4)) + "0" + "".join(rng.choice(_ALNUM) for _ in range(6))


def upi(rng: random.Random) -> str:
    name = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 9)))
    return f"{name}@{rng.choice(UPI_HANDLES)}"


def gstin(rng: random.Random) -> str:
    first14 = rng.choice(GST_STATES) + pan(rng) + rng.choice("123456789") + "Z"
    return first14 + gstin_check_char(first14)


def luhn_check_digit(partial: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(partial)):
        d = int(ch)
        if i % 2 == 0:  # these digits become odd-position digits once the check digit is appended
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def card(rng: random.Random) -> str:
    partial = "4" + random_digits(rng, 14)
    return partial + luhn_check_digit(partial)


def bank_account(rng: random.Random) -> str:
    return rng.choice("123456789") + random_digits(rng, rng.randint(10, 13))


def group(s: str, size: int = 4, sep: str = " ") -> str:
    return sep.join(s[i:i + size] for i in range(0, len(s), size))


_DEVANAGARI = str.maketrans("0123456789", "०१२३४५६७८९")


def to_devanagari(s: str) -> str:
    return s.translate(_DEVANAGARI)
