import random

from gateway import verhoeff
from tests import synth


def test_known_example():
    # Standard published example: "236" takes check digit 3.
    assert verhoeff.check_digit("236") == "3"
    assert verhoeff.is_valid("2363")
    assert not verhoeff.is_valid("2364")


def test_generated_numbers_validate():
    rng = random.Random(1)
    for _ in range(500):
        assert verhoeff.is_valid(synth.aadhaar(rng))


def test_every_single_digit_error_is_caught():
    rng = random.Random(2)
    for _ in range(100):
        n = synth.aadhaar(rng)
        for i in range(len(n)):
            for d in "0123456789":
                if d != n[i]:
                    assert not verhoeff.is_valid(n[:i] + d + n[i + 1:])


def test_adjacent_swaps_are_caught():
    rng = random.Random(3)
    for _ in range(200):
        n = synth.aadhaar(rng)
        for i in range(len(n) - 1):
            if n[i] != n[i + 1]:
                swapped = n[:i] + n[i + 1] + n[i] + n[i + 2:]
                assert not verhoeff.is_valid(swapped)


def test_rejects_non_digits():
    assert not verhoeff.is_valid("")
    assert not verhoeff.is_valid("23a3")
    assert not verhoeff.is_valid("२३६३")  # Devanagari digits need normalising first
