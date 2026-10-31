import pytest

from gateway.attacks import SCENARIOS, Harness, evaluate, leaks
from gateway.bankbot import OTHER_CUSTOMERS, VICTIM, BankLLM
from gateway.detectors import detect

# (without gateway, with gateway). True = leaked. None = not applicable.
EXPECTED = {
    "paste": (True, False),
    "otp": (True, False),
    "obfuscated": (True, False),
    "replay": (None, False),
    "injection": (True, False),  # closed in step 5: raw records in a reply are redacted
    "exfil": (True, False),      # closed in step 5: external images removed, no restore inside URLs
}


def test_every_scenario_has_an_expectation():
    assert {s.key for s in SCENARIOS} == set(EXPECTED)


def test_results_match_expectations():
    for s, direct, gw in evaluate():
        assert (direct, gw) == EXPECTED[s.key], s.key


def test_no_attack_leaks_through_the_gateway_any_more():
    assert not any(gw for _, gw in EXPECTED.values())
    assert not any(s.closes_in for s in SCENARIOS)


def test_synthetic_data_is_detectable_as_the_real_thing():
    # so step 5 can find leaked records by detector alone
    for c in OTHER_CUSTOMERS + [VICTIM]:
        types = {e.type for e in detect(f"aadhaar {c['aadhaar']} pan {c['pan']}")}
        assert {"AADHAAR", "PAN"} <= types


def test_leaks_helper_ignores_formatting_and_scripts():
    assert leaks("id 2384 1295 6108 ok", ["238412956108"])
    assert leaks("२३८४ १२९५ ६१०८", ["238412956108"])
    assert leaks("abcpk1234d", ["ABCPK1234D"])
    assert not leaks("nothing here", ["238412956108"])


def test_bot_behaves_normally_without_attack():
    reply = BankLLM()([{"role": "user", "content": "how do I check my loan status?"}])
    assert reply.startswith("Thanks for contacting")
    assert not leaks(reply, [c["aadhaar"] for c in OTHER_CUSTOMERS])


def test_harness_blocked_request_never_reaches_model():
    h = Harness(True)
    reply = h.ask("my otp is 482913")
    assert reply.startswith("[HTTP 400]") and h.bot.calls == []
