import random

import pytest

from gateway.detectors import detect
from gateway.policy import Action, Policy
from gateway.protect import partial_mask, protect, restore
from gateway.vault import TokenVault
from tests import synth


@pytest.fixture
def rng():
    return random.Random(20261106)


@pytest.fixture
def vault():
    return TokenVault()


def test_clean_text_passes_through_unchanged(vault):
    r = protect("Why was my loan rejected?", "s", vault)
    assert not r.blocked and r.text == "Why was my loan rejected?" and r.applied == ()
    assert len(vault) == 0


def test_aadhaar_is_tokenized_and_restored(rng, vault):
    a = synth.group(synth.aadhaar(rng))
    msg = f"mera aadhaar {a} hai"
    r = protect(msg, "s", vault)
    assert r.text == "mera aadhaar <AADHAAR_1> hai"
    assert a not in r.text
    assert restore(f"Verified <AADHAAR_1>.", "s", vault) == f"Verified {a}."


def test_same_aadhaar_twice_gets_one_token(rng, vault):
    a = synth.aadhaar(rng)
    r = protect(f"aadhaar {a}, again aadhaar {synth.group(a)}", "s", vault)
    assert r.text == "aadhaar <AADHAAR_1>, again aadhaar <AADHAAR_1>"


def test_multiple_entities_keep_surrounding_text_exactly(rng, vault):
    p, u = synth.pan(rng), synth.upi(rng)
    r = protect(f"A: pan {p}; B: upi {u}!", "s", vault)
    assert r.text == "A: pan <PAN_1>; B: upi <UPI_1>!"


def test_mobile_is_partially_masked_not_reversible(rng, vault):
    m = synth.mobile(rng)
    r = protect(f"call +91 {m[:5]} {m[5:]} now", "s", vault)
    assert m[:6] not in r.text.replace(" ", "")
    assert r.text.endswith(f"{m[-4:]} now")
    assert "X" in r.text
    assert len(vault) == 0  # nothing stored for PARTIAL


def test_otp_blocks_and_stores_nothing(vault):
    r = protect("pan ABCPK1234D and my otp is 482913", "s", vault)
    assert r.blocked and r.text is None and r.blocked_types == ("OTP",)
    assert "otp" in r.user_message
    assert "482913" not in r.user_message
    assert len(vault) == 0  # the PAN in the same message was NOT stored


def test_card_blocks(rng, vault):
    r = protect(f"card {synth.card(rng)}", "s", vault)
    assert r.blocked and r.blocked_types == ("CARD",)


def test_block_combo_aadhaar_plus_bank_account(rng, vault):
    a, acct = synth.aadhaar(rng), synth.bank_account(rng)
    r = protect(f"aadhaar {a} and a/c no {acct}", "s", vault)
    assert r.blocked and r.blocked_types == ("AADHAAR", "BANK_ACCOUNT")
    assert len(vault) == 0
    # each alone is fine
    assert not protect(f"aadhaar {a}", "s", vault).blocked
    assert not protect(f"a/c no {acct}", "s", vault).blocked


def test_low_confidence_entity_below_threshold_is_left_alone(rng, vault):
    bad = next(n for n in (synth.random_12_digit(rng) for _ in range(1000))
               if not __import__("gateway.verhoeff", fromlist=["x"]).is_valid(n))
    msg = f"aadhaar {bad}"
    assert detect(msg)[0].confidence == 0.5
    strict = Policy.default().with_rule("AADHAAR", Action.TOKENIZE, min_confidence=0.9)
    r = protect(msg, "s", vault, strict)
    assert r.text == msg and r.applied[0].action is Action.ALLOW
    assert protect(msg, "s", vault).text == "aadhaar <AADHAAR_1>"  # default threshold protects it


def test_redact_action(rng, vault):
    pol = Policy.default().with_rule("PAN", Action.REDACT)
    r = protect(f"pan {synth.pan(rng)}", "s", vault, pol)
    assert r.text == "pan [PAN]" and len(vault) == 0


def test_allow_action_keeps_value_but_is_audited(rng, vault):
    i = synth.ifsc(rng)
    r = protect(f"IFSC {i}", "s", vault)
    assert r.text == f"IFSC {i}"
    assert r.summary() == {"IFSC": {"ALLOW": 1}}


def test_summary_and_result_contain_no_raw_values(rng, vault):
    a, p = synth.aadhaar(rng), synth.pan(rng)
    r = protect(f"aadhaar {a} pan {p}", "s", vault)
    dump = repr(r) + repr(r.summary())
    assert a not in dump and p not in dump
    assert r.summary() == {"AADHAAR": {"TOKENIZE": 1}, "PAN": {"TOKENIZE": 1}}


def test_sessions_are_isolated_end_to_end(rng, vault):
    a = synth.aadhaar(rng)
    protect(f"aadhaar {a}", "alice", vault)
    assert restore("<AADHAAR_1>", "bob", vault) == "<AADHAAR_1>"


def test_llm_reply_that_repeats_a_token_gets_real_value_back(rng, vault):
    p = synth.pan(rng)
    r = protect(f"my pan is {p}", "s", vault)
    reply = f"Thanks. {r.text.split()[-1]} looks valid; <PAN_1> again."
    assert restore(reply, "s", vault) == f"Thanks. {p} looks valid; {p} again."


# ---------------------------------------------------------------- partial_mask
def test_partial_mask_keeps_last_four_and_separators():
    assert partial_mask("2384 1295 6108") == "XXXX XXXX 6108"
    assert partial_mask("+91 98765 43210") == "+XX XXXXX X3210"
    assert partial_mask("98765-43210") == "XXXXX-X3210"


def test_partial_mask_short_values_fully_masked():
    assert partial_mask("12345") == "XXXXX"


def test_known_limit_luhn_valid_account_number_is_treated_as_a_card(vault):
    # ~10% of random 13-19 digit account numbers pass Luhn and look like cards.
    # Failing safe (blocking) is the intended behaviour; this test documents it.
    r = protect("a/c no 50100123456789", "s", vault)
    assert r.blocked and r.blocked_types == ("CARD",)
