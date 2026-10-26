import random

import pytest

from gateway.detectors import Entity, detect, gstin_check_char, luhn_valid, normalize
from gateway import verhoeff
from tests import synth


def types(text, **kw):
    return [e.type for e in detect(text, **kw)]


@pytest.fixture
def rng():
    return random.Random(20261106)


# ---------------------------------------------------------------- normalisation
def test_normalize_preserves_length():
    for s in ["abc", "१२३४ 5678", "2384​1295", "ＡＢＣ１２３", "आधार: ०१२३"]:
        assert len(normalize(s)) == len(s)


def test_normalize_maps_indic_digits():
    assert normalize("०१२३४५६७८९") == "0123456789"
    assert normalize("١٢٣") == "123"


# ---------------------------------------------------------------- Aadhaar
def test_aadhaar_formats(rng):
    a = synth.aadhaar(rng)
    forms = {
        "plain": a,
        "grouped": synth.group(a),
        "hyphen": synth.group(a, sep="-"),
        "devanagari": synth.to_devanagari(synth.group(a)),
        "single_spaced": " ".join(a),
    }
    for name, form in forms.items():
        text = f"my aadhaar number is {form} please check"
        hits = detect(text)
        assert [e.type for e in hits] == ["AADHAAR"], name
        assert hits[0].slice(text) == form, name  # offsets index the ORIGINAL text


def test_aadhaar_hinglish_and_devanagari_context(rng):
    a = synth.group(synth.aadhaar(rng))
    assert types(f"mera aadhaar {a} hai") == ["AADHAAR"]
    assert types(f"मेरा आधार नंबर {synth.to_devanagari(a)} है") == ["AADHAAR"]


def test_aadhaar_zero_width_evasion(rng):
    a = synth.aadhaar(rng)
    sneaky = "​".join([a[:4], a[4:8], a[8:]])
    assert types(f"aadhaar {sneaky}") == ["AADHAAR"]


def test_aadhaar_never_starts_with_0_or_1(rng):
    body = "1" + synth.random_digits(rng, 10)
    n = body + verhoeff.check_digit(body)
    assert "AADHAAR" not in types(f"aadhaar {n}")


def test_random_12_digits_without_context_mostly_ignored(rng):
    n = 2000
    flagged = sum("AADHAAR" in types(f"value {synth.random_12_digit(rng)} end") for _ in range(n))
    rate = flagged / n
    print(f"false-positive rate on random 12-digit numbers, no context: {rate:.3f}")
    assert rate < 0.15  # Verhoeff passes ~10% of random numbers; this is the honest floor


def test_order_context_suppresses_aadhaar_shaped_numbers(rng):
    for _ in range(50):
        a = synth.aadhaar(rng)
        assert "AADHAAR" not in types(f"order id {a} has shipped")


def test_keyword_with_bad_checksum_is_low_confidence(rng):
    bad = next((n for n in (synth.random_12_digit(rng) for _ in range(1000)) if not verhoeff.is_valid(n)), None)
    assert bad is not None, "Verhoeff accepted 1000 random numbers; checksum is broken"
    hits = detect(f"aadhaar {bad}")
    assert [e.type for e in hits] == ["AADHAAR"]
    assert hits[0].confidence == 0.5
    assert hits[0].meta["checksum_ok"] is False
    assert detect(f"aadhaar {bad}", min_confidence=0.6) == []


def test_card_number_is_not_mistaken_for_aadhaar(rng):
    for _ in range(100):
        c = synth.card(rng)
        assert types(f"pay with {synth.group(c)} now") == ["CARD"]


def test_pieces_of_a_longer_number_are_not_flagged(rng):
    # 16 digits that are NOT a card (fail Luhn): no 12-digit or 10-digit slice
    # of the run may be reported as an Aadhaar or a mobile number.
    n, checked = 0, 0
    while checked < 400 and n < 20000:
        n += 1
        s = rng.choice("23456789") + synth.random_digits(rng, 15)
        if luhn_valid(s):
            continue
        checked += 1
        assert types(f"reference {synth.group(s)} noted") == [], s
        assert types(f"reference {synth.group(s, 5, '-')} noted") == [], s


# ---------------------------------------------------------------- PAN
def test_pan(rng):
    p = synth.pan(rng)
    text = f"my pan is {p}."
    hits = detect(text)
    assert [e.type for e in hits] == ["PAN"]
    assert hits[0].slice(text) == p
    assert hits[0].confidence > 0.9


def test_pan_lowercase_is_lower_confidence(rng):
    p = synth.pan(rng).lower()
    hits = detect(f"pan: {p}")
    assert [e.type for e in hits] == ["PAN"]
    assert hits[0].confidence < 0.9


def test_pan_with_invalid_holder_type_rejected():
    assert "PAN" not in types("code ABCXK1234D here")  # 4th letter X is not a holder type


def test_pan_inside_longer_token_rejected(rng):
    assert "PAN" not in types(f"ref X{synth.pan(rng)}9")


# ---------------------------------------------------------------- mobile
def test_mobile_formats(rng):
    m = synth.mobile(rng)
    for form in [m, f"+91{m}", f"+91 {m[:5]} {m[5:]}", f"0{m}", f"{m[:5]}-{m[5:]}", f"91{m}"]:
        assert types(f"call me on {form} tonight") == ["MOBILE"], form


def test_prefixed_mobile_is_not_also_an_aadhaar(rng):
    for _ in range(300):
        assert types(f"phone +91 {synth.mobile(rng)}") == ["MOBILE"]


def test_bare_mobile_without_context_scores_lower_than_prefixed(rng):
    m = synth.mobile(rng)
    bare = detect(f"x {m} y")[0].confidence
    prefixed = detect(f"x +91{m} y")[0].confidence
    assert bare < prefixed


def test_order_number_is_not_a_mobile(rng):
    for _ in range(50):
        assert "MOBILE" not in types(f"order {synth.mobile(rng)} confirmed")


# ---------------------------------------------------------------- UPI / email
def test_upi_and_email_are_distinguished(rng):
    u = synth.upi(rng)
    assert types(f"pay {u} now") == ["UPI"]
    assert types("write to rahul.k@gmail.com please") == ["EMAIL"]


def test_vpa_made_of_a_mobile_number_is_one_upi_entity(rng):
    vpa = f"{synth.mobile(rng)}@ybl"
    hits = detect(f"upi id {vpa}")
    assert [e.type for e in hits] == ["UPI"]
    assert hits[0].slice(f"upi id {vpa}") == vpa


def test_upi_followed_by_full_stop(rng):
    u = synth.upi(rng)
    assert types(f"send it to {u}.") == ["UPI"]


# ---------------------------------------------------------------- IFSC / GSTIN
def test_ifsc(rng):
    for _ in range(100):
        i = synth.ifsc(rng)
        text = f"branch IFSC {i}"
        hits = detect(text)
        assert [e.type for e in hits] == ["IFSC"]
        assert hits[0].slice(text) == i


def test_gstin_and_embedded_pan_is_one_entity(rng):
    for _ in range(100):
        g = synth.gstin(rng)
        text = f"GSTIN {g}"
        hits = detect(text)
        assert [e.type for e in hits] == ["GSTIN"]
        assert hits[0].slice(text) == g


def test_gstin_known_public_example():
    assert gstin_check_char("27AAPFU0939F1Z") == "V"


def test_gstin_bad_checksum_needs_keyword(rng):
    g = synth.gstin(rng)
    bad = g[:-1] + ("0" if g[-1] != "0" else "1")
    assert "GSTIN" not in types(f"id {bad}")
    assert types(f"GST number {bad}") == ["GSTIN"]


# ---------------------------------------------------------------- card / bank account
def test_luhn():
    assert luhn_valid("4111111111111111")  # standard public test card number
    assert not luhn_valid("4111111111111112")


def test_bank_account_requires_keyword(rng):
    acct = synth.bank_account(rng)
    assert types(f"a/c no {acct}") == ["BANK_ACCOUNT"]
    assert types(f"मेरा खाता नंबर {acct}") == ["BANK_ACCOUNT"]
    assert "BANK_ACCOUNT" not in types(f"number {acct}")


# ---------------------------------------------------------------- OTP / CVV
def test_otp_covers_only_the_digits(rng):
    code = synth.random_digits(rng, 6)
    for text in [f"mera otp {code} hai", f"OTP is {code}", f"{code} is your OTP", f"ओटीपी {code}"]:
        hits = detect(text)
        assert [e.type for e in hits] == ["OTP"], text
        assert hits[0].slice(text) == code


def test_cvv(rng):
    code = synth.random_digits(rng, 3)
    text = f"card cvv {code}"
    hits = [e for e in detect(text) if e.type == "CVV"]
    assert len(hits) == 1 and hits[0].slice(text) == code


def test_pin_code_is_not_an_otp():
    assert types("my pin code is 110001") == []


# ---------------------------------------------------------------- mixed text
def test_clean_text_has_no_entities():
    for t in [
        "Please explain how UPI works and why my loan was rejected.",
        "Meeting at 4:30 on 25.10.2026, room 12, floor 3.",
        "The invoice total is 12,450 rupees for 3 items.",
        "",
    ]:
        assert detect(t) == []


def test_several_entities_are_ordered_and_non_overlapping(rng):
    a, p, m, u = synth.aadhaar(rng), synth.pan(rng), synth.mobile(rng), synth.upi(rng)
    text = f"aadhaar {a}, pan {p}, mobile +91{m}, upi {u}"
    hits = detect(text)
    assert [e.type for e in hits] == ["AADHAAR", "PAN", "MOBILE", "UPI"]
    for first, second in zip(hits, hits[1:]):
        assert first.end <= second.start


def test_entity_repr_never_contains_the_value(rng):
    a = synth.aadhaar(rng)
    e = detect(f"aadhaar {a}")[0]
    assert isinstance(e, Entity)
    assert a not in repr(e) and a not in str(e)


# ---------------------------------------------------------------- recall sweep
@pytest.mark.parametrize(
    "etype,make,template",
    [
        ("AADHAAR", synth.aadhaar, "my aadhaar is {}"),
        ("PAN", synth.pan, "PAN {}"),
        ("MOBILE", synth.mobile, "call me on {}"),
        ("IFSC", synth.ifsc, "IFSC {}"),
        ("UPI", synth.upi, "pay to {}"),
        ("GSTIN", synth.gstin, "GSTIN {}"),
        ("CARD", synth.card, "card {}"),
    ],
)
def test_recall_on_synthetic_samples(etype, make, template, rng):
    n = 300
    found = sum(etype in types(template.format(make(rng))) for _ in range(n))
    print(f"{etype}: recall {found}/{n}")
    assert found == n
