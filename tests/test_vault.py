import threading

import pytest
from cryptography.fernet import Fernet

from gateway.vault import TokenVault


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_round_trip():
    v = TokenVault()
    t = v.tokenize("s1", "AADHAAR", "2384 1295 6108")
    assert t == "<AADHAAR_1>"
    assert v.restore("s1", f"id {t} ok") == "id 2384 1295 6108 ok"


def test_same_value_same_token_whatever_the_formatting():
    v = TokenVault()
    a = v.tokenize("s", "AADHAAR", "2384 1295 6108")
    assert v.tokenize("s", "AADHAAR", "238412956108") == a
    assert v.tokenize("s", "AADHAAR", "2384-1295-6108") == a
    assert v.tokenize("s", "AADHAAR", "२३८४ १२९५ ६१०८") == a


def test_different_values_get_numbered_tokens_per_type():
    v = TokenVault()
    assert v.tokenize("s", "PAN", "ABCPK1234D") == "<PAN_1>"
    assert v.tokenize("s", "PAN", "ABCPK1234E") == "<PAN_2>"
    assert v.tokenize("s", "UPI", "a@ybl") == "<UPI_1>"


def test_pan_case_and_email_case_do_not_create_new_tokens():
    v = TokenVault()
    assert v.tokenize("s", "PAN", "abcpk1234d") == v.tokenize("s", "PAN", "ABCPK1234D")
    assert v.tokenize("s", "EMAIL", "Rahul@Gmail.com") == v.tokenize("s", "EMAIL", "rahul@gmail.com")


def test_mobile_with_and_without_country_code_is_one_token():
    v = TokenVault()
    assert v.tokenize("s", "MOBILE", "+91 98765 43210") == v.tokenize("s", "MOBILE", "9876543210")


def test_tokens_do_not_cross_sessions():
    v = TokenVault()
    t = v.tokenize("alice", "AADHAAR", "2384 1295 6108")
    assert v.restore("bob", f"x {t}") == f"x {t}"
    assert v.resolve("bob", t) is None
    # bob gets his own <AADHAAR_1> for a different value
    assert v.tokenize("bob", "AADHAAR", "5555 5555 5555") == "<AADHAAR_1>"
    assert v.restore("alice", "<AADHAAR_1>") == "2384 1295 6108"


def test_unknown_token_left_alone():
    v = TokenVault()
    v.tokenize("s", "PAN", "ABCPK1234D")
    assert v.restore("s", "see <PAN_9> and <PAN_1>") == "see <PAN_9> and ABCPK1234D"


def test_values_are_encrypted_at_rest():
    v = TokenVault()
    v.tokenize("s", "AADHAAR", "238412956108")
    blob = repr(v._sessions["s"].tokens) + repr(v._sessions["s"].by_digest)
    assert "238412956108" not in blob
    assert "238412956108" not in repr(v)


def test_same_key_decrypts_other_key_cannot():
    key = Fernet.generate_key()
    v1 = TokenVault(key=key)
    v1.tokenize("s", "PAN", "ABCPK1234D")
    ct = v1._sessions["s"].tokens["<PAN_1>"]
    assert Fernet(key).decrypt(ct) == b"ABCPK1234D"
    with pytest.raises(Exception):
        Fernet(Fernet.generate_key()).decrypt(ct)


def test_key_from_env(monkeypatch):
    monkeypatch.setenv("PII_VAULT_KEY", Fernet.generate_key().decode())
    assert TokenVault().ephemeral is False
    monkeypatch.delenv("PII_VAULT_KEY")
    assert TokenVault().ephemeral is True


def test_invalid_key_rejected():
    with pytest.raises(ValueError):
        TokenVault(key="not-a-key")


def test_dedupe_digest_depends_on_the_key():
    a, b = TokenVault(), TokenVault()
    assert a._digest("AADHAAR", "238412956108") != b._digest("AADHAAR", "238412956108")


def test_session_expires_after_ttl():
    clock = FakeClock()
    v = TokenVault(ttl_seconds=60, clock=clock)
    t = v.tokenize("s", "PAN", "ABCPK1234D")
    clock.now += 30
    assert v.resolve("s", t) == "ABCPK1234D"  # touching it extends the session
    clock.now += 59
    assert v.resolve("s", t) == "ABCPK1234D"
    clock.now += 61
    assert v.resolve("s", t) is None
    assert len(v) == 0


def test_expiry_is_per_session():
    clock = FakeClock()
    v = TokenVault(ttl_seconds=60, clock=clock)
    v.tokenize("old", "PAN", "ABCPK1234D")
    clock.now += 100
    v.tokenize("new", "PAN", "ABCPK1234E")
    assert len(v) == 1


def test_clear():
    v = TokenVault()
    t = v.tokenize("s", "PAN", "ABCPK1234D")
    v.clear("s")
    assert v.restore("s", t) == t


def test_bad_arguments():
    v = TokenVault()
    with pytest.raises(ValueError):
        v.tokenize("", "PAN", "x")
    with pytest.raises(ValueError):
        v.tokenize("s", "pan!", "x")


def test_thread_safety_unique_tokens():
    v = TokenVault()
    out = []

    def work(i):
        out.append(v.tokenize("s", "BANK_ACCOUNT", f"{50100000000000 + i}"))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(out)) == 50
