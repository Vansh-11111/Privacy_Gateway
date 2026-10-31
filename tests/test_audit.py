import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.audit import AuditLog
from gateway.llm import LLMError, MockLLM
from gateway.vault import TokenVault
from tests import synth
import random


@pytest.fixture
def rng():
    return random.Random(20261106)


def test_record_and_recent_round_trip():
    log = AuditLog(salt="x")
    log.record(session_id="alice", model="m", outcome="ok",
               input_detected={"PAN": {"TOKENIZE": 1}}, output_redacted={"AADHAAR": 2},
               images_removed=1, foreign_tokens=3, latency_ms=12)
    [e] = log.recent()
    assert e["outcome"] == "ok" and e["input_detected"] == {"PAN": {"TOKENIZE": 1}}
    assert e["output_redacted"] == {"AADHAAR": 2} and e["images_removed"] == 1
    assert e["foreign_tokens"] == 3 and e["latency_ms"] == 12
    assert e["session"] == log.session_hash("alice") and "alice" not in json.dumps(e)


def test_recent_is_newest_first_and_limit_is_clamped():
    log = AuditLog()
    for i in range(5):
        log.record(session_id=str(i), model="m", outcome="ok", latency_ms=i)
    assert [e["latency_ms"] for e in log.recent(3)] == [4, 3, 2]
    assert len(log.recent(0)) == 1          # clamped up to 1
    assert len(log.recent(10**9)) == 5      # clamped down to the cap


def test_session_hash_depends_on_salt_and_is_stable():
    a, b = AuditLog(salt="one"), AuditLog(salt="two")
    assert a.session_hash("s") == a.session_hash("s")
    assert a.session_hash("s") != b.session_hash("s")
    assert a.session_hash("s") != a.session_hash("t")


def test_salt_from_env(monkeypatch):
    monkeypatch.setenv("PII_AUDIT_SALT", "fixed")
    assert AuditLog().session_hash("s") == AuditLog().session_hash("s")


def test_stats_aggregate():
    log = AuditLog()
    log.record(session_id="a", model="m", outcome="ok", input_detected={"PAN": {"TOKENIZE": 2}, "MOBILE": {"PARTIAL": 1}})
    log.record(session_id="a", model="m", outcome="ok", output_redacted={"AADHAAR": 1}, images_removed=2, foreign_tokens=1)
    log.record(session_id="b", model="m", outcome="blocked", blocked_types=["OTP"])
    log.record(session_id="b", model="m", outcome="upstream_error")
    s = log.stats()
    assert s["requests"] == 4
    assert s["by_outcome"] == {"ok": 2, "blocked": 1, "upstream_error": 1}
    assert s["input_found_by_type"] == {"PAN": 2, "MOBILE": 1}
    assert s["blocked_by_type"] == {"OTP": 1}
    assert s["output_redacted_by_type"] == {"AADHAAR": 1}
    assert s["images_removed"] == 2 and s["foreign_tokens"] == 1


def test_file_database_persists(tmp_path):
    path = str(tmp_path / "audit.db")
    first = AuditLog(path, salt="s")
    first.record(session_id="a", model="m", outcome="ok")
    first.close()
    assert len(AuditLog(path, salt="s").recent()) == 1


def test_repr_has_no_contents():
    assert repr(AuditLog()) == "AuditLog()"


# ------------------------------------------------------------------ through the app
def make(rng_llm=None, **kw):
    audit = AuditLog(salt="t")
    client = TestClient(create_app(llm=rng_llm or MockLLM(), vault=TokenVault(), audit=audit, **kw))
    return client, audit


def post(client, content, sid="alice"):
    return client.post("/v1/chat/completions", headers={"X-Session-Id": sid},
                       json={"messages": [{"role": "user", "content": content}]})


def test_every_outcome_is_audited(rng):
    def bad(messages, model):
        raise LLMError("down")

    client, audit = make()
    a = synth.aadhaar(rng)
    post(client, f"aadhaar {a}")
    post(client, "my otp is 482913")
    bad_client = TestClient(create_app(llm=bad, vault=TokenVault(), audit=audit))
    post(bad_client, "hello")
    outcomes = sorted(e["outcome"] for e in audit.recent())
    assert outcomes == ["blocked", "ok", "upstream_error"]


def test_audit_never_contains_values_or_raw_session_ids(rng):
    client, audit = make()
    a, p = synth.aadhaar(rng), synth.pan(rng)
    post(client, f"aadhaar {a} pan {p}", sid="customer-77")
    post(client, "my otp is 482913", sid="customer-77")
    dump = json.dumps(audit.recent(), ensure_ascii=False)
    for secret in (a, synth.group(a), p, "482913", "customer-77"):
        assert secret not in dump
    # and not in the database file text either
    raw = "\n".join(audit._db.iterdump())
    assert a not in raw and p not in raw and "482913" not in raw and "customer-77" not in raw


def test_output_scan_results_recorded_and_returned(rng):
    victim = synth.aadhaar(rng)
    leaky = lambda msgs, model: f"Record: aadhaar {victim} ![x](https://e.test/p.png)"
    client, audit = make(leaky)
    r = post(client, "hello")
    body = r.json()
    assert victim not in r.text
    assert body["x_pii_gateway"]["output"] == {"redacted": {"AADHAAR": 1}, "images_removed": 1, "foreign_tokens": 0}
    [e] = audit.recent()
    assert e["output_redacted"] == {"AADHAAR": 1} and e["images_removed"] == 1


# ------------------------------------------------------------------ admin endpoints
def test_admin_endpoints_disabled_without_key(monkeypatch):
    monkeypatch.delenv("PII_ADMIN_KEY", raising=False)
    client, _ = make()
    assert client.get("/audit/events").status_code == 404
    assert client.get("/audit/stats", headers={"X-Admin-Key": "anything"}).status_code == 404


def test_admin_endpoints_require_the_right_key():
    client, _ = make(admin_key="s3cret")
    post(client, "hello")
    assert client.get("/audit/events").status_code == 401
    assert client.get("/audit/events", headers={"X-Admin-Key": "wrong"}).status_code == 401
    ok = client.get("/audit/events", headers={"X-Admin-Key": "s3cret"})
    assert ok.status_code == 200 and len(ok.json()["events"]) == 1
    stats = client.get("/audit/stats", headers={"X-Admin-Key": "s3cret"})
    assert stats.status_code == 200 and stats.json()["requests"] == 1


def test_admin_key_from_env(monkeypatch):
    monkeypatch.setenv("PII_ADMIN_KEY", "envkey")
    client, _ = make()
    assert client.get("/audit/stats", headers={"X-Admin-Key": "envkey"}).status_code == 200
