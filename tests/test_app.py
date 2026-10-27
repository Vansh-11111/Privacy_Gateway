import json
import random

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.llm import LLMError, MockLLM
from gateway.vault import TokenVault
from tests import synth


@pytest.fixture
def rng():
    return random.Random(20261106)


@pytest.fixture
def llm():
    return MockLLM()


@pytest.fixture
def client(llm):
    return TestClient(create_app(llm=llm, vault=TokenVault()))


def ask(client, content, sid="alice", role="user", **extra):
    body = {"model": "mock-1", "messages": [{"role": role, "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body, headers={"X-Session-Id": sid})


def reply_of(r):
    return r.json()["choices"][0]["message"]["content"]


def test_health(client):
    assert client.get("/healthz").json()["ok"] is True


def test_clean_message_round_trip(client):
    r = ask(client, "hello there")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert reply_of(r) == "You said: hello there"
    assert body["x_pii_gateway"]["detected"] == {}


def test_llm_never_sees_raw_value_but_user_gets_it_back(client, llm, rng):
    a = synth.group(synth.aadhaar(rng))
    p = synth.pan(rng)
    r = ask(client, f"aadhaar {a} and pan {p}")
    assert r.status_code == 200
    shown = json.dumps(llm.calls)
    assert a not in shown and p not in shown
    assert "<AADHAAR_1>" in shown and "<PAN_1>" in shown
    assert reply_of(r) == f"You said: aadhaar {a} and pan {p}"
    assert r.json()["x_pii_gateway"]["detected"] == {"AADHAAR": {"TOKENIZE": 1}, "PAN": {"TOKENIZE": 1}}


def test_response_extension_has_no_raw_values(client, rng):
    a = synth.aadhaar(rng)
    r = ask(client, f"aadhaar {a}")
    assert a not in json.dumps(r.json()["x_pii_gateway"])


def test_blocked_request_is_400_and_model_not_called(client, llm):
    r = ask(client, "my otp is 482913")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "pii_blocked"
    assert "482913" not in r.text
    assert llm.calls == []


def test_conversation_history_is_protected_and_tokens_stay_stable(client, llm, rng):
    a = synth.aadhaar(rng)
    body = {"messages": [
        {"role": "user", "content": f"aadhaar {a}"},
        {"role": "assistant", "content": f"Noted {a}."},
        {"role": "user", "content": f"is aadhaar {synth.group(a)} valid?"},
    ]}
    r = client.post("/v1/chat/completions", json=body, headers={"X-Session-Id": "s"})
    assert r.status_code == 200
    sent = [m["content"] for m in llm.calls[0]]
    assert sent == ["aadhaar <AADHAAR_1>", "Noted <AADHAAR_1>.", "is aadhaar <AADHAAR_1> valid?"]
    assert a not in json.dumps(llm.calls)


def test_blocked_message_anywhere_in_history_blocks_request(client, llm):
    body = {"messages": [{"role": "user", "content": "cvv 123 for my card"},
                         {"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=body, headers={"X-Session-Id": "s"})
    assert r.status_code == 400 and llm.calls == []


def test_sessions_are_isolated(rng, llm):
    c = TestClient(create_app(llm=lambda msgs, model: "your id is <AADHAAR_1>", vault=TokenVault()))
    a = synth.aadhaar(rng)
    ask(c, f"aadhaar {a}", sid="alice")
    # a model that "leaks" a token must not resolve it for another user
    r = ask(c, "hello", sid="bob")
    assert reply_of(r) == "your id is <AADHAAR_1>"
    assert a not in r.text
    r = ask(c, "hello", sid="alice")
    assert reply_of(r) == f"your id is {a}"


def test_session_from_user_field_and_oneshot_default(client, rng):
    a = synth.aadhaar(rng)
    r = client.post("/v1/chat/completions", json={"user": "u-42", "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["X-Session-Id"] == "u-42"
    assert r.json()["x_pii_gateway"]["session_supplied"] is True
    r1 = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    r2 = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r1.headers["X-Session-Id"] != r2.headers["X-Session-Id"]
    assert r1.json()["x_pii_gateway"]["session_supplied"] is False


def test_content_parts_are_protected(client, llm, rng):
    a = synth.aadhaar(rng)
    parts = [{"type": "text", "text": f"aadhaar {a}"}, {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]
    r = ask(client, parts)
    assert r.status_code == 200
    assert a not in json.dumps(llm.calls)
    assert llm.calls[0][0]["content"][0]["text"] == "aadhaar <AADHAAR_1>"


def test_stream_is_refused_cleanly(client):
    r = ask(client, "hi", stream=True)
    assert r.status_code == 400 and r.json()["error"]["code"] == "stream_unsupported"


def test_bad_requests_rejected(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert client.post("/v1/chat/completions", json={"messages": [{"role": "boss", "content": "x"}]}).status_code == 422
    assert client.post("/v1/chat/completions", json={}).status_code == 422


def test_oversized_conversation_rejected(client, llm):
    r = ask(client, "x" * 20_001)
    assert r.status_code == 413 and llm.calls == []


def test_upstream_failure_does_not_leak_details(rng):
    a = synth.aadhaar(rng)

    def boom(messages, model):
        raise RuntimeError(f"secret detail {a}")

    c = TestClient(create_app(llm=boom, vault=TokenVault()), raise_server_exceptions=False)
    r = ask(c, f"aadhaar {a}")
    assert r.status_code == 502 and a not in r.text and "secret" not in r.text

    def known(messages, model):
        raise LLMError("down")

    r = ask(TestClient(create_app(llm=known, vault=TokenVault())), "hi")
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"


def test_logs_contain_no_raw_values(client, rng, caplog):
    a = synth.aadhaar(rng)
    with caplog.at_level("INFO", logger="pii_gateway"):
        ask(client, f"aadhaar {a}", sid="alice-session")
        ask(client, "my otp is 482913", sid="alice-session")
    text = caplog.text
    assert a not in text and "482913" not in text and "alice-session" not in text
