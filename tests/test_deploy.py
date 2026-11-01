"""Step 7: real model adapter, settings checks, client keys, rate limits, public demo, entry point."""
import json

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.audit import AuditLog
from gateway.config import ConfigError, check, load_settings, new_secrets
from gateway.demo_app import create_demo_app
from gateway.llm import LLMError, MockLLM
from gateway.llm_openai import OpenAICompatLLM
from gateway.ratelimit import RateLimiter
from gateway.server import build_app
import random

from tests.synth import aadhaar as _aadhaar, group


def aadhaar() -> str:
    return _aadhaar(random.Random(7))

KEY = "pk_" + "a" * 24


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retries wait briefly in real life; tests must not."""
    import gateway.llm_openai as mod
    monkeypatch.setattr(mod, "_sleep", lambda s: None)


# ------------------------------------------------------------------ RateLimiter
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_ratelimiter_allows_up_to_limit_then_blocks():
    rl = RateLimiter(3, window=60, clock=Clock())
    assert [rl.allow("a") for _ in range(4)] == [True, True, True, False]


def test_ratelimiter_keys_are_independent():
    rl = RateLimiter(1, clock=Clock())
    assert rl.allow("a") and rl.allow("b") and not rl.allow("a")


def test_ratelimiter_window_slides():
    c = Clock()
    rl = RateLimiter(2, window=60, clock=c)
    rl.allow("a"); rl.allow("a")
    assert not rl.allow("a")
    c.t += 61
    assert rl.allow("a")


def test_ratelimiter_over_does_not_record():
    rl = RateLimiter(1, clock=Clock())
    assert not rl.over("a") and not rl.over("a")
    rl.hit("a")
    assert rl.over("a")


def test_ratelimiter_retry_after_is_positive_when_blocked_and_zero_otherwise():
    c = Clock()
    rl = RateLimiter(1, window=60, clock=c)
    assert rl.retry_after("a") == 0
    rl.allow("a")
    c.t += 20
    assert 39 <= rl.retry_after("a") <= 42


def test_ratelimiter_memory_is_bounded():
    c = Clock()
    rl = RateLimiter(5, window=60, clock=c, max_keys=100)
    for i in range(1000):
        rl.allow(f"ip{i}")
        c.t += 0.001
    assert len(rl._hits) <= 100


def test_ratelimiter_rejects_zero_limit():
    with pytest.raises(ValueError):
        RateLimiter(0)


# ------------------------------------------------------------------ OpenAICompatLLM
def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _ok(text="hello"):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})


def test_llm_sends_openai_shaped_request_with_bearer_key():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return _ok("hi there")

    llm = OpenAICompatLLM("https://api.example.com/v1/", "sk-test-123456789", client=_client(handler))
    assert llm([{"role": "user", "content": "yo"}], "gpt-x") == "hi there"
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-123456789"
    assert seen["body"] == {"model": "gpt-x", "messages": [{"role": "user", "content": "yo"}]}


def test_llm_configured_model_overrides_callers_choice():
    seen = {}

    def handler(req):
        seen["model"] = json.loads(req.content)["model"]
        return _ok()

    OpenAICompatLLM("https://api.example.com/v1", model="cheap-model", client=_client(handler))([], "expensive-model")
    assert seen["model"] == "cheap-model"


def test_llm_no_key_means_no_auth_header():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        return _ok()

    OpenAICompatLLM("http://localhost:11434/v1", client=_client(handler))([], "m")
    assert seen["auth"] is None


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_llm_non_200_becomes_llmerror_without_body(status):
    secret_body = f"echoing the prompt: {aadhaar()}"

    def handler(req):
        return httpx.Response(status, text=secret_body)

    with pytest.raises(LLMError) as e:
        OpenAICompatLLM("https://api.example.com/v1", "k", client=_client(handler))([], "m")
    assert secret_body not in str(e.value) and str(status) in str(e.value)


def test_llm_network_failure_becomes_llmerror():
    def handler(req):
        raise httpx.ConnectError("boom sk-live-secret")

    with pytest.raises(LLMError) as e:
        OpenAICompatLLM("https://api.example.com/v1", "sk-live-secret", client=_client(handler))([], "m")
    assert "sk-live-secret" not in str(e.value) and e.value.__cause__ is None


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {"content": None}}]}, [1, 2]])
def test_llm_malformed_responses_are_errors(payload):
    def handler(req):
        return httpx.Response(200, json=payload)

    with pytest.raises(LLMError):
        OpenAICompatLLM("https://api.example.com/v1", client=_client(handler))([], "m")


def test_llm_non_json_response_is_error():
    with pytest.raises(LLMError):
        OpenAICompatLLM("https://api.example.com/v1", client=_client(lambda r: httpx.Response(200, text="<html>")))([], "m")


def test_llm_list_content_parts_are_joined():
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]})

    assert OpenAICompatLLM("https://api.example.com/v1", client=_client(handler))([], "m") == "a b"


@pytest.mark.parametrize("url", ["", "api.example.com/v1", "ftp://x.com/v1", "http://api.example.com/v1", "https://"])
def test_llm_rejects_bad_or_insecure_urls(url):
    with pytest.raises(ValueError):
        OpenAICompatLLM(url)


@pytest.mark.parametrize("url", ["http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "https://api.openai.com/v1"])
def test_llm_accepts_local_http_and_any_https(url):
    OpenAICompatLLM(url)


def test_llm_repr_hides_key():
    assert "sk-very-secret" not in repr(OpenAICompatLLM("https://api.example.com/v1", "sk-very-secret"))


def test_llm_does_not_follow_redirects():
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(302, headers={"location": "https://evil.example/steal"})

    llm = OpenAICompatLLM("https://api.example.com/v1", "k", client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False))
    with pytest.raises(LLMError):
        llm([], "m")
    assert calls == ["https://api.example.com/v1/chat/completions"]


def test_default_client_is_built_without_redirects():
    assert OpenAICompatLLM("https://api.example.com/v1")._client.follow_redirects is False


# ------------------------------------------------------------------ settings
def _prod_env(**over):
    env = {"PII_MODE": "production", **new_secrets(), "LLM_BASE_URL": "https://api.example.com/v1", "LLM_API_KEY": "sk-x",
           "PII_AUDIT_DB": "x.db", "PII_TRUST_PROXY": "1"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_new_secrets_pass_the_production_check():
    problems, warnings = check(load_settings(_prod_env()))
    assert problems == [] and warnings == []


def test_default_mode_is_demo_and_needs_no_secrets():
    s = load_settings({})
    assert s.mode == "demo" and check(s)[0] == [] and s.audit_db == ":memory:"


def test_unknown_mode_is_rejected():
    with pytest.raises(ConfigError):
        load_settings({"PII_MODE": "prod"})


@pytest.mark.parametrize("name,value,fragment", [
    ("PII_ADMIN_KEY", "demo", "PII_ADMIN_KEY"),
    ("PII_ADMIN_KEY", "short", "PII_ADMIN_KEY"),
    ("PII_ADMIN_KEY", None, "PII_ADMIN_KEY"),
    ("PII_VAULT_KEY", None, "PII_VAULT_KEY"),
    ("PII_VAULT_KEY", "not-a-fernet-key", "PII_VAULT_KEY"),
    ("PII_AUDIT_SALT", None, "PII_AUDIT_SALT"),
    ("PII_API_KEYS", None, "PII_API_KEYS"),
    ("PII_API_KEYS", "tooshort", "PII_API_KEYS"),
    ("LLM_BASE_URL", None, "LLM_BASE_URL"),
])
def test_production_refuses_unsafe_settings(name, value, fragment):
    problems, _ = check(load_settings(_prod_env(**{name: value})))
    assert any(fragment in p for p in problems)


def test_problems_never_contain_secret_values():
    env = _prod_env(PII_ADMIN_KEY="demo", PII_VAULT_KEY="bad-vault-secret-value", PII_API_KEYS="short-key-secret")
    text = " ".join(check(load_settings(env))[0])
    assert "bad-vault-secret-value" not in text and "short-key-secret" not in text


def test_warnings_for_memory_audit_and_missing_proxy_flag():
    _, w = check(load_settings(_prod_env(PII_AUDIT_DB=":memory:", PII_TRUST_PROXY=None)))
    assert any("PII_AUDIT_DB" in x for x in w) and any("PII_TRUST_PROXY" in x for x in w)


def test_bad_number_is_a_config_error():
    with pytest.raises(ConfigError):
        load_settings({"PII_RATE_LIMIT_PER_MIN": "lots"})


def test_api_keys_are_split_and_trimmed():
    assert load_settings({"PII_API_KEYS": " a1 , b2,,c3 "}).api_keys == ("a1", "b2", "c3")


def test_settings_repr_has_no_secrets():
    s = load_settings(_prod_env(PII_ADMIN_KEY="super-secret-admin-key-123"))
    assert "super-secret" not in repr(s) and "sk-x" not in repr(s)


def test_generated_vault_key_is_a_real_fernet_key():
    Fernet(new_secrets()["PII_VAULT_KEY"].encode())


def test_secrets_differ_each_time():
    assert new_secrets() != new_secrets()


# ------------------------------------------------------------------ client keys and limits on the gateway
def _gw(**kw):
    return TestClient(create_app(llm=MockLLM(), audit=AuditLog(salt="t"), **kw))


def _msg(text="hello"):
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


def test_open_gateway_when_no_keys_configured():
    assert _gw().post("/v1/chat/completions", json=_msg()).status_code == 200


def test_missing_key_is_401_with_gateway_error_shape():
    r = _gw(api_keys=[KEY]).post("/v1/chat/completions", json=_msg())
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert r.headers["www-authenticate"] == "Bearer"


def test_wrong_key_is_401():
    r = _gw(api_keys=[KEY]).post("/v1/chat/completions", json=_msg(), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_right_key_works_and_scheme_is_case_insensitive():
    c = _gw(api_keys=[KEY])
    assert c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": f"bearer {KEY}"}).status_code == 200


def test_any_of_several_keys_works():
    c = _gw(api_keys=[KEY, "pk_" + "b" * 24])
    assert c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": "Bearer pk_" + "b" * 24}).status_code == 200


def test_key_check_happens_before_body_validation():
    r = _gw(api_keys=[KEY]).post("/v1/chat/completions", json={"nonsense": True})
    assert r.status_code == 401


def test_raw_key_prefix_without_bearer_is_rejected():
    assert _gw(api_keys=[KEY]).post("/v1/chat/completions", json=_msg(), headers={"Authorization": KEY}).status_code == 401


def test_blocked_pii_is_still_checked_after_auth():
    r = _gw(api_keys=[KEY]).post("/v1/chat/completions", json=_msg("my otp is 482913"), headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "pii_blocked"


def test_failed_attempts_lock_out_the_address_even_for_the_right_key():
    c = _gw(api_keys=[KEY])
    codes = [c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": "Bearer guess"}).status_code for _ in range(25)]
    assert codes[0] == 401 and codes[-1] == 429
    r = c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_good_requests_do_not_count_as_failures():
    c = _gw(api_keys=[KEY])
    for _ in range(30):
        assert c.post("/v1/chat/completions", json=_msg(), headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_rate_limit_per_key():
    c = _gw(api_keys=[KEY], rate_limit_per_min=3)
    h = {"Authorization": f"Bearer {KEY}"}
    codes = [c.post("/v1/chat/completions", json=_msg(), headers=h).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = c.post("/v1/chat/completions", json=_msg(), headers=h)
    assert r.json()["error"]["code"] == "rate_limited" and "retry-after" in r.headers


def test_rate_limit_per_address_when_open():
    c = _gw(rate_limit_per_min=2)
    assert [c.post("/v1/chat/completions", json=_msg()).status_code for _ in range(3)] == [200, 200, 429]


def test_proxy_header_is_used_only_when_trusted_and_uses_the_last_entry():
    def burst(c, first_hop):
        return [c.post("/v1/chat/completions", json=_msg(), headers={"X-Forwarded-For": f"{first_hop}, 203.0.113.9"}).status_code
                for _ in range(3)]

    trusted = _gw(rate_limit_per_min=2, trust_proxy=True)
    assert burst(trusted, "1.1.1.1") == [200, 200, 429]
    # a forged left-hand entry must not buy a fresh allowance
    assert trusted.post("/v1/chat/completions", json=_msg(), headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9"}).status_code == 429
    # untrusted: the header is ignored and everyone is the socket address
    untrusted = _gw(rate_limit_per_min=2, trust_proxy=False)
    codes = [untrusted.post("/v1/chat/completions", json=_msg(), headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]


def test_oversized_body_is_refused_before_parsing():
    big = "x" * 1_100_000
    r = _gw().post("/v1/chat/completions", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_oversized_body_is_refused_without_reading_it():
    # claims 2 MB but sends 5 bytes: refused on the header alone, so nothing was buffered
    r = _gw().post("/v1/chat/completions", content=b"{}", headers={"Content-Length": "2000000"})
    assert r.status_code in (413, 400)


def test_post_without_content_length_is_refused():
    r = _gw().post("/v1/chat/completions", content=iter([b"{}"]), headers={"Content-Type": "application/json"})   # sent chunked
    assert r.status_code == 411


def test_oversized_body_is_refused_on_demo_endpoints_too():
    r = TestClient(create_demo_app(audit=AuditLog(salt="t"))).post("/demo/direct", content="x" * 1_100_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_get_requests_need_no_content_length():
    assert _gw().get("/healthz").status_code == 200


def test_healthz_reports_auth_and_llm_kind_without_secrets():
    body = _gw(api_keys=[KEY], llm_kind="upstream").get("/healthz").json()
    assert body["auth"] == "api_key" and body["llm"] == "upstream" and KEY not in json.dumps(body)
    assert _gw().get("/healthz").json()["auth"] == "open"


def test_healthz_is_not_rate_limited_or_authenticated():
    c = _gw(api_keys=[KEY], rate_limit_per_min=1)
    assert all(c.get("/healthz").status_code == 200 for _ in range(5))


# ------------------------------------------------------------------ full path with the real adapter (mocked network)
def test_end_to_end_model_sees_tokens_user_gets_real_value_back():
    seen = {}
    num = aadhaar()

    def handler(req):
        body = json.loads(req.content)
        seen["sent"] = json.dumps(body)
        last = body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": f"Noted: {last}"}}]})

    llm = OpenAICompatLLM("https://api.example.com/v1", "sk-x", client=_client(handler))
    c = TestClient(create_app(llm=llm, audit=AuditLog(salt="t"), api_keys=[KEY]))
    r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {KEY}", "X-Session-Id": "s1"},
               json=_msg(f"my aadhaar number is {group(num)}"))
    assert r.status_code == 200
    assert num not in seen["sent"].replace(" ", "") and "<AADHAAR_1>" in seen["sent"]
    assert group(num) in r.json()["choices"][0]["message"]["content"].replace(num, group(num)) or num in r.json()["choices"][0]["message"]["content"]


def test_upstream_failure_gives_502_and_no_details():
    llm = OpenAICompatLLM("https://api.example.com/v1", "sk-secret-key", client=_client(lambda r: httpx.Response(500, text="oops")))
    r = TestClient(create_app(llm=llm, audit=AuditLog(salt="t"))).post("/v1/chat/completions", json=_msg())
    assert r.status_code == 502 and "sk-secret-key" not in r.text and "oops" not in r.text


# ------------------------------------------------------------------ public demo
def _public(**kw):
    return TestClient(create_demo_app(public=True, admin_key="k" * 20, audit=AuditLog(salt="t"), **kw))


def test_public_demo_cannot_clear_the_log():
    c = _public()
    c.post("/v1/chat/completions", json=_msg())
    assert c.post("/demo/reset", json={}).status_code == 403
    assert c.get("/audit/stats", headers={"X-Admin-Key": "k" * 20}).json()["by_outcome"]["ok"] == 1


def test_private_demo_can_still_clear_the_log():
    c = TestClient(create_demo_app(audit=AuditLog(salt="t")))
    c.post("/v1/chat/completions", json=_msg())
    assert c.post("/demo/reset", json={}).status_code == 200


def test_public_flag_is_injected_into_the_page():
    pub = _public().get("/dashboard").text
    priv = TestClient(create_demo_app(audit=AuditLog(salt="t"))).get("/dashboard").text
    assert "const PUBLIC_DEMO = true;" in pub and "const PUBLIC_DEMO = false;" in priv
    assert "__PUBLIC__" not in pub + priv


def test_public_demo_rate_limits_the_no_gateway_endpoint_too():
    c = _public(rate_limit_per_min=2)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert [c.post("/demo/direct", json=body).status_code for _ in range(3)] == [200, 200, 429]


def test_demo_direct_caps_message_count():
    r = TestClient(create_demo_app(audit=AuditLog(salt="t"))).post("/demo/direct", json={"messages": [{"role": "user", "content": "x"}] * 101})
    assert r.status_code == 413


def test_public_demo_still_blocks_and_tokenises():
    c = _public()
    assert c.post("/v1/chat/completions", json=_msg("my otp is 482913")).status_code == 400
    assert c.get("/healthz").json()["llm"] == "demo-bot"


# ------------------------------------------------------------------ entry point
def test_build_app_default_is_a_public_demo_with_random_admin_key():
    c = TestClient(build_app({}))
    page = c.get("/dashboard").text
    assert "const PUBLIC_DEMO = true;" in page
    assert 'const ADMIN_KEY = "demo"' not in page
    assert c.post("/demo/reset", json={}).status_code == 403


def test_build_app_demo_audit_is_in_memory_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_app({})
    assert not (tmp_path / "audit.db").exists()


def test_two_demo_apps_get_different_admin_keys():
    a = TestClient(build_app({})).get("/dashboard").text.split("const ADMIN_KEY = ")[1].split(";")[0]
    b = TestClient(build_app({})).get("/dashboard").text.split("const ADMIN_KEY = ")[1].split(";")[0]
    assert a != b


def test_build_app_production_refuses_to_start_with_defaults():
    with pytest.raises(ConfigError) as e:
        build_app({"PII_MODE": "production"})
    assert "PII_ADMIN_KEY" in str(e.value) and "PII_VAULT_KEY" in str(e.value)


def test_build_app_production_rejects_insecure_llm_url():
    with pytest.raises(ConfigError):
        build_app(_prod_env(LLM_BASE_URL="http://api.example.com/v1"))


def test_build_app_production_requires_client_key_and_has_no_demo_routes(tmp_path):
    env = _prod_env(PII_AUDIT_DB=str(tmp_path / "a.db"))
    c = TestClient(build_app(env))
    assert c.post("/v1/chat/completions", json=_msg()).status_code == 401
    assert c.get("/demo/presets").status_code == 404 and c.get("/dashboard").status_code == 404
    assert c.post("/demo/direct", json={"messages": []}).status_code == 404
    body = c.get("/healthz").json()
    assert body == {"ok": True, "vault_key": "configured", "llm": "upstream", "auth": "api_key"}


def test_build_app_production_admin_endpoints_use_the_configured_key(tmp_path):
    env = _prod_env(PII_AUDIT_DB=str(tmp_path / "a.db"))
    c = TestClient(build_app(env))
    assert c.get("/audit/stats").status_code == 401
    assert c.get("/audit/stats", headers={"X-Admin-Key": "demo"}).status_code == 401
    assert c.get("/audit/stats", headers={"X-Admin-Key": env["PII_ADMIN_KEY"]}).status_code == 200


def test_build_app_production_vault_uses_the_configured_key(tmp_path):
    env = _prod_env(PII_AUDIT_DB=str(tmp_path / "a.db"))
    assert TestClient(build_app(env)).get("/healthz").json()["vault_key"] == "configured"


def test_upstream_failure_reason_is_logged_without_secrets_or_prompt(caplog):
    import logging
    num = aadhaar()
    llm = OpenAICompatLLM("https://api.example.com/v1", "sk-secret-key", client=_client(lambda r: httpx.Response(404, text=f"model not found {num}")))
    with caplog.at_level(logging.WARNING, logger="pii_gateway"):
        TestClient(create_app(llm=llm, audit=AuditLog(salt="t"))).post("/v1/chat/completions", json=_msg(f"aadhaar {group(num)}"))
    text = caplog.text
    assert "HTTP 404" in text and "sk-secret-key" not in text and num not in text.replace(" ", "")


# ------------------------------------------------------------------ retries and fallback models
def _scripted(statuses_by_model, seen=None):
    """Handler answering per model: a list of statuses consumed in order (last one repeats)."""
    counters = {}

    def handler(req):
        m = json.loads(req.content)["model"]
        if seen is not None:
            seen.append(m)
        script = statuses_by_model[m]
        i = counters.get(m, 0)
        counters[m] = i + 1
        status = script[min(i, len(script) - 1)]
        return _ok(f"from {m}") if status == 200 else httpx.Response(status, text="nope")

    return handler


def _fb(handler, **kw):
    return OpenAICompatLLM("https://api.example.com/v1", "k", client=_client(handler), **kw)


def test_busy_model_is_retried_once_then_succeeds():
    seen = []
    llm = _fb(_scripted({"a": [503, 200]}, seen), model="a")
    assert llm([], "x") == "from a" and seen == ["a", "a"]


def test_after_retries_the_next_model_is_used():
    seen = []
    llm = _fb(_scripted({"a": [503], "b": [200]}, seen), model="a", fallback_models=["b"])
    assert llm([], "x") == "from b" and seen == ["a", "a", "b"]


def test_retry_waits_between_attempts(monkeypatch):
    import gateway.llm_openai as mod
    waits = []
    monkeypatch.setattr(mod, "_sleep", waits.append)
    _fb(_scripted({"a": [503, 200]}), model="a", backoff=0.7)([], "x")
    assert waits == [0.7]


def test_no_wait_when_first_try_works(monkeypatch):
    import gateway.llm_openai as mod
    waits = []
    monkeypatch.setattr(mod, "_sleep", waits.append)
    _fb(_scripted({"a": [200]}), model="a")([], "x")
    assert waits == []


def test_unknown_model_404_skips_retry_and_moves_on():
    seen = []
    llm = _fb(_scripted({"a": [404], "b": [200]}, seen), model="a", fallback_models=["b"])
    assert llm([], "x") == "from b" and seen == ["a", "b"]


@pytest.mark.parametrize("status", [400, 401, 403])
def test_key_or_request_problems_fail_at_once_without_fallback(status):
    seen = []
    llm = _fb(_scripted({"a": [status], "b": [200]}, seen), model="a", fallback_models=["b"])
    with pytest.raises(LLMError) as e:
        llm([], "x")
    assert seen == ["a"] and str(status) in str(e.value)


def test_all_models_failing_reports_last_status_and_attempt_count():
    llm = _fb(_scripted({"a": [503], "b": [429]}), model="a", fallback_models=["b"])
    with pytest.raises(LLMError) as e:
        llm([], "x")
    assert "429" in str(e.value) and "4 attempts" in str(e.value)


def test_retries_zero_means_one_try_per_model():
    seen = []
    llm = _fb(_scripted({"a": [503], "b": [503]}, seen), model="a", fallback_models=["b"], retries=0)
    with pytest.raises(LLMError):
        llm([], "x")
    assert seen == ["a", "b"]


def test_time_budget_stops_further_attempts():
    c = Clock()
    seen = []

    def handler(req):
        seen.append(json.loads(req.content)["model"])
        c.t += 40                                  # each attempt "takes" 40 s
        return httpx.Response(503)

    llm = _fb(handler, model="a", fallback_models=["b", "c"], total_budget=60, clock=c)
    with pytest.raises(LLMError) as e:
        llm([], "x")
    assert len(seen) == 2 and "gave up" in str(e.value)


def test_duplicate_models_are_tried_once_each():
    seen = []
    llm = _fb(_scripted({"a": [503]}, seen), model="a", fallback_models=["a", "a"], retries=0)
    with pytest.raises(LLMError):
        llm([], "x")
    assert seen == ["a"]


def test_callers_model_is_the_primary_when_none_is_pinned():
    seen = []
    llm = _fb(_scripted({"asked": [503], "b": [200]}, seen), fallback_models=["b"], retries=0)
    assert llm([], "asked") == "from b" and seen == ["asked", "b"]


def test_pinned_model_beats_callers_choice_even_with_fallbacks():
    seen = []
    _fb(_scripted({"pinned": [200]}, seen), model="pinned", fallback_models=["b"])([], "expensive")
    assert seen == ["pinned"]


def test_odd_empty_reply_is_retried():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={}) if len(calls) == 1 else _ok("fine")

    assert _fb(handler, model="a")([], "x") == "fine" and len(calls) == 2


def test_network_error_is_retried_then_falls_back():
    seen = []

    def handler(req):
        m = json.loads(req.content)["model"]
        seen.append(m)
        if m == "a":
            raise httpx.ReadTimeout("slow")
        return _ok("from b")

    assert _fb(handler, model="a", fallback_models=["b"])([], "x") == "from b" and seen == ["a", "a", "b"]


def test_retry_logs_name_model_and_status_but_no_prompt_or_key(caplog):
    import logging
    num = aadhaar()
    llm = OpenAICompatLLM("https://api.example.com/v1", "sk-secret-key", model="a", fallback_models=["b"],
                          client=_client(_scripted({"a": [503], "b": [200]})))
    with caplog.at_level(logging.WARNING, logger="pii_gateway"):
        llm([{"role": "user", "content": f"aadhaar {num}"}], "x")
    assert "model=a" in caplog.text and "HTTP 503" in caplog.text
    assert "sk-secret-key" not in caplog.text and num not in caplog.text


def test_repr_lists_fallbacks_but_not_key():
    r = repr(OpenAICompatLLM("https://api.example.com/v1", "sk-very-secret", model="a", fallback_models=["b"]))
    assert "'b'" in r and "sk-very-secret" not in r


def test_gateway_user_gets_an_answer_when_primary_is_overloaded():
    llm = _fb(_scripted({"a": [503], "b": [200]}), model="a", fallback_models=["b"])
    r = TestClient(create_app(llm=llm, audit=AuditLog(salt="t"))).post("/v1/chat/completions", json=_msg())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from b"


def test_gateway_still_returns_clean_502_when_every_model_is_down():
    llm = _fb(_scripted({"a": [503], "b": [503]}), model="a", fallback_models=["b"])
    r = TestClient(create_app(llm=llm, audit=AuditLog(salt="t"))).post("/v1/chat/completions", json=_msg())
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error" and "503" not in r.text


def test_fallback_models_setting_is_parsed_and_trimmed():
    s = load_settings(_prod_env(LLM_FALLBACK_MODELS=" m1 , m2,,m3 "))
    assert s.llm_fallback_models == ("m1", "m2", "m3")
    assert load_settings(_prod_env()).llm_fallback_models == ()


def test_timeout_setting_default_and_validation():
    assert load_settings(_prod_env()).llm_timeout == 30
    assert any("LLM_TIMEOUT_SECONDS" in p for p in check(load_settings(_prod_env(LLM_TIMEOUT_SECONDS="0")))[0])
    with pytest.raises(ConfigError):
        load_settings(_prod_env(LLM_TIMEOUT_SECONDS="fast"))


def test_server_passes_model_fallbacks_and_timeout_to_the_adapter(tmp_path, monkeypatch):
    import gateway.server as srv
    seen = {}

    class Spy(srv.OpenAICompatLLM):
        def __init__(self, *a, **kw):
            seen["args"], seen["kw"] = a, kw
            super().__init__(*a, **kw)

    monkeypatch.setattr(srv, "OpenAICompatLLM", Spy)
    env = _prod_env(PII_AUDIT_DB=str(tmp_path / "a.db"), LLM_MODEL="m1", LLM_FALLBACK_MODELS=" m2, m3 ,", LLM_TIMEOUT_SECONDS="12")
    build_app(env)
    assert seen["args"][2] == "m1"
    assert tuple(seen["kw"]["fallback_models"]) == ("m2", "m3") and seen["kw"]["timeout"] == 12


def test_timeout_must_be_positive():
    assert any("LLM_TIMEOUT" in p for p in check(load_settings(_prod_env(LLM_TIMEOUT_SECONDS="0")))[0])
