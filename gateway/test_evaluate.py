import json

from gateway import evaluate as ev


def test_wilson_interval_known_values():
    lo, hi = ev.wilson(0, 300)
    assert lo == 0.0 and abs(hi - 0.0127) < 0.0005
    lo, hi = ev.wilson(300, 300)
    assert abs(lo - 0.9873) < 0.0005 and abs(hi - 1.0) < 1e-9
    lo, hi = ev.wilson(50, 100)
    assert lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 0.01
    assert ev.wilson(0, 0) == (0.0, 0.0)


def test_results_have_every_section_and_are_json_serialisable():
    res = ev.run(n=30)
    assert {"recall", "false_positives", "leak_through", "latency", "attacks", "caveats"} <= set(res)
    assert len(res["recall"]) == len(ev.POSITIVES)
    assert len(res["false_positives"]) == len(ev.NEGATIVES)
    json.dumps(res)


def test_results_are_reproducible():
    assert ev.measure_recall(25, 7) == ev.measure_recall(25, 7)
    assert ev.measure_false_positives(25, 7) == ev.measure_false_positives(25, 7)


def test_no_identifier_type_leaks_through_the_default_policy():
    for row in ev.measure_leak_through(60, 1):
        assert row["leaked"] == 0, row["type"]
        assert row["leaked"] + row["blocked"] + row["protected"] + row["allowed_by_policy"] == row["n"]


def test_allowed_by_policy_is_not_counted_as_a_leak():
    ifsc = next(r for r in ev.measure_leak_through(30, 1) if r["type"] == "IFSC")
    assert ifsc["allowed_by_policy"] == 30 and ifsc["leaked"] == 0


def test_clean_text_is_not_flagged():
    rows = {r["case"]: r for r in ev.measure_false_positives(100, 3)}
    assert rows["clean sentences (amounts, dates, pincodes, flight no.)"]["flagged"] == 0
    assert rows["order/invoice number shaped like an Aadhaar"]["flagged"] == 0


def test_unavoidable_cases_are_labelled():
    labelled = {r["case"] for r in ev.measure_false_positives(30, 3) if r["note"]}
    assert "PAN-shaped code with a valid holder letter" in labelled
    assert "random 10-digit number starting 6-9, no context" in labelled


def test_core_recall_rows_are_near_perfect():
    for r in ev.measure_recall(60, 5):
        if r["type"] != "BANK_ACCOUNT":
            assert r["rate"] >= 0.95, (r["type"], r["style"])


def test_latency_is_small():
    lat = ev.measure_latency(50, 1)
    assert lat["protect"]["p95_ms"] < 50 and lat["reply_scan"]["p95_ms"] < 50


def test_report_prints(capsys):
    ev.print_report(ev.run(n=20))
    out = capsys.readouterr().out
    assert "RECALL" in out and "FALSE POSITIVES" in out and "CAVEATS" in out


def test_measurement_can_fail(monkeypatch):
    """The evaluation must report misses: with a detector that finds nothing, recall is 0 and leak-through is 100%."""
    monkeypatch.setattr(ev, "detect", lambda text, **kw: [])
    assert all(r["found"] == 0 for r in ev.measure_recall(10, 1))
    import gateway.protect as protect_mod
    monkeypatch.setattr(protect_mod, "detect", lambda text, **kw: [])
    leak = {r["type"]: r for r in ev.measure_leak_through(10, 1)}
    assert leak["AADHAAR"]["leaked"] == 10 and leak["PAN"]["leaked"] == 10
