import pytest

from gateway.policy import Action, Policy, Rule


def test_default_policy_covers_every_detector_type():
    p = Policy.default()
    for t in ["AADHAAR", "PAN", "MOBILE", "IFSC", "EMAIL", "UPI", "GSTIN", "CARD", "BANK_ACCOUNT", "OTP", "CVV"]:
        assert t in p.rules, t


def test_otp_cvv_card_are_blocked_by_default():
    p = Policy.default()
    for t in ["OTP", "CVV", "CARD"]:
        assert p.rule_for(t).action is Action.BLOCK


def test_unknown_type_falls_back_to_tokenize():
    assert Policy.default().rule_for("SOMETHING_NEW").action is Action.TOKENIZE


@pytest.mark.parametrize("t", ["OTP", "CVV"])
def test_otp_and_cvv_cannot_be_weakened(t):
    for action in [Action.ALLOW, Action.TOKENIZE, Action.REDACT, Action.PARTIAL]:
        with pytest.raises(ValueError):
            Policy.default().with_rule(t, action)
    with pytest.raises(ValueError):
        Policy.default().with_rule(t, Action.BLOCK, min_confidence=0.9)


def test_otp_cvv_rules_cannot_be_omitted():
    with pytest.raises(ValueError):
        Policy(rules={"AADHAAR": Rule(Action.TOKENIZE)})


def test_with_rule_returns_new_policy_and_leaves_original_alone():
    base = Policy.default()
    changed = base.with_rule("MOBILE", Action.REDACT)
    assert changed.rule_for("MOBILE").action is Action.REDACT
    assert base.rule_for("MOBILE").action is Action.PARTIAL


def test_rule_rejects_bad_confidence():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            Rule(Action.REDACT, bad)
