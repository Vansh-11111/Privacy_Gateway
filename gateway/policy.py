from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    ALLOW = "ALLOW"        # send unchanged (still counted in the audit log)
    PARTIAL = "PARTIAL"    # keep the last 4 characters: XXXX XXXX 6108 (not reversible)
    REDACT = "REDACT"      # replace with [AADHAAR] (not reversible)
    TOKENIZE = "TOKENIZE"  # replace with <AADHAAR_1>; real value kept in the vault (reversible)
    BLOCK = "BLOCK"        # refuse the whole message


@dataclass(frozen=True)
class Rule:
    action: Action
    min_confidence: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


# These can never be anything but BLOCK, and their threshold cannot be raised
# above the detector default.
NEVER_ALLOW = frozenset({"OTP", "CVV"})

# Message is refused when ALL types in a combination are present.
# Example default: tune it per use case.
DEFAULT_BLOCK_COMBOS = (frozenset({"AADHAAR", "BANK_ACCOUNT"}),)


def _default_rules() -> dict[str, Rule]:
    return {
        "AADHAAR": Rule(Action.TOKENIZE),
        "PAN": Rule(Action.TOKENIZE),
        "GSTIN": Rule(Action.TOKENIZE),
        "UPI": Rule(Action.TOKENIZE),
        "EMAIL": Rule(Action.TOKENIZE),
        "BANK_ACCOUNT": Rule(Action.TOKENIZE),
        "MOBILE": Rule(Action.PARTIAL),
        "IFSC": Rule(Action.ALLOW),     # identifies a bank branch, not a person
        "CARD": Rule(Action.BLOCK),
        "OTP": Rule(Action.BLOCK),
        "CVV": Rule(Action.BLOCK),
    }


@dataclass(frozen=True)
class Policy:
    rules: dict
    default_rule: Rule = Rule(Action.TOKENIZE)
    block_combos: tuple = DEFAULT_BLOCK_COMBOS

    def __post_init__(self):
        for t in sorted(NEVER_ALLOW):
            r = self.rules.get(t)
            if r is None or r.action is not Action.BLOCK or r.min_confidence > 0.5:
                raise ValueError(f"{t} must always be BLOCK with min_confidence <= 0.5")

    @classmethod
    def default(cls) -> "Policy":
        return cls(rules=_default_rules())

    def rule_for(self, entity_type: str) -> Rule:
        return self.rules.get(entity_type, self.default_rule)

    def with_rule(self, entity_type: str, action: Action, min_confidence: float = 0.5) -> "Policy":
        """Return a NEW policy with one rule changed; this one is untouched."""
        rules = dict(self.rules)
        rules[entity_type] = Rule(action, min_confidence)
        return Policy(rules=rules, default_rule=self.default_rule, block_combos=self.block_combos)

