from __future__ import annotations

from dataclasses import dataclass

from .detectors import detect
from .policy import Action, Policy
from .vault import TokenVault

_MIN_LEN_FOR_PARTIAL = 10


@dataclass(frozen=True)
class Applied:
    type: str
    action: Action
    confidence: float
    token: str | None = None


@dataclass(frozen=True)
class ProtectResult:
    text: str | None                 # None when blocked: there is nothing safe to forward
    blocked: bool
    blocked_types: tuple[str, ...]
    applied: tuple[Applied, ...]

    @property
    def block_reason(self) -> str | None:
        return "+".join(self.blocked_types) if self.blocked else None

    @property
    def user_message(self) -> str | None:
        if not self.blocked:
            return None
        what = ", ".join(t.replace("_", " ").lower() for t in self.blocked_types)
        return (f"Your message contains information that can't be sent to the assistant ({what}). "
                "Please remove it and try again.")

    def summary(self) -> dict[str, dict[str, int]]:
        """Counts by type and action, ready for the audit log. No values."""
        out: dict[str, dict[str, int]] = {}
        for a in self.applied:
            by_action = out.setdefault(a.type, {})
            by_action[a.action.value] = by_action.get(a.action.value, 0) + 1
        return out


def partial_mask(value: str, keep: int = 4) -> str:
    """Mask every letter/digit except the last `keep`; separators stay.
    Values shorter than 10 characters are masked completely."""
    positions = [i for i, ch in enumerate(value) if ch.isalnum()]
    if len(positions) < _MIN_LEN_FOR_PARTIAL:
        keep = 0
    visible = set(positions[len(positions) - keep:]) if keep else set()
    return "".join("X" if ch.isalnum() and i not in visible else ch for i, ch in enumerate(value))


def _blocked(types) -> ProtectResult:
    return ProtectResult(text=None, blocked=True, blocked_types=tuple(sorted(types)), applied=())


def protect(text: str, session_id: str, vault: TokenVault, policy: Policy | None = None) -> ProtectResult:
    policy = policy or Policy.default()

    # 1. Decide what happens to each detected entity.
    plan = []
    for e in detect(text):
        rule = policy.rule_for(e.type)
        confident = e.confidence >= rule.min_confidence
        plan.append((e, rule.action if confident else Action.ALLOW, confident))

    # 2. Refuse BEFORE touching the vault, so a blocked message stores nothing.
    blocking = {e.type for e, action, _ in plan if action is Action.BLOCK}
    if blocking:
        return _blocked(blocking)
    present = {e.type for e, _, confident in plan if confident}
    for combo in policy.block_combos:
        if combo <= present:
            return _blocked(combo)

    # 3. Build the output left to right; offsets always refer to the ORIGINAL text.
    out, applied, cursor = [], [], 0
    for e, action, _ in plan:
        out.append(text[cursor:e.start])
        original = e.slice(text)
        token = None
        if action is Action.ALLOW:
            out.append(original)
        elif action is Action.PARTIAL:
            out.append(partial_mask(original))
        elif action is Action.REDACT:
            out.append(f"[{e.type}]")
        else:  # TOKENIZE
            token = vault.tokenize(session_id, e.type, original)
            out.append(token)
        applied.append(Applied(e.type, action, e.confidence, token))
        cursor = e.end
    out.append(text[cursor:])
    return ProtectResult(text="".join(out), blocked=False, blocked_types=(), applied=tuple(applied))


def restore(text: str, session_id: str, vault: TokenVault) -> str:
    """Swap tokens in an LLM reply back to the user's real values."""
    return vault.restore(session_id, text)