from __future__ import annotations

import sys

from .protect import protect, restore
from .vault import TokenVault

SAMPLES = [
    "mera aadhaar 2384 1295 6108 hai, mera loan kyun reject hua?",
    "pan ABCPK1234D, call +91 98765 43210, upi rahul@okhdfcbank",
    "my otp is 482913",
    "aadhaar 2384 1295 6108 and a/c no 50100123456781",
    "order id 238412956108 shipped",
]


def run(messages: list[str]) -> None:
    vault = TokenVault()  # random key, this run only
    for i, msg in enumerate(messages, 1):
        result = protect(msg, "demo", vault)
        print(f"--- message {i}")
        print(f"  user typed : {msg}")
        if result.blocked:
            print(f"  BLOCKED    : {result.block_reason}")
            print(f"  user sees  : {result.user_message}")
            continue
        reply = f"Noted. You said: {result.text}"
        print(f"  LLM gets   : {result.text}")
        print(f"  LLM replies: {reply}")
        print(f"  user sees  : {restore(reply, 'demo', vault)}")
        print(f"  audit      : {result.summary() or 'nothing detected'}")


if __name__ == "__main__":
    run(sys.argv[1:] or SAMPLES)
