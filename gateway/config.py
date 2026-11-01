"""Settings read from environment variables, plus the checks that stop an unsafe start.

    python -m gateway.config keys    # print fresh secrets to paste into your host's settings
    python -m gateway.config check   # check the current environment (never prints secrets)
"""
from __future__ import annotations

import argparse
import os
import secrets
from dataclasses import dataclass
from typing import Mapping

from cryptography.fernet import Fernet

MIN_SECRET_LEN = 16


class ConfigError(Exception):
    """Raised at startup with a plain list of what to fix. Never contains a secret value."""


@dataclass(frozen=True)
class Settings:
    mode: str                          # "demo" or "production"
    admin_key: str | None
    vault_key: str | None
    audit_salt: str | None
    audit_db: str
    api_keys: tuple[str, ...]
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_fallback_models: tuple[str, ...]
    llm_timeout: int
    rate_limit_per_min: int
    trust_proxy: bool
    vault_ttl_seconds: int
    eval_file: str

    def __repr__(self) -> str:   # secrets must not leak through logging or a traceback
        return f"Settings(mode={self.mode!r}, ...)"


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number") from None


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    mode = env.get("PII_MODE", "demo").strip().lower() or "demo"
    if mode not in ("demo", "production"):
        raise ConfigError("PII_MODE must be 'demo' or 'production'")
    keys = tuple(k.strip() for k in env.get("PII_API_KEYS", "").split(",") if k.strip())
    return Settings(
        mode=mode,
        admin_key=env.get("PII_ADMIN_KEY") or None,
        vault_key=env.get("PII_VAULT_KEY") or None,
        audit_salt=env.get("PII_AUDIT_SALT") or None,
        audit_db=env.get("PII_AUDIT_DB") or ("audit.db" if mode == "production" else ":memory:"),
        api_keys=keys,
        llm_base_url=env.get("LLM_BASE_URL") or None,
        llm_api_key=env.get("LLM_API_KEY") or None,
        llm_model=env.get("LLM_MODEL") or None,
        llm_fallback_models=tuple(m.strip() for m in env.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()),
        llm_timeout=_int(env, "LLM_TIMEOUT_SECONDS", 30),
        rate_limit_per_min=_int(env, "PII_RATE_LIMIT_PER_MIN", 30 if mode == "demo" else 60),
        trust_proxy=_flag(env, "PII_TRUST_PROXY"),
        vault_ttl_seconds=_int(env, "PII_VAULT_TTL_SECONDS", 3600),
        eval_file=env.get("PII_EVAL_FILE") or "eval_results.json",
    )


def check(s: Settings) -> tuple[list[str], list[str]]:
    """Returns (problems that stop startup, warnings). Messages never include secret values."""
    problems: list[str] = []
    warnings: list[str] = []
    if s.rate_limit_per_min < 1:
        problems.append("PII_RATE_LIMIT_PER_MIN must be at least 1")
    if s.llm_timeout < 1:
        problems.append("LLM_TIMEOUT_SECONDS must be at least 1")
    if s.vault_ttl_seconds < 60:
        problems.append("PII_VAULT_TTL_SECONDS must be at least 60")

    if s.mode == "demo":
        if s.llm_base_url or s.llm_api_key:
            warnings.append("LLM_* settings are ignored in demo mode: the demo always uses the scripted bot")
        return problems, warnings

    # ---- production: fail closed
    if not s.admin_key or s.admin_key == "demo" or len(s.admin_key) < MIN_SECRET_LEN:
        problems.append(f"PII_ADMIN_KEY must be set to a secret of at least {MIN_SECRET_LEN} characters (not 'demo')")
    if not s.vault_key:
        problems.append("PII_VAULT_KEY must be set (create one with: python -m gateway.config keys)")
    else:
        try:
            Fernet(s.vault_key.encode())
        except Exception:
            problems.append("PII_VAULT_KEY is not a valid key (create one with: python -m gateway.config keys)")
    if not s.audit_salt or len(s.audit_salt) < MIN_SECRET_LEN:
        problems.append(f"PII_AUDIT_SALT must be set to a secret of at least {MIN_SECRET_LEN} characters")
    if not s.api_keys:
        problems.append("PII_API_KEYS must list at least one client key, separated by commas")
    elif any(len(k) < MIN_SECRET_LEN for k in s.api_keys):
        problems.append(f"every key in PII_API_KEYS must be at least {MIN_SECRET_LEN} characters")
    if not s.llm_base_url:
        problems.append("LLM_BASE_URL must be set, e.g. https://api.openai.com/v1")
    if s.audit_db == ":memory:":
        warnings.append("PII_AUDIT_DB is ':memory:', so the audit log disappears on restart")
    if not s.trust_proxy:
        warnings.append("PII_TRUST_PROXY is off: behind a load balancer every client looks like one address")
    if not s.llm_api_key:
        warnings.append("LLM_API_KEY is empty: fine for a local model, an error for a hosted one")
    return problems, warnings


def new_secrets() -> dict[str, str]:
    return {
        "PII_ADMIN_KEY": secrets.token_urlsafe(24),
        "PII_VAULT_KEY": Fernet.generate_key().decode(),
        "PII_AUDIT_SALT": secrets.token_urlsafe(24),
        "PII_API_KEYS": "pk_" + secrets.token_urlsafe(24),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PII gateway settings helper")
    ap.add_argument("command", choices=["keys", "check"])
    cmd = ap.parse_args(argv).command
    if cmd == "keys":
        print("# New secrets. Put them in your host's environment settings. Never commit them.")
        for k, v in new_secrets().items():
            print(f"{k}={v}")
        return 0
    try:
        s = load_settings()
    except ConfigError as e:
        print(f"PROBLEM: {e}")
        return 1
    problems, warnings = check(s)
    print(f"mode: {s.mode}")
    for w in warnings:
        print(f"warning: {w}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if not problems:
        print("settings look OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
