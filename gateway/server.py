"""The one entry point for running the gateway as a website or an API.

    python -m uvicorn gateway.server:build_app --factory --host 0.0.0.0 --port 8000

PII_MODE=demo (the default): the public dashboard with the scripted bank bot. Safe to put
online: invented data only, no real model, no secrets needed.
PII_MODE=production: the gateway API in front of a real model. It refuses to start unless
every secret is set (see `python -m gateway.config check`).

Run ONE worker. The token vault is in memory, so a second worker would not know the tokens
the first one issued.
"""
from __future__ import annotations

import logging
import secrets
from typing import Mapping

from fastapi import FastAPI

from .app import create_app
from .audit import AuditLog
from .config import ConfigError, Settings, check, load_settings
from .demo_app import create_demo_app
from .llm_openai import OpenAICompatLLM
from .vault import TokenVault

log = logging.getLogger("pii_gateway")


def build_app(env: Mapping[str, str] | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")   # no-op if already set up
    settings: Settings = load_settings(env)
    problems, warnings = check(settings)
    if problems:
        raise ConfigError("Cannot start:\n  - " + "\n  - ".join(problems))
    for w in warnings:
        log.warning("config: %s", w)

    if settings.mode == "demo":
        app = create_demo_app(
            eval_path=settings.eval_file, audit=AuditLog(settings.audit_db, salt=settings.audit_salt),
            public=True, admin_key=settings.admin_key or secrets.token_urlsafe(24),
            rate_limit_per_min=settings.rate_limit_per_min, trust_proxy=settings.trust_proxy)
    else:
        try:
            llm = OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                                  timeout=settings.llm_timeout, fallback_models=settings.llm_fallback_models)
        except ValueError as e:   # the message names the setting, never a secret
            raise ConfigError(f"Cannot start:\n  - {e}") from None
        app = create_app(
            llm=llm, vault=TokenVault(key=settings.vault_key, ttl_seconds=settings.vault_ttl_seconds),
            audit=AuditLog(settings.audit_db, salt=settings.audit_salt), admin_key=settings.admin_key,
            api_keys=settings.api_keys, rate_limit_per_min=settings.rate_limit_per_min,
            trust_proxy=settings.trust_proxy, llm_kind="upstream")

    log.info("startup mode=%s llm=%s client_keys=%d rate_limit=%d/min audit_db=%s",
             settings.mode, "scripted-demo-bot" if settings.mode == "demo" else "upstream",
             len(settings.api_keys), settings.rate_limit_per_min,
             "memory" if settings.audit_db == ":memory:" else "file")
    return app

