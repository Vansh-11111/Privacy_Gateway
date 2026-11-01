# India PII Gateway

A privacy gateway that sits between your app and a large language model (LLM). It finds Indian personal data in prompts, masks, tokenises or blocks it before the model sees it, then restores the original values in the reply. Built by team **Cosmicpixel** (MAIT) for Build with Bharat 5.0.

**Live demo:** https://india-pii-gateway.onrender.com  
(Free hosting: the first load after 15 minutes idle takes about a minute.)

> All data in the demo, tests and evaluation set is **invented**. Do not send real customer data through the public demo.

## The problem

Apps that call an LLM often pass customer messages straight through. A bank-support chat can carry an Aadhaar number, PAN, UPI ID or card number to a third-party model, where it may be logged or stored. Generic PII tools miss Indian formats or flag random digits as IDs.

## What it does

- **Detects** Indian PII with validation, not just patterns: Aadhaar (Verhoeff checksum), PAN, GSTIN, UPI ID, IFSC, mobile number, email, card number (Luhn), bank account, OTP, CVV.
- **Acts per type:** mask, tokenise (reversible), or block the request.
- **Restores** tokens in the model's reply so the user sees a normal answer.
- **Scans the output** too, in case the model produces sensitive data.
- **Keeps tokens private:** an encrypted in-memory vault (Fernet: AES-128-CBC with HMAC-SHA256) with an expiry time, separate per session.
- **Audit log** in SQLite with hashed session IDs and **no raw values**.
- **OpenAI-compatible API** (`/v1/chat/completions`): an app changes its base URL and key, nothing else.

## How a request flows

```
App  ->  Gateway: detect -> mask / tokenise / block  ->  LLM
App  <-  Gateway: scan output -> restore tokens      <-  LLM
```

## Two modes (same code)

| Mode | Use | Notes |
|---|---|---|
| `demo` (default) | Public website | Dashboard, six attack presets, scripted bank bot, random admin key, no log reset, rate-limited |
| `production` | Real gateway API | Refuses to start without strong secrets, client API keys and an LLM URL |

Security controls: client API keys (stored as SHA-256 digests, constant-time compare), lockout after repeated failed auth, per-client rate limits, request-size limit, https-only model URL (except localhost), no redirects followed, errors and logs never contain prompts, responses or keys.

## Run it locally

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest                                   # expect 279 passed
python -m uvicorn gateway.server:build_app --factory
```
Open http://127.0.0.1:8000.

## Use a real model (production mode)

```powershell
python -m gateway.config keys     # prints secrets, saves nothing
python -m gateway.config check    # tells you what is missing
```
Set `PII_MODE=production`, the four secrets, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. Optional: `LLM_FALLBACK_MODELS` (tried in order if the first model is busy). Any OpenAI-compatible provider works. We tested against a fake provider and Google's Gemini API with invented data. Full guide: [DEPLOY.md](DEPLOY.md).

```python
from openai import OpenAI
client = OpenAI(base_url="https://YOUR-GATEWAY/v1", api_key="<client key>")
client.chat.completions.create(model="any", messages=[{"role": "user", "content": "..."}],
                               extra_headers={"X-Session-Id": "customer-123"})
```
Use a different `X-Session-Id` per end user.

## Project layout

```
gateway/
  app.py          gateway API (auth, rate limit, protect, restore)
  demo_app.py     public demo, dashboard routes, evaluation
  dashboard.html  demo website
  server.py       entry point for demo and production
  config.py       settings and startup checks
  llm.py          model interface and demo bot
  llm_openai.py   OpenAI-compatible adapter (retry, fallback)
  ratelimit.py    sliding-window rate limiter
tests/            279 automated tests
render.yaml       Render deployment blueprint
Dockerfile        provided, not yet built or tested
```

## Limitations (stated plainly)

- **Single worker.** Tokens live in memory, so a restart forgets them and a second worker would not know them. Scaling needs a shared vault such as Redis (not built).
- **Public demo uses a scripted bot**, not a real model. Real-model mode was tested only from a laptop with invented data.
- **Free-tier models can be busy** (HTTP 503). Retries and fallback models reduce this but cannot remove it.
- **Not detected:** personal names, postal addresses and other free-text identifiers. The detectors cover structured Indian identifiers.
- **No streaming**, no user accounts or billing, and **images and files are not inspected**.
- **English and digit-based formats.** Numbers written in words or spelled out in other scripts may be missed.
- **Pattern and checksum detection** can miss deliberately obfuscated values (for example digits split across a sentence) and may flag a number that happens to pass a checksum.
- **Rate limits and lockouts are per process** and in memory, so they reset on restart.
- **Dockerfile is untested.**
- **Hosting location:** the free host's nearest region is Singapore. For real use, pick hosting and a model provider that meet your customer's data-location rules.
- **Not legal advice or a compliance certificate.** The design follows data-minimisation ideas, but compliance with India's DPDP Act or sector rules (for example RBI) needs a proper legal and security review.

## Future work

- Shared vault (Redis or a managed key store) and multiple workers.
- Name and address detection (an Indian-language NER model) alongside the rule-based detectors.
- Streaming responses.
- Hindi and other Indian-language prompts, and numbers written in words.
- Inspect images and documents (OCR) before they reach a model.
- Per-client policies (which types to mask, tokenise or block) and a policy editor in the dashboard.
- Persistent, tamper-evident audit log with export for compliance teams.
- Metrics and alerting, and a published evaluation with precision and recall on a larger invented dataset.
- Tested Docker image and one-click deployment on Indian cloud regions.
- Independent security review before any production use.

## Team

Cosmicpixel, Maharaja Agrasen Institute of Technology (MAIT).
