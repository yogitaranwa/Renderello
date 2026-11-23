# Deployment

## Website (Render free web service)
- Private GitLab repository; Render free instance, no card,
  no auto-upgrade (verified at setup; stop if terms change).
- Env secrets (Render dashboard, never in source/Drive/chat):
  - RENDERELLO_ACCESS_CODE - app gate, vault-generated
  - RENDERELLO_COOKIE_SECURE=1
  - MAGIC_HOUR_API_KEY - optional; enables the Magic Hour engine
  - RENDERELLO_MH_LIVE=1 - only with an owner spend approval
  - RENDERELLO_MH_BUDGET_CREDITS=<n> - server-side aggregate cap; live
    generation refuses with zero bill when unset or exhausted. Spend ledger:
    RENDERELLO_MH_LEDGER (default runs/mh-spend-ledger.jsonl); uncertain
    charges count at the listed rate.
- Uploads: PNG/JPEG/WebP by magic bytes, 12 MB cap, per-run temp dir,
  janitor-deleted after 24 h. Uploads ARE stored server-side up to 24 h;
  do not claim otherwise.
- Start: `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT`

## Colab notebook
- Renderello_Studio.ipynb vendors the whole package inline (one cell writes
  all modules) - runs on a fresh runtime with no repo checkout.
- INTERACTIVE ONLY: Colab's free runtime terms (FAQ, verified 2026-11-20,
  https://research.google.com/colaboratory/faq.html) disallow remote proxies
  and serving content-generation web UIs. No tunnel backend exists or may
  be added.
- Verification status: vendored-package reconstruction and imports verified
  locally (165 modules); a full interactive Colab run has NOT been executed
  yet - treat end-to-end Colab behavior as untested until that run happens.

## Test status
Merged tree (Nov 14 base + v1.1 modules): 3,676 passed, 0 failed, 4 skipped.
Reproducible: three disjoint pytest invocations over tests/ (chunks of
62/62/60 files), each recorded above; no rerun double-counting.

## Login migration (2026-11-20, v1.3.0)
The site now signs in with email + password (modular login page:
`static/login.html` + `static/js/login.js` + `static/css/login.css`,
`webapp/auth.py`). Env vars:
- `RENDERELLO_LOGIN_EMAIL` - the login identifier (no mail is sent to it).
- `RENDERELLO_LOGIN_PASSWORD_HASH` - `pbkdf2_sha256$600000$<salt_hex>$<hash_hex>`.
  The plaintext password is never stored in source, env, logs or manifests.
- Legacy `RENDERELLO_ACCESS_CODE` mode remains only when the two vars above
  are absent; remove it after the password login is verified.
- Sessions: random-token cookies, 12 h, HttpOnly, SameSite=strict; 10 login
  attempts per 10 minutes per IP.

## Scaling rule (security)

Run SINGLE PROCESS ONLY (`uvicorn webapp.main:app` with no `--workers` flag):
the job registry, the single-live-MH-job rule and the budget reservation lock
are per-process by design. The credit hard cap itself is anchored in the live
Magic Hour account balance (RENDERELLO_MH_CAP_ANCHOR), so it holds across
deploys; the per-process rules only need one process to be correct.
