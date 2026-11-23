# Renderello: Technical Documentation

Last updated: 21 November 2026. Covers the deployed web studio and the Colab notebook at the `ac6114d` source checkpoint.

Companion documents in this folder: `ARCHITECTURE.md` (system map), `FLOW.md` (execution and data-flow traces), `DECISIONS.md` (decision log with rationale), `CONSTRAINTS.md` (change boundaries), `TESTING.md` (verification checklist), `ROLLBACK.md` (undo plan), `HANDOVER.md` (current state at a glance). This file is the deep reference; those are the working documents.

---

## 1. System overview

Renderello produces candidate product shots from real photos: a generative model builds the scene, and fully opaque product pixels are preserved bit-exact rather than repainted. The same pipeline ships in two forms:

- **Web studio** (`webapp/`): a FastAPI app with a vanilla JS frontend, deployed on a Render free web service at https://renderello-studio.onrender.com (private, sign-in required).
- **Colab notebook** (`Renderello_Studio.ipynb`): the full pipeline in one self-contained notebook, built from the same `renderello/` package code, with additional batch and manifest tooling.

Pipeline stages: intake and validation of product photos (up to 4 products plus optional style references, PNG/JPEG/WebP, 12 MB each) → product/background separation and cutout verification → scene generation (procedural CPU engine or an AI provider) → compositing of the preserved product pixels onto the scene → side-by-side review against the source photo.

### Product preservation contract

On the scene-compose path (the `scene` route, and `hybrid` before its harmonize step), the provider generates the scene only. Fully opaque product pixels are composited bit-exact on top of the scene; this scope is verified by tests. The whole-image route (`whole`) and the `hybrid` harmonize step reconstruct the product from model output and sit outside this contract (4.3). Semi-transparent edges and mask semantics are excluded from the guarantee and remain subject to human review. No quality score is claimed for generated scenes.

---

## 2. Architecture map

The system is a layered monolith. One Python package (`renderello/`) holds all domain logic; two entry surfaces sit on top of it and share it unchanged.

```
webapp/ (FastAPI site)          Renderello_Studio.ipynb (Colab)      CLI / research tools
        |                                |                                |
        +--------------+-----------------+----------------+---------------+
                       v                 v
              renderello.studio    (run_studio: one high-level generate call)
              renderello.pipeline  (run_pipeline: research/audit harness)
                       |
        +--------------+---------------------------+
        v              v            v               v
   presets.py    prompts.py   scene_providers   multinput.py
   templates.py  longprompt.py  + backends/     (multi-product)
        |              |            |               |
        +--------------+-----+------+---------------+
                           v
        image_intake.py -> cutout.py -> matting.py -> compose.py
                           |                            |
                           +---------> qa.py (fidelity metrics)
        Platform guards: webapp/auth.py, mh_ledger.py, process_memory.py,
        resources.py. Evidence/review: benchmark.py, gallery.py, review.py,
        brief_acceptance.py, brief_economics.py, human review pack modules.
```

Dependency direction is strictly downward: entry surfaces import the package; the package never imports the webapp. `schemas.py` and `config.py` sit at the bottom and hold the shared record shapes (`EditRequest`, `CutoutResult`, `QAReport`, `VariantResult`, `RunResult`) and model IDs, so the benchmark, QA and gallery all speak the same shapes.

### Module areas

| Area | Representative modules | Responsibility |
|------|------------------------|----------------|
| Entry | `webapp/main.py`, `webapp/auth.py`, notebook cells, `renderello/cli.py` | HTTP surface, auth, notebook assembly, command-line workflows |
| Orchestration | `renderello/studio.py`, `renderello/pipeline.py` | One-call generate flow; route x variant research harness with manifests |
| Domain | `renderello/presets.py`, `templates.py`, `prompts.py`, `longprompt.py` | 12 shoot templates, prompt plans, negative prompts, long-prompt chunking |
| Imaging core | `renderello/image_intake.py`, `cutout.py`, `matting.py`, `compose.py` | Validated intake, subject cutout, trimap matting, deterministic compositing |
| Providers | `renderello/scene_providers.py`, `backends/magichour.py`, `backends/qwen_local.py`, `backends/klein_local.py` | Scene generation and editor backends, with honest availability and cost reporting |
| Quality and evidence | `renderello/qa.py`, `review.py`, `benchmark.py`, `gallery.py`, `brief_acceptance.py`, `brief_economics.py` | Automated fidelity metrics, review-sheet tooling, benchmark runs, private galleries |
| Human review | `renderello/blind_review.py`, `blind_review_return.py`, `blind_pack_cli.py` | Anonymized review packs for human raters (module names predate the current terminology) |
| Platform guards | `renderello/mh_ledger.py`, `process_memory.py`, `resources.py` | Durable spend ledger, memory telemetry, resource reporting |

---

## 3. Repository components

```
Renderello/
├── Renderello_Studio.ipynb   # Self-contained Colab notebook (clean, unrun)
├── README.md
├── renderello/               # Core pipeline package (flat module layout + backends/ + checked_research/)
├── webapp/
│   ├── main.py               # FastAPI app: routes, middleware, job registry
│   ├── auth.py               # Credentials, sessions, rate limiting, logout denylist
│   └── static/               # login.html, index.html, js/, css/, covers/ (template style previews)
├── tests/                    # Pytest suite (see TESTING.md)
├── benchmark/                # Benchmark harness manifest and sources
├── research/                 # Research scratch (gps_shadow_v485)
├── scripts/                  # Capture utilities and the notebook builder
├── outputs/                  # Existing-results gallery (see outputs/README.md)
├── docs/                     # This documentation set, reports, screenshots, saved-run notebook
└── requirements.txt          # Web app dependencies (pinned; see section 10)
```

Additional pinned requirement sets sit beside `requirements.txt` for optional local features: `requirements-web.lock.txt`, `requirements-semantic.txt`, `requirements-pyramid.txt`, `requirements-prompted.txt`, `requirements-matting.txt`, `requirements-dev.txt`, `requirements-candidate-studio.txt`, `requirements-bilateral.txt`.

---

## 4. Execution flow traces

This section traces the two web requests that matter most, step by step with file and function references. `FLOW.md` carries the same traces plus the notebook flow and the credit-spend data flow.

### 4.1 POST /api/generate (webapp/main.py)

1. **Middleware, before any form parsing.** `generate_gate` rejects unauthenticated or oversized `/api/generate` POSTs before the multipart body is parsed, because FastAPI parses form and file parameters before the handler runs. `BodyLimitMiddleware` then streams-counts actual bytes (which also bounds chunked requests that send no Content-Length) and answers 413 itself when the cap is hit.
2. **Handler validation** (`generate`): session check, template keys against `PRESETS` (1 to 8 per batch, de-duplicated, order kept), file count (1 to 8 across products plus references, at least one product), Magic Hour route in {scene, whole, hybrid}, text-field caps (2,000 chars), variants clamped to 1 to 4.
3. **Job slot claim.** A random job id is registered under a lock BEFORE any await, upload or spend, so concurrent requests cannot all pass the running-job checks. Two rules apply: total running jobs are capped (`RENDERELLO_MAX_RUNNING_JOBS`, fail-safe 1), and only one live Magic Hour job may run at a time.
4. **Budget pre-check (Magic Hour only).** `_enforce_budget` computes the whole batch cost (templates x variants x scene attempts, plus one harmonize call per variant on the hybrid route) and refuses with HTTP 402 before any upload work or spend if the batch does not fit.
5. **Upload staging.** Files land in a per-run temp dir. Each is size-capped (12 MB), sniffed by magic bytes (PNG/JPEG/WebP), decoded and verified by Pillow with explicit caps (25M pixels, 8,000 px per side; Pillow's own 40M-pixel decompression-bomb limit also applies). Failures delete the temp dir.
6. **Work dispatch.** The job runs on a bounded `ThreadPoolExecutor` sized to the running-job cap. For each selected template, `run_studio` (renderello/studio.py) executes the pipeline and writes a run directory with a manifest.
7. **Result readout.** `GET /api/jobs/{job_id}` returns status and, when done, each run's manifest. Images are served through `GET /api/runs/{run_dir}/{name}`, which enforces an id pattern and path-traversal checks.
8. **Cleanup.** The upload temp dir is removed when the job ends. A background janitor deletes run directories older than 24 hours. Errors reaching the client are filtered through `_SAFE_ERROR_PREFIXES`; anything else becomes the generic "generation failed".

### 4.2 POST /api/login (webapp/main.py + webapp/auth.py)

1. `GET /api/login-mode` (unauthenticated) tells the login page which form to render: `password` when `RENDERELLO_LOGIN_EMAIL` and `RENDERELLO_LOGIN_PASSWORD_HASH` are configured, else legacy `code`.
2. The login handler re-checks the body cap, parses JSON, and reserves rate-limit slots BEFORE checking credentials: one slot in the global circuit breaker, plus one in a per-email bucket in password mode. A full bucket answers HTTP 429.
3. Password mode verifies with a constant-time email comparison and a PBKDF2-HMAC-SHA256 check (600,000 iterations). The slow hash check always runs, so response time does not reveal whether the email matched.
4. A successful login refunds only its own slots; failed attempts keep theirs, so brute-force limits are unchanged by interleaved good logins.
5. The session cookie (`renderello_session`, 12 h, HttpOnly, SameSite=lax, Secure by default) carries a stateless HMAC-signed token `s1.<expiry>.<nonce>.<sig>` signed with `RENDERELLO_SESSION_SECRET`, so deploys and restarts do not end sessions. Without that secret, legacy in-memory random tokens are used (invalid after restart).
6. `POST /api/logout` revokes a stateless token by recording only its SHA-256 id, with the token's own expiry, in an in-memory set plus a JSONL append file (mode 0600, atomic whole-file compaction). The raw token never touches disk.

### 4.3 Pipeline core (renderello/studio.py `run_studio`)

`run_studio` is the single high-level generate call shared by the website and the notebook:

1. Resolve the preset and aspect; validate the instruction.
2. Merge optional reference photos into a style descriptor (steers the scene and prompt only; never composited).
3. Cut out each uploaded product photo (`cutout_products`), recording per-photo errors honestly.
4. Route by provider. `magichour` dispatches to the whole-image editor route or the scene-compose route (with optional harmonize on hybrid). Other providers (procedural, sdxl_lightning, hf_inference) are explicit, availability-checked, and never silently substituted.
5. Per variant: build the detailed prompt from preset, product description, material, instruction, reference style and knobs; plan it (`longprompt.plan_prompt`) with the negative prompt; generate the scene; composite the verified cutout on top (`compose.composite`, or `compose_group` for multi-product); run automated QA (`qa.evaluate`).
6. Write the run manifest (provider requested and used, prompt plan, QA verdicts, warnings, failures, runtime versions) and a contact sheet. Failures are listed in the manifest, not dropped.

---

## 5. API reference

All routes live in `webapp/main.py`. Interactive API docs and the schema are all disabled (`docs_url=None`, `redoc_url=None`, `openapi_url=None`: no Swagger UI, no ReDoc, no `openapi.json`), so this table is the catalog.

| Route | Auth | Purpose |
|-------|------|---------|
| `GET /` | session-aware | Login page when signed out, studio UI when signed in |
| `GET /static/{path}` | session (except the login assets and favicons in `_PUBLIC_STATIC`) | Frontend assets, with containment checks |
| `GET /favicon.ico` | no | Real ICO for browsers that probe the path |
| `GET /api/login-mode` | no | Which login form to render |
| `POST /api/login` | rate-limited | Establish a session (see 4.2) |
| `POST /api/logout` | no | Revoke the session token (denylist) |
| `GET /api/presets` | yes | Template catalog for the picker |
| `GET /api/providers` | yes | Provider availability report |
| `GET /api/estimate` | yes | Cost preview before Generate; never spends |
| `POST /api/generate` | yes, gated | Start a generation job (see 4.1) |
| `GET /api/jobs/{job_id}` | yes | Job status and run manifests |
| `GET /api/runs/{run_dir}/{name}` | yes | Serve a generated file, traversal-checked |

`/api/estimate` deserves note: the frontend calls it on every selection change. Procedural runs report free; Magic Hour runs report the listed per-image credits, worst-case images (variants x scene attempts), the budget state, and whether the batch fits, so the cost is visible before the button is pressed.

---

## 6. Web app security model

### 6.1 Authentication and sessions

- Email + password sign-in. The password is never stored or compared in plaintext: the server reads a PBKDF2-HMAC-SHA256 hash (600,000 iterations) from `RENDERELLO_LOGIN_PASSWORD_HASH`. The email comparison is constant-time, and the slow hash check always runs so timing does not reveal whether the email matched.
- A legacy single-access-code mode (`RENDERELLO_ACCESS_CODE`) remains as a fallback when email credentials are not configured.
- Sessions are 12-hour HttpOnly, SameSite=lax cookies holding stateless HMAC-signed tokens (`s1.<expiry>.<nonce>.<sig>`) signed with `RENDERELLO_SESSION_SECRET` (required to be at least 32 random characters when set), so deploys and restarts do not end sessions. The Secure flag defaults on (`RENDERELLO_COOKIE_SECURE`).

### 6.2 Login rate limiting

Ten attempts per ten-minute window, enforced in two buckets:

1. A **global circuit breaker** bounding the total login attempt rate service-wide. The global key is deliberate: per-client-IP attribution has not been verified under the deployed proxy configuration (the connection peer has not been observed directly in the deployed environment; it is presumed to be the platform load balancer, and trusting forwarded headers without verification would be spoofable). Render's documented ingress topology and the deployed uvicorn defaults support that inference, but the code does not rely on it.
2. In password mode, a **per-email bucket** bounding per-account guessing.

A slot is reserved before credentials are checked, so concurrent guesses all count. Failed attempts keep their slot; a successful login refunds only its own. The response at the limit is HTTP 429.

### 6.3 Logout revocation (denylist)

- Logout revokes a stateless session token by recording only its SHA-256 id (never the raw token) with the token's own expiry.
- Storage is an in-memory set plus a JSONL append file (`RENDERELLO_DENYLIST_PATH`, default under `RENDERELLO_RUNS`), mode 0600, with atomic whole-file replacement for compaction.
- Durability is best-effort: a same-host process restart keeps revocations via file reload; a write failure keeps the in-memory revocation for the process lifetime; an instance replacement on an ephemeral filesystem clears the file. Cross-replacement revocation is not solved; the residual window is bounded by the 12-hour token expiry.
- The 64 KB file size is a compaction trigger, not a cap: live revocations are never evicted. Compaction runs at most once per 300 s, prunes expired/corrupt/duplicate lines, and skips no-op rewrites. The file is read as UTF-8 with replacement, so undecodable bytes corrupt only their own line.

### 6.4 Headers and content security policy

Every response carries `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store`, and `Strict-Transport-Security: max-age=15552000; includeSubDomains`. The CSP is enforcing when `RENDERELLO_CSP_ENFORCE=1` (set on the deployment) and report-only otherwise:

```
default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline';
script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'
```

`img-src` allows `blob:` for client-side upload thumbnails and `data:` for inline previews. `style-src 'unsafe-inline'` covers inline component styles. The full authenticated UI flow (login, presets, covers, upload previews, result images, downloads) was exercised locally in a real browser with JavaScript enabled, using a synthetic generated result and its download: zero violations. On the live deployment, bounded checks (login, presets, page responses, CSP headers) showed zero violations; a full paid generation flow was not run live for CSP testing.

### 6.5 Concurrency, storage and input limits

- `RENDERELLO_MAX_RUNNING_JOBS=1`: one generation job at a time on the free instance. Two full in-process generations exceeded the 512 MB free instance on 21 November 2026 (OOM restart), so missing or out-of-range values fail safe to 1; only an explicit value 2 to 8 opts into concurrency on hosts with the headroom. The worker pool is sized from the same value, so admission and pool stay consistent.
- The job registry keeps at most 200 entries and trims finished ones; jobs run on a bounded persistent thread pool (measured RSS growth from thread-per-job under burst load motivated the pool; pool threads are non-daemon so process exit waits for a running job rather than abandoning it mid-write).
- Request body caps on login (4 KB) and generate (file caps plus 1 MB form overhead) routes.
- Uploads: magic-byte sniffed, 12 MB cap each, 8 files per request, per-run temp dir removed when the job ends; upload contents are never logged.
- Generated files live under `RENDERELLO_RUNS` on instance storage, which is ephemeral on the free host: a redeploy or instance replacement clears past results. A janitor thread deletes run directories older than 24 hours. Downloaded variants and notebook outputs are the durable copies.
- Error hygiene: only messages with known-safe prefixes reach the client; everything else is the generic "generation failed".

### 6.6 Environment variables

| Variable | Purpose |
|----------|---------|
| `RENDERELLO_LOGIN_EMAIL`, `RENDERELLO_LOGIN_PASSWORD_HASH` | Password-mode credentials (hash only) |
| `RENDERELLO_ACCESS_CODE` | Legacy code-mode fallback |
| `RENDERELLO_SESSION_SECRET` | Signs stateless session tokens (32+ random chars) |
| `RENDERELLO_COOKIE_SECURE` | Secure cookie flag (default 1) |
| `RENDERELLO_CSP_ENFORCE` | 1 sends the enforcing CSP; unset sends report-only |
| `RENDERELLO_MAX_RUNNING_JOBS` | Concurrent generation cap (1 on the free host) |
| `RENDERELLO_RUNS`, `RENDERELLO_DENYLIST_PATH` | Storage locations |
| `MAGIC_HOUR_API_KEY` | Enables the Magic Hour engine |
| `RENDERELLO_MH_LIVE`, `RENDERELLO_MH_BUDGET_CREDITS`, `RENDERELLO_MH_MAX_BATCH_CREDITS`, `RENDERELLO_MH_CAP_ANCHOR`, `RENDERELLO_MH_SPENT_BASELINE`, `RENDERELLO_MH_DURABLE_LEDGER`, `RENDERELLO_MH_LEDGER_PROJECT`, `RENDERELLO_MH_LEDGER_TOKEN` | Magic Hour budget, batch and durable-ledger controls |

Secrets live only in the hosting dashboard environment, never in source.

---

## 7. Pipeline internals

### 7.1 Intake (renderello/image_intake.py)

Opt-in image intake, not automatic acceptance. Original bytes are never rewritten. Reads are bounded; dimensions are checked; ICC profiles are honoured through Pillow's ImageCms, and colour-changing conversions require explicit policy. Unknown profiles and high-bit-depth images fail rather than being silently truncated. Image identity, material and print readability still need human review.

### 7.2 Cutout (renderello/cutout.py)

Subject cutout with an explicit backend order: **BiRefNet** (MIT license, the default) → **RMBG-2.0** (restricted license, only when explicitly enabled) → **OpenCV GrabCut** (CPU fallback, no download). Model backends lazy-load on the resolved device. The GrabCut fallback keeps the whole pipeline runnable and testable on CPU with zero downloads. Matte quality differs by backend and is recorded on the `CutoutResult` for QA honesty. A supplied-mask path exists for externally prepared masks, and the bounding box always contains every nonzero alpha pixel so soft edges are not silently discarded by later cropping.

### 7.3 Matting (renderello/matting.py)

Opt-in trimap matting and foreground recovery, not automatic segmentation. Closed-form and learning-based matting (CF/LKM) plus multi-level foreground estimation use the official MIT PyMatting 1.1.16; the guided filter follows He, Sun and Tang (ECCV 2010). No network access. Worker guards bound a refinement job to 1.5 GB RSS and 90 seconds. This stage cannot recover a missing semantic object or pose.

### 7.4 Compositing (renderello/compose.py)

Deterministic compositing, CPU-fast, no models:

1. Procedural background from the template's background hint.
2. Product placement scaled to the template's framing, on the aspect canvas.
3. Synthetic contact shadow matched to the template's shadow spec.
4. Alpha blend where product pixels are pasted, never resampled twice, so the untransformed interior stays bit-identical to the source cutout.

For generated scenes, `composite_with_scene` places the cutout over the provider's scene using the same contact-shadow language, with a reference path that keeps the verified product pixels source-exact. This module is the fidelity-safe path the benchmark uses to separate model quality from pipeline correctness.

### 7.5 Scene providers (renderello/scene_providers.py + backends/)

Every provider declares cost and limits honestly; an unavailable provider says so and nothing silently falls back to a weaker one:

| Provider | Cost | Availability | Note |
|----------|------|--------------|------|
| `procedural` | free | always (CPU) | Not photoreal-AI quality |
| `sdxl_lightning` | free | needs a GPU (Colab/Kaggle) | AI-quality empty scenes |
| `hf_inference` | free tier | needs an HF token | rate-limited hosted API |
| `magichour` | paid credits | needs API key, `RENDERELLO_MH_LIVE=1`, and an authorized budget | guarded, see section 8 |

### 7.6 Templates and prompts (renderello/presets.py, prompts.py, longprompt.py)

Twelve researched shoot templates ship in the preset library:

`clean_studio`, `luxury_hero`, `editorial_beauty`, `lifestyle_kitchen`, `natural_outdoor`, `minimal_tabletop`, `seasonal_holiday`, `social_ad`, `reflective_surface`, `flat_lay`, `macro_detail`, `hero_negative_space`.

Each preset carries a detailed commercial-brief prompt, a template spec (aspect, framing, shadow language) and per-template knobs (aspect, variants, shadow strength, props). `build_detailed_prompt` composes preset, product description, material, free instruction and reference-style fragment into the full prompt; `longprompt.plan_prompt` supports long instructions through chunked embedding on the GPU path, and `prompts.build_negative_prompt` supplies the negative prompt. Template cover images in the picker are illustrative style previews, not sample outputs.

### 7.7 Quality metrics (renderello/qa.py)

Automated fidelity QA runs inside the aligned product region only; background differences are expected and never penalized. Deterministic outputs embed the source crop at a recorded offset (crop-vs-crop comparison); generative routes are re-segmented and aligned by bounding box, which is approximate and flagged as such. Metrics: masked SSIM, masked L1, colour drift, edge retention, plus critical-error checks and matte sanity. `fidelity_score` maps these to the brief's 1 to 5 rubric. These scores are automated proxies: human review remains the acceptance gate (section 11).

---

## 8. Magic Hour integration and spend controls

- The Magic Hour engine is optional and paid. The client never holds the API key; the server holds it, estimates cost before any call (`/api/estimate`), requires an explicit typed spend confirmation per run in the notebook flow, and records spend against a durable 1,000-credit budget with a server-side ledger.
- The verified live route is **scene-compose**: background-only generation plus the preserved product cutout. One verified live run (21 November 2026) charged exactly 30 credits for one image.
- **Mock by default.** Without `live=True` and `MAGIC_HOUR_API_KEY`, every backend call is served by an offline deterministic mock and records zero credits. The key is read from the environment per call and never written to disk or logs.
- **Model table** (edit-capable models, from the Magic Hour docs as fetched 14 and 20 November 2026; listed rates are single-image lowest-resolution "from" rates, and actual charge varies with resolution and image count): flux-2-klein 5, qwen-edit 10, krea-2 10, qwen-image-2.1 10, seedream-v4 40, seedream-v4.5 50, seedream-v5-flash 30, nano-banana family 50, seedream-v5-pro 75, gpt-image-2 50, gpt-image-2.5-flare 100, nano-banana-pro 150 credits per image. Free-tier models are 640 px only. A source-only `gpt-image-2.5-flare-1k` entry (200 credits, from 13 November receipts) is not on the live allowlist.
- **Routes**: `whole` (whole-image editor call; reconstructs the product, fidelity risk), `scene` (background-only generation plus source-exact composite; the fidelity-safe live route), `hybrid` (scene plus one harmonize call per variant). The editor API has no mask input (schema-verified), so mask-constrained edits are not available through this backend.
- **Budget enforcement** (`studio._enforce_budget` and friends): the whole batch must fit before any upload work or spend. Reserve before any paid call; a failed or unreachable reserve fails closed. After the call, reconcile moves the reservation to the observed charge, down or up: the listed price is a floor, not a ceiling, and an overcharge is real spend.
- **Durable ledger** (`renderello/mh_ledger.py`): the on-host JSONL ledger lives on ephemeral storage and resets on redeploy, so cumulative committed spend is also stored as a single JSON document in an isolated, ledger-only private GitLab project with no deployment hook, updated with strict compare-and-swap through the Repository Files API (GET for content plus last commit id, PUT naming that id, bounded 409 retry, six attempts). The hard cap is additionally anchored in the live account balance (`RENDERELLO_MH_CAP_ANCHOR`), so it holds across deploys.
- Template cover images in the picker are illustrative style previews, not sample outputs.

---

## 9. Colab notebook

`Renderello_Studio.ipynb` (185 code cells) is self-contained: dependency check, pipeline source, preset library, provider configuration, batch generation, review and manifest tooling. `scripts/build_colab_notebook.py` builds it by embedding every package module inline (one cell writes all modules), so it runs on a fresh runtime with no repo checkout. The dependency-check cell may restart the kernel on a fresh runtime; rerunning all cells afterward is expected, and Colab VM migrations or further restarts can require repeating Run all more than once.

The shipped notebook configuration cell selects the free `sdxl_lightning` engine (GPU on Colab); `procedural` (free, CPU) and `magichour` (dry-run mock in the notebook) are the explicit alternatives. The web studio's generate form defaults to `procedural`. The Magic Hour bridge ships disabled (`bridgeEnabled = False`) and turns on only per session with sign-in through the hosted bridge and a per-run spend confirmation.

Colab's free runtime terms (FAQ, verified 20 November 2026) disallow remote proxies and serving content-generation web UIs, so the notebook is interactive only: no tunnel backend exists or may be added.

The root notebook ships clean: no saved outputs and no execution counts. An executed copy with retained outputs from the first verified Magic Hour run is kept at `docs/Renderello_Studio_LIVE_magichour_20261121.ipynb`; its execution counts are not recorded, so it evidences the run without proving a sequential full run-all.

---

## 10. Dependencies and tech stack

| Area | Choice |
|------|--------|
| Runtime | Python 3.14 (web app), Colab CPU/GPU runtimes (notebook) |
| Web stack | FastAPI 0.165, uvicorn 0.54, vanilla JS frontend, Pillow 13 |
| Imaging | numpy 2.5, opencv-python-headless 5.1, scikit-image 0.26, onnxruntime 1.32 |
| AI providers | Procedural engine (built in), Magic Hour API (optional, paid), SDXL Lightning config (notebook) |
| Matting / compositing | Custom pipeline with PyMatting 1.1.16 (MIT) for CF/LKM, guided filter per He/Sun/Tang ECCV 2010 |
| Cutout | BiRefNet (MIT, default), RMBG-2.0 (restricted, opt-in), OpenCV GrabCut (fallback) |
| Hosting | Render free web service, Google Colab |

`requirements.txt` pins 38 packages with exact versions, locked to the resolved set of the deployed build (Python 3.14). This is version-set equality with the deployed build's install log, not a bit-for-bit reproducibility guarantee. GPU and model dependencies (torch, diffusers, transformers, accelerate, safetensors) are deliberately commented out of the web build and installed on the GPU host only.

---

## 11. Human review and evaluation status

Human review is the acceptance gate for output quality. Automated metrics and development-side scores are proxies and preparation, not acceptance.

Current state, honestly:

- The seven-dimension review sheet (`HUMAN_REVIEW.md`) is prepared tooling: export, fill, and check steps exist, with provenance rules that keep development-side scores from being counted as human evidence.
- Anonymized human review packs (`HUMAN_REVIEW_PACKS.md`) prepare candidates for a human rater without revealing which method made each image. The underlying code modules keep their original `blind_review` names.
- The scores that exist today are development-side judgments (rater `agent_eye`) over the 20 baseline outputs, awaiting human review; two outputs were tentative 4/5 usefulness candidates. They are not independent human ratings.
- The brief's 80 percent no-critical-alteration and 70 percent publication-usefulness gates remain unestablished, as do a completed matched comparison against two leading specialist tools, the template polish targets, and a verified tiny-text fidelity case. See `outputs/BRIEF_COVERAGE.csv` for the full requirement-by-requirement status.

---

## 12. Testing

At the 21 November 2026 checkpoint: 3,913 tests passed, 6 skipped, 0 failed on the pinned dependency set (Python 3.14), covering pipeline units, web routes, authentication, logout revocation (including corrupt-file and write-failure cases), budget guards and compositing contracts. This is a regression result for the tested code paths, not a photo-quality score or a production load test. The full checklist, including how to reproduce the count and what to run before any change ships, is in `TESTING.md`.

---

## 13. Deployment

- Render free web service, no card, auto-deploys on push to the connected branch (observed on the 21 November 2026 deploys). Build and start commands live in the service dashboard, not in the repo. Start command: `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT`.
- **Run single process only** (no `--workers`): the job registry, the single-live-Magic-Hour-job rule and the budget reservation lock are per-process by design.
- All runtime configuration is dashboard environment variables (see 6.6). No credentials in source.
- Rollback is redeploy-based from known-good commits; see `ROLLBACK.md`.

---

## 14. Known limitations, fragile areas and safe first changes

Known limitations:

- Free-host reliability over long sessions is unproven; the one-job cap reduces risk but does not guarantee availability.
- Generated results on the web app are ephemeral (instance storage, 24 h janitor).
- Product preservation covers fully opaque pixels only; edge and mask quality needs human review.
- The global login breaker trades per-attacker attribution for honest accounting: sustained attack traffic can temporarily slow legitimate logins, and counters reset on process restart.
- Cross-instance revocation of logged-out sessions is bounded by token expiry, not solved by the denylist file.
- A long local stress test shows memory growth (about 160 MB to 700 MB over roughly 2,000 generations); the cause is not yet found.

Fragile areas (read before editing):

- The job slot claim in `generate` must stay before any await or spend, or concurrent requests can overspend.
- The denylist append and compaction are serialized by a dedicated IO lock; bypassing it can lose revocations.
- `_MAX_RUNNING_JOBS`, the executor size and the budget reservation all assume one process; a second worker silently breaks all three.
- The Magic Hour listed price is a floor; removing the reconcile step under-records real spend.
- `compose` relies on paste-without-resample for bit-exactness; adding an intermediate transform voids the preservation contract.

Safe first changes: frontend copy and styles in `webapp/static/`; new templates in `presets.py` (they are data, and the suite checks the count); additional read-only provider diagnostics; new tests.

---

## 15. Development workflow

This project follows the documentation habits in `HANDOVER.md`, `DECISIONS.md`, `FLOW.md`, `CONSTRAINTS.md`, `TESTING.md` and `ROLLBACK.md`: current state is written down, decisions carry their reasoning, execution paths are traced, boundaries are explicit, verification is a checklist of real commands, and every change has a way back. Reviews read the actual diff, requests stay small, and each session ends with a short handoff note. The full build history and evidence (8 to 21 November 2026) is in `docs/Renderello-Complete-Methods-and-Evidence-Reading.pdf` with editable sources in `docs/report-sources/`.
