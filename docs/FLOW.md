# Flow

How execution actually travels through Renderello: the three request/run traces and the three data flows. File and function references are load-bearing; check them before changing a step.

## Trace 1: POST /api/generate

1. `webapp/main.py` middleware `generate_gate`: rejects unauthenticated or oversized `/api/generate` POSTs before multipart parsing.
2. `BodyLimitMiddleware.__call__`: streams-counts bytes; answers 413 itself over the cap.
3. `generate()` handler: session check, template validation against `PRESETS`, file-count cap, route validation, text-field caps, variant clamp.
4. Job slot claim under `_lock`: cap checks (`_MAX_RUNNING_JOBS`, one live Magic Hour job), registry trim at 200 entries. This must stay before any await or spend.
5. Magic Hour only: `_enforce_budget(model, whole_batch_calls)` refuses with 402 if the batch does not fit the budget.
6. Upload staging: per-run temp dir, 12 MB per file, magic-byte sniff (`_sniff`), Pillow verify (`_decodable`: 25M pixels, 8,000 px per side).
7. `_executor.submit(_work)`: bounded persistent pool sized from the running-job cap.
8. Per template: `renderello.studio.run_studio(StudioRequest, RUNS_ROOT)`.
9. Inside `run_studio`: preset and aspect resolution, reference-style merge, `cutout_products`, provider dispatch, per-variant prompt build (`build_detailed_prompt` + `plan_prompt`), scene generation, `compose.composite` (or `compose_group`), `qa.evaluate`, manifest and contact sheet write.
10. `GET /api/jobs/{job_id}` reads the manifest back; `GET /api/runs/{run_dir}/{name}` serves files with id-pattern and traversal checks.
11. Temp dir removed in the job's `finally`; the janitor thread deletes run directories older than 24 h.

## Trace 2: POST /api/login

1. `GET /api/login-mode` -> `auth.mode()`: `password` or legacy `code`.
2. Login handler: body cap, JSON parse, slot reservation via `auth.check_rate("global")` plus `auth.check_rate("email:<email>")` in password mode; HTTP 429 when a bucket is full.
3. `auth.verify_password_login`: constant-time email compare, PBKDF2-HMAC-SHA256 (600,000 iterations) verify that always runs.
4. Success refunds only its own slots (`auth.refund_rate`); failures keep theirs.
5. `auth.new_session()`: stateless `s1.<expiry>.<nonce>.<sig>` token when `RENDERELLO_SESSION_SECRET` is set, else a legacy in-memory random token.
6. Cookie set: 12 h, HttpOnly, SameSite=lax, Secure by default.
7. `POST /api/logout` -> `auth.drop_session`: records the token's SHA-256 id in the denylist (memory plus JSONL file, mode 0600); `auth.valid_session` consults it on every request.

## Trace 3: Colab notebook run

1. Dependency-check cell installs packages and may restart the kernel on a fresh runtime; repeat Run all afterward.
2. A single cell writes the full embedded package to disk (built by `scripts/build_colab_notebook.py`), so no repo checkout is needed.
3. Configuration cell selects the engine: `sdxl_lightning` (free, GPU) by default, `procedural` (free, CPU) or `magichour` (dry-run mock) as explicit alternatives.
4. Generation calls the same `run_studio` as the website. The Magic Hour bridge ships disabled (`bridgeEnabled = False`) and turns on per session through the hosted bridge with a typed per-run spend confirmation.
5. Outputs, manifests and contact sheets are written per run and downloadable.

## Data flow: a product photo

Upload (multipart) -> per-run temp dir (sniffed, decode-verified, never logged) -> `cutout_products` (intermediates in the run dir) -> verified cutout -> composited onto the scene -> variant PNG in the run dir -> manifest records hashes and QA -> served through the authenticated run-file route -> janitor deletes after 24 h. The durable copies are the ones downloaded or produced by the notebook. Original bytes are never rewritten.

## Data flow: a session token

Created at login (stateless `s1.` token signed with `RENDERELLO_SESSION_SECRET`) -> sent only as an HttpOnly cookie -> validated per request by signature, expiry and denylist membership -> on logout, only its SHA-256 id is persisted (JSONL, mode 0600, atomic compaction) -> expiry bounds every token at 12 h regardless of revocation durability.

## Data flow: a Magic Hour credit

`/api/estimate` shows the worst-case batch cost before Generate -> `_enforce_budget` verifies the whole batch fits -> `reserve_budget` reserves before any paid call (a failed reserve fails closed) -> the provider call runs -> `reconcile_budget` moves the reservation to the observed charge (up or down; listed price is a floor) -> spend is appended to the on-host JSONL ledger and compare-and-swap written to the durable ledger in an isolated private GitLab project -> the hard cap is anchored to the live account balance (`RENDERELLO_MH_CAP_ANCHOR`) so it holds across deploys.
