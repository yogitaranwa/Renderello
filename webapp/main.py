"""Renderello web app - FastAPI backend serving the modular frontend and the
generate API. The SAME studio pipeline powers this app and the Colab
notebook (renderello.studio.run_studio).

Security model (owner rules, 2026-11-20):
- Every page and API sits behind an access code. The code lives ONLY in the
  RENDERELLO_ACCESS_CODE environment variable (Render dashboard secret),
  never in source, Drive, logs or URLs. Sessions are HMAC-signed cookies.
- Uploads are validated by magic bytes (PNG/JPEG/WebP), size-capped, and
  stored in a per-run temp dir that a janitor deletes after 24 h. Upload
  contents are never logged.
- No secrets in URLs: runs are addressed by random ids, files are served
  only through the authenticated endpoint with path-traversal checks.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import (FastAPI, File, Form, HTTPException, Request, Response,
                     UploadFile)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from PIL import Image as _PILImage

# Decompression-bomb guards: Pillow hard-limit at 40M pixels (raises), plus
# explicit decoded-size caps verified in _decodable (25M pixels, 8000/side).
_PILImage.MAX_IMAGE_PIXELS = 40_000_000
_MAX_PIXELS_EXPLICIT = 25_000_000
_MAX_SIDE = 8000

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # webapp/ itself, for `import auth`

import auth                                          # noqa: E402
from renderello.presets import PRESETS, presets_for_ui            # noqa: E402
from renderello.scene_providers import provider_report            # noqa: E402
from renderello.studio import StudioRequest, run_studio           # noqa: E402

RUNS_ROOT = Path(os.environ.get("RENDERELLO_RUNS", "/tmp/renderello-runs"))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_FILES = 8
SESSION_HOURS = 12
_MAGIC = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpg"}
_MAX_TEXT_FIELD = 2000
_MAX_KNOBS = 40
_MAX_LOGIN_BODY = 4096
def _max_running_jobs() -> int:
    # Concurrent generation jobs admitted at once. Two full in-process
    # generations exceeded the 512 MB free instance on 2026-11-21 (OOM
    # restart), so small hosts set RENDERELLO_MAX_RUNNING_JOBS=1. Missing,
    # unparsable, or out-of-range values fail SAFE to 1 (serialized jobs);
    # only an explicit in-range value 2-8 opts into concurrency, which is
    # an operator choice for hosts with the memory headroom, not a
    # free-tier-safe claim. The worker pool is sized from this same value,
    # so admission and pool stay consistent.
    raw = os.environ.get("RENDERELLO_MAX_RUNNING_JOBS", "").strip()
    try:
        n = int(raw)
    except ValueError:
        return 1
    return n if 1 <= n <= 8 else 1


_MAX_RUNNING_JOBS = _max_running_jobs()
_MAX_JOBS_KEPT = 200
_RUN_RE = __import__("re").compile(r"[A-Za-z0-9_-]{1,64}")

_sess_secret = os.environ.get("RENDERELLO_SESSION_SECRET", "")
if _sess_secret and len(_sess_secret) < 32:
    raise RuntimeError("RENDERELLO_SESSION_SECRET must be >= 32 random chars")

app = FastAPI(title="Renderello", docs_url=None, redoc_url=None, openapi_url=None)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
# Persistent job workers. Measured locally under a mixed sequential+burst
# workload, a fresh OS thread per job showed step-wise RSS growth while this
# bounded pool stayed flat (see tests/test_job_worker_pool.py); the retention
# mechanism is uncertain (consistent with per-thread allocator arenas, not
# proven). Admission still caps running jobs at _MAX_RUNNING_JOBS, so the
# pool never queues beyond it and concurrency/admission behavior is the same.
# Pool threads are non-daemon: process exit now waits for a running job
# instead of abandoning it mid-write.
_executor = ThreadPoolExecutor(max_workers=_MAX_RUNNING_JOBS,
                               thread_name_prefix="renderello-job")


def _authed(request: Request) -> bool:
    return auth.valid_session(request.cookies.get("renderello_session"))


def _require_auth(request: Request) -> None:
    if not _authed(request):
        raise HTTPException(status_code=401, detail="Your session ended or the server restarted. Sign in again to continue.")


# Upper bound for one /api/generate request body: MAX_FILES files of
# MAX_FILE_BYTES plus form overhead.
_MAX_GENERATE_BODY = MAX_FILES * MAX_FILE_BYTES + 1024 * 1024


class BodyLimitMiddleware:
    """Streaming request-body cap (counts actual bytes, so it also bounds
    chunked requests that send no Content-Length). Applies to the paths in
    `limits`; everything else passes through untouched. When the cap is hit
    it answers 413 itself and tells the app the client went away, so no
    framework error handler can turn it into a 500."""

    def __init__(self, app, limits: dict[str, int]):
        self.app = app
        self.limits = limits

    async def __call__(self, scope, receive, send):
        limit = self.limits.get(scope.get("path")) \
            if scope["type"] == "http" and scope.get("method") == "POST" else None
        if limit is None:
            return await self.app(scope, receive, send)
        seen = 0
        started = False
        rejected = False

        async def counted():
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    rejected = True
                    if not started:
                        await JSONResponse({"detail": "request too large"},
                                           status_code=413)(scope, receive, send)
                    return {"type": "http.disconnect"}
            return msg

        async def tracked(msg):
            nonlocal started
            if rejected:
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counted, tracked)
        except Exception:
            if not rejected:
                raise


@app.middleware("http")
async def generate_gate(request: Request, call_next):
    """Reject unauthenticated or oversized /api/generate requests BEFORE the
    multipart body is parsed (FastAPI parses form/file parameters before the
    handler runs, so the handler's own auth check comes too late)."""
    if request.method == "POST" and request.url.path == "/api/generate":
        if not _authed(request):
            return JSONResponse({"detail": "Your session ended or the server "
                "restarted. Sign in again to continue."}, status_code=401)
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "bad request"}, status_code=400)
        if length > _MAX_GENERATE_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
    return await call_next(request)


app.add_middleware(BodyLimitMiddleware, limits={
    "/api/generate": _MAX_GENERATE_BODY, "/api/login": _MAX_LOGIN_BODY + 1024})


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Strict-Transport-Security"] = "max-age=15552000; includeSubDomains"
    # Report-only while the frontend evolves: violations are observable in the
    # browser console without breaking the UI. Promote to enforcing after the
    # frontend settles (inline styles/scripts must be gone first).
    csp_header = ("Content-Security-Policy"
                  if os.environ.get("RENDERELLO_CSP_ENFORCE") == "1"
                  else "Content-Security-Policy-Report-Only")
    resp.headers[csp_header] = (
        "default-src 'self'; img-src 'self' blob: data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
    return resp


@app.get("/api/login-mode")
async def login_mode():
    # Unauthenticated: tells the login page which form to render.
    return {"mode": auth.mode()}


@app.post("/api/login")
async def login(request: Request):
    raw = await request.body()
    if len(raw) > _MAX_LOGIN_BODY:
        raise HTTPException(413, "request too large")
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(400, "bad request")
    if not isinstance(body, dict):
        raise HTTPException(400, "bad request")
    slots: list[tuple[str, float]] = []
    try:
        # Global circuit breaker, honestly named. Static configuration
        # plus Render's documented ingress topology (Cloudflare edge ->
        # Render load balancer -> instance) strongly indicate the
        # connection peer is the platform LB for every request: the start
        # command sets no proxy flags, deployed uvicorn 0.54.0 defaults
        # forwarded_allow_ips to loopback only, and FORWARDED_ALLOW_IPS is
        # not set. request.client.host on a live request was NOT observed;
        # if the inference holds, an IP-keyed bucket was already one
        # shared bucket service-wide. A trusted-proxy fix has no safe
        # basis: Render LB addresses are dynamic and unpublished, and
        # trusting '*' makes X-Forwarded-For client-spoofable (uvicorn
        # then picks the leftmost entry). So no per-IP key is used at
        # all: this bucket bounds the total login attempt rate, and the
        # email bucket below bounds per-account guessing. Same thresholds
        # and window as before; failed attempts keep their slot (no
        # fail-open), successes refund only their own.
        slots.append(("global", auth.check_rate("global")))
        if auth.mode() == "password":
            ekey = "email:" + str(body.get("email", ""))[:254].lower()
            slots.append((ekey, auth.check_rate(ekey)))
    except RuntimeError:
        raise HTTPException(429, "too many attempts; wait 10 minutes")
    if auth.mode() == "password":
        email = str(body.get("email", ""))
        password = str(body.get("password", ""))
        if not auth.verify_password_login(email, password):
            raise HTTPException(401, "wrong email or password")
    else:
        code = str(body.get("code", ""))
        if not os.environ.get("RENDERELLO_ACCESS_CODE"):
            raise HTTPException(503, "server login not configured")
        if not auth.verify_code_login(code):
            raise HTTPException(401, "wrong code")
    # Good login: give the reserved slots back (failures and in-flight
    # guesses keep theirs, so brute-force limits are unchanged).
    for key, slot in slots:
        auth.refund_rate(key, slot)
    token = auth.new_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie("renderello_session", token,
                    max_age=auth.SESSION_HOURS * 3600,
                    httponly=True, samesite="lax",
                    secure=os.environ.get("RENDERELLO_COOKIE_SECURE", "1") == "1")
    return resp


@app.post("/api/logout")
async def logout(request: Request):
    auth.drop_session(request.cookies.get("renderello_session"))
    response = JSONResponse({"ok": True})
    response.delete_cookie("renderello_session", httponly=True, samesite="lax",
                           secure=os.environ.get("RENDERELLO_COOKIE_SECURE", "1") == "1")
    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not _authed(request):
        return (STATIC / "login.html").read_text()
    return (STATIC / "index.html").read_text()


# Login-page assets must load before the user has a session. The favicon is
# a static brand icon, safe to serve pre-auth.
_PUBLIC_STATIC = {"js/login.js", "css/login.css", "favicon.svg",
                  "favicon-32.png", "favicon.ico"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    # Browsers that ignore the <link> tags still probe this path; a real ICO
    # (16+32 px) keeps older/strict browsers covered.
    return FileResponse(STATIC / "favicon.ico", media_type="image/x-icon")


@app.get("/static/{path:path}")
async def static_files(request: Request, path: str):
    if path not in _PUBLIC_STATIC:
        _require_auth(request)
    try:
        target = (STATIC / path).resolve()
        ok = target.is_relative_to(STATIC.resolve()) and target.is_file()
    except (ValueError, OSError):
        ok = False
    if not ok:
        raise HTTPException(404)
    return FileResponse(target)


@app.get("/api/presets")
async def presets(request: Request):
    _require_auth(request)
    return {"presets": presets_for_ui()}


@app.get("/api/providers")
async def providers(request: Request):
    _require_auth(request)
    return {"providers": provider_report()}


def _sniff(data: bytes) -> str | None:
    for magic, ext in _MAGIC.items():
        if data.startswith(magic):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _decodable(path: Path) -> bool:
    try:
        with _PILImage.open(path) as im:
            im.verify()
            w, h = im.size
        return w * h <= _MAX_PIXELS_EXPLICIT and max(w, h) <= _MAX_SIDE
    except Exception:
        return False


MAX_PRESETS_PER_BATCH = 8


@app.get("/api/estimate")
async def estimate(request: Request, provider: str = "procedural",
                   mh_model: str = "seedream-v5-flash", variants: int = 4,
                   presets: str = "", mh_route: str = "scene",
                   knob_scene_max_attempts: str = "1"):
    """Cost preview BEFORE Generate. The frontend calls this on every
    selection change and shows the result next to the button. Procedural
    runs are free; Magic Hour runs report listed per-image credits x worst
    case images (variants x scene attempts), the budget state and whether
    the batch fits. Never spends anything."""
    _require_auth(request)
    from renderello.studio import budget_status, scene_attempts_from
    from renderello.backends.magichour import MODEL_TABLE
    n_presets = max(1, min(len([p for p in presets.split(",") if p]),
                           MAX_PRESETS_PER_BATCH))
    n_variants = max(1, min(int(variants), 4))
    if provider != "magichour":
        return {"provider": provider, "credits": 0, "free": True,
                "images": n_presets * n_variants}
    if mh_model not in MODEL_TABLE:
        raise HTTPException(400, f"unknown Magic Hour model {mh_model!r}")
    b = budget_status(mh_model)
    per_image = b["credits_per_image"]
    attempts = scene_attempts_from({"scene_max_attempts": knob_scene_max_attempts})
    if mh_route not in ("scene", "whole", "hybrid"):
        raise HTTPException(400, f"unknown Magic Hour route {mh_route!r}")
    calls_per_image = 1 if mh_route == "whole" else attempts + int(mh_route == "hybrid")
    images = n_presets * n_variants
    worst = images * calls_per_image * (per_image or 0)
    return {"provider": provider, "model": mh_model, "route": mh_route,
            "credits_per_image": per_image, "images": images,
            "scene_max_attempts": attempts, "calls_per_image": calls_per_image,
            "credits_expected": images * (per_image or 0),
            "credits_worst_case": worst,
            "budget": b, "within_budget": worst <= b["remaining"],
            "within_batch_limit": worst <= b["max_batch"]}


@app.post("/api/generate")
async def generate(request: Request,
                   products: list[UploadFile] = File(...),
                   references: list[UploadFile] = File(default=[]),
                   preset: list[str] = Form(...),
                   provider: str = Form("procedural"),
                   instruction: str = Form(""),
                   product_description: str = Form(""),
                   material: str = Form(""),
                   aspect: str = Form(""),
                   variants: int = Form(4),
                   mh_model: str = Form("seedream-v5-flash"),
                   mh_route: str = Form("scene")):
    _require_auth(request)
    presets = []
    for p in preset:
        presets.extend(x.strip() for x in p.split(",") if x.strip())
    presets = list(dict.fromkeys(presets))  # de-dupe, keep order
    if not presets or len(presets) > MAX_PRESETS_PER_BATCH:
        raise HTTPException(400, f"select 1-{MAX_PRESETS_PER_BATCH} templates")
    for p in presets:
        if p not in PRESETS:
            raise HTTPException(400, f"unknown preset {p!r}")
    if not products or len(products) + len(references) > MAX_FILES:
        raise HTTPException(400, "1-8 files: at least one product photo")
    if mh_route not in ("scene", "whole", "hybrid"):
        raise HTTPException(400, f"unknown Magic Hour route {mh_route!r}")
    for val in (instruction, product_description, material, aspect, provider,
                mh_model, mh_route):
        if len(val) > _MAX_TEXT_FIELD:
            raise HTTPException(400, "text field too long")
    variants = max(1, min(int(variants), 4))
    # Claim the job slot atomically BEFORE any await/upload/spend work, so
    # concurrent requests cannot all pass the running-job checks.
    job_id = secrets.token_hex(8)
    with _lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
        if len(running) >= _MAX_RUNNING_JOBS:
            raise HTTPException(429, "too many jobs running; try again shortly")
        if provider == "magichour" and any(j.get("mh") for j in running):
            raise HTTPException(429, "one live Magic Hour job at a time; "
                                "wait for the current run to finish")
        if len(_jobs) >= _MAX_JOBS_KEPT:
            for k in [k for k, j in _jobs.items()
                      if j["status"] != "running"][:50]:
                _jobs.pop(k, None)
        _jobs[job_id] = {"status": "running", "run_dir": None, "error": None,
                         "mh": provider == "magichour"}
    try:
      return await _generate_claimed(request, job_id, products, references,
          presets, provider, instruction, product_description, material,
          aspect, variants, mh_model, mh_route)
    except BaseException:
        with _lock:
            _jobs.pop(job_id, None)
        raise


async def _generate_claimed(request, job_id, products, references, presets,
                            provider, instruction, product_description,
                            material, aspect, variants, mh_model, mh_route="scene"):
    if provider == "magichour":
        from renderello.backends.magichour import MODEL_TABLE
        if mh_model not in MODEL_TABLE:
            raise HTTPException(400, f"unknown Magic Hour model {mh_model!r}")
        # Fail fast before any upload work or spend: the WHOLE batch
        # (templates x variants x scene attempts, +1 harmonize call per
        # variant on the hybrid route) must fit the budget.
        from renderello.studio import _enforce_budget, scene_attempts_from
        try:
            attempts = scene_attempts_from(
                {k[5:]: v for k, v in (await request.form()).items()
                 if k.startswith("knob_")})
            _enforce_budget(mh_model,
                            len(presets) * max(1, min(variants, 4)) *
                            (1 if mh_route == "whole" else attempts + int(mh_route == "hybrid")))
        except RuntimeError as e:
            raise HTTPException(402, str(e))

    run_id = secrets.token_hex(6)
    work = RUNS_ROOT / f"up-{run_id}"
    work.mkdir(parents=True)
    product_paths, reference_paths = [], []
    try:
        for group, dest in ((products, product_paths), (references, reference_paths)):
            for up in group:
                data = await up.read()
                if len(data) > MAX_FILE_BYTES:
                    raise HTTPException(413, "file exceeds 12 MB")
                ext = _sniff(data)
                if not ext:
                    raise HTTPException(415, "file is not a PNG/JPEG/WebP image")
                p = work / f"{dest is product_paths and 'p' or 'r'}{len(dest)}.{ext}"
                p.write_bytes(data)
                if not _decodable(p):
                    raise HTTPException(415, "image could not be decoded or is too large")
                dest.append(p)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise

    knobs = {"mh_model": mh_model}
    form = await request.form()
    for k, v in form.items():
        if k.startswith("knob_") and isinstance(v, str) and len(v) <= 200 \
                and len(knobs) < _MAX_KNOBS:
            knobs[k[5:][:60]] = v
    def _work():
        run_dirs: list[str] = []
        try:
            from renderello.resources import memoryUsage
            for preset_key in presets:
                req = StudioRequest(
                    product_paths=product_paths, preset_key=preset_key,
                    instruction=instruction, reference_paths=reference_paths,
                    variants=variants, aspect=aspect or None, provider=provider,
                    mh_route=mh_route,
                    material=material or None,
                    product_description=product_description, knob_values=knobs)
                result = run_studio(req, RUNS_ROOT)
                manifestPath = result.run_dir / "manifest.json"
                manifestData = json.loads(manifestPath.read_text())
                manifestData["resources"] = memoryUsage()
                manifestPath.write_text(json.dumps(manifestData, indent=2))
                run_dirs.append(result.run_dir.name)
                with _lock:
                    _jobs[job_id] = {"status": "running", "run_dir": run_dirs[0],
                                     "run_dirs": list(run_dirs), "error": None,
                                     "mh": provider == "magichour"}
            with _lock:
                _jobs[job_id] = {"status": "done", "run_dir": run_dirs[0],
                                 "run_dirs": list(run_dirs), "error": None}
        except Exception as e:
            with _lock:
                _jobs[job_id] = {"status": "failed",
                                 "run_dir": run_dirs[0] if run_dirs else None,
                                 "run_dirs": list(run_dirs),
                                 "error": _job_error(e)}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    try:
        _executor.submit(_work)
    except BaseException:
        # Submission failed (e.g. executor shut down): the job body will never
        # run its own cleanup, so drop the temp upload dir here. The caller
        # releases the claimed job slot.
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def job_status(request: Request, job_id: str):
    _require_auth(request)
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    if job["status"] != "done":
        return job
    run_dirs = job.get("run_dirs") or ([job["run_dir"]] if job.get("run_dir") else [])
    runs = []
    for rd in run_dirs:
        mpath = RUNS_ROOT / rd / "manifest.json"
        if not mpath.is_file():
            raise HTTPException(410, "run files expired (24 h cleanup); generate again")
        m = json.loads(mpath.read_text())
        m["run_dir"] = rd
        runs.append({"preset": m.get("preset"), "run_dir": rd, "manifest": m})
    out = {**job, "runs": runs}
    if runs:
        # Back-compat: single-template clients read top-level "manifest".
        out["manifest"] = runs[0]["manifest"]
    return out


@app.get("/api/runs/{run_dir}/{name}")
async def run_file(request: Request, run_dir: str, name: str):
    _require_auth(request)
    if not _RUN_RE.fullmatch(run_dir) or "/" in name or "\\" in name \
            or name.startswith("."):
        raise HTTPException(404)
    root = RUNS_ROOT.resolve()
    try:
        base = (root / run_dir).resolve()
        target = (base / name).resolve()
        ok = (base.parent == root and target.is_relative_to(base)
              and target.is_file())
    except (ValueError, OSError):   # NUL byte, over-long name, ...
        ok = False
    if not ok:
        raise HTTPException(404)
    return FileResponse(target)


def _janitor():
    while True:
        time.sleep(3600)
        cutoff = time.time() - 24 * 3600
        for d in RUNS_ROOT.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


_SAFE_ERROR_PREFIXES = (
    "Magic Hour", "batch too large", "unknown ", "provider unavailable",
    "no product cutout", "scene-compose", "RENDERELLO_",
    "could not verify live account balance", "live balance",
    "text field too long", "select 1-", "1-8 files",
)


def _job_error(e: Exception) -> str:
    if isinstance(e, RuntimeError) and str(e).startswith(_SAFE_ERROR_PREFIXES):
        return str(e)[:300]
    return "generation failed"


threading.Thread(target=_janitor, daemon=True).start()
