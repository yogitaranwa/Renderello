"""Web gate regressions: auth/size before upload parsing, MH one-job flag kept
across multi-template runs, login does equal hash work for any email."""
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))
import auth  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "code-12345")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    monkeypatch.setenv("RENDERELLO_RUNS", str(tmp_path))
    for k in ("RENDERELLO_LOGIN_EMAIL", "RENDERELLO_LOGIN_PASSWORD_HASH"):
        monkeypatch.delenv(k, raising=False)
    auth._attempts.clear()
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    r = c.post("/api/login", json={"code": "code-12345"})
    assert r.status_code == 200
    yield main, c
    auth._attempts.clear()


def test_unauth_generate_rejected_before_body_parsing(app_client):
    main, authed = app_client
    from fastapi.testclient import TestClient
    anon = TestClient(main.app)
    # Malformed body: with the old order FastAPI validated (422) first.
    assert anon.post("/api/generate", data={"preset": "x"}).status_code == 401


def _gate(main, content_length):
    import asyncio

    class R:
        method = "POST"
        url = SimpleNamespace(path="/api/generate")
        headers = {"content-length": content_length}
        cookies = {"renderello_session": auth.new_session()}

    async def nxt(_):
        raise AssertionError("must not reach handler")
    return asyncio.new_event_loop().run_until_complete(main.generate_gate(R, nxt))


def test_oversized_declared_body_rejected_when_authed(app_client):
    main, _ = app_client
    assert _gate(main, str(main._MAX_GENERATE_BODY + 1)).status_code == 413


def test_bad_content_length_is_400(app_client):
    main, _ = app_client
    assert _gate(main, "abc").status_code == 400


def test_normal_authed_upload_still_reaches_handler(app_client):
    main, c = app_client
    r = c.post("/api/generate", files=[("products", ("a.png", PNG, "image/png"))],
               data={"preset": "no-such-template"})
    assert r.status_code == 400  # reached the handler's own validation


def test_mh_flag_survives_progress_updates(app_client, monkeypatch, tmp_path):
    main, c = app_client
    import renderello.studio as studio
    monkeypatch.setattr(studio, "_enforce_budget", lambda *a, **k: None)
    gate = threading.Event()
    calls = []

    def fake_run(req, root):
        calls.append(req.preset_key)
        d = Path(root) / f"run-{len(calls)}"
        d.mkdir(exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({"preset": req.preset_key}))
        if len(calls) == 2:
            gate.wait(10)
        return SimpleNamespace(run_dir=d)
    monkeypatch.setattr(main, "run_studio", fake_run)
    presets = list(main.PRESETS)[:2]
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, "PNG")
    files = [("products", ("a.png", buf.getvalue(), "image/png"))]
    r = c.post("/api/generate", files=files,
               data={"preset": presets, "provider": "magichour",
                     "mh_route": "scene", "variants": "1"})
    assert r.status_code == 200, r.text
    # wait until preset 1 finished (progress update replaced the job dict)
    for _ in range(100):
        j = c.get(f"/api/jobs/{r.json()['job_id']}").json()
        if j.get("run_dirs"):
            break
        time.sleep(0.05)
    assert j["status"] == "running" and j.get("run_dirs")
    r2 = c.post("/api/generate", files=files,
                data={"preset": presets[0], "provider": "magichour",
                      "mh_route": "scene", "variants": "1"})
    gate.set()
    assert r2.status_code == 429, r2.text


def test_login_runs_hash_even_for_wrong_email(monkeypatch):
    monkeypatch.setenv("RENDERELLO_LOGIN_EMAIL", "admin@magichour.ai")
    monkeypatch.setenv("RENDERELLO_LOGIN_PASSWORD_HASH", auth.hash_password("pw-12345"))
    calls = []
    real = auth._verify_hash
    monkeypatch.setattr(auth, "_verify_hash",
                        lambda p, s: calls.append(1) or real(p, s))
    assert not auth.verify_password_login("nobody@example.com", "pw-12345")
    assert calls == [1]
    assert auth.verify_password_login("admin@magichour.ai", "pw-12345")
    assert not auth.verify_password_login("admin@magichour.ai", "bad")
    assert calls == [1, 1, 1]


# ---- streaming body cap (chunked / no Content-Length) ----------------------

def _chunks(total, size=1024):
    sent = 0
    while sent < total:
        n = min(size, total - sent)
        sent += n
        yield b"x" * n


def test_body_limit_counts_chunked_bodies_without_content_length(app_client):
    main, _ = app_client
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    async def echo(request):
        return PlainTextResponse(str(len(await request.body())))
    inner = Starlette(routes=[Route("/up", echo, methods=["POST"]),
                              Route("/free", echo, methods=["POST"])])
    c = TestClient(main.BodyLimitMiddleware(inner, {"/up": 4096}))
    assert c.post("/up", content=_chunks(4000)).text == "4000"
    r = c.post("/up", content=_chunks(10_000))      # chunked, no length header
    assert r.status_code == 413
    assert c.post("/free", content=_chunks(50_000)).text == "50000"  # other path untouched
    assert c.get("/up").status_code == 405           # GET unaffected


def test_login_body_cap_applies_to_chunked_unauth_requests(app_client):
    main, c = app_client
    from fastapi.testclient import TestClient
    anon = TestClient(main.app)
    r = anon.post("/api/login", content=_chunks(20_000),
                  headers={"content-type": "application/json"})
    assert r.status_code == 413
    ok = anon.post("/api/login", json={"code": "wrong"})
    assert ok.status_code == 401  # normal-size requests still handled


def test_generate_chunked_unauth_gets_401_not_parsed(app_client):
    main, _ = app_client
    from fastapi.testclient import TestClient
    anon = TestClient(main.app)
    r = anon.post("/api/generate", content=_chunks(200_000),
                  headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 401


def test_csp_header_is_report_only_by_default_and_enforced_by_flag(monkeypatch):
    import main
    from fastapi.testclient import TestClient
    monkeypatch.delenv("RENDERELLO_CSP_ENFORCE", raising=False)
    h = TestClient(main.app).get("/api/login-mode").headers
    assert "content-security-policy-report-only" in h
    assert "content-security-policy" not in h
    monkeypatch.setenv("RENDERELLO_CSP_ENFORCE", "1")
    h = TestClient(main.app).get("/api/login-mode").headers
    assert "content-security-policy" in h
    assert "content-security-policy-report-only" not in h
