"""Auth matrix for every API/static route and run-file traversal checks."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))
import auth  # noqa: E402


@pytest.fixture
def env(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "code-12345")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    monkeypatch.setenv("RENDERELLO_SESSION_SECRET", "s" * 40)
    monkeypatch.setenv("RENDERELLO_RUNS", str(runs))
    for k in ("RENDERELLO_LOGIN_EMAIL", "RENDERELLO_LOGIN_PASSWORD_HASH"):
        monkeypatch.delenv(k, raising=False)
    auth._attempts.clear()
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    anon = TestClient(main.app)
    user = TestClient(main.app)
    assert user.post("/api/login", json={"code": "code-12345"}).status_code == 200
    (runs / "abc123").mkdir()
    (runs / "abc123" / "variant_0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (runs / "abc123" / ".hidden").write_text("h")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    (runs / "abc123" / "link.txt").symlink_to(secret)
    (runs / "outside").symlink_to(tmp_path, target_is_directory=True)
    (tmp_path / "x.png").write_bytes(b"x")
    yield main, anon, user, runs, tmp_path
    auth._attempts.clear()


PROTECTED_GET = ["/", "/api/presets", "/api/providers", "/api/estimate",
                 "/api/jobs/deadbeef", "/api/runs/abc123/variant_0.png",
                 "/static/js/app.js", "/static/css/app.css",
                 "/static/index.html"]


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_protected_routes_need_a_session(env, path):
    _, anon, user, *_ = env
    r = anon.get(path)
    # "/" serves the login page (200) but never the app shell.
    if path == "/":
        assert "Renderello" in r.text and "app.js" not in r.text
    else:
        assert r.status_code == 401, (path, r.status_code)


def test_public_assets_are_only_login_assets(env):
    _, anon, *_ = env
    assert anon.get("/static/js/login.js").status_code == 200
    assert anon.get("/static/css/login.css").status_code == 200
    assert anon.get("/api/login-mode").status_code == 200
    assert anon.get("/static/js/app.js").status_code == 401


def test_post_routes_need_a_session(env):
    _, anon, *_ = env
    assert anon.post("/api/generate", data={"preset": "clean_studio"}).status_code == 401
    # logout is harmless without a session
    assert anon.post("/api/logout").status_code == 200


def test_forged_and_tampered_cookies_rejected(env):
    main, anon, user, *_ = env
    good = user.cookies.get("renderello_session")
    for bad in ("", "garbage", "s1.9999999999.nonce." + "0" * 64,
                good[:-1] + ("0" if good[-1] != "0" else "1"),
                # expiry rewritten, signature kept
                ".".join(["s1", "9" * 10] + good.split(".")[2:])):
        r = anon.get("/api/presets", cookies={"renderello_session": bad})
        assert r.status_code == 401, bad


def test_run_file_happy_path(env):
    _, _, user, *_ = env
    assert user.get("/api/runs/abc123/variant_0.png").status_code == 200


@pytest.mark.parametrize("url", [
    "/api/runs/abc123/..%2fsecret.txt",
    "/api/runs/abc123/%2e%2e%2fsecret.txt",
    "/api/runs/abc123/..%5csecret.txt",
    "/api/runs/..%2f/x.png",
    "/api/runs/abc123/.hidden",
    "/api/runs/abc123/link.txt",          # symlink leading outside the run
    "/api/runs/outside/x.png",            # run dir that is a symlink
    "/api/runs/abc.123/variant_0.png",    # dot not allowed in run id
    "/api/runs/%2Fetc/passwd",
    "/api/runs/abc123/variant_0.png%00.txt",
    "/api/runs/abc123/nope.png",
    "/api/runs/abc123/" + "a" * 5000,     # over-long name
])
def test_run_file_traversal_and_symlinks_are_404(env, url):
    _, _, user, *_ = env
    r = user.get(url)
    assert r.status_code in (404, 400, 422), (url, r.status_code)
    assert b"TOP-SECRET" not in r.content


@pytest.mark.parametrize("url", [
    "/static/../main.py", "/static/%2e%2e/main.py", "/static/..%2fmain.py",
    "/static/js/../../auth.py", "/static//etc/passwd",
    "/static/js/app.js%00.txt", "/static/" + "a" * 5000,
])
def test_static_traversal_is_blocked(env, url):
    _, _, user, *_ = env
    r = user.get(url)
    assert r.status_code in (404, 400)
    assert b"def " not in r.content


def test_job_status_unknown_is_404_not_500(env):
    _, _, user, *_ = env
    assert user.get("/api/jobs/does-not-exist").status_code == 404
    assert user.get("/api/jobs/" + "a" * 5000).status_code == 404


def test_security_headers_present_on_errors_too(env):
    _, anon, *_ = env
    r = anon.get("/api/presets")
    assert r.status_code == 401
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"
