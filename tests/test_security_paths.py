"""Run: pytest tests/test_security_paths.py (uses a temp RUNS root; no network, no spend)."""
import os, io, importlib
from pathlib import Path
import pytest
pytest.importorskip("fastapi"); pytest.importorskip("httpx")
from fastapi.testclient import TestClient

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "test-code-not-real")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import webapp.main as m
    importlib.reload(m)
    m.auth._attempts.clear()
    (tmp_path / "secret.txt").write_text("nope")
    (m.RUNS_ROOT / "run1").mkdir(parents=True, exist_ok=True)
    (m.RUNS_ROOT / "run1" / "ok.txt").write_text("ok")
    c = TestClient(m.app)
    assert c.post("/api/login", json={"code": "test-code-not-real"}).status_code == 200
    return c, m, tmp_path

def test_run_file_ok(client):
    c, m, _ = client
    assert c.get("/api/runs/run1/ok.txt").text == "ok"

@pytest.mark.parametrize("url", ["/api/runs/../x", "/api/runs/%2e%2e/secret.txt", "/api/runs/run1/..%2fsecret.txt", "/api/runs/run1/%2e%2e%2fsecret.txt", "/api/runs/.../secret.txt"])
def test_traversal_blocked(client, url):
    c, *_ = client
    assert c.get(url).status_code in (404, 400)

def test_login_body_limits(client):
    c, m, _ = client
    assert c.post("/api/login", content=b"x" * 10000).status_code in (413, 400, 429)
    assert c.post("/api/login", content=b"[1]").status_code in (400, 429)

def test_upload_rejects_riff_non_webp(client):
    c, m, _ = client
    assert m._sniff(b"RIFF\0\0\0\0WAVEfmt ") is None
    assert m._sniff(b"RIFF\0\0\0\0WEBPVP8 ") == "webp"

def test_unauth_denied(client):
    c, m, _ = client
    m.auth._attempts.clear()
    c2 = TestClient(m.app)
    for u in ["/api/presets", "/api/providers", "/api/runs/run1/ok.txt", "/static/js/app.js"]:
        assert c2.get(u).status_code == 401
