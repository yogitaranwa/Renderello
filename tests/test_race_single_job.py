import importlib, io, threading, time
import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from PIL import Image

def png():
    b = io.BytesIO(); Image.new("RGB", (64, 64), "red").save(b, "PNG"); return b.getvalue()

def test_single_live_job_race(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "test-code-not-real")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import webapp.main as m
    importlib.reload(m); m.auth._attempts.clear()
    import renderello.studio as st
    monkeypatch.setattr(st, "_enforce_budget", lambda *a, **k: None)
    started = []
    def slow(req, root):
        started.append(1); time.sleep(1.5); raise RuntimeError("Magic Hour stub")
    monkeypatch.setattr(m, "run_studio", slow)
    key = next(iter(m.PRESETS))
    c = TestClient(m.app)
    assert c.post("/api/login", json={"code": "test-code-not-real"}).status_code == 200
    res = []
    def go():
        r = c.post("/api/generate", files={"products": ("a.png", png(), "image/png")},
                   data={"preset": key, "provider": "magichour", "variants": "1"})
        res.append(r.status_code)
    ts = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    time.sleep(2)
    print("statuses", res, "jobs started", len(started))
    assert len(started) == 1, f"single live MH job rule violated: {len(started)} jobs ran"
