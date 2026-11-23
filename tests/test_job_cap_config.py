"""Regression: configurable concurrent-job cap.

Two concurrent full in-process generations OOM-restarted the 512 MB free
instance on 2026-11-21. RENDERELLO_MAX_RUNNING_JOBS now bounds admitted
concurrent jobs; small hosts set it to 1. The worker pool is sized from the
same value so admission and pool stay consistent.
"""
import importlib, io, threading, time
import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from PIL import Image


def png():
    b = io.BytesIO(); Image.new("RGB", (64, 64), "red").save(b, "PNG")
    return b.getvalue()


def _client(tmp_path, monkeypatch, cap):
    monkeypatch.setenv("RENDERELLO_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "test-code-not-real")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    if cap is None:
        monkeypatch.delenv("RENDERELLO_MAX_RUNNING_JOBS", raising=False)
    else:
        monkeypatch.setenv("RENDERELLO_MAX_RUNNING_JOBS", cap)
    import webapp.main as m
    importlib.reload(m); m.auth._attempts.clear()
    c = TestClient(m.app)
    assert c.post("/api/login", json={"code": "test-code-not-real"}).status_code == 200
    return c, m


@pytest.mark.parametrize("raw,expected", [
    (None, 1), ("1", 1), ("3", 3), ("8", 8), ("2", 2),
    ("abc", 1), ("0", 1), ("-2", 1), ("9", 1), ("", 1), (" 1 ", 1),
])
def test_cap_env_parsing(tmp_path, monkeypatch, raw, expected):
    _, m = _client(tmp_path, monkeypatch, raw)
    assert m._MAX_RUNNING_JOBS == expected
    assert m._executor._max_workers == expected


def test_cap_one_rejects_second_concurrent_job(tmp_path, monkeypatch):
    c, m = _client(tmp_path, monkeypatch, "1")
    started = []
    release = threading.Event()

    def slow(req, root):
        started.append(1)
        release.wait(timeout=10)
        raise RuntimeError("stub")

    monkeypatch.setattr(m, "run_studio", slow)
    key = next(iter(m.PRESETS))
    statuses = []

    def go():
        r = c.post("/api/generate",
                   files={"products": ("a.png", png(), "image/png")},
                   data={"preset": key, "provider": "local", "variants": "1"})
        statuses.append(r.status_code)

    t1 = threading.Thread(target=go); t1.start()
    for _ in range(200):
        if started:
            break
        time.sleep(0.02)
    assert started, "first job did not start"
    go()  # second concurrent request, same client
    release.set(); t1.join(timeout=15)
    assert sorted(statuses) == [200, 429], (
        f"cap=1 must admit exactly one job, got {statuses}")


def test_single_job_slot_released_after_finish(tmp_path, monkeypatch):
    """After a job fails, the slot frees and the next job is admitted."""
    c, m = _client(tmp_path, monkeypatch, "1")
    monkeypatch.setattr(m, "run_studio",
                        lambda req, root: (_ for _ in ()).throw(
                            RuntimeError("stub")))
    key = next(iter(m.PRESETS))
    for i in range(2):
        r = c.post("/api/generate",
                   files={"products": ("a.png", png(), "image/png")},
                   data={"preset": key, "provider": "local", "variants": "1"})
        assert r.status_code == 200, f"job {i} rejected after prior finish"
        job_id = r.json()["job_id"]
        for _ in range(200):
            if c.get(f"/api/jobs/{job_id}").json()["status"] != "running":
                break
    with m._lock:
        stuck = [j for j in m._jobs.values() if j["status"] == "running"]
    assert stuck == []
