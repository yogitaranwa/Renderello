"""Regression: jobs must run on a bounded persistent worker pool.

Measured locally under a mixed sequential+burst workload, a fresh OS thread
per generation job showed step-wise RSS growth while a bounded pool stayed
flat (mechanism uncertain, consistent with per-thread allocator arenas).
These tests pin the semantics that matter: worker reuse, pool size matching
the admission cap, submit-failure cleanup, and clean pool shutdown.
"""
import importlib, io, threading
import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from PIL import Image


def png():
    b = io.BytesIO(); Image.new("RGB", (64, 64), "red").save(b, "PNG")
    return b.getvalue()


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDERELLO_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "test-code-not-real")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import webapp.main as m
    importlib.reload(m); m.auth._attempts.clear()
    c = TestClient(m.app)
    assert c.post("/api/login", json={"code": "test-code-not-real"}).status_code == 200
    return c, m


def test_sequential_jobs_reuse_bounded_workers(tmp_path, monkeypatch):
    c, m = _client(tmp_path, monkeypatch)
    names = []
    done = []

    def fake_run(req, root):
        names.append(threading.current_thread().name)
        done.append(1)
        raise RuntimeError("stub")  # job ends "failed"; fine for this test

    monkeypatch.setattr(m, "run_studio", fake_run)
    key = next(iter(m.PRESETS))
    for _ in range(12):
        r = c.post("/api/generate",
                   files={"products": ("a.png", png(), "image/png")},
                   data={"preset": key, "provider": "local", "variants": "1"})
        assert r.status_code == 200
        # wait for this job to finish before posting the next (sequential)
        job_id = r.json()["job_id"]
        for _ in range(200):
            if c.get(f"/api/jobs/{job_id}").json()["status"] != "running":
                break
    assert len(done) == 12
    distinct = set(names)
    assert len(distinct) <= m._MAX_RUNNING_JOBS, (
        f"expected worker reuse, got {len(distinct)} distinct job threads: {distinct}")
    assert all(n.startswith("renderello-job") for n in distinct)


def test_pool_size_matches_admission_cap(tmp_path, monkeypatch):
    _, m = _client(tmp_path, monkeypatch)
    assert m._executor._max_workers == m._MAX_RUNNING_JOBS


def test_submit_failure_cleans_up(tmp_path, monkeypatch):
    """If job submission fails, no job slot and no temp upload dir may stick."""
    c, m = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(m._executor, "submit",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("executor unavailable")))
    c2 = TestClient(m.app, raise_server_exceptions=False)
    assert c2.post("/api/login",
                   json={"code": "test-code-not-real"}).status_code == 200
    r = c2.post("/api/generate",
                files={"products": ("a.png", png(), "image/png")},
                data={"preset": next(iter(m.PRESETS)), "provider": "local",
                      "variants": "1"})
    assert r.status_code == 500
    with m._lock:
        stuck = [j for j in m._jobs.values() if j["status"] == "running"]
    assert stuck == [], f"stuck running jobs after failed submit: {stuck}"
    leftovers = list(m.RUNS_ROOT.glob("up-*")) if m.RUNS_ROOT.is_dir() else []
    assert leftovers == [], f"temp upload dirs left behind: {leftovers}"


def test_pool_shutdown_drains_cleanly(tmp_path, monkeypatch):
    """After completed jobs, shutdown(wait=True) returns promptly: no wedged
    workers, so reload/exit cannot hang on an idle pool."""
    import time
    c, m = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "run_studio",
                        lambda req, root: (_ for _ in ()).throw(
                            RuntimeError("stub")))
    key = next(iter(m.PRESETS))
    for _ in range(3):
        r = c.post("/api/generate",
                   files={"products": ("a.png", png(), "image/png")},
                   data={"preset": key, "provider": "local", "variants": "1"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        for _ in range(200):
            if c.get(f"/api/jobs/{job_id}").json()["status"] != "running":
                break
    flag = []
    t = threading.Thread(
        target=lambda: (m._executor.shutdown(wait=True), flag.append(1)),
        daemon=True)
    t.start(); t.join(timeout=10)
    assert flag, "pool shutdown(wait=True) did not return within 10s"
