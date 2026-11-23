"""Mock end-to-end execution of the Colab bridge cells 7.1 -> 7.5.

The notebook code cells are exec'd in order against a fake requests module,
so the success, timeout, 5xx, still-running and download-failure sequences
are exercised as the runtime would run them - not just statically scanned.
No real network, no real Magic Hour spend."""
import json
import sys
import types

import pytest
from PIL import Image

from scripts.build_colab_notebook import build


def _bridge_cells():
    nb = json.loads(build().read_text())
    cells = ["".join(c["source"]) for c in nb["cells"]
             if c["cell_type"] == "code"
             and "".join(c["source"]).lstrip().startswith("# 7.")]
    assert [c[2:5].strip() for c in cells] == ["7.1", "7.2", "7.3", "7.4", "7.5"]
    return cells


class FakeResponse:
    def __init__(self, status_code=200, data=None, content=b"", text=""):
        self.status_code = status_code
        self._data = data or {}
        self.content = content
        self.text = text or json.dumps(self._data)
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Scripted session: behaviors keyed by url suffix."""
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []          # (method, url) in order
    def post(self, url, **kw):
        self.calls.append(("POST", url))
        for key, b in self.behaviors.items():
            if key in url:
                if isinstance(b, Exception):
                    raise b
                if callable(b):
                    return b()
                return b
        return FakeResponse(200, {"ok": True})
    def get(self, url, **kw):
        self.calls.append(("GET", url))
        for key, b in self.behaviors.items():
            if key in url:
                if isinstance(b, Exception):
                    raise b
                if callable(b):
                    return b()
                return b
        return FakeResponse(200, {})


def _fake_requests(session):
    mod = types.ModuleType("requests")
    mod.Session = lambda: session
    mod.Timeout = TimeoutError
    return mod


def _run_bridge(tmp_path, monkeypatch, session, *, job_behaviors,
                time_limit_s=900):
    """Exec 7.1-7.4 (and 7.5 when downloads succeed) with all I/O faked."""
    cells = [c.replace("bridgeEnabled = False", "bridgeEnabled = True") for c in _bridge_cells()]
    monkeypatch.setitem(sys.modules, "requests", _fake_requests(session))
    getpass = types.ModuleType("getpass")
    getpass.getpass = lambda prompt: "secret-input"
    monkeypatch.setitem(sys.modules, "getpass", getpass)

    png = tmp_path / "sheet.png"
    Image.new("RGB", (8, 8), (9, 9, 9)).save(png)
    png_bytes = png.read_bytes()
    manifest_bytes = json.dumps({"mode": "scene-compose", "provider_used":
                                 "magichour", "budget": {"spent": 105}}).encode()

    job_polls = {"n": 0}

    def job_response():
        job_polls["n"] += 1
        behavior = job_behaviors[min(job_polls["n"] - 1, len(job_behaviors) - 1)]
        return FakeResponse(200, behavior)

    session.behaviors["/api/estimate"] = FakeResponse(200, {
        "credits_per_image": 30, "within_budget": True,
        "within_batch_limit": True, "budget": {"spent": 105, "remaining": 895}})
    session.behaviors["/api/jobs/"] = job_response
    session.behaviors["contact_sheet.png"] = FakeResponse(200, content=png_bytes)
    session.behaviors["manifest.json"] = FakeResponse(200, content=manifest_bytes)

    clock = {"t": 0.0}
    time_mod = types.ModuleType("time")
    time_mod.monotonic = lambda: clock["t"]
    def _sleep(s):
        clock["t"] += s
    time_mod.sleep = _sleep
    monkeypatch.setitem(sys.modules, "time", time_mod)

    product = tmp_path / "product.png"
    Image.new("RGB", (16, 16), (200, 30, 30)).save(product)

    import pathlib as _pathlib
    ns = {
        "mhModel": "seedream-v5-flash",
        "products": [product],
        "instructionText": "",
        "productDescription": "test bottle",
        "input": lambda prompt: "SPEND",
        "display": lambda obj: None,
        "json": json,
        "pathlib": _pathlib,
        "__name__": "__main__",
    }
    # Cells write to /content/...; redirect that literal prefix to a tmp dir.
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    cells = [c.replace("/content/", str(content_dir) + "/") for c in cells]

    completed = []
    for i, cell in enumerate(cells):
        exec(compile(cell, f"cell{i}", "exec"), ns)
        completed.append(i)
    return ns, completed, job_polls, content_dir


def _done_job():
    return {"status": "done", "runs": [{"run_dir": "20261120-152520-abc123"}]}


def test_success_sequence_downloads_then_logout(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": FakeResponse(200, {"job_id": "j1"})})
    ns, completed, polls, content = _run_bridge(
        tmp_path, monkeypatch, session, job_behaviors=[_done_job()])
    assert completed == [0, 1, 2, 3, 4]
    assert (content / "live_contact_sheet.png").exists()
    assert (content / "live_manifest.json").exists()
    urls = [u for _, u in session.calls]
    # Downloads happened BEFORE logout; login first, logout last.
    assert urls[0].endswith("/api/login")
    assert urls[-1].endswith("/api/logout")
    assert urls.index("https://renderello-studio.onrender.com/api/logout") > \
        max(i for i, u in enumerate(urls) if "/api/runs/" in u)
    # Exactly ONE paid POST.
    assert sum(1 for m, u in session.calls
               if m == "POST" and u.endswith("/api/generate")) == 1
    assert ns["typedConfirmation"] is None  # consumed before POST


def test_timeout_never_resubmits_and_still_logs_out(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": TimeoutError("ambig")})
    with pytest.raises(SystemExit, match="DO NOT RERUN"):
        _run_bridge(tmp_path, monkeypatch, session, job_behaviors=[])
    posts = [u for m, u in session.calls if m == "POST"]
    assert posts == ["https://renderello-studio.onrender.com/api/login",
                     "https://renderello-studio.onrender.com/api/generate",
                     "https://renderello-studio.onrender.com/api/logout"]


def test_5xx_unknown_paid_state_stops_and_logs_out(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": FakeResponse(503, text="boom")})
    with pytest.raises(SystemExit, match="paid state UNKNOWN"):
        _run_bridge(tmp_path, monkeypatch, session, job_behaviors=[])
    assert sum(1 for m, u in session.calls
               if m == "POST" and u.endswith("/api/generate")) == 1
    assert session.calls[-1][1].endswith("/api/logout")


def test_still_running_at_deadline_systemexit_with_job_id(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": FakeResponse(200, {"job_id": "j9"})})
    with pytest.raises(SystemExit, match="j9 still running"):
        _run_bridge(tmp_path, monkeypatch, session,
                    job_behaviors=[{"status": "running"}])
    # Polls stopped at the 900s deadline; logout still ran.
    assert session.calls[-1][1].endswith("/api/logout")
    get_polls = [u for m, u in session.calls if m == "GET" and "/api/jobs/" in u]
    assert len(get_polls) == 180  # 900s / 5s cadence, no runaway loop


def test_download_failure_after_paid_success_no_rerun(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": FakeResponse(200, {"job_id": "j2"})})
    ns, *_ = (None,), None, None, None
    with pytest.raises(SystemExit, match="paid job COMPLETED"):
        cells = [c.replace("bridgeEnabled = False", "bridgeEnabled = True") for c in _bridge_cells()]
        monkeypatch.setitem(sys.modules, "requests", _fake_requests(session))
        getpass = types.ModuleType("getpass")
        getpass.getpass = lambda prompt: "x"
        monkeypatch.setitem(sys.modules, "getpass", getpass)
        session.behaviors["/api/estimate"] = FakeResponse(200, {
            "credits_per_image": 30, "within_budget": True,
            "within_batch_limit": True, "budget": {"spent": 105}})
        session.behaviors["/api/jobs/"] = FakeResponse(200, _done_job())
        session.behaviors["contact_sheet.png"] = FakeResponse(401, text="expired")
        clock = {"t": 0.0}
        time_mod = types.ModuleType("time")
        time_mod.monotonic = lambda: clock["t"]
        time_mod.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
        monkeypatch.setitem(sys.modules, "time", time_mod)
        product = tmp_path / "product.png"
        Image.new("RGB", (16, 16)).save(product)
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        cells = [c.replace("/content/", str(content_dir) + "/") for c in cells]
        import pathlib as _pathlib
        ns = {"mhModel": "seedream-v5-flash", "products": [product],
              "instructionText": "", "productDescription": "d",
              "input": lambda p: "SPEND", "display": lambda o: None,
              "json": json, "pathlib": _pathlib}
        for i, cell in enumerate(cells):
            exec(compile(cell, f"cell{i}", "exec"), ns)
    # The paid POST happened once; logout after the failed download.
    assert sum(1 for m, u in session.calls
               if m == "POST" and u.endswith("/api/generate")) == 1
    assert session.calls[-1][1].endswith("/api/logout")


def test_stale_confirmation_cannot_spend_twice(tmp_path, monkeypatch):
    session = FakeSession({"/api/generate": FakeResponse(200, {"job_id": "j3"})})
    ns, completed, polls, content = _run_bridge(
        tmp_path, monkeypatch, session, job_behaviors=[_done_job()])
    # Re-running the paid cell without a fresh 7.3 must refuse BEFORE any POST.
    cells = [c.replace("bridgeEnabled = False", "bridgeEnabled = True") for c in _bridge_cells()]
    before = len(session.calls)
    with pytest.raises(AssertionError, match="No fresh spending confirmation"):
        exec(compile(cells[3], "cell3-rerun", "exec"), ns)
    assert len(session.calls) == before  # not even a logout; the assert is first


@pytest.mark.parametrize("stale", [False, True])
def test_run_all_default_off_even_with_stale_globals(monkeypatch, stale):
    session = FakeSession({})
    monkeypatch.setitem(sys.modules, "requests", _fake_requests(session))
    prompts = []
    gp = types.ModuleType("getpass")
    gp.getpass = lambda prompt: prompts.append(prompt) or "unused"
    monkeypatch.setitem(sys.modules, "getpass", gp)
    ns = {"input": lambda prompt: prompts.append(prompt) or "SPEND"}
    if stale:
        ns.update(bridgeEnabled=True, typedConfirmation="SPEND", site=session,
                  bridgeSessionToken="old", bridgeQuoteToken="old", spendConfirmationToken="old")
    # Simulate Run all continuing after every guard error.
    errors = []
    for cell in _bridge_cells():
        try:
            exec(compile(cell, "bridge-default-off", "exec"), ns)
        except (AssertionError, SystemExit) as error:
            errors.append(str(error))
    assert len(errors) == 5
    assert session.calls == []
    assert prompts == []
    assert ns["typedConfirmation"] is None
    assert ns["site"] is None


def test_opt_in_without_spend_never_generates(monkeypatch):
    session = FakeSession({"/api/estimate": FakeResponse(200, {
        "credits_per_image":30,"within_budget":True,"within_batch_limit":True,
        "budget":{"spent":105}})})
    monkeypatch.setitem(sys.modules,"requests",_fake_requests(session))
    gp=types.ModuleType("getpass");gp.getpass=lambda prompt:"fake"
    monkeypatch.setitem(sys.modules,"getpass",gp)
    ns={"mhModel":"seedream-v5-flash","input":lambda prompt:"NO", "typedConfirmation":"SPEND",
        "spendConfirmationToken":"stale"}
    for cell in _bridge_cells()[:4]:
        cell=cell.replace("bridgeEnabled = False", "bridgeEnabled = True")
        try:
            exec(compile(cell,"opt-in-no-spend","exec"),ns)
        except (AssertionError,SystemExit):
            pass
    assert not any(url.endswith("/api/generate") for _,url in session.calls)
    assert ns["spendConfirmationToken"] is None
