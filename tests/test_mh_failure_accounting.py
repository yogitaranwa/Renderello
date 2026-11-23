"""Provider failure/timeout accounting with a FAKE backend (no network):
a failed or timed-out paid call is never refunded below the listed price, and
a failed batch never frees reservation for calls that may have been accepted."""
import numpy as np
import pytest
from PIL import Image

import renderello.studio as st
from renderello.studio import StudioRequest, run_studio

LISTED = 30


class Boom(RuntimeError):
    pass


class FakeBackend:
    mode = "raise"
    instances = []

    def __init__(self, model=None, live=False, receipt_path=None, **kw):
        self.live = live
        self.edit_calls = 0
        self.last_result = {}
        FakeBackend.instances.append(self)

    def balance(self):
        return {"credits": 10_000}

    def edit(self, source, *a, **k):
        self.edit_calls += 1
        if FakeBackend.mode == "raise":
            raise Boom("Magic Hour timeout (test)")
        if FakeBackend.mode == "base":
            raise KeyboardInterrupt()
        return source

    def attempt_evidence(self):
        return []


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDERELLO_MH_LIVE", "1")
    monkeypatch.delenv("RENDERELLO_MH_DEV_MOCK", raising=False)
    FakeBackend.instances.clear()
    FakeBackend.mode = "raise"
    rec = {"reserve": [], "reconcile": []}
    monkeypatch.setattr(st, "reserve_budget",
                        lambda m, n: rec["reserve"].append(n) or n * LISTED)
    monkeypatch.setattr(st, "reconcile_budget",
                        lambda m, r, a: rec["reconcile"].append((r, a)))
    monkeypatch.setattr("renderello.backends.magichour.MagicHourBackend", FakeBackend)
    a = np.zeros((180, 200, 3), np.uint8)
    a[50:150, 40:140] = (30, 144, 200)
    p = tmp_path / "p.png"
    Image.fromarray(a).save(p)
    return rec, p, tmp_path


def _go(p, tmp, route, variants=2, attempts="1"):
    return run_studio(StudioRequest(
        product_paths=[p], preset_key="clean_studio", variants=variants,
        provider="magichour", mh_route=route,
        knob_values={"mh_model": "seedream-v5-flash",
                     "scene_max_attempts": attempts}), tmp / "runs")


@pytest.mark.parametrize("route", ["whole", "scene", "hybrid"])
def test_failed_calls_are_billed_at_listed_price_not_refunded(live, route):
    rec, p, tmp = live
    try:
        _go(p, tmp, route)
    except Exception:
        pass
    assert len(rec["reconcile"]) == 1, rec
    reserved, actual = rec["reconcile"][0]
    calls = sum(b.edit_calls for b in FakeBackend.instances)
    assert calls >= 1
    assert actual >= calls * LISTED          # every attempted call counted
    assert actual <= reserved                # never above the reservation


@pytest.mark.parametrize("route", ["whole", "scene", "hybrid"])
def test_non_exception_abort_does_not_refund_inflight_call(live, route):
    """A BaseException (e.g. interrupt) raised from inside the provider call
    must still leave that call counted: it may already have been accepted."""
    rec, p, tmp = live
    FakeBackend.mode = "base"
    with pytest.raises(BaseException):
        _go(p, tmp, route, variants=1)
    assert len(rec["reconcile"]) == 1, rec
    reserved, actual = rec["reconcile"][0]
    # None = unknown charge: the whole reservation stays (reconcile no-ops).
    assert actual is None or actual >= LISTED, (route, reserved, actual)
