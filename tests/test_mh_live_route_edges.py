"""Live-mode Magic Hour route edges with a FAKE backend and a recording
reserve_budget. No network, no spend: proves refusals happen before any
reservation or provider call and that reservations match the worst case."""
import numpy as np
import pytest
from PIL import Image

import renderello.studio as st
from renderello.studio import StudioRequest, run_studio


class FakeBackend:
    instances = []

    def __init__(self, model=None, live=False, receipt_path=None, **kw):
        self.edit_calls = 0
        self.edit_kwargs = []
        self.last_result = {}
        self.model = model
        self.live = live
        FakeBackend.instances.append(self)

    def balance(self):
        return {"credits": 10_000}

    def edit(self, *a, **k):
        self.edit_calls += 1
        self.edit_kwargs.append(k)
        raise RuntimeError("Magic Hour fake failure (test)")

    def attempt_evidence(self):
        return []


def _img(path, color=(30, 144, 200)):
    a = np.zeros((180, 200, 3), np.uint8)
    a[50:150, 40:140] = color
    Image.fromarray(a).save(path)
    return path


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDERELLO_MH_LIVE", "1")
    monkeypatch.delenv("RENDERELLO_MH_DEV_MOCK", raising=False)
    FakeBackend.instances.clear()
    reserved = []

    def fake_reserve(model, count):
        reserved.append((model, count))
        return count * 30
    monkeypatch.setattr(st, "reserve_budget", fake_reserve)
    monkeypatch.setattr("renderello.backends.magichour.MagicHourBackend", FakeBackend)
    return reserved, tmp_path


def _req(tmp, route, extra_products=0, refs=0, variants=2, **knobs):
    p = [_img(tmp / "p0.png")]
    for i in range(extra_products):
        p.append(_img(tmp / f"p{i + 1}.png", (200, 60, 60)))
    r = [_img(tmp / f"r{i}.png", (60, 200, 60)) for i in range(refs)]
    return StudioRequest(product_paths=p, reference_paths=r,
                         preset_key="clean_studio", variants=variants,
                         provider="magichour", mh_route=route,
                         knob_values={"mh_model": "seedream-v5-flash", **knobs})


def _run(req, tmp):
    return run_studio(req, tmp / "runs")


@pytest.mark.parametrize("route", ["whole"])
def test_multi_product_refused_before_reserve_or_call(live, route):
    reserved, tmp = live
    with pytest.raises(RuntimeError, match="exactly ONE input"):
        _run(_req(tmp, route, extra_products=1), tmp)
    assert reserved == []
    assert all(b.edit_calls == 0 for b in FakeBackend.instances)


@pytest.mark.parametrize("route", ["whole"])
def test_reference_image_refused_before_reserve_or_call(live, route):
    reserved, tmp = live
    with pytest.raises(RuntimeError, match="exactly ONE input"):
        _run(_req(tmp, route, refs=1), tmp)
    assert reserved == []
    assert all(b.edit_calls == 0 for b in FakeBackend.instances)


@pytest.mark.parametrize("route,attempts,expected", [
    ("whole", "1", 2), ("whole", "3", 2),        # whole: 1 call per variant
    ("scene", "1", 2), ("scene", "3", 6),        # scene: attempts per variant
    ("hybrid", "1", 4), ("hybrid", "3", 8),      # hybrid: attempts + 1 harmonize
])
def test_reservation_equals_worst_case_calls(live, route, attempts, expected):
    reserved, tmp = live
    try:
        _run(_req(tmp, route, variants=2, scene_max_attempts=attempts), tmp)
    except Exception:
        pass  # fake provider fails; only the reservation matters here
    assert reserved == [("seedream-v5-flash", expected)]
    # No more provider calls than were reserved for.
    assert sum(b.edit_calls for b in FakeBackend.instances) <= expected


def test_junk_attempts_knob_cannot_undercount(live):
    reserved, tmp = live
    for junk in ("0", "-5", "99", "abc", ""):
        reserved.clear()
        try:
            _run(_req(tmp, "scene", variants=1, scene_max_attempts=junk), tmp)
        except Exception:
            pass
        assert reserved == [("seedream-v5-flash", 1)], junk


def test_reserve_refusal_means_zero_provider_calls(live, monkeypatch):
    reserved, tmp = live

    def refuse(model, count):
        raise RuntimeError("batch too large (test)")
    monkeypatch.setattr(st, "reserve_budget", refuse)
    with pytest.raises(RuntimeError, match="batch too large"):
        _run(_req(tmp, "hybrid"), tmp)
    assert all(b.edit_calls == 0 for b in FakeBackend.instances)


@pytest.mark.parametrize("route", ["scene", "hybrid"])
def test_scene_and_hybrid_never_forward_extra_inputs(live, route):
    """Only the whole-image route sends product/reference photos to the
    provider; scene and hybrid send one generated canvas, so extra uploads
    cannot change the per-call input shape or price."""
    reserved, tmp = live
    # Hardened contract: unsupported multi-source composition now fails
    # before reserving, rather than reserving then failing inside a variant.
    with pytest.raises(RuntimeError, match="scene-compose requires exactly one"):
        _run(_req(tmp, route, extra_products=1, refs=1, variants=1), tmp)
    assert reserved == []
    for b in FakeBackend.instances:
        for k in b.edit_kwargs:
            assert not k.get("extra_images") and not k.get("reference")
