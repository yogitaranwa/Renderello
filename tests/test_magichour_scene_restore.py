"""Magic Hour studio routes after the 13 Nov 2026 restore: environment-only
scene prompt, product-free conditioning, native pixel audit, hybrid pass."""
import json

import numpy as np
import pytest
from PIL import Image

from renderello.studio import StudioRequest, run_studio


def _square(color, size=(200, 180)):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[50:150, 40:140] = color
    return Image.fromarray(a)


@pytest.fixture
def product(tmp_path):
    p = tmp_path / "p.png"
    _square((30, 144, 200)).save(p)
    return p


def test_scene_route_uses_empty_guide_and_audits_pixels(tmp_path, monkeypatch,
                                                        product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    res = run_studio(StudioRequest(product_paths=[product],
                                   preset_key="clean_studio", variants=1,
                                   provider="magichour", mh_route="scene"),
                     tmp_path / "runs")
    v = res.variants[0]
    assert v.error is None
    m = json.loads((res.run_dir / "manifest.json").read_text())
    assert m["mode"].startswith("scene-compose")
    diag = v.qa["scene_diag"]
    assert diag["conditioning"] == "empty_guide"
    assert "emptiness_screen" in diag
    audit = v.qa["native_pixel_audit"]
    # Mock scene is the unmodified two-tone guide, so the paste is exact.
    assert audit["status"] == "MATCH"
    assert audit["native_rgb_exact"] is True
    assert audit["quality_acceptance"] == "NOT_EVALUATED"
    assert any("eye review" in w for w in v.warnings)


def test_scene_prompt_never_names_product(tmp_path, monkeypatch, product):
    """The contradictory product-prompt-plus-'no product' tail is gone: the
    scene model receives the environment-only prompt."""
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    sent = []
    import renderello.studio as st
    real_composite = st.compose.composite_with_scene

    def spy(*args, **kwargs):
        sent.append(kwargs.get("prompt", args[6] if len(args) > 6 else None))
        return real_composite(*args, **kwargs)

    monkeypatch.setattr(st.compose, "composite_with_scene", spy)
    res = run_studio(StudioRequest(product_paths=[product],
                                   preset_key="clean_studio", variants=1,
                                   provider="magichour", mh_route="scene",
                                   product_description="amber spray bottle"),
                     tmp_path / "runs")
    assert res.variants[0].error is None
    assert sent, "scene model was never called"
    for prompt in sent:
        assert "amber spray bottle" not in prompt
        assert "no product" in prompt
        assert ", empty background scene, no product, no text" not in prompt


def test_product_naming_instruction_dropped_with_warning(tmp_path, monkeypatch,
                                                         product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    res = run_studio(StudioRequest(product_paths=[product],
                                   preset_key="clean_studio", variants=1,
                                   provider="magichour", mh_route="scene",
                                   instruction="put the bottle on the left"),
                     tmp_path / "runs")
    v = res.variants[0]
    assert v.error is None
    assert any("named the product" in w for w in v.warnings)


def test_hybrid_route_adds_harmonize_pass(tmp_path, monkeypatch, product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    res = run_studio(StudioRequest(product_paths=[product],
                                   preset_key="clean_studio", variants=1,
                                   provider="magichour", mh_route="hybrid"),
                     tmp_path / "runs")
    v = res.variants[0]
    assert v.error is None
    assert v.image.endswith("_magichour_hybrid.png")
    assert (res.run_dir / v.image).exists()
    # The pre-harmonize source-exact composite is preserved alongside.
    assert (res.run_dir / "variant_0_magichour_scene.png").exists()
    m = json.loads((res.run_dir / "manifest.json").read_text())
    assert m["mode"].startswith("hybrid")
    assert any("harmonize pass" in w and "EYE REVIEW" in w
               for w in v.warnings)
    assert v.qa["native_pixel_audit"]["status"] in (
        "MATCH", "MISMATCH", "NATIVE_SIZE_CHANGED", "NO_FULLY_OPAQUE_PIXELS")


def test_hybrid_mock_does_not_touch_budget(tmp_path, monkeypatch, product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    monkeypatch.setenv("RENDERELLO_MH_LEDGER", str(tmp_path / "led.jsonl"))
    run_studio(StudioRequest(product_paths=[product],
                             preset_key="clean_studio", variants=1,
                             provider="magichour", mh_route="hybrid"),
               tmp_path / "runs")
    assert not (tmp_path / "led.jsonl").exists()  # zero credits in mock


def test_unknown_mh_route_refused(tmp_path, monkeypatch, product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    with pytest.raises(ValueError, match="route"):
        run_studio(StudioRequest(product_paths=[product],
                                 preset_key="clean_studio", variants=1,
                                 provider="magichour", mh_route="bogus"),
                   tmp_path / "runs")


def test_flare_1k_refused_off_default_allowlist(tmp_path, monkeypatch):
    """gpt-image-2.5-flare-1k is in MODEL_TABLE (source-verified 200cr @1k)
    but must stay OFF the live path: the default allowlist is only
    seedream-v5-flash, and a live reserve for it refuses with zero bill."""
    monkeypatch.delenv("RENDERELLO_MH_LIVE_MODELS", raising=False)
    from renderello.studio import _live_model_allowlist, reserve_budget
    assert _live_model_allowlist() == {"seedream-v5-flash"}
    with pytest.raises(RuntimeError, match="allowlist"):
        reserve_budget("gpt-image-2.5-flare-1k", 1)
    with pytest.raises(RuntimeError, match="allowlist"):
        reserve_budget("gpt-image-2.5-flare", 1)


def test_estimate_prices_hybrid_and_whole_routes(monkeypatch):
    """Hybrid worst case = attempts+1 calls per image; whole = 1."""
    from fastapi.testclient import TestClient
    import webapp.main as wm
    monkeypatch.setattr(wm, "_require_auth", lambda request: None)
    client = TestClient(wm.app)
    base = "/api/estimate?provider=magichour&mh_model=seedream-v5-flash&variants=1&presets=clean_studio"
    scene = client.get(base + "&mh_route=scene").json()
    hybrid = client.get(base + "&mh_route=hybrid").json()
    whole = client.get(base + "&mh_route=whole").json()
    per = scene["credits_per_image"]
    assert scene["credits_worst_case"] == 1 * per     # default attempts = 1
    assert hybrid["credits_worst_case"] == 2 * per    # attempts + harmonize
    assert whole["credits_worst_case"] == 1 * per
    assert client.get(base + "&mh_route=bogus").status_code == 400


def test_estimate_attempt_knob_matches_reservation_parser(monkeypatch):
    from fastapi.testclient import TestClient
    import webapp.main as wm
    from renderello.studio import scene_attempts_from
    monkeypatch.setattr(wm, "_require_auth", lambda request: None)
    client = TestClient(wm.app)
    for raw in ("1", "2", "3", "bad", "0", "99"):
        quote = client.get("/api/estimate", params={
            "provider": "magichour", "variants": 1, "mh_route": "scene",
            "knob_scene_max_attempts": raw}).json()
        attempts = scene_attempts_from({"scene_max_attempts": raw})
        assert quote["scene_max_attempts"] == attempts
        assert quote["credits_worst_case"] == attempts * 30


def test_one_attempt_never_retries_suspect_scene(tmp_path, monkeypatch, product):
    monkeypatch.setenv("RENDERELLO_MH_DEV_MOCK", "1")
    import renderello.scene_emptiness as se
    import renderello.backends.magichour as mh
    monkeypatch.setattr(se, "screen_scene_emptiness", lambda *a, **k: {
        "suspect_object_in_scene": True, "test_forced": True})
    calls = []
    import renderello.compose as comp
    real_compose = comp.composite_with_scene
    def counted(*args, **kwargs):
        args = list(args)
        real_scene = args[4]
        def scene(*a, **k):
            calls.append(1)
            return real_scene(*a, **k)
        args[4] = scene
        return real_compose(*args, **kwargs)
    monkeypatch.setattr(comp, "composite_with_scene", counted)
    result = run_studio(StudioRequest(product_paths=[product],
        preset_key="clean_studio", variants=1, provider="magichour",
        mh_route="scene", knob_values={"scene_max_attempts": "1"}), tmp_path / "runs")
    assert len(calls) == 1
    assert result.variants[0].qa["scene_diag"]["emptiness_screen"]["suspect_object_in_scene"] is True
