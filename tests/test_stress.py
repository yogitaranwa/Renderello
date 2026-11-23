"""Stress tests: hostile inputs the benchmark's 20-source gate will actually
contain - edge-touching products, extreme aspects, tiny products, multi-object
scenes, transparent-glass-like low-contrast subjects, huge sources. These guard
pipeline robustness, not output beauty (that needs the GPU routes)."""
from __future__ import annotations

import json
import numpy as np
import pytest
from PIL import Image, ImageDraw

from renderello.backends.base import StubEditor
from renderello.config import RunConfig
from renderello.cutout import _bbox_from_matte, cutout
from renderello.pipeline import run_pipeline
from renderello.schemas import Aspect, EditRequest, Route
from renderello.templates import get_template


def _scene(size, shapes) -> Image.Image:
    img = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(img)
    for box, color in shapes:
        d.rectangle(box, fill=color)
    return img


def _run(source: Image.Image, tmp_path, stem="s"):
    p = tmp_path / f"{stem}.png"
    source.save(p)
    req = EditRequest(source_path=p, template=get_template("clean_studio"), seed=1)
    return run_pipeline(req, [Route.CUTOUT_COMPOSE],
                      RunConfig(seed=1, variants=1, out_root=tmp_path / "r"))


def test_product_touching_frame_edge(tmp_path):
    # Product flush against the left frame edge - the GrabCut rect seed must
    # still find it (classic failure mode noted in cutout.py).
    img = _scene((400, 400), [((0, 120, 140, 300), (40, 90, 160))])
    res = _run(img, tmp_path, "edge")
    v = res.variants[0]
    assert v.output_path.exists()
    assert any("touches the source frame on left" in e for e in v.qa.critical_errors)
    assert v.qa.score_fidelity <= 2
    assert json.loads(res.manifest_path.read_text())["source_provenance"]["match_basis"] == "no_registered_content_match"


def test_tiny_product(tmp_path):
    img = _scene((800, 800), [((370, 380, 430, 440), (160, 40, 40))])
    res = _run(img, tmp_path, "tiny")
    assert res.variants[0].output_path.exists()


def test_extreme_aspect_source(tmp_path):
    img = _scene((1600, 300), [((700, 40, 900, 260), (30, 120, 60))])
    res = _run(img, tmp_path, "wide")
    assert res.variants[0].output_path.exists()


def test_multi_object_scene(tmp_path):
    # The benchmark gate includes multi-product arrangements; pipeline must not
    # crash on disjoint matte components.
    img = _scene((600, 400), [((80, 120, 200, 300), (180, 60, 60)),
                              ((400, 100, 520, 320), (60, 60, 180))])
    res = _run(img, tmp_path, "multi")
    assert res.variants[0].output_path.exists()


def test_low_contrast_subject(tmp_path):
    # Transparent-glass proxy: subject barely differs from background.
    img = _scene((400, 400), [((150, 100, 250, 320), (205, 208, 212))])
    try:
        res = _run(img, tmp_path, "glass")
        assert res.variants[0].output_path.exists()
    except ValueError as e:
        # An empty matte must fail loudly, not silently ship a blank output.
        assert "empty cutout matte" in str(e)


def test_huge_cpu_source_is_recorded_without_unsafe_graph_allocation(tmp_path, monkeypatch):
    import sys
    # The GrabCut allocation budget is the guard under test; force the CPU
    # fallback chain regardless of locally installed learned backends.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    img = _scene((3000, 2400), [((1200, 900, 1800, 1600), (90, 60, 140))])
    res = _run(img, tmp_path, "huge")
    variant = res.variants[0]
    assert not variant.output_path.exists()
    assert any("CPU GrabCut allocation budget exceeded" in flag for flag in variant.qa.critical_errors)
    assert Image.open(tmp_path / "huge.png").size == (3000, 2400)
    assert json.loads(res.manifest_path.read_text())["variants"][0]["success"] is False


def test_flat_field_fails_loud(tmp_path):
    img = Image.new("RGB", (300, 300), (128, 128, 128))
    p = tmp_path / "flat.png"
    img.save(p)
    req = EditRequest(source_path=p, template=get_template("clean_studio"))
    res = run_pipeline(req, [Route.CUTOUT_COMPOSE],
                       RunConfig(seed=1, variants=1, out_root=tmp_path / "r2"))
    v = res.variants[0]
    # Either a degenerate matte or a recorded failure - never a silent success.
    if v.output_path.exists():
        assert v.qa.critical_errors or (v.qa.score_fidelity or 5) <= 3
    else:
        assert ("variant failed" in v.qa.critical_errors[0]
                or "cutout failed" in v.qa.critical_errors[0])


def test_all_aspects_render(tmp_path):
    img = _scene((500, 500), [((180, 150, 320, 380), (70, 130, 90))])
    p = tmp_path / "prod.png"
    img.save(p)
    for aspect in Aspect:
        req = EditRequest(source_path=p, template=get_template("clean_studio"),
                          aspect=aspect)
        res = run_pipeline(req, [Route.CUTOUT_COMPOSE],
                           RunConfig(seed=2, variants=1,
                                     out_root=tmp_path / f"r_{aspect.name}"))
        out = Image.open(res.variants[0].output_path)
        assert out.size == aspect.size


def test_bbox_rejects_degenerate_matte():
    with pytest.raises(ValueError):
        _bbox_from_matte(np.zeros((10, 10), np.uint8))


def test_stub_editor_is_deterministic(tmp_path):
    img = _scene((200, 200), [((60, 60, 140, 140), (200, 50, 50))])
    a = StubEditor().edit(img, "p", "n", 7, Aspect.SQUARE)
    b = StubEditor().edit(img, "p", "n", 7, Aspect.SQUARE)
    assert np.array_equal(np.array(a), np.array(b))


def test_grabcut_budget_checks_before_opencv_allocation(monkeypatch):
    import renderello.cutout as module
    called = []
    monkeypatch.setattr(module.cv2, "cvtColor", lambda *a, **k: called.append(True))
    with pytest.raises(ValueError, match="allocation budget exceeded"):
        module.grabcut_matte(np.zeros((100, 100, 3), np.uint8), max_pixels=9999)
    assert not called
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            module.grabcut_matte(np.zeros((2,2,3),np.uint8), max_pixels=invalid)


def test_huge_source_with_explicit_native_matte_preserves_input_and_exports(tmp_path):
    img = _scene((3000, 2400), [((1200,900,1800,1600),(90,60,140))])
    source = tmp_path / "huge-supplied.png"; img.save(source)
    alpha = Image.new("L",img.size,0)
    ImageDraw.Draw(alpha).rectangle((1200,900,1800,1600),fill=255)
    mask = tmp_path / "explicit-native-alpha.png"; alpha.save(mask)
    request=EditRequest(source_path=source, mask_path=mask, template=get_template("clean_studio"),seed=1)
    result=run_pipeline(request,[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/"supplied-runs"))
    assert result.variants[0].output_path.exists()
    manifest=json.loads(result.manifest_path.read_text())
    assert manifest['cutout']['backend']=='supplied_mask'
    assert Image.open(source).size==(3000,2400)
    assert Image.open(manifest['cutout']['rgba_path']).size==(3000,2400)
    assert 'SUPPLIED MASK' in result.variants[0].qa.notes
