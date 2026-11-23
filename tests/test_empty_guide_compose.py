"""empty_guide conditioning: the scene model receives no product pixels and a
non-contradictory environment-only prompt, and the pre-paste emptiness screen
feeds the regeneration budget."""
import numpy as np
import pytest
from PIL import Image

from renderello.compose import composite_with_scene
from renderello.empty_scene_prompt import build_empty_scene_prompt
from renderello.schemas import Aspect, CutoutResult
from renderello.templates import get_template


def _cutout(tmp_path):
    a = np.zeros((150, 150, 4), np.uint8)
    a[30:100, 40:110, :3] = (255, 0, 30)
    a[30:100, 40:110, 3] = 255
    p = tmp_path / 'p.png'
    Image.fromarray(a).save(p)
    return CutoutResult(p, p, (40, 30, 70, 70), 'test', .2)


def _prompt():
    return build_empty_scene_prompt(get_template('clean_studio'),
                                    aspect_key="1:1")


def _clean(h, w):
    s = np.full((h, w, 3), 140, np.uint8)
    s[int(h * 0.6):] = 180
    return s


class Capture:
    """One mode per expected generation: 'clean' or 'dirty'."""
    def __init__(self, modes):
        self.modes = list(modes)
        self.canvases = []
        self.prompts = []
    def __call__(self, canvas, prompt):
        self.canvases.append(canvas.copy())
        self.prompts.append(prompt)
        mode = self.modes.pop(0)
        h, w = canvas.shape[:2]
        scene = _clean(h, w)
        if mode == "dirty":
            # Long straight dark lines across the whole lower half: wherever
            # the placement rectangle lands it catches several, and the luma
            # contrast reads as an object-like blob.
            for y in range(h // 2, h - 4, 10):
                scene[y:y + 3, :] = 25
        return scene


def test_empty_guide_sends_product_free_canvas_and_caller_prompt(tmp_path):
    co = _cutout(tmp_path)
    tmpl = get_template('clean_studio')
    cb = Capture(["clean"])
    img, d = composite_with_scene(co, tmpl, Aspect.SQUARE, 0, cb,
                                  prompt=_prompt(), conditioning="empty_guide")
    # Exactly one generation; canvas is ONLY the two-tone guide.
    assert len(cb.canvases) == 1
    assert set(np.unique(cb.canvases[0])) == {127, 176}
    # The caller's environment-only prompt passes through unchanged.
    assert cb.prompts[0] is not None and "no product" in cb.prompts[0]
    assert d["conditioning"] == "empty_guide"
    assert d["scene_mode"] == "empty_guide"
    assert "emptiness" in d["scene_acceptance"]
    assert "emptiness_screen" in d
    assert d["paste_bit_exact"]


def test_empty_guide_refuses_product_bearing_prompt(tmp_path):
    co = _cutout(tmp_path)
    cb = Capture([])
    with pytest.raises(ValueError, match="EmptyScenePrompt"):
        composite_with_scene(co, get_template('clean_studio'), Aspect.SQUARE,
                             0, cb, prompt="a red bottle on a table, product shot",
                             conditioning="empty_guide")
    assert not cb.canvases          # refused BEFORE any generation call


def test_empty_guide_refuses_20nov_failure_shape(tmp_path):
    """The exact 20 Nov 2026 failure: a product-bearing prompt WITH a
    contradictory 'no product' tail. The old substring guard passed this
    shape; the provenance guard must refuse it before any generation."""
    co = _cutout(tmp_path)
    cb = Capture([])
    bad = ("Keep the exact product unchanged: identical shape, packaging, "
           "color, logos, label text. Place the amber spray bottle on a "
           "marble table, soft window light, empty background scene, "
           "no product, no text")
    with pytest.raises(ValueError, match="EmptyScenePrompt"):
        composite_with_scene(co, get_template('clean_studio'), Aspect.SQUARE,
                             0, cb, prompt=bad, conditioning="empty_guide")
    assert not cb.canvases


def test_builder_output_accepted(tmp_path):
    p = _prompt()
    from renderello.empty_scene_prompt import EmptyScenePrompt
    assert isinstance(p, EmptyScenePrompt)
    co = _cutout(tmp_path)
    cb = Capture(["clean"])
    img, d = composite_with_scene(co, get_template('clean_studio'),
                                  Aspect.SQUARE, 0, cb, prompt=p,
                                  conditioning="empty_guide")
    assert d["conditioning"] == "empty_guide"


def test_suspect_object_scene_regenerated_within_budget(tmp_path):
    co = _cutout(tmp_path)
    tmpl = get_template('clean_studio')
    cb = Capture(["dirty", "clean"])
    img, d = composite_with_scene(co, tmpl, Aspect.SQUARE, 0, cb,
                                  prompt=_prompt(), conditioning="empty_guide",
                                  scene_max_attempts=2)
    assert len(cb.canvases) == 2    # suspect scene rejected, one regeneration
    assert d["scene_attempts"] == 2
    assert d["emptiness_screen"]["suspect_object_in_scene"] is False
    assert d["emptiness_screen"]["needs_eye_review"] is True


def test_suspect_object_exhausts_budget_and_still_flags(tmp_path):
    co = _cutout(tmp_path)
    tmpl = get_template('clean_studio')
    cb = Capture(["dirty", "dirty"])
    img, d = composite_with_scene(co, tmpl, Aspect.SQUARE, 0, cb,
                                  prompt=_prompt(), conditioning="empty_guide",
                                  scene_max_attempts=2)
    assert d["scene_attempts"] == 2
    assert d["emptiness_screen"]["suspect_object_in_scene"] is True
    assert d["needs_eye_review"] is True    # never auto-cleared


def test_product_gray_default_unchanged(tmp_path):
    co = _cutout(tmp_path)
    cb = Capture(["clean"])
    img, d = composite_with_scene(co, get_template('clean_studio'),
                                  Aspect.SQUARE, 0, cb)
    assert d["conditioning"] == "product_gray"
    # Legacy canvas carries product-modulated gray, not the bare two-tone.
    assert set(np.unique(cb.canvases[0])) != {127, 176}


def test_unknown_conditioning_rejected(tmp_path):
    with pytest.raises(ValueError, match="conditioning"):
        composite_with_scene(_cutout(tmp_path), get_template('clean_studio'),
                             Aspect.SQUARE, 0, Capture([]),
                             conditioning="mystery")
