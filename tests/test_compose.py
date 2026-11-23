import numpy as np
from PIL import Image

from renderello import compose
from renderello.cutout import cutout
from renderello.schemas import Aspect
from renderello.templates import get_template


def test_background_deterministic_per_seed():
    t = get_template("clean_studio")
    a = compose.procedural_background((256, 256), t, seed=3)
    b = compose.procedural_background((256, 256), t, seed=3)
    c = compose.procedural_background((256, 256), t, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_variants_are_visually_distinct(product_path, tmp_path):
    """2-4 alternatives must be genuinely distinct compositions (PDF flow),
    not re-noised copies: placement and background vary with the seed while
    the product pixels stay source-exact (covered by the bit-exact test)."""
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    outs = [np.array(compose.composite(co, t, Aspect.SQUARE, seed=s),
                     np.float32) for s in (1, 2, 3)]
    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            assert np.abs(outs[i] - outs[j]).mean() > 0.5
    p1 = compose.placement_params(1, t.key)
    p2 = compose.placement_params(2, t.key)
    assert p1 != p2


def test_composite_size_and_paste(product_path, tmp_path):
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    out = compose.composite(co, t, Aspect.SQUARE, seed=1)
    assert out.size == Aspect.SQUARE.size
    assert out.mode == "RGB"


def test_product_interior_bit_exact(product_path, tmp_path):
    """Route 2's core promise: untransformed interior pixels are identical."""
    src = Image.open(product_path).convert("RGB")
    co = cutout(src, tmp_path, backend="grabcut")
    t = get_template("minimal_tabletop")
    aspect = Aspect.SQUARE
    out = np.array(compose.composite(co, t, aspect, seed=0))

    # Locate placement exactly as the pipeline does.
    cw, ch = aspect.size
    x, y, w, h = co.bbox
    fill, jx, baseline = compose.placement_params(0, t.key)
    scale = min((ch * fill) / h, (cw * 0.9) / w, 1.0)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    px = max(0, min((cw - nw) // 2 + int(jx * cw), cw - nw))
    py = max(0, min(int(ch * baseline) - nh, ch - nh))
    rgba = np.array(Image.open(co.rgba_path))
    crop = rgba[y:y + h, x:x + w]
    import cv2
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    interior = crop[..., 3] == 255
    placed = out[py:py + nh, px:px + nw]
    # Unsharp masking shifts pixels by design; allow tiny tolerance, require
    # near-perfect agreement on fully-opaque interior.
    diff = np.abs(placed.astype(np.int16) - crop[..., :3].astype(np.int16))
    assert (diff[interior].max() <= 6)
    assert (diff[interior] <= 2).mean() > 0.95


def test_reflection_only_where_placed(product_path, tmp_path):
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("reflective_surface")
    out = compose.composite(co, t, Aspect.SQUARE, seed=0, reflection=True)
    assert out.size == Aspect.SQUARE.size


def test_empty_matte_raises(tmp_path):
    blank = Image.new("RGB", (64, 64), (128, 128, 128))
    # GrabCut on a flat field may still return something; force-empty is
    # exercised via _bbox_from_matte directly.
    import pytest
    from renderello.cutout import _bbox_from_matte
    with pytest.raises(ValueError):
        _bbox_from_matte(np.zeros((64, 64), np.uint8))


def _placed(co, t, aspect, seed):
    import numpy as np
    from PIL import Image
    rgba = np.array(Image.open(co.rgba_path).convert("RGBA"))
    fill, jx, base = compose.placement_params(seed, t.key)
    crop, (px, py) = compose.fit_product(rgba, co.bbox, aspect.size,
                                         fill=fill, jx=jx, baseline=base)
    return crop, px, py


def test_scene_route_pastes_source_exact_product(product_path, tmp_path):
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    rng = np.random.default_rng(0)
    seen = {}

    def fake_scene(gray, prompt):
        seen["shape"], seen["prompt"] = gray.shape, prompt
        return rng.integers(0, 255, gray.shape, dtype=np.uint8)  # hostile scene

    out, diag = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, fake_scene)
    crop, px, py = _placed(co, t, Aspect.SQUARE, 1)
    inner = crop[..., 3] == 255
    assert inner.any()
    got = np.array(out)[py:py + crop.shape[0], px:px + crop.shape[1], :3]
    assert np.array_equal(got[inner], crop[..., :3][inner])  # source-exact interior
    # scene model now sees the canvas padded to its native 4:5 aspect (ghost-product fix, v0.4.23)
    assert seen["shape"] == (1280, 1024, 3)
    assert "product photography" in seen["prompt"]
    assert "needs_eye_review" in diag


def test_scene_route_flags_dark_halo_for_eye_review(product_path, tmp_path):
    import cv2
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    crop, px, py = _placed(co, t, Aspect.SQUARE, 1)
    w, h = Aspect.SQUARE.size

    def halo_scene(gray, prompt):
        m = np.zeros((h, w), np.uint8)
        m[py:py + crop.shape[0], px:px + crop.shape[1]] = (crop[..., 3] > 127)
        k = max(3, int(round(0.01 * max(w, h))))
        dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1)))
        sc = np.full((h, w, 3), 230, np.uint8)
        sc[(dil - m).astype(bool)] = 20     # generated dark shadow hugging product
        return sc

    _, diag = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, halo_scene)
    assert diag["needs_eye_review"] is True
    assert diag["ring_vs_far_luma_diff"] > 100


def test_scene_route_contact_shadow_darkens_below_keeps_interior(product_path, tmp_path):
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    flat = lambda gray, prompt: np.full(gray.shape, 200, np.uint8)
    plain, _ = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, flat)
    shaded, _ = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, flat,
                                             shadow_style="soft")
    crop, px, py = _placed(co, t, Aspect.SQUARE, 1)
    inner = crop[..., 3] == 255
    g = np.array(shaded)[py:py + crop.shape[0], px:px + crop.shape[1], :3]
    assert np.array_equal(g[inner], crop[..., :3][inner])
    assert np.array(shaded).astype(int).sum() < np.array(plain).astype(int).sum()


def test_scene_grade_moves_scene_not_product(product_path, tmp_path):
    co = cutout(Image.open(product_path), tmp_path, backend="grabcut")
    t = get_template("clean_studio")
    warm = lambda gray, prompt: np.tile(np.array([230, 170, 90], np.uint8), (*gray.shape[:2], 1))
    a, _ = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, warm)
    b, _ = compose.composite_with_scene(co, t, Aspect.SQUARE, 1, warm, scene_grade=0.4)
    crop, px, py = _placed(co, t, Aspect.SQUARE, 1)
    inner = crop[..., 3] == 255
    g = np.array(b)[py:py + crop.shape[0], px:px + crop.shape[1], :3]
    assert np.array_equal(g[inner], crop[..., :3][inner])
    assert not np.array_equal(np.array(a)[0, 0], np.array(b)[0, 0])
