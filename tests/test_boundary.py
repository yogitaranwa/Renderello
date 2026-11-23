import numpy as np
from PIL import Image
from renderello import boundary


def _alpha():
    a = np.zeros((80, 80), np.uint8)
    a[30:50, 30:50] = 255
    return a


def test_band_is_ring_around_edge():
    b = boundary.edge_band(_alpha())
    assert b[40, 40] == 0 and b[40, 25] and not b[2, 2]


def test_stats_identical_images_zero():
    im = np.random.default_rng(0).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    s = boundary.band_stats(im, im, _alpha())
    assert s["mean_abs_final_vs_raw_outside"] == 0.0


def test_strip_shape():
    im = Image.new("RGB", (80, 80))
    s = boundary.boundary_strip(im, im, _alpha(), (30, 30, 50, 50), zoom=2, pad=5)
    assert s.size == (3 * 30 * 2, 30 * 2)


def test_frame_edge_contact_detects_cropped_product_and_passes_centered():
    import numpy as np
    from renderello.boundary import frame_edge_contact
    centered = np.zeros((100, 100), np.uint8); centered[30:70, 30:70] = 255
    r = frame_edge_contact(centered)
    assert r["touches"] == [] and r["needs_review"] is False
    cropped = centered.copy(); cropped[40:60, 80:] = 255  # strap runs off the right edge
    r = frame_edge_contact(cropped)
    assert r["touches"] == ["right"] and r["needs_review"] is True
    tiny = np.zeros((100, 100), np.uint8); tiny[50, 99] = 255  # 1px speck must not trigger
    assert frame_edge_contact(tiny)["needs_review"] is False


def test_matte_sanity_frame_cut_is_opt_in():
    import numpy as np
    from renderello import qa
    m = np.zeros((100, 100), np.uint8); m[30:70, 30:100] = 255
    bbox = (30, 30, 70, 40)
    assert qa.matte_sanity(m, bbox) == []
    f = qa.matte_sanity(m, bbox, flag_frame_cut=True)
    assert len(f) == 1 and "right" in f[0]


def test_snap_to_canvas_edges_only_moves_frame_cut_sides():
    import numpy as np
    from renderello.compose import snap_to_canvas_edges
    a = np.zeros((100, 100), np.uint8); a[30:70, 30:70] = 255
    assert snap_to_canvas_edges(a, (50, 60), (40, 40), (200, 300)) == (50, 60)  # centered: unchanged
    b = a.copy(); b[40:60, 70:] = 255  # touches right
    assert snap_to_canvas_edges(b, (50, 60), (40, 40), (200, 300)) == (160, 60)
    c = a.copy(); c[70:, 40:60] = 255  # touches bottom
    assert snap_to_canvas_edges(c, (50, 60), (40, 40), (200, 300)) == (50, 260)


def test_composite_with_scene_snap_is_opt_in_and_interior_exact(tmp_path):
    import numpy as np
    from PIL import Image
    from renderello import compose
    from renderello.cutout import cutout
    from renderello.schemas import Aspect
    from renderello.templates import get_template
    img = np.full((300, 300, 3), 235, np.uint8); img[80:220, 160:300] = (40, 90, 160)  # product runs off right edge
    co = cutout(Image.fromarray(img), tmp_path, backend="grabcut")
    t = get_template("minimal_tabletop")
    fn = lambda g, p: np.full(g.shape, 140, np.uint8)
    a, _ = compose.composite_with_scene(co, t, Aspect.PORTRAIT, 1, fn)
    b, _ = compose.composite_with_scene(co, t, Aspect.PORTRAIT, 1, fn, snap_frame_cut=True)
    A, B = np.array(a), np.array(b)
    assert A.shape == B.shape
    # snapped: product's right edge sits on the canvas border (blue pixels in last column)
    assert (np.abs(B[:, -1, :].astype(int) - [40, 90, 160]).sum(axis=1) < 30).any()


def test_snap_edge_cases_empty_and_both_sides():
    import numpy as np
    from renderello.compose import snap_to_canvas_edges
    empty = np.zeros((50, 50), np.uint8)
    assert snap_to_canvas_edges(empty, (7, 9), (10, 10), (100, 100)) == (7, 9)
    wide = np.zeros((50, 50), np.uint8); wide[20:30, :] = 255  # runs off both left and right
    assert snap_to_canvas_edges(wide, (7, 9), (10, 10), (100, 100)) == (7, 9)  # ambiguous: leave placement alone


def test_frame_edge_contact_small_margin_speck_and_threshold():
    import numpy as np
    from renderello.boundary import frame_edge_contact
    m = np.zeros((100, 100), np.uint8); m[:, 98:] = 255  # full-height strip at right, within margin 2
    assert frame_edge_contact(m)["touches"] == ["right"]
    m2 = np.zeros((100, 100), np.uint8); m2[0, 10:13] = 255  # 3px speck on top = 3% < 4%
    assert frame_edge_contact(m2)["touches"] == []


def test_cover_fit_only_for_opposite_side_cuts():
    import numpy as np
    from renderello.compose import cover_fit_if_cut_both_sides
    a = np.zeros((100, 100, 4), np.uint8); a[30:70, 30:70, 3] = 255
    assert cover_fit_if_cut_both_sides(a, (60, 80)) is None
    b = a.copy(); b[40:60, :, 3] = 255  # strap across the whole width: left and right cut
    out = cover_fit_if_cut_both_sides(b, (60, 80))
    assert out.shape == (80, 60, 4)
    c = a.copy(); c[40:60, 70:, 3] = 255  # only right: snap case, not cover
    assert cover_fit_if_cut_both_sides(c, (60, 80)) is None


def test_transmission_composite_uses_new_scene_not_old_backdrop():
    import numpy as np
    from renderello.compose import estimate_clean_backdrop, silhouette_from_backdrop, transmission_composite
    src = np.full((120, 100, 3), 250, np.uint8)
    src[30:100, 30:70] = 250            # clear interior = backdrop
    src[30:100, 30:34] = 120; src[30:100, 66:70] = 120   # dark glass edges
    src[30:34, 30:70] = 120; src[96:100, 30:70] = 120
    B, ok = estimate_clean_backdrop(src); assert ok and abs(B[0] - 250) < 2
    m = silhouette_from_backdrop(src, B); assert m[60, 50] > 0.9 and m[5, 5] < 0.05
    scene = np.zeros((120, 100, 3), np.uint8); scene[...] = (10, 100, 200)
    out = transmission_composite(src, scene, m, B)
    assert abs(int(out[60, 50, 2]) - 200) <= 3        # interior shows NEW scene
    assert out[60, 31, 2] < 120                        # glass edge darkens the scene
    clutter = np.random.default_rng(0).integers(0, 255, (120, 100, 3), dtype=np.uint8)
    assert estimate_clean_backdrop(clutter)[1] is False
