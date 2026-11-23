import numpy as np
from PIL import Image

from renderello import qa


def _mask(img: np.ndarray) -> np.ndarray:
    # Whole-image matte for unit tests (alignment is explicit in qa.evaluate).
    return np.full(img.shape[:2], 255, np.uint8)


def test_identical_images_score_five(tmp_path):
    from tests.conftest import make_product
    src = make_product()
    arr = np.array(src)
    p = tmp_path / "same.png"
    src.save(p)
    h, w = arr.shape[:2]
    r = qa.evaluate(src, p, "cutout_compose", "clean_studio", 0,
                    src_crop_box=(0, 0, w, h), matte=_mask(arr))
    assert r.color_drift < 0.02
    assert r.edge_retained > 0.98
    assert not r.critical_errors
    assert r.score_fidelity == 5


def test_recolored_label_flags_critical(tmp_path):
    from tests.conftest import make_product
    src = make_product()
    arr = np.array(src).copy()
    arr[300:420, 280:360] = (250, 40, 40)  # repaint the label area
    p = tmp_path / "bad.png"
    Image.fromarray(arr).save(p)
    # Align to the product bbox, as real runs do; evaluate() takes the
    # FULL-SIZE matte and crops internally.
    box = (231, 166, 179, 333)  # bottle bbox from the conftest drawing
    x, y, w, h = box
    matte = np.zeros((640, 640), np.uint8)
    matte[y:y + h, x:x + w] = 255
    r = qa.evaluate(src, p, "pure_prompt", "clean_studio", 0,
                    src_crop_box=box, matte=matte)
    assert r.color_drift > 0.1
    assert r.critical_errors, "expected at least one critical flag"
    assert r.score_fidelity <= 2  # critical flags cap the rubric


def test_edge_loss_detected(tmp_path):
    from tests.conftest import make_product
    src = make_product()
    arr = np.array(src)
    blurred = np.array(src.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(6)))
    p = tmp_path / "blur.png"
    Image.fromarray(blurred).save(p)
    h, w = arr.shape[:2]
    r = qa.evaluate(src, p, "hybrid", "clean_studio", 0,
                    src_crop_box=(0, 0, w, h), matte=_mask(arr))
    assert r.edge_retained < 0.9


def test_no_alignment_records_unscored(tmp_path):
    from tests.conftest import make_product
    src = make_product()
    p = tmp_path / "any.png"
    src.save(p)
    r = qa.evaluate(src, p, "pure_prompt", "clean_studio", 0)
    assert "metrics skipped" in r.notes
    assert r.critical_errors  # unscored is itself a flag


def test_histogram_drift_monotonic():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 140, (64, 64, 3), dtype=np.uint8)
    m = _mask(base)
    slight = np.clip(base.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    heavy = np.clip(base.astype(np.int16) + 90, 0, 255).astype(np.uint8)
    d1 = qa.color_drift(base, slight, m)
    d2 = qa.color_drift(base, heavy, m)
    assert d1 < d2


def test_matte_sanity_flags_background_slab():
    """The realistic-CPU-stress finding: GrabCut on busy scenes keeps a
    full-frame slab whose pixels are source-exact, so pixel QA alone scored it
    5/5. The matte itself must trip the flag."""
    import numpy as np
    from renderello.qa import matte_sanity
    matte = np.full((800, 800), 255, np.uint8)  # everything kept
    flags = matte_sanity(matte, (0, 0, 800, 800))
    assert any("slab" in f for f in flags)


def test_matte_sanity_passes_isolated_bottle():
    import numpy as np
    from renderello.qa import matte_sanity
    matte = np.zeros((800, 800), np.uint8)
    matte[150:700, 320:480] = 255  # tall narrow bottle
    flags = matte_sanity(matte, (320, 150, 160, 550))
    assert flags == []
