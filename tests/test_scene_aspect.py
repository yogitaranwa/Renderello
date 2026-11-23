import numpy as np
from renderello.compose import aspect_safe_scene_call

def _crop_like_iclight(img, tw=512, th=640):
    # emulate ICLight: center-crop to native aspect, resize, return native size
    import cv2
    h, w = img.shape[:2]; s = max(tw / w, th / h)
    r = cv2.resize(img, (round(w * s), round(h * s)))
    l, t = (r.shape[1] - tw) // 2, (r.shape[0] - th) // 2
    return r[t:t + th, l:l + tw]

def test_landscape_canvas_alignment_preserved():
    h, w = 768, 1120
    gray = np.full((h, w, 3), 127, np.uint8); gray[300:500, 400:700] = (200, 30, 30)  # "product"
    out = aspect_safe_scene_call(lambda x, p: _crop_like_iclight(x), gray, "p")
    assert out.shape == gray.shape
    red = (out[..., 0] > 150) & (out[..., 1] < 80)
    ys, xs = np.where(red)
    assert abs(ys.min() - 300) < 6 and abs(ys.max() - 499) < 6
    assert abs(xs.min() - 400) < 6 and abs(xs.max() - 699) < 6

def test_native_aspect_passthrough():
    g = np.full((640, 512, 3), 127, np.uint8)
    calls = []
    def fn(x, p): calls.append(x.shape); return x
    aspect_safe_scene_call(fn, g, "p")
    assert calls == [(640, 512, 3)]


def test_pipeline_scene_route_same_coordinate_qa(tmp_path):
    """Landscape source through the real pipeline with a ghost-free scene model:
    placement is reported, paste is bit-exact at those coordinates, and the QA
    pixel metrics compare the placed region (not the whole canvas)."""
    import numpy as np, cv2
    from PIL import Image
    from renderello.config import RunConfig
    from renderello.pipeline import run_pipeline
    from renderello.schemas import EditRequest, Route
    from renderello.templates import get_template
    rng = np.random.default_rng(1)
    img = np.full((600, 900, 3), 230, np.uint8)
    img[200:420, 300:600] = rng.integers(40, 200, (220, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (300, 200), (600, 420), (20, 60, 160), 6)
    src = tmp_path / "s.jpg"; Image.fromarray(img).save(src, quality=98)
    cfg = RunConfig(seed=0, variants=1, out_root=tmp_path / "o")
    def scene(gray, prompt):  # mock model: flat scene, no product
        return np.full_like(gray, 90)
    res = run_pipeline(EditRequest(source_path=src, template=get_template("lifestyle_kitchen"), seed=0),
                       [Route.SCENE_COMPOSE], cfg, scene_fn=scene)
    v = res.variants[0]
    assert "paste_bit_exact=True" in v.qa.notes
    assert "same-coordinate QA region=(" in v.qa.notes
    assert not any("NOT bit-exact" in e for e in v.qa.critical_errors)


def test_all_aspects_alignment_with_cropping_scene_model():
    """For every Aspect, a model that center-crops to 4:5 must still keep the generated
    marker under the pasted product (regression test for the ghost-product failure)."""
    import numpy as np
    from renderello.schemas import Aspect
    for asp in Aspect:
        w, h = asp.size
        g = np.full((h, w, 3), 127, np.uint8)
        y0, y1, x0, x1 = h // 3, h // 2, w // 3, w // 2
        g[y0:y1, x0:x1] = (210, 20, 20)
        out = aspect_safe_scene_call(lambda x, p: _crop_like_iclight(x), g, "p")
        import cv2
        if out.shape != g.shape:  # native-aspect passthrough returns model size; compose resizes (same aspect)
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
        assert out.shape == g.shape, asp
        red = (out[..., 0] > 150) & (out[..., 1] < 90)
        ys, xs = np.where(red)
        tol = max(8, int(0.01 * max(w, h)))
        assert abs(ys.min() - y0) <= tol and abs(ys.max() - (y1 - 1)) <= tol, asp
        assert abs(xs.min() - x0) <= tol and abs(xs.max() - (x1 - 1)) <= tol, asp


def test_cover_fit_geometry_and_none_case():
    import numpy as np
    from renderello.compose import cover_fit_if_cut_both_sides
    rgba = np.zeros((300, 600, 4), np.uint8); rgba[..., :3] = 100
    rgba[100:200, :, 3] = 255            # strap touching left and right edges
    out = cover_fit_if_cut_both_sides(rgba, (512, 640))
    assert out is not None and out.shape == (640, 512, 4)
    rgba2 = np.zeros((300, 600, 4), np.uint8); rgba2[100:200, 200:400, 3] = 255
    assert cover_fit_if_cut_both_sides(rgba2, (512, 640)) is None


def test_cover_to_canvas_and_reference_composite():
    """Reference/backdrop photo (PDF section 4 optional): cover-fits to the canvas
    and the product pastes source-exact on top of it."""
    import numpy as np
    from renderello.compose import cover_to_canvas, composite
    from PIL import Image
    import tempfile, pathlib
    ref = np.zeros((300, 600, 3), np.uint8); ref[..., 0] = 200
    out = cover_to_canvas(ref, (512, 640))
    assert out.shape == (640, 512, 3) and (out[..., 0] == 200).all()

    # product: opaque red square in a transparent frame
    rgba = np.zeros((100, 100, 4), np.uint8); rgba[25:75, 25:75] = (10, 10, 220, 255)
    d = pathlib.Path(tempfile.mkdtemp())
    rp = d / "p.png"; Image.fromarray(rgba).save(rp)
    from renderello.schemas import CutoutResult
    from renderello.templates import get_template
    mp = d / "m.png"; Image.fromarray(rgba[..., 3]).save(mp)
    co = CutoutResult(rgba_path=rp, matte_path=mp, bbox=(25, 25, 50, 50), backend="test", coverage=0.25)
    img = composite(co, get_template("clean_studio"), __import__("renderello.schemas", fromlist=["Aspect"]).Aspect.SQUARE,
                    0, reference=ref)
    a = np.array(img)
    assert a.shape == (1024, 1024, 3)
    reds = (a[..., 0] < 60) & (a[..., 2] > 180)
    assert reds.sum() > 1000                      # product present
    assert (a[0, :, 0] == 200).all()              # backdrop is the reference, not procedural


def test_scene_route_reference_backdrop_no_gpu(tmp_path):
    """scene_compose with a reference image and NO scene_fn: reference acts as the
    scene (GPU-free path), paste still bit-exact at placed coordinates."""
    import numpy as np, cv2
    from PIL import Image
    from renderello.config import RunConfig
    from renderello.pipeline import run_pipeline
    from renderello.schemas import EditRequest, Route
    from renderello.templates import get_template
    rng = np.random.default_rng(2)
    img = np.full((500, 700, 3), 235, np.uint8)
    img[150:350, 250:450] = rng.integers(30, 200, (200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (250, 150), (450, 350), (200, 40, 40), 5)
    src = tmp_path / "s.jpg"; Image.fromarray(img).save(src, quality=98)
    ref = np.full((480, 800, 3), (60, 120, 30), np.uint8)
    rp = tmp_path / "ref.jpg"; Image.fromarray(ref).save(rp, quality=95)
    cfg = RunConfig(seed=0, variants=1, out_root=tmp_path / "o")
    res = run_pipeline(EditRequest(source_path=src, template=get_template("natural_outdoor"),
                                   seed=0, reference_path=rp),
                       [Route.SCENE_COMPOSE], cfg, scene_fn=None)
    v = res.variants[0]
    assert "REFERENCE-BACKDROP" in v.qa.notes
    assert "paste_bit_exact=True" in v.qa.notes
    assert not any("NOT bit-exact" in e for e in v.qa.critical_errors)


def test_reference_backdrop_cover_fit_matches_every_aspect(tmp_path):
    from PIL import Image
    from renderello.schemas import Aspect
    """A direct reference must not go through model padding/stretching."""
    from renderello.pipeline import run_pipeline
    from renderello.schemas import EditRequest, Route
    from renderello.config import RunConfig
    from tests.conftest import make_product
    from renderello.templates import get_template
    from renderello.compose import cover_to_canvas
    source=tmp_path/'source.png';make_product().save(source)
    y,x=np.mgrid[:200,:600]
    pattern=np.stack([x%256,y%256,(x//30%2)*255],axis=-1).astype(np.uint8)
    ref=tmp_path/'ref.png';Image.fromarray(pattern).save(ref)
    for aspect in Aspect:
        result=run_pipeline(EditRequest(source_path=source,template=get_template('clean_studio'),
                         aspect=aspect,reference_path=ref),[Route.SCENE_COMPOSE],
                         RunConfig(variants=1,out_root=tmp_path/aspect.name),scene_fn=None)
        actual=np.array(Image.open(result.variants[0].output_path))
        expected=cover_to_canvas(pattern,aspect.size)
        # Top strip and far sides are outside the placed synthetic product.
        assert np.array_equal(actual[:20],expected[:20])
        assert np.array_equal(actual[:,:20],expected[:,:20])


def test_canvas_dimensions_match_all_advertised_ratios():
    from renderello.schemas import Aspect
    for aspect in Aspect:
        w, h = aspect.size
        rw, rh = map(int, aspect.value.split(":"))
        assert w * rh == h * rw
        assert w % 32 == h % 32 == 0
