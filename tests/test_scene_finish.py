"""v0.4.30 scene finishing: contact shadow + dark-scene auto-lift on the
scene_compose route. Product pixels must stay bit-exact in both cases."""
import numpy as np
from PIL import Image

from renderello import compose
from renderello.schemas import Aspect
from renderello.templates import get_template
from renderello.cutout import CutoutResult


def _cutout(tmp_path):
    rgba = np.zeros((300, 400, 4), np.uint8)
    rgba[100:200, 150:250, :3] = (30, 30, 160)
    rgba[100:200, 150:250, 3] = 255
    p = tmp_path / "c.png"
    Image.fromarray(rgba).save(p)
    return CutoutResult(rgba_path=p, bbox=(150, 100, 100, 100),
                        matte_path=p, coverage=0.083, backend="test")


def test_contact_shadow_darkens_scene_only(tmp_path):
    co = _cutout(tmp_path)
    tpl = get_template("clean_studio")
    scene_fn = lambda g, pr: np.full(g.shape, 200, np.uint8)
    img, diag = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, scene_fn,
                                             shadow_style=tpl.shadow)
    arr = np.array(img)
    px, py, pw, ph = diag["placement"]
    # product interior stays exact
    assert diag["paste_bit_exact"] is True
    # scene near the product's actual matte bottom is darkened somewhere
    col = arr[py:min(py + ph + 60, arr.shape[0]), px + pw // 2].mean(axis=1)
    # Compare against the no-shadow output, rather than counting dark product pixels.
    base, _ = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, scene_fn)
    assert np.any(arr != np.array(base))
    assert diag["shadow_style"] == tpl.shadow


def test_dark_scene_auto_lift(tmp_path):
    co = _cutout(tmp_path)
    tpl = get_template("clean_studio")
    scene_fn = lambda g, pr: np.full(g.shape, 30, np.uint8)  # near-black scene
    img, diag = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, scene_fn,
                                             scene_auto_lift=True)
    arr = np.array(img)
    px, py, pw, ph = diag["placement"]
    far = arr[10, 10].mean()  # far from product: scene only
    assert far > 60, f"dark scene not lifted: {far}"
    assert diag["scene_lift"] > 1.0
    assert diag["paste_bit_exact"] is True


def test_bright_scene_not_lifted(tmp_path):
    co = _cutout(tmp_path)
    tpl = get_template("clean_studio")
    scene_fn = lambda g, pr: np.full(g.shape, 200, np.uint8)
    img, diag = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, scene_fn)
    assert diag["scene_lift"] == 1.0


def test_shadow_does_not_wrap_to_opposite_edge():
    matte = np.full((40, 40), 255, np.uint8)
    sh = compose.contact_shadow((100, 100), (60, 40), (40, 40), matte, 'hard')
    assert sh.max() > 0
    assert not sh[:20].any()
    assert not sh[:, :20].any()


def test_padded_matte_shadow_anchors_at_foreground():
    matte = np.zeros((100, 100), np.uint8)
    matte[10:40, 20:80] = 255
    sh = compose.contact_shadow((200, 200), (50, 50), (100, 100), matte, 'soft')
    assert sh.max() > 0
    assert sh[90:110].max() > 0
    assert not sh[140:].any()


def test_shadow_does_not_use_product_as_evidence(tmp_path):
    co = _cutout(tmp_path)
    tpl = get_template('clean_studio')
    fn = lambda g, p: np.full(g.shape, 200, np.uint8)
    before, _ = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, fn)
    after, diag = compose.composite_with_scene(co, tpl, Aspect.SQUARE, 0, fn,
                                              shadow_style=tpl.shadow)
    b, a = np.array(before), np.array(after)
    changed = np.any(a != b, axis=2)
    assert changed.any()  # fails if shadow missing; product cannot make it pass
    assert np.all(a[changed] <= b[changed])
    assert diag['paste_bit_exact']
    Image.fromarray(b).save(tmp_path/'before.png')
    Image.fromarray(a).save(tmp_path/'after.png')


def test_cached_replay_exact_layers(tmp_path):
    co = _cutout(tmp_path)
    tpl = get_template('clean_studio')
    fn = lambda g,p: np.full(g.shape,200,np.uint8)
    cache = tmp_path/'layers'
    before, _ = compose.composite_with_scene(co,tpl,Aspect.SQUARE,0,fn,cache_dir=cache)
    replay, d = compose.replay_scene(cache)
    assert np.array_equal(np.array(before),np.array(replay))
    assert d['paste_bit_exact']
    candidate, d = compose.replay_scene(cache,shadow_style=tpl.shadow)
    assert not np.array_equal(np.array(before),np.array(candidate))
    assert d['paste_bit_exact']


def test_cli_replay_no_generation(tmp_path):
    from renderello.cli import main
    co=_cutout(tmp_path)
    cache=tmp_path/'layers'
    original,_=compose.composite_with_scene(co,get_template('clean_studio'),Aspect.SQUARE,0,
                        lambda g,p:np.full(g.shape,200,np.uint8),cache_dir=cache)
    output=tmp_path/'replay.png'
    assert main(['scene-replay','--cache',str(cache),'--output',str(output)])==0
    assert np.array_equal(np.array(original),np.array(Image.open(output)))
