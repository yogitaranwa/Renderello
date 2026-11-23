import numpy as np
import pytest
from PIL import Image
from renderello.cached_native_scene import compose_cached_native


def arrays():
    rng = np.random.default_rng(55)
    source = rng.integers(0,256,(40,30,3),dtype=np.uint8)
    alpha = np.full((40,30),255,np.uint8)
    alpha[:5] = 0
    alpha[5:10] = 128
    scene = rng.integers(0,256,(100,120,3),dtype=np.uint8)
    return source,alpha,scene


def test_native_paste_matches_pillow_without_resize_or_repaint():
    source,alpha,scene = arrays()
    out,report=compose_cached_native(source,alpha,scene,translation=(20,30))
    expected=Image.fromarray(scene).convert('RGBA')
    expected.alpha_composite(Image.fromarray(np.dstack((source,alpha)),'RGBA'),(20,30))
    assert np.array_equal(out,np.array(expected.convert('RGB')))
    assert report['opaque_source_rgb_exact_pixels']==900
    assert report['source_resampled'] is False and report['source_rgb_repainted'] is False
    assert report['extra_api_calls']==0 and report['photo_accepted'] is False
    assert np.array_equal(out[30:70,20:50][alpha==255],source[alpha==255])
    assert np.array_equal(out[:20],scene[:20])


def test_zero_shadow_exactly_matches_no_shadow_and_does_not_mutate_inputs():
    source,alpha,scene=arrays();saved=[a.copy()for a in (source,alpha,scene)]
    first,_=compose_cached_native(source,alpha,scene,translation=(1,1))
    second,_=compose_cached_native(source,alpha,scene,translation=(1,1),shadow=np.zeros(scene.shape[:2],float))
    assert np.array_equal(first,second)
    for original,prior in zip((source,alpha,scene),saved):assert np.array_equal(original,prior)


def test_shadow_only_changes_declared_background_and_keeps_opaque_rgb():
    source,alpha,scene=arrays();shadow=np.zeros(scene.shape[:2],float);shadow[75:80,20:50]=.3
    result,_=compose_cached_native(source,alpha,scene,translation=(20,30),shadow=shadow)
    assert np.array_equal(result[30:70,20:50][alpha==255],source[alpha==255])
    assert np.any(result[75:80,20:50]!=scene[75:80,20:50])
    assert np.array_equal(result[:20],scene[:20])


@pytest.mark.parametrize('translation',[(-1,0),(0,-1),(True,0),(0.,0),(100,90),(0,),None])
def test_invalid_placement_is_refused_not_fitted(translation):
    source,alpha,scene=arrays()
    with pytest.raises(ValueError):compose_cached_native(source,alpha,scene,translation=translation)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,.51])
def test_invalid_shadow_refused_not_clamped(value):
    source,alpha,scene=arrays();shadow=np.zeros(scene.shape[:2],float);shadow[90,90]=value
    with pytest.raises(ValueError):compose_cached_native(source,alpha,scene,translation=(0,0),shadow=shadow)


def test_shadow_crossing_product_is_refused():
    source,alpha,scene=arrays();shadow=np.zeros(scene.shape[:2],float);shadow[20,20]=.1
    with pytest.raises(ValueError):compose_cached_native(source,alpha,scene,translation=(0,0),shadow=shadow)


def test_empty_product_refused():
    source,alpha,scene=arrays();alpha[:]=0
    with pytest.raises(ValueError):compose_cached_native(source,alpha,scene,translation=(0,0))
