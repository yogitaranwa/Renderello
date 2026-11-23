"""Visible downsized pixels cannot depend on RGB hidden by zero alpha."""
import cv2
import numpy as np
import pytest
from renderello.compose import fit_product


def render_white(rgba):
    alpha=rgba[...,3:4].astype(np.float64)/255
    return np.rint(rgba[...,:3]*alpha+255*(1-alpha)).astype(np.uint8)


@pytest.mark.parametrize('hidden',[(0,255,0),(20,40,255),(0,0,0)])
def test_downscale_matches_simple_area_weighted_alpha_oracle(hidden):
    # One opaque red-orange and three invisible texels per 2x2 footprint.
    rgba=np.empty((4,4,4),np.uint8);rgba[...,:3]=hidden;rgba[...,3]=0
    rgba[::2,::2]=(240,40,20,255)
    before=rgba.copy()
    crop,_=fit_product(rgba,(0,0,4,4),(10,4),fill=.5)
    expected=np.empty((2,2,4),np.uint8);expected[:]=(240,40,20,64)
    assert np.array_equal(crop,expected)
    assert np.array_equal(rgba,before)


def test_hidden_rgb_invariance_under_noninteger_downsize():
    a=np.zeros((200,200,4),np.uint8);a[:,:,:3]=(0,255,0)
    a[20:180,20:101]=(230,40,20,255)
    b=a.copy();b[b[...,3]==0,:3]=(20,40,255)
    ac,_=fit_product(a,(0,0,200,200),(101,101),fill=.7)
    bc,_=fit_product(b,(0,0,200,200),(101,101),fill=.7)
    assert np.array_equal(ac[...,3],bc[...,3])
    assert np.array_equal(render_white(ac),render_white(bc))


def test_unscaled_rgba_is_bit_exact_including_hidden_colours():
    rng=np.random.default_rng(7);a=rng.integers(0,256,(25,17,4),dtype=np.uint8)
    crop,_=fit_product(a,(0,0,17,25),(101,101),fill=.7)
    assert np.array_equal(a,crop)


def test_fully_opaque_downscale_matches_previous_opencv_area():
    rng=np.random.default_rng(4);a=rng.integers(0,256,(55,71,4),dtype=np.uint8);a[...,3]=255
    crop,_=fit_product(a,(0,0,71,55),(30,30),fill=.7)
    assert np.array_equal(crop,cv2.resize(a,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_AREA))


def test_fully_transparent_downscale_is_invisible_and_finite():
    a=np.full((20,20,4),255,np.uint8);a[...,3]=0
    crop,_=fit_product(a,(0,0,20,20),(10,10),fill=.7)
    assert crop[...,3].max()==0
    assert np.all(render_white(crop)==255)


def test_mixed_alpha_preserves_previous_fully_covered_interior_resize():
    rng=np.random.default_rng(19);a=rng.integers(0,256,(200,200,4),dtype=np.uint8)
    a[...,3]=255;a[:10,:,3]=0;a[-10:,:,3]=0;a[:,:10,3]=0;a[:,-10:,3]=0
    crop,_=fit_product(a,(0,0,200,200),(101,101),fill=.7)
    size=(crop.shape[1],crop.shape[0]);coverage=cv2.resize(a[...,3].astype(np.float64)/255,size,interpolation=cv2.INTER_AREA)
    legacy=cv2.resize(a,size,interpolation=cv2.INTER_AREA)
    assert np.array_equal(crop[coverage>=1-1e-10],legacy[coverage>=1-1e-10])
