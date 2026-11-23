import cv2
import numpy as np
import pytest
from renderello.alpha_resample import resize_rgba
from renderello.compose import cover_fit_if_cut_both_sides

@pytest.mark.parametrize('interp',[cv2.INTER_AREA,cv2.INTER_LANCZOS4])
@pytest.mark.parametrize('size',[(6,7),(43,39)])
def test_hidden_colour_cannot_leak(interp,size):
    a=np.zeros((17,19,4),np.uint8);a[4:13,5:14]=[35,91,147,255]
    b=a.copy();b[b[...,3]==0,:3]=[255,0,255]
    before=b.copy()
    x=resize_rgba(a,size,interpolation=interp);y=resize_rgba(b,size,interpolation=interp)
    assert np.array_equal(x,y)
    assert np.array_equal(b,before)
    assert not y[y[...,3]==0,:3].any()
    visible=y[...,3]>0
    assert np.max(np.abs(y[visible,:3].astype(int)-[35,91,147]))<=1

@pytest.mark.parametrize('interp',[cv2.INTER_AREA,cv2.INTER_LANCZOS4])
def test_opaque_exact_legacy(interp):
    a=np.random.default_rng(11).integers(0,256,(17,19,4),dtype=np.uint8);a[...,3]=255
    assert np.array_equal(resize_rgba(a,(33,31),interpolation=interp),cv2.resize(a,(33,31),interpolation=interp))

def test_cover_fit_filters_premultiplied():
    a=np.zeros((20,30,4),np.uint8);a[8:12,:]=[45,90,120,255]
    b=a.copy();b[b[...,3]==0,:3]=[255,0,255]
    assert np.array_equal(cover_fit_if_cut_both_sides(a,(73,53)),cover_fit_if_cut_both_sides(b,(73,53)))

def test_no_resize_preserves_all_bytes_and_returns_copy():
    a=np.random.default_rng(2).integers(0,256,(9,11,4),dtype=np.uint8)
    b=resize_rgba(a,(11,9),interpolation=cv2.INTER_AREA)
    assert np.array_equal(a,b) and not np.shares_memory(a,b)

@pytest.mark.parametrize('size',[(True,2),(0,3),(3000,3000),(3.5,4)])
def test_size_refused(size):
    with pytest.raises(ValueError):resize_rgba(np.zeros((3,4,4),np.uint8),size,interpolation=cv2.INTER_AREA)
