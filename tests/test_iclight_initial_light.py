import numpy as np
import pytest
from renderello.iclight_gpu import lighting_gradient, ICLight

@pytest.mark.parametrize('direction',['left','right','top','bottom'])
def test_official_endpoints_and_native_shape(direction):
    a=lighting_gradient(512,640,direction)
    assert a.shape==(640,512,3) and a.dtype==np.uint8
    assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2])
    if direction=='left':assert a[0,0,0]==255 and a[0,-1,0]==0 and np.all(np.diff(a[0,:,0].astype(int))<=0)
    if direction=='right':assert a[0,0,0]==0 and a[0,-1,0]==255 and np.all(np.diff(a[0,:,0].astype(int))>=0)
    if direction=='top':assert a[0,0,0]==255 and a[-1,0,0]==0 and np.all(np.diff(a[:,0,0].astype(int))<=0)
    if direction=='bottom':assert a[0,0,0]==0 and a[-1,0,0]==255 and np.all(np.diff(a[:,0,0].astype(int))>=0)


def test_none_and_exact_mirrors():
    assert lighting_gradient(512,640,'none') is None
    assert np.array_equal(lighting_gradient(512,640,'left')[:,::-1],lighting_gradient(512,640,'right'))
    assert np.array_equal(lighting_gradient(512,640,'top')[::-1],lighting_gradient(512,640,'bottom'))


@pytest.mark.parametrize('w,h,d',[(0,640,'left'),(256,1025,'left'),(257,640,'left'),(512.,640,'left'),(True,640,'left'),(512,640,'unknown')])
def test_invalid(w,h,d):
    with pytest.raises(ValueError):lighting_gradient(w,h,d)


@pytest.mark.parametrize('value',[0,1.1,float('nan'),True])
def test_denoise_refused_before_gpu(value):
    m=ICLight()
    with pytest.raises(ValueError,match='lowres_denoise'):m.relight(np.zeros((640,512,3),np.uint8),'test',lowres_denoise=value)
    assert not m._loaded
