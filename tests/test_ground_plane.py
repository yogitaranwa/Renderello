import numpy as np
import pytest
from renderello.ground_plane import studio_foreground


def fixture():
    scene=np.full((100,120,3),111,np.uint8)
    crop=np.full((50,30,4),255,np.uint8)
    return scene,crop,(40,30)


def test_foreground_only_and_inputs_unchanged():
    s,c,p=fixture();out,d=studio_foreground(s,c,p,horizon=60,color=(170,165,150))
    assert np.array_equal(out[:60],s[:60])
    assert (s==111).all() and (c==255).all()
    assert d['foot_row']==79 and d['automatic_geometry'] is False
    assert not np.array_equal(out[60:],s[60:])


@pytest.mark.parametrize('horizon',[-1,79,100,60.2])
def test_invalid_horizon(horizon):
    s,c,p=fixture()
    with pytest.raises(ValueError):studio_foreground(s,c,p,horizon=horizon,color=(1,2,3))


@pytest.mark.parametrize('color',[(256,0,0),(-1,2,3),(1,2),(float('nan'),2,3)])
def test_invalid_color(color):
    s,c,p=fixture()
    with pytest.raises(ValueError):studio_foreground(s,c,p,horizon=60,color=color)


def test_no_clipping_or_empty_alpha():
    s,c,p=fixture()
    with pytest.raises(ValueError):studio_foreground(s,c,(110,30),horizon=60,color=(1,2,3))
    c[...,3]=0
    with pytest.raises(ValueError):studio_foreground(s,c,p,horizon=60,color=(1,2,3))
