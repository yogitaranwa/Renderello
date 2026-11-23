import numpy as np
import pytest
from PIL import Image
from renderello import compose
from renderello.schemas import CutoutResult,Aspect
from renderello.templates import get_template


def fixture(tmp_path):
    a=np.zeros((200,100,4),np.uint8);a[:,:,:3]=(180,80,30);a[:,:,3]=255
    Image.fromarray(a).save(tmp_path/'rgba.png');Image.fromarray(a[:,:,3]).save(tmp_path/'alpha.png')
    return CutoutResult(tmp_path/'rgba.png',tmp_path/'alpha.png',(0,0,100,200),'fixture',1.)


@pytest.mark.parametrize('baseline,expected_rows',[(.86,144),(.4,200),(1.,0)])
def test_reflection_is_clipped_to_visible_floor_not_silently_omitted(tmp_path,monkeypatch,baseline,expected_rows):
    co=fixture(tmp_path);t=get_template('reflective_surface')
    monkeypatch.setattr(compose,'placement_params',lambda *_:(.72,0.,baseline))
    plain=np.array(compose.composite(co,t,Aspect.SQUARE,0,reflection=False))
    reflected=np.array(compose.composite(co,t,Aspect.SQUARE,0,reflection=True))
    crop,(px,py)=compose.fit_product(np.array(Image.open(co.rgba_path)),co.bbox,Aspect.SQUARE.size,fill=.72,jx=0.,baseline=baseline)
    ph,pw=crop.shape[:2];ry=py+ph
    visible=min(ph,1024-ry);assert visible==expected_rows
    difference=np.any(plain!=reflected,axis=2)
    outside=difference.copy();outside[ry:ry+visible,px:px+pw]=False
    assert not outside.any()
    assert np.array_equal(reflected[py:py+ph,px:px+pw],crop[:,:,:3])
    if visible:
        assert difference[ry:ry+visible,px:px+pw].any()
        # The visible slice must keep the original full-reflection fade,
        # not squeeze the entire fade into the shortened canvas region.
        ra=np.linspace(.5,0.,ph,dtype=np.float32)[:visible,None,None]*.35
        expected=(plain[ry:ry+visible,px:px+pw].astype(np.float32)*(1-ra)+crop[::-1,:][:visible,:,:3].astype(np.float32)*ra).clip(0,255).astype(np.uint8)
        assert np.max(np.abs(reflected[ry:ry+visible,px:px+pw].astype(int)-expected.astype(int)))<=1
    else:assert not difference.any()
