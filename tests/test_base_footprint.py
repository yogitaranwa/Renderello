import numpy as np
import pytest
from renderello.compose import base_footprint_shadow,finish_scene


def test_base_footprint_is_local_soft_and_preserves_opaque_rgb():
    alpha=np.zeros((200,100),np.uint8);alpha[10:185,10:90]=255
    alpha[185:198,10:40]=255;alpha[185:198,60:90]=255
    sh=base_footprint_shadow((400,400),(130,100),alpha)
    assert sh.dtype==np.uint8 and sh.max()<=64
    assert sh[:285].max()==0 # no rectangular cast block beside lower product
    assert sh[:,0].max()==0 and sh[:,-1].max()==0
    crop=np.dstack([np.full((200,100,3),150,np.uint8),alpha]);scene=np.full((400,400,3),220,np.uint8)
    out,_=finish_scene(scene,crop,(130,100),shadow_style='base footprint soft')
    assert np.array_equal(out[100:300,130:230][alpha==255],crop[...,:3][alpha==255])
    with pytest.raises(ValueError):base_footprint_shadow((100,100),(0,0),alpha)
