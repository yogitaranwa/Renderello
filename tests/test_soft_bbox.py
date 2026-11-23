import numpy as np
import pytest
from renderello.cutout import _bbox_from_matte,_save_cutout
from renderello.source_contract import source_mask_preflight,SourceContract


def test_bbox_retains_all_nonzero_soft_alpha_not_only_thresholded_interior(tmp_path):
    alpha=np.zeros((100,110),np.uint8)
    alpha[30:70,40:80]=255
    alpha[28:72,38:82]=np.maximum(alpha[28:72,38:82],8)
    alpha[27,37]=1
    assert _bbox_from_matte(alpha)==(37,27,45,45)
    rgb=np.full((100,110,3),180,np.uint8)
    co=_save_cutout(rgb,alpha,tmp_path,'soft','test_known_alpha')
    check=source_mask_preflight(rgb,alpha,co.bbox,SourceContract('front','opaque','complete'))
    assert not any('outside declared bbox' in w for w in check['warnings'])


def test_bbox_does_not_silently_remove_disconnected_low_alpha_pixels():
    alpha=np.zeros((60,80),np.uint8);alpha[20:50,20:60]=255;alpha[3,5]=1
    assert _bbox_from_matte(alpha)==(5,3,55,47)
    with pytest.raises(ValueError,match='empty'):_bbox_from_matte(np.zeros_like(alpha))
