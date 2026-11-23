import numpy as np
import pytest
from renderello.scene_finish import finish_background,validate_support_plane


def test_scene_noop_exact_and_never_mutates():
    im=np.arange(240,dtype=np.uint8).reshape(10,8,3);before=im.copy()
    out=finish_background(im);assert np.array_equal(out,im) and out is not im
    lifted=finish_background(im,exposure_ev=.25,white_balance=(1.04,1.,.96))
    assert lifted.shape==im.shape and lifted.dtype==np.uint8
    assert np.array_equal(im,before) and (lifted.astype(int)-im).mean()>0

@pytest.mark.parametrize('settings',[dict(exposure_ev=float('nan')),dict(exposure_ev=.51),dict(white_balance=[1,1]),dict(white_balance=[1,1,float('inf')]),dict(white_balance=[1,1,.5])])
def test_scene_bad_gains_rejected(settings):
    with pytest.raises(ValueError):finish_background(np.full((5,5,3),128,np.uint8),**settings)


def test_support_plane_detects_previously_floating_placement():
    a=np.full((50,30),255,np.uint8);mask=np.zeros((150,150),np.uint8);mask[110:]=255
    with pytest.raises(ValueError,match='misses'):validate_support_plane(mask,a,(40,20))
    assert validate_support_plane(mask,a,(40,70))==30


def test_contact_strength_and_scene_finish_preserve_product_and_clip_plane(tmp_path):
    from renderello.reviewed_source import prepare_reviewed_source
    from renderello.scene_plate import compose_plate
    from PIL import Image,ImageDraw
    src=Image.new('RGB',(80,100),'white');alpha=Image.new('L',src.size,0)
    ImageDraw.Draw(src).rectangle((20,10,60,90),fill=(25,80,140));ImageDraw.Draw(alpha).rectangle((20,10,60,90),fill=255)
    src.save(tmp_path/'source.png');alpha.save(tmp_path/'alpha.png')
    prepare_reviewed_source(tmp_path/'source.png',tmp_path/'prepared',mask=tmp_path/'alpha.png',mask_coordinates='display')
    Image.new('RGB',(1024,1024),(190,190,190)).save(tmp_path/'plate.png')
    plane=Image.new('L',(1024,1024),0);ImageDraw.Draw(plane).rectangle((0,800,1023,1023),fill=255);plane.save(tmp_path/'plane.png')
    with pytest.raises(ValueError,match='misses'):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'bad',baseline=700,support_plane_path=tmp_path/'plane.png')
    assert not (tmp_path/'bad').exists()
    m=compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'ok',baseline=900,support_plane_path=tmp_path/'plane.png',scene_exposure_ev=.2,scene_white_balance=(1.04,1,.97),contact_strength=1.7,background_kind='photograph')
    assert m['background_kind']=='photograph'
    assert m['native_pixel_audit']['status']=='MATCH' and m['support_plane_samples']==41
    assert not np.array(Image.open(tmp_path/'ok/shadow-alpha.png'))[:800].any()
