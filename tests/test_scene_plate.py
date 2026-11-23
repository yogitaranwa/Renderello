import numpy as np
import pytest
from PIL import Image
from renderello.scene_plate import support_shadow


def test_contact_attaches_to_two_actual_feet_without_product_mutation():
    a=np.zeros((100,90),np.uint8);a[10:70,10:80]=255;a[70:90,10:25]=255;a[70:90,65:80]=255
    before=a.copy();s=support_shadow(a,(256,256),(70,100))
    assert np.array_equal(a,before)
    assert s[190,85]>s[190,115]*2
    assert s[190,140]>s[190,115]*2
    assert s.max()<=.45 and np.isfinite(s).all()


def test_cast_direction_follows_explicit_floor_vector():
    a=np.full((100,50),255,np.uint8)
    plain=support_shadow(a,(256,256),(50,70))
    projected=support_shadow(a,(256,256),(50,70),cast_offset=(80,35))
    assert projected[185,130]>plain[185,130]
    assert projected[185,130]>.02


def test_empty_support_and_outside_placement_refused():
    with pytest.raises(ValueError):support_shadow(np.zeros((10,10),np.uint8),(50,50),(10,10))
    with pytest.raises(ValueError):support_shadow(np.full((10,10),255,np.uint8),(50,50),(45,45))


def test_plate_saved_pixels_native_audit_and_no_overwrite(tmp_path):
    from renderello.reviewed_source import prepare_reviewed_source
    from renderello.scene_plate import compose_plate
    from PIL import ImageDraw
    src=Image.new('RGB',(80,100),'white');a=Image.new('L',src.size,0)
    ImageDraw.Draw(src).rectangle((20,10,60,90),fill=(25,80,140));ImageDraw.Draw(a).rectangle((20,10,60,90),fill=255)
    src.save(tmp_path/'source.png');a.save(tmp_path/'a.png')
    prepare_reviewed_source(tmp_path/'source.png',tmp_path/'prepared',mask=tmp_path/'a.png',mask_coordinates='display')
    plate=tmp_path/'plate.png';Image.new('RGB',(1024,1024),(210,200,190)).save(plate)
    before={p.name:p.read_bytes() for p in (tmp_path/'prepared').iterdir()}
    result=compose_plate(tmp_path/'prepared',plate,tmp_path/'out',baseline=850,cast_offset=(70,25))
    assert result['native_pixel_audit']['status']=='MATCH' and result['source_scale']==1
    im=np.array(Image.open(tmp_path/'out/candidate.png'));x,y=result['placement']
    assert np.all(im[y:y+81,x:x+41]==(25,80,140))
    assert before=={p.name:p.read_bytes() for p in (tmp_path/'prepared').iterdir()}
    with pytest.raises(ValueError,match='exists'):compose_plate(tmp_path/'prepared',plate,tmp_path/'out',baseline=850)
    with pytest.raises(ValueError,match='fit'):compose_plate(tmp_path/'prepared',plate,tmp_path/'outside',baseline=2)
    assert not (tmp_path/'outside').exists()
