import json
import numpy as np
from PIL import Image
import pytest
from renderello.compose import anchored_contact_shadow,finish_scene,replay_scene


def test_contact_centers_on_lowest_matte_support_not_padding():
    a=np.zeros((100,100),np.uint8);a[10:70,30:70]=255
    s=anchored_contact_shadow((200,200),(40,30),a)
    assert s[100,89]>100
    assert not s[140:].any() and not s[:50].any()


def test_separate_products_get_distinct_support_rows():
    a=np.zeros((100,100),np.uint8);a[5:40,10:30]=255;a[20:80,65:85]=255
    s=anchored_contact_shadow((150,150),(10,10),a)
    assert s[50,30]>100 and s[90,85]>100
    assert s[90,30]<5 and s[50,85]<5


def test_finish_preserves_product_and_changes_only_scene():
    scene=np.full((200,200,3),200,np.uint8);crop=np.zeros((80,80,4),np.uint8);crop[5:60,15:65,:3]=[35,80,150];crop[5:60,15:65,3]=255
    out,_=finish_scene(scene,crop,(50,50),shadow_style='anchored soft')
    assert np.array_equal(out[55:110,65:115],crop[5:60,15:65,:3])
    assert out[110:118,80:100].min()<180
    assert np.array_equal(out[:40],scene[:40])


def test_empty_and_frame_bottom_no_wrap():
    a=np.zeros((60,60),np.uint8);assert not anchored_contact_shadow((60,60),(0,0),a).any()
    a[10:,20:40]=255;s=anchored_contact_shadow((60,60),(0,0),a)
    assert s.max()>0 and not s[:30].any()


def test_bounds_rejected():
    with pytest.raises(ValueError):anchored_contact_shadow((50,50),(40,0),np.ones((30,30),np.uint8))


def test_cached_callback_metadata_is_not_a_finish_argument(tmp_path):
    Image.new('RGB',(80,80),'white').save(tmp_path/'scene.png');Image.new('RGBA',(20,20),(10,80,90,255)).save(tmp_path/'product_rgba.png')
    (tmp_path/'layers.json').write_text(json.dumps({'placement':[10,10],'scene_mode':'foreground_conditioned_or_reference','cover_used':False,'glass_backdrop':None,'shadow_style':None,'scene_grade':0,'scene_auto_lift':False}))
    im,d=replay_scene(tmp_path,shadow_style='anchored soft');assert d['paste_bit_exact'] and im.size==(80,80)
