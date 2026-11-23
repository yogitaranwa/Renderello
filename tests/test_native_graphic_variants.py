import json
import numpy as np
import pytest
from PIL import Image
from renderello.native_graphic_variants import render_variants,PRESETS

@pytest.fixture
def case(tmp_path):
    s=tmp_path/'source.png'; r=tmp_path/'rgba.png'
    rgb=np.full((120,150,3),245,np.uint8);rgb[25:100,20:130]=[30,75,110]
    a=np.zeros((120,150),np.uint8);a[25:100,20:130]=255;a[24,20:130]=90
    Image.fromarray(rgb).save(s);Image.fromarray(np.dstack([rgb,a])).save(r)
    return s,r,tmp_path/'output',rgb,a

def test_variants_are_distinct_source_locked_and_honest(case):
    s,r,o,rgb,a=case;m=render_variants(s,r,o)
    assert len(m['variants'])==4
    assert len(set(x['sha256']for x in m['variants']))==4
    assert m['accepted']==0 and not m['product_resampled'] and not m['physical_contact']
    for row in m['variants']:
        im=Image.open(o/row['output']);assert im.size==(800,800)
        assert 'NOT_PHOTOGRAPHIC' in im.info['status']
        px,py,pw,ph=row['placement'];v=np.array(im)[py:py+ph,px:px+pw]
        assert np.array_equal(v[1:],[30,75,110]*np.ones_like(v[1:]))
        assert row['opaque_changed_pixels']==0
    assert json.loads((o/'manifest.json').read_text())==m

def test_existing_output_is_preserved(case):
    s,r,o,*_=case;o.mkdir();sentinel=o/'keep';sentinel.write_text('old')
    with pytest.raises(FileExistsError):render_variants(s,r,o)
    assert sentinel.read_text()=='old'

def test_repainted_input_is_refused(case):
    s,r,o,*_=case;im=np.array(Image.open(r));im[0,0,0]=17;Image.fromarray(im).save(r)
    with pytest.raises(ValueError,match='repaints'):render_variants(s,r,o)
    assert not o.exists()

def test_empty_mask_is_refused(case):
    s,r,o,rgb,_=case;Image.fromarray(np.dstack([rgb,np.zeros(rgb.shape[:2],np.uint8)])).save(r)
    with pytest.raises(ValueError,match='Empty'):render_variants(s,r,o)
    assert not o.exists()

def test_size_mismatch_is_refused(case):
    s,r,o,*_=case;Image.new('RGBA',(20,30)).save(r)
    with pytest.raises(ValueError,match='dimensions'):render_variants(s,r,o)

def test_deterministic_replay(case):
    s,r,o,*_=case;a=render_variants(s,r,o);b=render_variants(s,r,o.with_name('other'))
    assert [x['sha256']for x in a['variants']]==[x['sha256']for x in b['variants']]


def test_source_icc_is_carried_without_color_conversion(case):
    from PIL import ImageCms
    s,r,o,rgb,_=case
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(rgb).save(s,icc_profile=profile)
    m=render_variants(s,r,o)
    assert m['source_icc_preserved']
    assert Image.open(o/m['variants'][0]['output']).info['icc_profile']==profile
