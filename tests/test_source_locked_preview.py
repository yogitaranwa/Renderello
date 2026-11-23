from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from renderello.source_locked_preview import compose_native

@pytest.fixture
def sample(tmp_path):
    a=np.arange(20*30*3,dtype='uint8').reshape((20,30,3));Image.fromarray(a).save(tmp_path/'source.png')
    m=np.zeros((20,30),dtype='uint8');m[2:18,4:26]=255;Image.fromarray(m).save(tmp_path/'mask.png')
    return tmp_path

def run(p,**kw):
    opts=dict(protected_regions={'label':[7,5,22,15]},canvas=(64,64));opts.update(kw)
    return compose_native(p/'source.png',p/'mask.png',p/'out',**opts)

def test_pixels_match_actual_output(sample):
    m=run(sample);assert m['all_opaque_audit']['changed_pixels']==0;assert m['all_opaque_audit']['pixels']==16*22
    a=np.array(Image.open(sample/'source.png'));b=np.array(Image.open(sample/'out/output.png'));dx,dy=m['integer_translation'];assert np.array_equal(a[5:15,7:22],b[5+dy:15+dy,7+dx:22+dx]);assert not m['source_resampled'];assert not m['rgb_relit']

def test_old_version_not_overwritten(sample):
    run(sample)
    with pytest.raises(ValueError,match='output exists'):run(sample)

def test_requires_regions(sample):
    with pytest.raises(ValueError,match='required'):run(sample,protected_regions={})

def test_translucent_protected_region_rejected(sample):
    m=Image.open(sample/'mask.png');m.putpixel((10,10),254);m.save(sample/'mask.png')
    with pytest.raises(ValueError,match='opaque'):run(sample)

@pytest.mark.parametrize('offset',[(0.1,0),(True,0),(-10,-10),(64,0)])
def test_bad_offset(sample,offset):
    with pytest.raises(ValueError):run(sample,offset=offset)

@pytest.mark.parametrize('canvas',[(10,),(-1,64),(64.5,64),(5000,5000)])
def test_invalid_canvas(sample,canvas):
    with pytest.raises(ValueError):run(sample,canvas=canvas)

def test_mask_size(sample):
    Image.new('L',(10,10),255).save(sample/'mask.png')
    with pytest.raises(ValueError,match='native'):run(sample)

def test_empty_mask(sample):
    Image.new('L',(30,20),0).save(sample/'mask.png')
    with pytest.raises(ValueError,match='empty'):run(sample)

def test_region_outside_source(sample):
    with pytest.raises(ValueError,match='outside'):run(sample,protected_regions={'label':[0,0,100,100]})

def test_manifest_limitations_and_alpha(sample):
    m=run(sample);assert len(m['limitations'])==4;assert m['kind'].endswith('NOT_QUALITY_APPROVAL');a=Image.open(sample/'out/source-native-cutout.png');assert a.mode=='RGBA';assert a.getpixel((10,10))[3]==255
