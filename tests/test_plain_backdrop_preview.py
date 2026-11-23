from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from renderello.plain_backdrop_preview import plain_backdrop_alpha,preview_plain_backdrop

def fixture():
    a=np.full((80,100,3),255,dtype=np.uint8)
    a[15:65,20:80]=[35,45,55]
    a[30:40,85:92]=[80,90,100]  # disconnected opaque part remains
    return Image.fromarray(a)

def test_retains_disconnected_parts():
    im=fixture();alpha,metadata=plain_backdrop_alpha(im,material='opaque')
    a=np.array(alpha)
    assert a[35,88]==255 and a[20,30]==255 and a[0,0]==0
    assert 0<metadata['coverage']<1

def test_transparency_refused():
    with pytest.raises(ValueError,match='explicitly opaque'):
        plain_backdrop_alpha(fixture(),material='transparent')

def test_unknown_material_refused():
    with pytest.raises(ValueError):plain_backdrop_alpha(fixture(),material='unknown')

def test_untested_threshold_refused():
    with pytest.raises(ValueError):plain_backdrop_alpha(fixture(),material='opaque',threshold=12)

def test_nonuniform_border_refused():
    a=np.array(fixture());a[:10,:,0]=0
    with pytest.raises(ValueError,match='Nonuniform'):
        plain_backdrop_alpha(Image.fromarray(a),material='opaque')

def test_empty_foreground_refused():
    with pytest.raises(ValueError,match='coverage'):
        plain_backdrop_alpha(Image.new('RGB',(80,100),'white'),material='opaque')

def test_saved_manifest_and_native_rgb_identity(tmp_path):
    src=tmp_path/'source.png';fixture().save(src)
    result=preview_plain_backdrop(src,tmp_path/'new',material='opaque')
    m=json.loads(Path(result['manifest']).read_text())
    assert m['paid_credits']==0 and m['opaque_rgb_changed']==0
    assert m['source_resampled'] is False and len(result['outputs'])==2
    assert all(type(x) is int for x in m['integer_translation'])
    alpha=np.asarray(Image.open(tmp_path/'new'/'experimental-alpha.png'));ys,xs=np.nonzero(alpha)
    dx,dy=m['integer_translation'];original=np.asarray(Image.open(src))
    for p in result['outputs']:
        saved=np.asarray(Image.open(p))
        assert np.array_equal(original[ys,xs],saved[ys+dy,xs+dx])

def test_preserves_existing_output(tmp_path):
    src=tmp_path/'source.png';fixture().save(src);root=tmp_path/'existing';root.mkdir()
    sentinel=root/'sentinel.txt';sentinel.write_text('preserve')
    with pytest.raises(ValueError,match='Preserve'):
        preview_plain_backdrop(src,root,material='opaque')
    assert sentinel.read_text()=='preserve'
