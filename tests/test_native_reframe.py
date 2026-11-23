import json
import numpy as np
import pytest
from PIL import Image
from renderello.native_reframe import native_reframe


def cache(tmp_path, size=(100,60), glass=None, cover=False):
    product=np.full((size[1],size[0],4),[99,27,184,255],np.uint8)
    scene=np.full((400,400,3),[180,210,150],np.uint8)
    Image.fromarray(product).save(tmp_path/'product_rgba.png')
    Image.fromarray(scene).save(tmp_path/'scene.png')
    (tmp_path/'layers.json').write_text(json.dumps(dict(placement=[20,30],cover_used=cover,
        glass_backdrop=glass,shadow_style='surface contact')))


def test_smaller_export_not_product_upscale(tmp_path):
    cache(tmp_path)
    im,d=native_reframe(tmp_path)
    assert im.size==(132,132) and d['paste_bit_exact']
    assert d['product_size']==[100,60] and not d['product_resampled']
    assert not d['scene_resampled'] and d['quality_status']=='REVIEW_REQUIRED'
    x,y=d['placement'];assert np.all(np.asarray(im)[y:y+60,x:x+100]==[99,27,184])


def test_never_enlarges_scene(tmp_path):
    cache(tmp_path,(390,350))
    im,d=native_reframe(tmp_path)
    assert im.size==(400,400) and d['paste_bit_exact']


@pytest.mark.parametrize('fill',[0,.99,float('nan'),float('inf')])
def test_invalid_fill(tmp_path,fill):
    with pytest.raises(ValueError):native_reframe(tmp_path,fill=fill)


@pytest.mark.parametrize('glass,cover',[([240,240,240],False),(None,True)])
def test_unsupported_layers(tmp_path,glass,cover):
    cache(tmp_path,glass=glass,cover=cover)
    with pytest.raises(ValueError):native_reframe(tmp_path)


def test_empty_alpha(tmp_path):
    cache(tmp_path)
    Image.fromarray(np.zeros((30,30,4),np.uint8)).save(tmp_path/'product_rgba.png')
    with pytest.raises(ValueError):native_reframe(tmp_path)


def test_transparent_padding_trim_preserves_soft_edges(tmp_path):
    cache(tmp_path)
    p=np.zeros((60,100,4),np.uint8);p[10:50,20:80]=[99,27,184,255];p[9,20:80]=[99,27,184,1]
    Image.fromarray(p).save(tmp_path/'product_rgba.png')
    im,d=native_reframe(tmp_path)
    assert d['product_size']==[60,41] and d['paste_bit_exact']
