import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello.templates import TEMPLATES
from renderello.schemas import Aspect
from renderello.template_locked_preview import preview_reviewed

@pytest.fixture
def inputs(tmp_path):
 yy,xx=np.mgrid[:40,:32];rgb=np.stack([xx*7,yy*5,(xx+yy)*3],axis=-1).astype(np.uint8)
 a=np.zeros((40,32),np.uint8);a[4:36,5:27]=255
 s=tmp_path/'source.png';m=tmp_path/'alpha.png';Image.fromarray(rgb).save(s);Image.fromarray(a).save(m)
 return dict(source=s,alpha_path=m,regions={'core':[7,7,25,33]},root=tmp_path/'out',reviewed=True,material='opaque'),rgb

@pytest.mark.parametrize('key',list(TEMPLATES))
def test_template_declared_aspect_and_native_core(inputs,key):
 kw,rgb=inputs;r=preview_reviewed(**kw,template=key,aspect='template');m=json.loads(Path(r['manifest']).read_text());size=TEMPLATES[key].aspect.size
 assert m['canvas']==list(size) and m['aspect']==TEMPLATES[key].aspect.value and m['aspect_selection']=='explicit'
 for path,audit in zip(r['outputs'],m['audits']):
  im=Image.open(path);assert im.size==size
  dx,dy=audit['integer_translation'];assert np.array_equal(np.array(im)[dy+7:dy+33,dx+7:dx+25],rgb[7:33,7:25])
  assert not audit['source_resampled'] and not audit['source_rgb_repainted']

@pytest.mark.parametrize('aspect',list(Aspect))
def test_explicit_override(inputs,aspect):
 kw,_=inputs;r=preview_reviewed(**kw,aspect=aspect.value);assert Image.open(r['outputs'][0]).size==aspect.size

@pytest.mark.parametrize('aspect',['9:16','',False,7])
def test_bad_aspect_no_output(inputs,aspect):
 kw,_=inputs
 with pytest.raises(ValueError):preview_reviewed(**kw,aspect=aspect)
 assert not kw['root'].exists()

def test_clipped_native_refused_before_output(inputs):
 kw,_=inputs
 with pytest.raises(ValueError,match='no silent'):preview_reviewed(**kw,aspect='3:2',offset=[1040,0])
 assert not kw['root'].exists()

def test_default_legacy_square(inputs):
 kw,_=inputs;r=preview_reviewed(**kw,template='luxury_hero');m=json.loads(Path(r['manifest']).read_text())
 assert m['aspect']=='1:1' and m['aspect_selection']=='legacy_square_default' and Image.open(r['outputs'][0]).size==(1024,1024)
