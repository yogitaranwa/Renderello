from dataclasses import replace
import numpy as np
import pytest
from renderello.contact_spans import ContactPolicy,ContactSpan,contact_map,render_contact
from renderello.candidate_diagnostics import digest

def fixture():
 rgb=np.full((100,100,3),155,np.uint8);alpha=np.zeros((100,100),np.uint8);alpha[10:60,20:80]=255
 p=ContactPolicy('front','opaque','complete','front',digest(rgb),digest(alpha),reviewed=True,length=20,lateral_sigma=0,cast_dx=0)
 return rgb,alpha,[ContactSpan('base',20,80,59,59)],p

def test_default_identical_and_taper_is_bounded_and_symmetric():
 rgb,a,spans,p=fixture();s,_=contact_map(rgb,a,spans,p);z,_=contact_map(rgb,a,spans,replace(p,span_taper_px=0));t,meta=contact_map(rgb,a,spans,replace(p,span_taper_px=10))
 assert np.array_equal(s,z) and np.all(t<=s) and t[60,20]<s[60,20]*.01
 assert t[60,50]==s[60,50] and np.allclose(t[:,20:80],t[:,20:80][:,::-1])
 assert meta['spans'][0]['span_taper_px']==10 and not np.any(t[a==255])

def test_render_preserves_opaque_rgb_and_source_arrays():
 rgb,a,spans,p=fixture();bg=np.full_like(rgb,220);before=rgb.copy();mask=a.copy()
 out,shade,meta=render_contact(rgb,a,bg,spans,replace(p,span_taper_px=16))
 assert np.array_equal(out[a==255],rgb[a==255]) and np.array_equal(rgb,before) and np.array_equal(a,mask)
 assert np.array_equal(out[shade==0],np.where((a==255)[...,None],rgb,bg)[shade==0])
 assert meta['opaque_rgb_exact'] and meta['automatic_acceptance'] is False

@pytest.mark.parametrize('value',[-1,65,True,1.5,None,'4'])
def test_invalid_taper_refused(value):
 *_,p=fixture()
 with pytest.raises(ValueError):replace(p,span_taper_px=value)

def test_flatlay_cannot_silently_consume_upright_taper():
 *_,p=fixture()
 with pytest.raises(ValueError):replace(p,camera='flatlay',source_pose='overhead',span_taper_px=4)
