from copy import deepcopy
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.source_context_extension import extend_source_context


def fixture():
 rgb=np.full((40,32,3),24,np.uint8);rgb[10:30,10:22]=[150,100,80];a=np.zeros((40,32),np.uint8);a[10:30,10:22]=255
 opts=dict(material='opaque',reviewed_source_sha256=digest(rgb),reviewed_alpha_sha256=digest(a),canvas=(64,64),offset=(16,12),band=4)
 return rgb,a,opts


def test_native_whole_scene_and_contact_retained_without_relighting():
 rgb,a,o=fixture();before=rgb.copy();old=a.copy();out,r=extend_source_context(rgb,a,**o)
 assert np.array_equal(out[12:52,16:48],rgb)and np.array_equal(rgb,before)and np.array_equal(a,old)
 assert np.all(out[:10]==24)and r['source_changed_pixels']==0 and r['source_pixels_preserved']==1280
 assert r['new_outer_pixels']==2816 and not r['background_replaced']and not r['photo_accepted']
 assert r['source_lighting_retained']and r['source_contact_retained']and not r['new_contact_synthesized']


@pytest.mark.parametrize('name,value',[('material','transparent'),('material','reflective'),('material','unknown'),('reviewed_source_sha256','0'*64),('reviewed_alpha_sha256','0'*64),('canvas',(31,64)),('canvas',(2000,2000)),('canvas',(64.0,64)),('offset',(-1,0)),('offset',(1.5,2)),('band',1),('band',9),('decay',False),('decay',float('nan')),('decay',float('inf')),('decay',3),('decay',257)])
def test_invalid_or_unverifiable_input_is_not_filled(name,value):
 rgb,a,o=fixture();o[name]=value
 with pytest.raises(ValueError):extend_source_context(rgb,a,**o)


def test_scene_border_foreground_refused():
 rgb,a,o=fixture();a[0,0]=1;o['reviewed_alpha_sha256']=digest(a)
 with pytest.raises(ValueError,match='intersects'):extend_source_context(rgb,a,**o)


def test_empty_product_refused():
 rgb,a,o=fixture();a[:]=0;o['reviewed_alpha_sha256']=digest(a)
 with pytest.raises(ValueError,match='nonempty'):extend_source_context(rgb,a,**o)


def test_complex_scene_not_synthetically_completed():
 rgb,a,o=fixture();rgb[:4:2]=255;o['reviewed_source_sha256']=digest(rgb)
 with pytest.raises(ValueError,match='nonuniform'):extend_source_context(rgb,a,**o)


def test_repeat_and_constant_scene_are_exact():
 rgb,a,o=fixture();x,r=extend_source_context(rgb,a,**o);y,s=extend_source_context(rgb,a,**o);assert np.array_equal(x,y)and r==s


def test_smoothed_outer_does_not_blur_or_reflect_the_product():
 rgb,a,o=fixture();out,r=extend_source_context(rgb,a,**o,smooth_outer=True)
 assert np.array_equal(out[12:52,16:48],rgb)
 assert np.all(out[:12]==24)and np.all(out[:,:16]==24)and np.all(out[:,48:]==24)and np.all(out[52:]==24)
 assert r['smooth_outer']and not r['photo_accepted']


def test_smoothing_option_requires_boolean():
 rgb,a,o=fixture()
 with pytest.raises(ValueError,match='boolean'):extend_source_context(rgb,a,**o,smooth_outer=1)
