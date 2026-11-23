from dataclasses import replace
import numpy as np
import pytest
from renderello.checked_research.context_transition import transition_context
from renderello.candidate_diagnostics import digest


def case():
 rgb=np.full((64,64,3),24,np.uint8);rgb[25:39,25:39]=150;a=np.zeros((64,64),np.uint8);a[25:39,25:39]=255;c=np.zeros_like(a,dtype=bool);c[23:41,23:41]=True
 opts=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),context_sha256=digest(c),reviewed=True,note='Synthetic context, not a real quality claim.',material='opaque',canvas=(80,80),offset=(8,8),feather=16)
 return rgb,a,c,opts


def test_context_and_all_alpha_support_exact():
 rgb,a,c,o=case();rgb[22,30]=25;o['source_sha256']=digest(rgb);out,r=transition_context(rgb,a,c,**o)
 assert np.array_equal(out[8:72,8:72][c],rgb[c])and np.array_equal(out[8:72,8:72][a>0],rgb[a>0])
 assert r['protected_context_changed_pixels']==0 and r['source_alpha_support_changed_pixels']==0 and not r['whole_source_frame_locked']and not r['photo_accepted']


@pytest.mark.parametrize('name,value',[('source_sha256','0'*64),('alpha_sha256','0'*64),('context_sha256','0'*64),('reviewed',False),('reviewed',1),('note',''),('material','transparent'),('feather',1),('feather',65),('feather',True)])
def test_invalid_scope_not_inferred(name,value):
 rgb,a,c,o=case();o[name]=value
 with pytest.raises(ValueError):transition_context(rgb,a,c,**o)


def test_partial_alpha_cannot_escape_preservation():
 rgb,a,c,o=case();a[20,20]=1;o['alpha_sha256']=digest(a)
 with pytest.raises(ValueError,match='every nonzero'):transition_context(rgb,a,c,**o)


def test_frame_seam_refused():
 rgb,a,c,o=case();c[4,4]=True;o['context_sha256']=digest(c)
 with pytest.raises(ValueError,match='seam'):transition_context(rgb,a,c,**o)
