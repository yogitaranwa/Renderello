import numpy as np, pytest
from scripts.preflight_clean_product import inspect_arrays

def fixture():
 s=np.full((80,60,3),79,np.uint8);a=np.zeros((80,60),np.uint8);a[10:70,15:45]=255
 return s,np.dstack([s,a]),a

def test_pass_does_not_grant_quality_or_spend():
 r=inspect_arrays(*fixture());assert r['source_rgb_changed']==0;assert r['quality_pass'] is False;assert r['paid_calls']==0;assert r['visual_review'].startswith('PENDING')

def test_pixel_change_rejected():
 s,r,a=fixture();r[30,30,0]+=1
 with pytest.raises(ValueError,match='RGB changed'):inspect_arrays(s,r,a)

def test_resizing_rejected():
 s,r,a=fixture()
 with pytest.raises(ValueError,match='dimensions'):inspect_arrays(s,r[:-1],a)

def test_alpha_mismatch_rejected():
 s,r,a=fixture();r[30,30,3]=0
 with pytest.raises(ValueError,match='alpha differs'):inspect_arrays(s,r,a)

@pytest.mark.parametrize('v',[0,255])
def test_degenerate_rejected(v):
 s,r,a=fixture();a[:]=v;r[:,:,3]=a
 with pytest.raises(ValueError,match='empty or full'):inspect_arrays(s,r,a)

def test_fragmented_warns_not_quality_pass():
 s,r,a=fixture();a[:]=0
 for y in [5,25,45,65]:a[y:y+8,10:18]=255
 r[:,:,3]=a;d=inspect_arrays(s,r,a);assert d['integrity_warning'];assert not d['quality_pass']
