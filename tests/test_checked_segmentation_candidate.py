import numpy as np
import pytest
from renderello.checked_research.segmentation_candidate import import_rgb_exact_candidate as f

def sample():
 s=np.full((12,14,3),127,np.uint8);a=np.zeros((12,14),np.uint8);a[2:10,3:11]=255;a[2,:]=100
 return s,np.dstack([s,a])

def call(s,c,**kw):return f(s,c,pairing_reviewed=True,internal_resampling_declared=True,**kw)

def test_exact_and_independent():
 s,c=sample();out,m=call(s,c);assert np.array_equal(out,c);out[0,0]=0;assert c[0,0,0]==127;assert m['source_rgb_exact'];assert not m['native_model_inference'];assert not m['publication_accepted'];assert not m['production_promoted'];assert m['alpha_accuracy'] is None

@pytest.mark.parametrize('flag',['pairing_reviewed','internal_resampling_declared'])
@pytest.mark.parametrize('value',[False,1,'yes',None])
def test_required_declarations(flag,value):
 s,c=sample();kw=dict(pairing_reviewed=True,internal_resampling_declared=True);kw[flag]=value
 with pytest.raises(ValueError):f(s,c,**kw)

@pytest.mark.parametrize('bad',[None,[],np.zeros((2,2),np.uint8),np.zeros((2,2,4),np.uint8),np.zeros((2,2,3),float),np.zeros((1,2,3),np.uint8),np.zeros((2049,2049,3),np.uint8)])
def test_bad_source(bad):
 _,c=sample()
 with pytest.raises(ValueError):call(bad,c)

@pytest.mark.parametrize('bad',[None,[],np.zeros((12,14,4),float),np.zeros((12,14,3),np.uint8),np.zeros((14,12,4),np.uint8)])
def test_bad_candidate(bad):
 s,_=sample()
 with pytest.raises(ValueError):call(s,bad)

@pytest.mark.parametrize('region',[(0,0),(4,5),(2,4)])
def test_any_source_rgb_change_rejected(region):
 s,c=sample();c[region[0],region[1],0]+=1
 with pytest.raises(ValueError,match='RGB differs'):call(s,c)

@pytest.mark.parametrize('alpha',[0,100,128,255])
def test_trivial_alpha_rejected(alpha):
 s,c=sample();c[:,:,3]=alpha
 with pytest.raises(ValueError):call(s,c)

def test_strided_and_immutable_inputs():
 s,c=sample();s=s[:,::-1];c=c[:,::-1];s.setflags(write=False);c.setflags(write=False);o,m=call(s,c);assert np.array_equal(o,c);assert m['fractional_pixels']==14
