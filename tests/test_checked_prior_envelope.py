import numpy as np
import pytest
from renderello.checked_research.prior_envelope import constrain_alpha as f

def sample():
 a=np.full((20,24),128,np.uint8);p=np.zeros_like(a);p[6:14,7:17]=255;return a,p

def call(a,p,**kw):return f(a,p,native_pairing_reviewed=True,prior_support_reviewed=True,**kw)

@pytest.mark.parametrize('radius',[0,1,2,4,12])
def test_distance_semantics(radius):
 a,p=sample();o,m=call(a,p,radius=radius);y,x=np.indices(p.shape);dy=np.maximum(np.maximum(6-y,y-13),0);dx=np.maximum(np.maximum(7-x,x-16),0);expected=dx*dx+dy*dy<=radius*radius;assert np.array_equal(o,np.where(expected,a,0));assert (o<=a).all();assert m['prior_positive_lost_pixels']==0;assert m['boundary_accuracy'] is None;assert not m['publication_accepted'];assert not m['production_promoted']
@pytest.mark.parametrize('flag',['native_pairing_reviewed','prior_support_reviewed'])
@pytest.mark.parametrize('v',[False,1,None,'yes'])
def test_review_required(flag,v):
 a,p=sample();kw=dict(native_pairing_reviewed=True,prior_support_reviewed=True);kw[flag]=v
 with pytest.raises(ValueError):f(a,p,**kw)
@pytest.mark.parametrize('v',[-1,13,2.,True,None])
def test_bad_radius(v):
 a,p=sample()
 with pytest.raises(ValueError):call(a,p,radius=v)
@pytest.mark.parametrize('a',[None,[],np.zeros((20,24),float),np.zeros((20,24,1),np.uint8),np.zeros((1,24),np.uint8),np.zeros((1500,1500),np.uint8)])
def test_bad_candidate(a):
 _,p=sample()
 with pytest.raises(ValueError):call(a,p)
@pytest.mark.parametrize('p',[None,[],np.zeros((20,24),float),np.zeros((24,20),np.uint8)])
def test_bad_prior(p):
 a,_=sample()
 with pytest.raises(ValueError):call(a,p)
@pytest.mark.parametrize('kind',['emptyprior','fullprior','emptycandidate'])
def test_nontrivial(kind):
 a,p=sample()
 if kind=='emptyprior':p[:]=0
 elif kind=='fullprior':p[:]=255
 else:a[:]=0
 with pytest.raises(ValueError):call(a,p)
def test_does_not_recover_missing_candidate():
 a,p=sample();a[8,9]=0;o,m=call(a,p);assert o[8,9]==0;assert m['prior_positive_lost_pixels']==1

def test_input_unchanged_and_output_independent():
 a,p=sample();aa=a.copy();pp=p.copy();o,m=call(a,p);o[:]=0;assert np.array_equal(a,aa);assert np.array_equal(p,pp)
