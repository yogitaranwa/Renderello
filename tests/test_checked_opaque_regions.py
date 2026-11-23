import numpy as np
import pytest
from renderello.checked_research.opaque_regions import restore_reviewed_opaque_regions as f

def inputs():
 s=np.full((16,20,3),80,np.uint8);a=np.zeros((16,20),np.uint8);a[2:14,2:18]=253;r=np.where(a>0,255,0).astype('uint8');return s,np.dstack([s,a]),r,{'label':[4,4,12,10]}
def call(s,c,r,regions,**kw):return f(s,c,r,regions,pairing_reviewed=True,internal_resampling_declared=True,opaque_regions_reviewed=True,**kw)
def test_only_reviewed_regions_change():
 s,c,r,regions=inputs();o,m=call(s,c,r,regions);assert m['alpha_override_pixels']==48;mask=np.zeros(r.shape,bool);mask[4:10,4:12]=True;assert np.array_equal(o[~mask],c[~mask]);assert np.array_equal(o[:,:,:3],s);assert (o[mask,3]==255).all();assert not m['publication_accepted'];assert not m['model_alpha_truth'];assert np.all(c[mask,3]==253)
@pytest.mark.parametrize('v',[False,1,None,'yes'])
def test_region_review_required(v):
 s,c,r,regions=inputs()
 with pytest.raises(ValueError):f(s,c,r,regions,pairing_reviewed=True,internal_resampling_declared=True,opaque_regions_reviewed=v)
@pytest.mark.parametrize('r',[None,[],np.zeros((16,20),float),np.zeros((20,16),np.uint8)])
def test_reference_shape(r):
 s,c,_,regions=inputs()
 with pytest.raises(ValueError):call(s,c,r,regions)
@pytest.mark.parametrize('regions',[None,[],{}, {'':[4,4,12,10]},{'x':[4,4,12]},{'x':[4.,4,12,10]},{'x':[-1,4,12,10]},{'x':[4,4,21,10]},{'x':[12,4,4,10]},{'x':[4,4,12,17]}])
def test_bad_regions(regions):
 s,c,r,_=inputs()
 with pytest.raises(ValueError):call(s,c,r,regions)
def test_reference_must_be_opaque():
 s,c,r,regions=inputs();r[5,5]=254
 with pytest.raises(ValueError):call(s,c,r,regions)
def test_candidate_contradiction_rejected():
 s,c,r,regions=inputs();c[5,5,3]=239
 with pytest.raises(ValueError):call(s,c,r,regions)
def test_overlap_counts_once():
 s,c,r,regions=inputs();regions['same']=[4,4,12,10];o,m=call(s,c,r,regions);assert m['alpha_override_pixels']==48;assert m['unique_region_pixels']==48
def test_rgb_change_rejected():
 s,c,r,regions=inputs();c[5,5,0]+=1
 with pytest.raises(ValueError):call(s,c,r,regions)
