from dataclasses import replace
import numpy as np
import pytest
from renderello.reviewed_penumbra import PenumbraProfile,cast_reviewed_penumbra
from renderello.reviewed_silhouette_cast import SilhouetteCast
from renderello.candidate_diagnostics import digest

@pytest.fixture
def case():
 rgb=np.full((160,180,3),120,np.uint8);a=np.zeros((160,180),np.uint8);a[30:101,50:71]=255
 bg=np.full(rgb.shape,220,np.uint8);contact=np.zeros(a.shape,float);contact[101:106,50:71]=.2
 r=SilhouetteCast(100,50,71,.3,.2,4.,.2,digest(rgb),digest(a),digest(bg),digest(contact),'Synthetic reviewed support, no physical truth.',reviewed=True)
 return [rgb,a,bg,contact,r,PenumbraProfile(.75,4.,.6,8)]

def test_deterministic_native_inputs_unchanged(case):
 copies=[x.copy() for x in case[:4]];a=cast_reviewed_penumbra(*case);b=cast_reviewed_penumbra(*case)
 assert all(np.array_equal(x,y) for x,y in zip(a[:3],b[:3]))
 assert all(np.array_equal(x,y) for x,y in zip(case[:4],copies))
 assert np.array_equal(a[0][case[1]==255],case[0][case[1]==255])
 assert np.isfinite(a[1]).all() and a[1].min()>=0 and a[1].max()<=case[4].opacity
 assert not a[1][:100].any() and not a[1][case[1]>=128].any()
 assert not a[3]['accepted'] and not a[3]['physical_relighting']

@pytest.mark.parametrize('kw',[{'near_sigma':0},{'far_sigma':13},{'far_sigma':.5},{'near_sigma':float('nan')},{'levels':3},{'levels':13},{'levels':True},{'far_opacity_ratio':0},{'far_opacity_ratio':float('inf')},{'far_opacity_ratio':True}])
def test_profile_refusals(case,kw):
 case[-1]=replace(case[-1],**kw)
 with pytest.raises(ValueError):cast_reviewed_penumbra(*case)

def test_wrong_margin(case):
 case[-1]=replace(case[-1],far_sigma=5)
 with pytest.raises(ValueError,match='margin'):cast_reviewed_penumbra(*case)

def test_stale_source(case):
 case[0][0,0]=0
 with pytest.raises(ValueError,match='binding'):cast_reviewed_penumbra(*case)

def test_untyped(case):
 case[-1]={}
 with pytest.raises(ValueError,match='typed'):cast_reviewed_penumbra(*case)

def test_constant_profile_agrees_uniform(case):
 from renderello.reviewed_silhouette_cast import cast_source_silhouette
 case[-1]=PenumbraProfile(4.,4.,1.,8);new=cast_reviewed_penumbra(*case);old=cast_source_silhouette(*case[:-1])
 assert np.allclose(new[1],old[1],atol=1e-15) and np.array_equal(new[0],old[0])
