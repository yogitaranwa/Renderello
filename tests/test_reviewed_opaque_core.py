import numpy as np
import pytest
from renderello.reviewed_opaque_core import restore_reviewed_opaque_core
from renderello.candidate_diagnostics import digest


def fixture():
 rgb=np.full((60,70,3),130,np.uint8);alpha=np.zeros((60,70),np.uint8);alpha[5:55,5:65]=254;region=np.zeros(alpha.shape,bool);region[20:40,20:50]=True
 kw=dict(source_sha256=digest(rgb),alpha_sha256=digest(alpha),region_sha256=digest(region),reviewed=True,material='opaque',review_note='Reviewed test opaque region')
 return [rgb,alpha,region],kw


def test_scope_exact_and_no_input_mutation():
 a,k=fixture();before=[x.copy()for x in a];rgb,out,m=restore_reviewed_opaque_core(*a,**k)
 assert all(np.array_equal(x,y)for x,y in zip(a,before));assert np.array_equal(rgb,a[0]);assert np.array_equal(out[~a[2]],a[1][~a[2]])
 assert np.all(out>=a[1]);assert np.all(out[25:35,25:45]==255);assert m['maximum_alpha_gain']==1 and m['new_opaque_pixels']>0;assert not m['accepted']


def test_feather_and_deficit():
 a,k=fixture();a[1][a[2]]=247;k['alpha_sha256']=digest(a[1]);_,out,m=restore_reviewed_opaque_core(*a,**k)
 assert out[20,20]==249 and out[30,30]==255;assert m['maximum_alpha_gain']==8


def test_already_opaque_identity():
 a,k=fixture();a[1][a[2]]=255;k['alpha_sha256']=digest(a[1]);_,out,m=restore_reviewed_opaque_core(*a,**k)
 assert np.array_equal(out,a[1]) and m['alpha_changed_pixels']==0


@pytest.mark.parametrize('field,value',[('reviewed',False),('reviewed',1),('material','glass'),('review_note',''),('review_note','x'*2001),('max_deficit',True),('max_deficit',0),('max_deficit',17),('max_deficit',8.),('clearance',True),('clearance',3),('clearance',65),('feather',True),('feather',1),('feather',17)])
def test_invalid_controls(field,value):
 a,k=fixture();k[field]=value
 with pytest.raises(ValueError):restore_reviewed_opaque_core(*a,**k)


@pytest.mark.parametrize('field',['source_sha256','alpha_sha256','region_sha256'])
def test_binding(field):
 a,k=fixture();k[field]='0'*64
 with pytest.raises(ValueError,match='binding'):restore_reviewed_opaque_core(*a,**k)


@pytest.mark.parametrize('case',['boundary','frame','deficit','empty','wrong_dtype','wrong_shape','thin','all_fg'])
def test_bad_regions(case):
 a,k=fixture()
 if case=='boundary':a[2][5,5]=True
 if case=='frame':a[1][:]=254;a[1][-1]=0;a[2][0,30]=True
 if case=='deficit':a[1][30,30]=246
 if case=='empty':a[2][:]=False
 if case=='wrong_dtype':a[2]=a[2].astype('uint8')
 if case=='wrong_shape':a[2]=a[2][:-1]
 if case=='thin':a[2][:]=False;a[2][30,20:50]=True
 if case=='all_fg':a[1][:]=254
 k['alpha_sha256']=digest(a[1]);k['region_sha256']=digest(a[2])
 with pytest.raises(ValueError):restore_reviewed_opaque_core(*a,**k)
