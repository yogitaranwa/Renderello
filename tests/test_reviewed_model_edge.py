import numpy as np
import pytest
from renderello.reviewed_model_edge import import_reviewed_model_edge
from renderello.candidate_diagnostics import digest


def fixture():
 rgb=np.full((20,24,3),100,np.uint8);alpha=np.zeros((20,24),np.uint8);alpha[4:16,5:19]=255;alpha[4:16,4]=100
 estimate=np.full_like(rgb,240);target=np.zeros(alpha.shape,bool);target[5:15,4:7]=True
 protected=np.zeros_like(target);protected[8:12,10:15]=True
 kw=dict(source_sha256=digest(rgb),alpha_sha256=digest(alpha),estimate_sha256=digest(estimate),target_sha256=digest(target),protected_sha256=digest(protected),reviewed=True,material='opaque',review_note='test review')
 return [rgb,alpha,estimate,target,protected],kw


def test_exact_scope_and_cap():
 a,k=fixture();before=[v.copy()for v in a];out,m=import_reviewed_model_edge(*a,**k)
 assert all(np.array_equal(x,y)for x,y in zip(a,before))
 assert np.array_equal(out[~a[3]],a[0][~a[3]])
 assert np.array_equal(out[a[4]],a[0][a[4]])
 assert 0<m['observed_max_code_delta']<=24
 assert m['changed_rgb_pixels']>0 and not m['accepted'] and not m['alpha_changed'] and not m['model_called']


def test_direction_and_identity():
 a,k=fixture();a[2][:]=[200,10,100];k['estimate_sha256']=digest(a[2]);out,m=import_reviewed_model_edge(*a,**k)
 assert (out[a[3],0]>=100).all() and (out[a[3],1]<=100).all() and (out[a[3],2]==100).all()
 a[2]=a[0].copy();k['estimate_sha256']=digest(a[2]);out,m=import_reviewed_model_edge(*a,**k)
 assert np.array_equal(out,a[0]) and m['changed_rgb_pixels']==0


def test_feather_tapers():
 a,k=fixture();out1,_=import_reviewed_model_edge(*a,**k,feather=1);out8,_=import_reviewed_model_edge(*a,**k,feather=8)
 assert np.abs(out8.astype(int)-a[0]).sum()<np.abs(out1.astype(int)-a[0]).sum()


@pytest.mark.parametrize('name,value',[('reviewed',False),('reviewed',1),('material','glass'),('review_note',''),('review_note','x'*2001),('edge_width',True),('edge_width',float('nan')),('edge_width',9),('edge_width',0),('strength',0),('strength',float('inf')),('strength',True),('strength',1.1),('max_code_delta',True),('max_code_delta',0),('max_code_delta',33),('feather',True),('feather',0),('feather',9),('feather',2.0)])
def test_refuses_controls(name,value):
 a,k=fixture();k[name]=value
 with pytest.raises(ValueError):import_reviewed_model_edge(*a,**k)


@pytest.mark.parametrize('name',['source','alpha','estimate','target','protected'])
def test_changed_binding(name):
 a,k=fixture();k[name+'_sha256']='0'*64
 with pytest.raises(ValueError,match='binding'):import_reviewed_model_edge(*a,**k)


@pytest.mark.parametrize('case',['overlap','empty','zero_alpha','interior','wrong_mask','estimate_shape','no_anchor','frame_all','background_all'])
def test_refuses_regions(case):
 a,k=fixture()
 if case=='overlap':a[3][8,10]=True
 if case=='empty':a[3][:]=False
 if case=='zero_alpha':a[3][0,0]=True
 if case=='interior':
  a[1][:]=255;a[1][0]=0;a[3][:]=False;a[3][10,10]=True
 if case=='wrong_mask':a[3]=a[3].astype(np.uint8)
 if case=='estimate_shape':a[2]=a[2][:-1]
 if case=='no_anchor':a[4][:]=False;a[4][0,0]=True
 if case=='frame_all':a[1][:]=255
 if case=='background_all':a[1][:]=0
 for n,v in zip(['source','alpha','estimate','target','protected'],a):k[n+'_sha256']=digest(v)
 with pytest.raises(ValueError):import_reviewed_model_edge(*a,**k)
