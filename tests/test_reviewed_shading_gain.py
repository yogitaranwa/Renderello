import numpy as np
import pytest
from renderello.reviewed_shading_gain import apply_reviewed_shading_gain
from renderello.candidate_diagnostics import digest

def case():
 rgb=np.full((32,32,3),190,np.uint8);a=np.full((32,32),255,np.uint8);s=np.tile(np.linspace(.2,1,32),(32,1));t=np.zeros((32,32),bool);t[3:28,3:28]=True;p=np.zeros_like(t);p[29,10]=True
 return rgb,a,s,t,p

def run(args=None,**kw):
 r,a,s,t,p=args or case();k=dict(source_sha256=digest(r),alpha_sha256=digest(a),shading_sha256=digest(s),target_sha256=digest(t),protected_sha256=digest(p),review_note='Explicit aligned smooth opaque sample; not physical light.',reviewed=True,material='opaque');k.update(kw);return apply_reviewed_shading_gain(r,a,s,t,p,**k)

def test_real():
 args=case();copies=[a.copy() for a in args];o,m=run(args);assert m['changed_pixels']>0;assert np.array_equal(o[~args[3]],args[0][~args[3]]);assert np.all(o<=args[0]);assert all(np.array_equal(a,b) for a,b in zip(args,copies));assert not m['physical_relighting'];assert not m['albedo_replaced']

def test_uniform_identity():
 r,a,s,t,p=case();s[:]=1;o,m=run((r,a,s,t,p));assert np.array_equal(o,r)

def test_boundary_exact():
 from scipy.ndimage import distance_transform_edt
 args=case();o,_=run(args);edge=args[3]&(distance_transform_edt(args[3])==1);assert np.array_equal(o[edge],args[0][edge])

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':'glass'},{'review_note':''},{'review_note':'a'*2001},*[{k:v} for k in ['source_sha256','alpha_sha256','shading_sha256','target_sha256','protected_sha256'] for v in ['0'*64]],*[{k:v} for k in ['sigma','feather'] for v in [True,0,1,33,float('nan'),float('inf'),'4']],*[{k:v} for k in ['strength'] for v in [True,0,-1,2,float('nan')]],*[{'minimum_gain':v} for v in [True,.49,1,float('nan')]],*[{'max_code_delta':v} for v in [True,0,33,2.5]]])
def test_controls(kw):
 with pytest.raises(ValueError):run(**kw)

@pytest.mark.parametrize('what',['shape','nan','negative','int','emptytarget','emptyprotected','overlap','nonopaque','frame','masktype','zero'])
def test_arrays(what):
 r,a,s,t,p=case()
 if what=='shape':s=s[:-1]
 if what=='nan':s[0,0]=np.nan
 if what=='negative':s[0,0]=-1
 if what=='int':s=s.astype(np.uint8)
 if what=='emptytarget':t[:]=False
 if what=='emptyprotected':p[:]=False
 if what=='overlap':p[5,5]=True
 if what=='nonopaque':a[5,5]=254
 if what=='frame':t[0,5]=True
 if what=='masktype':t=t.astype(np.uint8)
 if what=='zero':s[:]=0
 with pytest.raises(ValueError):run((r,a,s,t,p))

def test_budget():
 with pytest.raises(ValueError):run(strength=1,minimum_gain=.5,feather=2,max_code_delta=1)
