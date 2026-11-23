import numpy as np
import pytest
from renderello.reviewed_local_edge_fit import fit_reviewed_local_edge
from renderello.candidate_diagnostics import digest


def sample():
 r=np.full((20,20,3),20,np.uint8);a=np.zeros((20,20),np.uint8);a[:,8:]=255;r[:,8:]=220;a[:,7]=90;r[:,7]=120;t=np.zeros_like(a,bool);t[4:16,7]=True;f=np.zeros_like(t);f[4:16,9]=True;b=np.zeros_like(t);b[4:16,5]=True;p=np.zeros_like(t);p[5:15,12]=True
 k=dict(zip(['source_sha256','alpha_sha256','target_sha256','foreground_sha256','background_sha256','protected_sha256'],map(digest,[r,a,t,f,b,p])));k.update(reviewed=True,material='opaque',review_note='synthetic local mixture',mixing_space='srgb',strength=1.,feather=1,max_rgb_delta=96)
 return r,a,t,f,b,p,k


def test_projection_invariants():
 r,a,t,f,b,p,k=sample();o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert m['eligible_pixels']==0 and m['rejected_rgb_budget']==12
 k['strength']=.75;r[t]=140;k['source_sha256']=digest(r);o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert m['eligible_pixels']==12;assert np.all(o[t]==200);assert np.all(aa[t]==137);assert np.array_equal(o[~t],r[~t])and np.array_equal(aa[~t],a[~t]);assert not m['accepted']


@pytest.mark.parametrize('key,value',[('reviewed',False),('reviewed',1),('material','glass'),('mixing_space',None),('mixing_space','auto'),('strength',True),('strength',0),('strength',float('nan')),('max_distance',33),('max_distance',True),('max_alpha_delta',129),('max_rgb_delta',97),('max_residual',25),('minimum_contrast',15),('source_sha256','bad'),('foreground_sha256','bad')])
def test_refuse_controls(key,value):
 r,a,t,f,b,p,k=sample();k[key]=value
 with pytest.raises(ValueError):fit_reviewed_local_edge(r,a,t,f,b,p,**k)


@pytest.mark.parametrize('case',['overlap','empty','float','shape','target_opaque','target_zero','foreground_low','background_nonzero','far_donor'])
def test_refuse_masks(case):
 r,a,t,f,b,p,k=sample()
 if case=='overlap':f|=t
 if case=='empty':t[:]=False
 if case=='float':f=f.astype(float)
 if case=='shape':b=b[:-1]
 if case=='target_opaque':a[t]=255
 if case=='target_zero':a[t]=0
 if case=='foreground_low':a[f]=249
 if case=='background_nonzero':a[b]=1
 if case=='far_donor':k['max_distance']=1
 k.update(dict(zip(['source_sha256','alpha_sha256','target_sha256','foreground_sha256','background_sha256','protected_sha256'],map(digest,[r,a,t,f,b,p]))))
 with pytest.raises(ValueError):fit_reviewed_local_edge(r,a,t,f,b,p,**k)


@pytest.mark.parametrize('space',['srgb','linear'])
def test_no_mutation(space):
 r,a,t,f,b,p,k=sample();before=[x.copy()for x in [r,a,t,f,b,p]];k['mixing_space']=space;o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k)
 assert all(np.array_equal(x,y)for x,y in zip([r,a,t,f,b,p],before));assert np.array_equal(aa>0,a>0);assert np.array_equal(aa==255,a==255)


def test_feather_reduces_boundary_change():
 r,a,t,f,b,p,k=sample();r[t]=140;k['source_sha256']=digest(r);k['feather']=4;o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert np.all(o[t]==160);assert np.array_equal(o[~t],r[~t])


def test_low_contrast_rejected():
 r,a,t,f,b,p,k=sample();r[f]=40;k['source_sha256']=digest(r);o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert m['eligible_pixels']==0 and m['rejected_contrast']==12;assert np.array_equal(o,r)


def test_alpha_budget_rejected():
 r,a,t,f,b,p,k=sample();r[t]=140;k['source_sha256']=digest(r);k['max_alpha_delta']=1;o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert m['rejected_alpha_budget']==12;assert np.array_equal(aa,a)


def test_residual_rejected():
 r,a,t,f,b,p,k=sample();r[t]=[140,20,140];k['source_sha256']=digest(r);o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert m['rejected_residual']==12;assert np.array_equal(o,r)


@pytest.mark.parametrize('seed',range(12))
def test_random_bounds(seed):
 r,a,t,f,b,p,k=sample();rng=np.random.default_rng(seed);r[t]=rng.integers(0,256,(int(t.sum()),3),dtype=np.uint8);k['source_sha256']=digest(r);k['strength']=.75;k['feather']=3;o,aa,m=fit_reviewed_local_edge(r,a,t,f,b,p,**k);assert np.max(np.abs(o.astype(int)-r))<=k['max_rgb_delta'];assert np.max(np.abs(aa.astype(int)-a))<=96;assert np.array_equal(o[~t],r[~t]);assert np.array_equal(aa[~t],a[~t])
