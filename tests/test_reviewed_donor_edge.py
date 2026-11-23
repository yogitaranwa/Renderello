import numpy as np
import pytest
from renderello.reviewed_donor_edge import transfer_reviewed_donor_edge
from renderello.candidate_diagnostics import digest


def data():
 r=np.full((30,30,3),100,np.uint8);a=np.zeros((30,30),np.uint8);a[5:25,5:25]=255;a[5:25,5]=160;r[5:25,8]=180;t=np.zeros_like(a,bool);t[8:22,5]=True;d=np.zeros_like(t);d[8:22,8]=True;p=np.zeros_like(t);p[10:20,12:20]=True
 k=dict(donor_sha256=digest(d),source_sha256=digest(r),alpha_sha256=digest(a),target_sha256=digest(t),protected_sha256=digest(p),reviewed=True,material='opaque',review_note='synthetic reviewed material',strength=1.,max_code_delta=16,feather=1)
 return r,a,t,d,p,k


def test_exact_bounds():
 r,a,t,d,p,k=data();o,m=transfer_reviewed_donor_edge(r,a,t,d,p,**k);assert np.all(o[t]==116);assert np.array_equal(o[~t],r[~t]);assert not m['accepted'];assert m['alpha_changed']is False


@pytest.mark.parametrize('case',['distance','donor_hash','donor_empty','donor_float','donor_shape','overlap','not_opaque','target_empty','target_float','review','material','source_hash'])
def test_refusals(case):
 r,a,t,d,p,k=data()
 if case=='distance':k['max_donor_distance']=2
 if case=='donor_hash':k['donor_sha256']='bad'
 if case=='donor_empty':d[:]=False
 if case=='donor_float':d=d.astype(float)
 if case=='donor_shape':d=d[:-1]
 if case=='overlap':d|=t;k['donor_sha256']=digest(d)
 if case=='not_opaque':a[d]=253;k['alpha_sha256']=digest(a)
 if case=='target_empty':t[:]=False
 if case=='target_float':t=t.astype(float)
 if case=='review':k['reviewed']=False
 if case=='material':k['material']='glass'
 if case=='source_hash':k['source_sha256']='bad'
 with pytest.raises(ValueError):transfer_reviewed_donor_edge(r,a,t,d,p,**k)


def test_near_opaque_donor():
 r,a,t,d,p,k=data();a[d]=254;k["alpha_sha256"]=digest(a);o,m=transfer_reviewed_donor_edge(r,a,t,d,p,**k);assert m["alpha_changed"]is False
