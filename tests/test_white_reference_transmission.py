import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.candidate_harmonization import linear_over
from renderello.white_reference_transmission import white_reference_hypothesis


def case():
    s = np.full((30,40,3),255,np.uint8)
    a = np.zeros((30,40),np.uint8)
    a[8:24,12:30] = 255
    s[8:24,12:30] = (180,190,210)
    t = np.full_like(s,35)
    m = np.zeros(a.shape,bool)
    m[:5] = True
    return s,a,t,m


def run(s,a,t,m,**kw):
    args = dict(bindings=dict(source=digest(s),alpha=digest(a),target=digest(t),white_samples=digest(m)),transmission_fraction=.9,reviewed=True,assume_white_reference=True,review_note='Research assumption, not a clean plate or optics.')
    args.update(kw)
    return white_reference_hypothesis(s,a,t,m,**args)


@pytest.mark.parametrize('k',[0,.5,.9,1])
def test_all_hypotheses_reconstruct_source_but_are_not_accepted(k):
    s,a,t,m=case();before=[x.copy()for x in [s,a,t,m]]
    candidate,trans,report=run(s,a,t,m,transmission_fraction=k)
    assert report['layer_reconstruction_changed_pixels']==0
    assert not report['real_glass_validated']and not report['photo_accepted']and not report['source_rgb_locked']
    assert all(np.array_equal(x,y)for x,y in zip(before,[s,a,t,m]))
    assert np.array_equal(candidate[a==0],t[a==0])
    assert np.all(trans>=0)and np.all(trans<=1)


def test_zero_transmission_matches_existing_alpha_compositor():
    s,a,t,m=case();a[8]=128
    candidate,_,_=run(s,a,t,m,transmission_fraction=0)
    assert np.array_equal(candidate,linear_over(s,a,t))


def test_equal_reconstruction_does_not_choose_identical_changed_background():
    s,a,t,m=case();x,_,r=run(s,a,t,m,transmission_fraction=0);y,_,q=run(s,a,t,m,transmission_fraction=1)
    assert r['layer_reconstruction_changed_pixels']==q['layer_reconstruction_changed_pixels']==0
    assert np.any(x[a==255]!=y[a==255])
    assert y[a==255].mean()<x[a==255].mean()


@pytest.mark.parametrize('k',[-.1,1.1,float('nan'),float('inf'),True,'0.9'])
def test_invalid_fraction(k):
    s,a,t,m=case()
    with pytest.raises(ValueError):run(s,a,t,m,transmission_fraction=k)


@pytest.mark.parametrize('field,value',[('reviewed',False),('reviewed',1),('assume_white_reference',False),('assume_white_reference',1),('review_note',''),('review_note','x'*2001),('bindings',{})])
def test_explicit_bound_assumptions_required(field,value):
    s,a,t,m=case()
    with pytest.raises(ValueError):run(s,a,t,m,**{field:value})


@pytest.mark.parametrize('kind',['not-white','overlap','small','shape','dtype','empty-alpha','target-shape','source-dtype'])
def test_bad_source_or_samples(kind):
    s,a,t,m=case()
    if kind=='not-white':s[m]=80
    elif kind=='overlap':a[m]=1
    elif kind=='small':m[:]=False;m[0,:20]=True
    elif kind=='shape':m=m[:-1]
    elif kind=='dtype':m=m.astype('uint8')
    elif kind=='empty-alpha':a[:]=0
    elif kind=='target-shape':t=t[:-1]
    elif kind=='source-dtype':s=s.astype('float32')
    with pytest.raises(ValueError):run(s,a,t,m)


def test_white_target_reconstructs_only_full_coverage_not_boundary_truth():
    s,a,t,m=case();t[:]=255;a[8]=128
    candidate,_,report=run(s,a,t,m)
    assert np.array_equal(candidate[a==255],s[a==255])
    assert report['layer_reconstruction_changed_pixels']==0
    assert np.any(candidate[a==128]!=s[a==128])
