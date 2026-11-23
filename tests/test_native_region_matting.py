import numpy as np,pytest,sys,types
from renderello.matting import refine_native_region,make_trimap
from renderello.candidate_diagnostics import digest


def fixture():
    rgb=np.full((80,100,3),[30,100,150],np.uint8);a=np.zeros((80,100),np.uint8);a[20:60,25:75]=255
    return rgb,a,(10,10,90,70)


def run(rgb,a,roi,**kw):
    args=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),reviewed=True,review_note='synthetic assistant ROI hypothesis',material='opaque',method='guided',feather=4)
    args.update(kw);return refine_native_region(rgb,a,roi,**args)


def test_native_roi_preserves_outside_and_border_without_resampling():
    rgb,a,roi=fixture();r,b,m=run(rgb,a,roi)
    mask=np.ones(a.shape,bool);mask[10:70,10:90]=False
    assert np.array_equal(r[mask],rgb[mask]) and np.array_equal(b[mask],a[mask])
    for y in [10,69]:assert np.array_equal(b[y],a[y])
    assert not m['resampled'] and m['outside_roi_rgb_alpha_exact']
    assert np.array_equal(r,rgb) and m['changed_alpha_pixels']>0
    assert not m['accepted'] and not m['default_pipeline_changed']


def test_foreground_estimation_never_changes_definite_seeds(monkeypatch):
    rgb,a,roi=fixture()
    fake=types.SimpleNamespace(estimate_alpha_cf=lambda i,t:t,estimate_alpha_lkm=lambda i,t:t,estimate_foreground_ml=lambda i,a:np.zeros_like(i))
    monkeypatch.setitem(sys.modules,'pymatting',fake)
    tri=make_trimap(a[10:70,10:90],2)
    r,b,m=run(rgb,a,roi,method='cf',edge_colors=True,trimap=tri)
    assert np.array_equal(r[10:70,10:90][tri!=128],rgb[10:70,10:90][tri!=128])
    assert m['changed_rgb_pixels']>0 and m['estimated_boundary_rgb']
    assert np.array_equal(r[10,:],rgb[10,:]) and np.array_equal(r[:,10],rgb[:,10])


@pytest.mark.parametrize('kw',[dict(source_sha256='bad'),dict(alpha_sha256='bad'),dict(reviewed=False),dict(review_note=''),dict(material='glass'),dict(feather=0),dict(feather=True),dict(feather=31),dict(edge_colors='false')])
def test_invalid_scope_refused(kw):
    rgb,a,roi=fixture()
    with pytest.raises(ValueError):run(rgb,a,roi,**kw)


@pytest.mark.parametrize('roi',[(0,0,101,10),(-1,0,10,10),(20,20,10,10),(1.,2,30,40),(True,2,30,40),(0,0,4,4)])
def test_invalid_roi_refused(roi):
    rgb,a,_=fixture()
    with pytest.raises(ValueError):run(rgb,a,roi)


def test_full_source_over_original_solver_cap_small_roi_only():
    rgb=np.full((1024,1024,3),100,np.uint8);a=np.zeros((1024,1024),np.uint8);a[20:60,25:75]=255
    r,b,m=run(rgb,a,(10,10,90,70))
    assert m['source_pixels']>1_000_000 and m['roi_pixels']==4800 and r.shape==rgb.shape
    assert np.array_equal(b[80:],a[80:])
