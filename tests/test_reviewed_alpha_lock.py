import numpy as np
import pytest
import renderello.reviewed_alpha_lock as lock
from renderello.candidate_diagnostics import digest

@pytest.fixture
def data(monkeypatch):
    rgb=np.full((56,64,3),110,np.uint8);a=np.zeros((56,64),np.uint8)
    a[8:48,10:54]=255;a[47:49,10:54]=170
    f=np.zeros(a.shape,bool);f[44:52,15:35]=True
    def solver(rgb,a,tri,method,edge_colors):
        assert method=='cf' and edge_colors is False
        out=np.full(a.shape,95,np.uint8);out[tri==0]=0;out[tri==255]=255
        return rgb.copy(),out,{'synthetic_fixture':True}
    monkeypatch.setattr(lock,'refine_arrays',solver)
    return rgb,a,f

def params(data,**kw):
    rgb,a,f=data
    d=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),frozen_sha256=digest(f) if isinstance(f,np.ndarray) else 'invalid',
        reviewed=True,material='opaque',review_note='Synthetic hem region only; prior matte not truth.')
    d.update(kw);return d

def call(data,**kw):return lock.recover_locked_alpha(*data,**params(data,**kw))

def test_preserves_fractional_opaque_and_background(data):
    before=[x.copy() for x in data];fg,a,tri,m=call(data);rgb,initial,f=data
    assert np.array_equal(fg,rgb) and np.array_equal(a[f],initial[f])
    assert (tri[f&(initial==0)]==0).all() and (tri[f&(initial==255)]==255).all()
    assert (tri[f&(initial==170)]==128).all()
    assert m['frozen_pixels_restored_after_solve']==40 and m['frozen_fractional_pixels']==40
    assert m['alpha_changed_pixels']>0 and m['all_source_rgb_exact'] and m['frozen_alpha_exact']
    assert not m['accepted'] and not m['default_pipeline_changed'] and not m['independent_ground_truth']
    assert all(np.array_equal(a,b) for a,b in zip(data,before))
    fg[:]=0;a[:]=0;tri[:]=0
    assert all(np.array_equal(a,b) for a,b in zip(data,before))

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':'glass'},{'material':None},
 {'review_note':''},{'review_note':'   '},{'review_note':None},{'review_note':'a'*2001},
 {'source_sha256':'x'},{'alpha_sha256':'x'},{'frozen_sha256':'x'},
 {'radius':True},{'radius':0},{'radius':13},{'radius':8.},{'radius':np.int32(8)}])
def test_bad_review(data,kw):
    with pytest.raises(ValueError):call(data,**kw)

@pytest.mark.parametrize('kind',['empty','wrongshape','wrongdtype','backgroundonly','notarray'])
def test_bad_frozen(data,kind):
    rgb,a,f=data
    if kind=='empty':f=np.zeros(a.shape,bool)
    if kind=='wrongshape':f=f[:-1]
    if kind=='wrongdtype':f=f.astype('uint8')
    if kind=='backgroundonly':f=np.zeros(a.shape,bool);f[:2,:2]=True
    if kind=='notarray':f=None
    with pytest.raises(ValueError):call((rgb,a,f))

@pytest.mark.parametrize('kind',['rgbshape','rgbfloat','ashape','afloat','nofg','nobg','large'])
def test_bad_native_arrays(data,kind):
    rgb,a,f=data
    if kind=='rgbshape':rgb=rgb[:,:,0]
    if kind=='rgbfloat':rgb=rgb.astype(float)
    if kind=='ashape':a=a[:-1]
    if kind=='afloat':a=a.astype(float)
    if kind=='nofg':a=np.zeros(a.shape,np.uint8)
    if kind=='nobg':a=np.full(a.shape,255,np.uint8)
    if kind=='large':rgb=np.zeros((1001,1001,3),np.uint8);a=np.zeros((1001,1001),np.uint8);f=np.ones(a.shape,bool)
    with pytest.raises(ValueError):call((rgb,a,f))

@pytest.mark.parametrize('kind',['fg_rgb','bg_rgb','unknown_rgb','fg_alpha','bg_alpha','rgb_shape','alpha_shape','alpha_float'])
def test_solver_cannot_bypass_contract(data,monkeypatch,kind):
    def bad(rgb,a,tri,**kw):
        out=a.copy();out[tri==0]=0;out[tri==255]=255
        if kind=='fg_rgb':rgb[tri==255]=0
        if kind=='bg_rgb':rgb[tri==0]=0
        if kind=='unknown_rgb':rgb[tri==128]=0
        if kind=='fg_alpha':out[tri==255]=0
        if kind=='bg_alpha':out[tri==0]=255
        if kind=='rgb_shape':rgb=rgb[:-1]
        if kind=='alpha_shape':out=out[:-1]
        if kind=='alpha_float':out=out.astype(float)
        return rgb,out,{}
    monkeypatch.setattr(lock,'refine_arrays',bad)
    with pytest.raises(ValueError):call(data)

@pytest.mark.parametrize('kind',['different','float','shape','notarray'])
def test_finish_rejects_unbound_trimap(data,kind):
    rgb,a,f=data;p=params(data);tri,_=lock.prepare_alpha_lock(*data,**p)
    out=a.copy();out[tri==0]=0;out[tri==255]=255
    if kind=='different':tri[0,0]=128
    if kind=='float':tri=tri.astype(float)
    if kind=='shape':tri=tri[:-1]
    if kind=='notarray':tri=None
    with pytest.raises(ValueError):lock.finish_alpha_lock(*data,rgb,out,tri,**p)

def test_prepare_independent_arrays(data):
    tri,m=lock.prepare_alpha_lock(*data,**params(data));tri[:]=0
    tri2,m2=lock.prepare_alpha_lock(*data,**params(data));assert tri2.any() and m==m2

def test_real_tiny_solver():
    pytest.importorskip('pymatting')
    rgb=np.full((48,48,3),230,np.uint8);a=np.zeros((48,48),np.uint8)
    a[10:38,12:36]=255;rgb[a==255]=(70,35,20)
    f=np.zeros(a.shape,bool);f[32:41,16:30]=True
    fg,out,tri,m=call((rgb,a,f),radius=2)
    assert np.array_equal(fg,rgb) and np.array_equal(out[f],a[f]) and not m['accepted']

@pytest.mark.parametrize('index',[0,1,2])
def test_binding_detects_changed_inputs(data,index):
    p=params(data);data[index].flat[0]=not bool(data[index].flat[0])
    with pytest.raises(ValueError):lock.recover_locked_alpha(*data,**p)

def test_unknown_band_budget():
    rgb=np.full((600,600,3),120,np.uint8);a=np.full((600,600),128,np.uint8)
    a[:50,:50]=0;a[-50:,-50:]=255;f=np.zeros(a.shape,bool);f[-2:,-2:]=True
    with pytest.raises(ValueError,match='unknown-pixel budget'):lock.prepare_alpha_lock(rgb,a,f,**params((rgb,a,f)))
