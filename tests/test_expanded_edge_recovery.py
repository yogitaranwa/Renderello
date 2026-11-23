import numpy as np
import pytest
import renderello.expanded_edge_recovery as recovery
from renderello.candidate_diagnostics import digest


@pytest.fixture
def data(monkeypatch):
    rgb=np.full((56,64,3),120,np.uint8);a=np.zeros((56,64),np.uint8)
    a[8:48,10:54]=255;a[7,10:54]=150
    p=np.zeros(a.shape,bool);p[16:25,22:35]=True
    def fixture(rgb,initial,tri,method,edge_colors):
        assert method=='cf' and edge_colors is True
        out=rgb.copy();out[tri==128]=121
        alpha=initial.copy();alpha[tri==255]=255;alpha[tri==0]=0
        return out,alpha,{'synthetic_solver_fixture':True}
    monkeypatch.setattr(recovery,'refine_arrays',fixture)
    return rgb,a,p


def call(data,**kw):
    rgb,a,_=data
    params=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),material='opaque',review_note='Synthetic core annotation, not image truth.',reviewed=True)
    params.update(kw)
    return recovery.recover_expanded_edge(*data,**params)


def test_locks_and_provenance(data):
    before=[v.copy() for v in data];fg,a,tri,m=call(data)
    rgb,initial,p=data
    assert np.array_equal(fg[p],rgb[p]) and (a[p]==255).all()
    assert np.array_equal(fg[tri!=128],rgb[tri!=128])
    assert m['rgb_changed_pixels']>0 and m['initially_opaque_rgb_changed']>0
    assert m['solver']['synthetic_solver_fixture']
    assert not m['fully_source_rgb_exact'] and not m['accepted']
    assert not m['physical_relighting'] and not m['independent_ground_truth']
    assert all(np.array_equal(x,y) for x,y in zip(data,before))


@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':'transparent'},{'material':'unknown'},
 {'review_note':''},{'review_note':None},{'review_note':'x'*2001},
 {'source_sha256':'0'*64},{'alpha_sha256':'0'*64},
 {'radius':True},{'radius':0},{'radius':13},{'radius':8.},{'radius':np.int32(8)}])
def test_bad_declarations(data,kw):
    with pytest.raises(ValueError):call(data,**kw)


@pytest.mark.parametrize('kind',['empty','shape','dtype','nonopaque'])
def test_bad_protected_core(data,kind):
    rgb,a,p=[v.copy() for v in data]
    if kind=='empty':p[:]=False
    elif kind=='shape':p=p[:-1]
    elif kind=='dtype':p=p.astype(np.uint8)
    elif kind=='nonopaque':a[p]=254
    with pytest.raises(ValueError):call((rgb,a,p))


@pytest.mark.parametrize('kind',['source_shape','source_dtype','alpha_shape','alpha_dtype','no_bg','no_fg','too_large'])
def test_bad_array_inputs(data,kind):
    rgb,a,p=data
    if kind=='source_shape':rgb=rgb[:,:,0]
    elif kind=='source_dtype':rgb=rgb.astype(float)
    elif kind=='alpha_shape':a=a[:-1]
    elif kind=='alpha_dtype':a=a.astype(float)
    elif kind=='no_bg':a=np.full(a.shape,255,np.uint8)
    elif kind=='no_fg':a=np.zeros(a.shape,np.uint8)
    elif kind=='too_large':rgb=np.zeros((1001,1001,3),np.uint8);a=np.zeros((1001,1001),np.uint8);p=np.ones(a.shape,bool)
    with pytest.raises(ValueError):call((rgb,a,p))


@pytest.mark.parametrize('kind',['bg_rgb','fg_rgb','fg_alpha','bg_alpha','shape','float'])
def test_solver_output_cannot_bypass_locks(data,monkeypatch,kind):
    def bad(rgb,initial,tri,**kwargs):
        fg=rgb.copy();a=initial.copy();a[tri==255]=255;a[tri==0]=0
        if kind=='bg_rgb':fg[tri==0]=0
        elif kind=='fg_rgb':fg[tri==255]=0
        elif kind=='fg_alpha':a[tri==255]=0
        elif kind=='bg_alpha':a[tri==0]=255
        elif kind=='shape':fg=fg[:-1]
        elif kind=='float':fg=fg.astype(float)
        return fg,a,{}
    monkeypatch.setattr(recovery,'refine_arrays',bad)
    with pytest.raises(ValueError):call(data)


def test_real_tiny_solver(monkeypatch):
    pytest.importorskip('pymatting')
    rgb=np.full((48,48,3),230,np.uint8);a=np.zeros((48,48),np.uint8);a[10:38,12:36]=255;rgb[a==255]=(100,50,20)
    p=np.zeros(a.shape,bool);p[18:30,20:28]=True
    fg,out,tri,m=call((rgb,a,p),radius=2)
    assert np.array_equal(fg[p],rgb[p]) and (out[p]==255).all()
    assert not m['accepted'] and not m['physical_relighting']


def test_protected_relocks_unknown_band(data):
    rgb,a,p=[v.copy() for v in data];p[9,11]=True
    fg,out,tri,m=call((rgb,a,p))
    assert m['protected_relocked_from_unknown']>0
    assert (tri[p]==255).all() and (out[p]==255).all()
    assert np.array_equal(fg[p],rgb[p])
