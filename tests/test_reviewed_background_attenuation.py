import numpy as np,pytest
from scipy.ndimage import distance_transform_edt
from renderello.reviewed_background_attenuation import attenuate_reviewed_background as apply
from renderello.candidate_diagnostics import digest
@pytest.fixture
def data():
 rgb=np.full((120,140,3),100,np.uint8);a=np.full((120,140),20,np.uint8);a[45:75,55:85]=255;p=np.zeros(a.shape,bool);p[40:80,50:90]=True;b=~p
 return rgb,a,b,p

def params(d,**kw):
 rgb,a,b,p=d;k=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),background_sha256=digest(b),protected_sha256=digest(p),reviewed=True,material='opaque',review_note='Synthetic known background only.');k.update(kw);return k

def run(d,**kw):return apply(*d,**params(d,**kw))
def test_independent_full_equation_and_immutable(data):
 before=[x.copy() for x in data];rgb,a,b,p=data;r,out,m=run(data)
 distance=distance_transform_edt(np.pad(b,1))[1:-1,1:-1];anchors=distance_transform_edt(~((a>=128)|p));x=np.clip((np.minimum(distance,anchors)-12)/32,0,1);w=x*x*(3-2*x);w[~b]=0
 assert np.array_equal(out,np.rint(a*(1-w)).astype(np.uint8))
 assert np.array_equal(r,rgb) and np.array_equal(out[p],a[p]) and np.all(out<=a) and m['alpha_changed_pixels']>0
 assert np.array_equal(out[:12],a[:12]) and np.array_equal(out[-12:],a[-12:])
 assert all(np.array_equal(x,y) for x,y in zip(data,before))
 assert not m['accepted'] and m['selection_boundary_alpha_exact'] and m['source_rgb_exact']
 r[:]=0;out[:]=0;assert rgb.any() and a.any()

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':'glass'},{'review_note':''},{'review_note':'x'*2001},{'ceiling':33},{'ceiling':0},{'ceiling':True},{'clearance':3},{'clearance':65},{'clearance':12.},{'feather':15},{'feather':65},{'feather':True},{'feather':np.nan}])
def test_controls(data,kw):
 with pytest.raises(ValueError):run(data,**kw)
@pytest.mark.parametrize('i',range(4))
def test_binding(data,i):
 k=params(data);data[i].flat[0]=not bool(data[i].flat[0])
 with pytest.raises(ValueError,match='binding'):apply(*data,**k)
@pytest.mark.parametrize('kind',['overlap','no_anchor','ceiling','emptybg','emptyprotected','maskdtype','maskshape','large','lowprotected'])
def test_regions(data,kind):
 rgb,a,b,p=data
 if kind=='overlap':b[50,60]=True
 if kind=='no_anchor':a[:]=20
 if kind=='ceiling':a[10,20]=33
 if kind=='emptybg':b[:]=False
 if kind=='emptyprotected':p[:]=False
 if kind=='maskdtype':b=b.astype(np.uint8)
 if kind=='maskshape':b=b[:-1]
 if kind=='large':rgb=np.zeros((1001,1001,3),np.uint8);a=np.zeros((1001,1001),np.uint8)
 if kind=='lowprotected':a[p]=0
 with pytest.raises(ValueError):apply(rgb,a,b,p,**params((rgb,a,b,p)))
def test_identity_when_selection_too_narrow(data):
 rgb,a,b,p=data;b[:]=False;b[10:20,15:25]=True;r,out,m=run(data)
 assert np.array_equal(a,out) and m['alpha_changed_pixels']==0

def test_no_hidden_partial_overceiling(data):
 before=data[1].copy();data[1][20,20]=33
 with pytest.raises(ValueError,match='ceiling'):run(data)
 assert data[1][20,20]==33 and before[20,20]==20
