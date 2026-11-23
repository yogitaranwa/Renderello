from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_silhouette_cast import SilhouetteCast,cast_source_silhouette

@pytest.fixture
def data():
 rgb=np.full((100,160,3),100,np.uint8);a=np.zeros((100,160),np.uint8);a[15:61,50:80]=255
 bg=np.full(rgb.shape,230,np.uint8);c=np.zeros(a.shape,float);c[61:65,55:75]=.3
 r=SilhouetteCast(60,55,75,.4,.15,2.,.25,digest(rgb),digest(a),digest(bg),digest(c),'Synthetic front opaque object, attached base; assumed left source key.',reviewed=True)
 return rgb,a,bg,c,r

def call(data,**kw):return cast_source_silhouette(*data[:4],replace(data[4],**kw))

def test_actual_projection_and_invariants(data):
 before=[x.copy() for x in data[:4]];out,cast,shade,m=call(data);rgb,a,bg,c,_=data
 assert np.array_equal(out[a==255],rgb[a==255]) and not cast[a>=128].any()
 assert not cast[:60].any() and cast[65:,80:].any()
 assert np.all(shade>=c-1e-15) and np.max(shade)<1 and m['cast_pixels']>0
 assert all(np.array_equal(a,b) for a,b in zip(before,data[:4]))
 assert not m['physical_relighting'] and not m['accepted'] and not m['default_pipeline_changed']
 a2,b2,c2,m2=call(data);assert np.array_equal(a2,out) and np.array_equal(b2,cast) and m2==m

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'floor_kind':'platform'},
 {'review_note':''},{'review_note':None},{'review_note':'a'*2001},
 {'source_sha256':'x'},{'alpha_sha256':'x'},{'background_sha256':'x'},{'contact_sha256':'x'},
 {'floor_row':True},{'support_x0':-1},{'support_x1':200},{'support_x0':75},{'floor_row':99},
 {'floor_row':55},{'support_x0':0},{'source_key_side':'top'},{'source_key_side':'right'},
 {'cast_x_per_height':0.},{'cast_x_per_height':.9},{'cast_x_per_height':float('nan')},
 {'cast_y_per_height':0.},{'cast_y_per_height':.4},{'blur_sigma':0.},{'blur_sigma':13.},
 {'opacity':True},{'opacity':.5},{'opacity':-.1}, {'floor_row':30.}])
def test_invalid_review(data,kw):
 with pytest.raises(ValueError):call(data,**kw)

@pytest.mark.parametrize('kind',['rgbfloat','ashape','bgshape','cshape','cint','cnan','cnegative','ctoohigh','contactinside','detached','frame','empty','belowfloor','badreview'])
def test_bad_arrays_and_support(data,kind):
 rgb,a,bg,c,r=data
 if kind=='rgbfloat':rgb=rgb.astype(float)
 if kind=='ashape':a=a[:-1]
 if kind=='bgshape':bg=bg[:-1]
 if kind=='cshape':c=c[:-1]
 if kind=='cint':c=c.astype('uint8')
 if kind=='cnan':c[0,0]=np.nan
 if kind=='cnegative':c[0,0]=-.1
 if kind=='ctoohigh':c[0,0]=.9
 if kind=='contactinside':c[30,60]=.1
 if kind=='detached':c[:]=0
 if kind=='frame':a[0,60]=255
 if kind=='empty':a[:]=0
 if kind=='belowfloor':a[70,95]=1
 r=replace(r,source_sha256=digest(rgb),alpha_sha256=digest(a),background_sha256=digest(bg),contact_sha256=digest(c))
 if kind=='badreview':r={}
 with pytest.raises(ValueError):cast_source_silhouette(rgb,a,bg,c,r)

def test_right_light_casts_left(data):
 rgb,a,bg,c,r=data;out,cast,shade,m=call(data,source_key_side='right',cast_x_per_height=-.4)
 assert cast[65:,:50].any() and m['review']['source_key_side']=='right'

def test_canvas_clipping_refused(data):
 with pytest.raises(ValueError,match='leaves canvas'):call(data,cast_y_per_height=.35,blur_sigma=12)
