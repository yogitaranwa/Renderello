import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_edge_lift import lift_reviewed_edge as lift

@pytest.fixture
def data():
 rgb=np.full((50,60,3),240,np.uint8);a=np.zeros((50,60),np.uint8);a[8:42,10:50]=255
 rgb[a==255]=[80,85,75];t=np.zeros(a.shape,bool);t[16:34,10:12]=True;rgb[t]=[30,35,25]
 d=np.zeros(a.shape,bool);d[16:34,15:17]=True;p=np.zeros(a.shape,bool);p[15:35,30:40]=True
 return rgb,a,t,d,p

def params(data,**kw):
 rgb,a,t,d,p=data;o=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),target_sha256=digest(t),donor_sha256=digest(d),protected_sha256=digest(p),reviewed=True,material='opaque',review_note='Synthetic reviewed opaque plastic, not a photo.')
 o.update(kw);return o

def run(data,**kw):return lift(*data,**params(data,**kw))

def test_local_brightening_preserves_every_other_pixel_and_alpha(data):
 before=[v.copy() for v in data];rgb,a,t,d,p=data;out,m=run(data)
 assert np.array_equal(out[~t],rgb[~t]) and np.array_equal(out[p],rgb[p])
 assert np.all(out[t]>=rgb[t]) and m['changed_pixels']==36 and m['changed_opaque_pixels']==36
 assert 1<=m['actual_maximum_linear_gain']<=1.125 and m['observed_maximum_code_delta']<=24
 assert all(np.array_equal(v,w) for v,w in zip(data,before))
 assert m['protected_rgb_exact'] and m['outside_target_rgb_exact'] and m['brighten_only']
 assert not any(m[k] for k in ['alpha_changed','accepted','fully_source_rgb_exact','default_pipeline_changed','physical_recovery','independent_ground_truth','source_resampled'])
 out[:]=0;assert rgb.any()

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':'glass'},{'material':None},
 {'review_note':''},{'review_note':None},{'review_note':' '*2},{'review_note':'x'*2001},
 {'source_sha256':'x'},{'alpha_sha256':'x'},{'target_sha256':'x'},{'donor_sha256':'x'},{'protected_sha256':'x'},
 {'edge_width':0},{'edge_width':9},{'edge_width':True},{'edge_width':np.nan},{'edge_width':complex(2,1)},
 {'donor_distance':0},{'donor_distance':49},{'donor_distance':np.inf},{'donor_distance':True},
 {'strength':0},{'strength':1.1},{'strength':True},{'strength':np.nan},{'strength':'.5'},
 {'maximum_gain':1.51},{'maximum_gain':1},{'maximum_gain':np.nan},{'maximum_gain':True},{'maximum_gain':1j},
 {'max_code_delta':0},{'max_code_delta':33},{'max_code_delta':True},{'max_code_delta':24.},{'max_code_delta':np.int64(24)}])
def test_bad_review_controls(data,kw):
 with pytest.raises(ValueError):run(data,**kw)

@pytest.mark.parametrize('idx',[0,1,2,3,4])
def test_changed_binding(data,idx):
 p=params(data);data[idx].flat[0]=not bool(data[idx].flat[0])
 with pytest.raises(ValueError,match='binding'):lift(*data,**p)

@pytest.mark.parametrize('idx,kind',[(i,k) for i in [2,3,4] for k in ['empty','dtype','shape','notarray']])
def test_bad_masks(data,idx,kind):
 d=list(data)
 if kind=='empty':d[idx]=np.zeros(d[1].shape,bool)
 if kind=='dtype':d[idx]=d[idx].astype(np.uint8)
 if kind=='shape':d[idx]=d[idx][:-1]
 if kind=='notarray':d[idx]=None
 with pytest.raises(ValueError):lift(*d,**params(data))

@pytest.mark.parametrize('kind',['td','tp','dp','lowtarget','softdonor','bgprotected','interior','far','allfg'])
def test_region_conflicts(data,kind):
 rgb,a,t,d,p=data;o={}
 if kind=='td':d[16,10]=True
 if kind=='tp':p[16,10]=True
 if kind=='dp':p[16,15]=True
 if kind=='lowtarget':a[t]=127
 if kind=='softdonor':a[d]=254
 if kind=='bgprotected':p[:]=False;p[0,0]=True
 if kind=='interior':t[20,20]=True
 if kind=='far':o['donor_distance']=1
 if kind=='allfg':a[:]=255
 with pytest.raises(ValueError):run(data,**o)

def test_fractional_protected_and_target_supported(data):
 data[1][data[2]]=128;data[1][data[4]]=30
 out,m=run(data);assert m['changed_pixels']==36 and m['changed_opaque_pixels']==0
 assert np.array_equal(out[data[4]],data[0][data[4]])

def test_donor_darker_does_not_darken_target(data):
 data[0][data[3]]=5;out,m=run(data)
 assert np.array_equal(out,data[0]) and m['fully_source_rgb_exact'] and m['actual_maximum_linear_gain']==1

def test_equal_donor_identity(data):
 data[0][data[3]]=data[0][data[2]][0];out,m=run(data)
 assert np.array_equal(out,data[0]) and m['changed_pixels']==0

def test_black_target_identity(data):
 data[0][data[2]]=0;out,m=run(data)
 assert np.array_equal(out,data[0]) and m['actual_maximum_linear_gain']==1

def test_delta_refusal_before_source_edit(data):
 before=data[0].copy()
 with pytest.raises(ValueError,match='code-delta'):run(data,max_code_delta=1)
 assert np.array_equal(before,data[0])

@pytest.mark.parametrize('kind',['rgbfloat','rgbshape','alphafloat','alphashape','large'])
def test_native_bounds(data,kind):
 rgb,a,t,d,p=data
 if kind=='rgbfloat':rgb=rgb.astype(float)
 if kind=='rgbshape':rgb=rgb[:,:,0]
 if kind=='alphafloat':a=a.astype(float)
 if kind=='alphashape':a=a[:-1]
 if kind=='large':rgb=np.zeros((1001,1001,3),np.uint8);a=np.zeros((1001,1001),np.uint8)
 with pytest.raises(ValueError):lift(rgb,a,t,d,p,**params(data))

def test_strong_valid_gain_still_bounded(data):
 out,m=run(data,strength=1,maximum_gain=1.5,max_code_delta=32)
 assert (out[data[2]]>=data[0][data[2]]).all() and m['actual_maximum_linear_gain']<=1.5

def test_independent_linear_gain_equation(data):
 rgb,a,t,d,p=data;out,m=run(data,strength=.75,maximum_gain=1.3)
 x=rgb[t].astype(float)/255.;x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 dn=rgb[d][0].astype(float)/255.;dn=np.where(dn<=.04045,dn/12.92,((dn+.055)/1.055)**2.4)
 w=np.array([.2126,.7152,.0722]);gain=1+.75*(np.minimum(1.3,np.maximum(1,(dn@w)/(x@w)))-1)
 y=x*gain[:,None];y=np.where(y<=.0031308,y*12.92,1.055*y**(1/2.4)-.055)
 assert np.array_equal(out[t],np.rint(np.clip(y,0,1)*255).astype(np.uint8))
 assert m['preencode_gamut_clipped_pixels']==0

def test_gamut_clipping_explicitly_counted(data):
 data[0][data[2]]=[250,20,20];data[0][data[3]]=255
 out,m=run(data,strength=1,maximum_gain=1.5,max_code_delta=32)
 assert m['preencode_gamut_clipped_pixels']==36
 assert np.all(out[data[2],0]==255) and m['observed_maximum_code_delta']<=32
 assert np.array_equal(out[~data[2]],data[0][~data[2]])

def test_white_identity(data):
 data[0][data[2]]=255;out,m=run(data)
 assert np.array_equal(out,data[0]) and m['actual_maximum_linear_gain']==1
