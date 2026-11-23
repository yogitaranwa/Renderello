import numpy as np
import pytest
from renderello.reviewed_background_exclusion import exclude_reviewed_background as exclude
from renderello.candidate_diagnostics import digest

@pytest.fixture
def data():
    rgb=np.full((80,100,3),120,np.uint8);a=np.zeros((80,100),np.uint8)
    a[24:58,35:65]=255;a[23:59,34]=70;a[4:8,4:8]=12
    p=np.zeros(a.shape,bool);p[22:60,32:68]=True
    b=np.zeros(a.shape,bool);b[:10,:20]=True
    return rgb,a,b,p

def params(data,**kw):
    rgb,a,b,p=data
    d=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),
        background_sha256=digest(b),protected_sha256=digest(p),
        reviewed=True,material='opaque',review_note='Synthetic reviewed background, not photo acceptance.')
    d.update(kw);return d

def run(data,**kw):return exclude(*data,**params(data,**kw))

def test_exact_contract_and_inputs_unmodified(data):
    before=[x.copy() for x in data];rgb,a,b,p=data;out_rgb,out,m=run(data)
    assert np.array_equal(out_rgb,rgb) and np.array_equal(out[~b],a[~b])
    assert np.array_equal(out[p],a[p]) and not out[b].any()
    assert m['alpha_changed_pixels']==16 and m['removed_alpha_units']==192
    assert m['maximum_removed_alpha']==12 and m['minimum_background_clearance']>=12
    assert m['protected_alpha_exact'] and m['outside_reviewed_background_exact'] and m['all_source_rgb_exact']
    assert not any(m[k] for k in ['accepted','default_pipeline_changed','independent_ground_truth','physical_recovery','source_resampled'])
    out_rgb[:]=0;out[:]=0
    assert all(np.array_equal(x,y) for x,y in zip(data,before))

def test_zero_alpha_noop_not_false_accept(data):
    data[1][data[2]]=0;out_rgb,out,m=run(data)
    assert np.array_equal(out,data[1]) and m['alpha_changed_pixels']==0 and m['maximum_removed_alpha']==0 and not m['accepted']

@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'material':None},{'material':'glass'},
 {'review_note':''},{'review_note':'   '},{'review_note':None},{'review_note':'a'*2001},
 {'source_sha256':'x'},{'alpha_sha256':'x'},{'background_sha256':'x'},{'protected_sha256':'x'},
 {'ceiling':0},{'ceiling':33},{'ceiling':True},{'ceiling':12.},{'ceiling':np.int64(12)},
 {'clearance':3},{'clearance':65},{'clearance':True},{'clearance':12.},{'clearance':np.int64(12)}])
def test_invalid_review_and_controls(data,kw):
    with pytest.raises(ValueError):run(data,**kw)

@pytest.mark.parametrize('idx,kind',[(i,k) for i in [2,3] for k in ['empty','dtype','shape','notarray']])
def test_bad_regions(data,idx,kind):
    d=list(data)
    if kind=='empty':d[idx]=np.zeros(d[1].shape,bool)
    if kind=='dtype':d[idx]=d[idx].astype(np.uint8)
    if kind=='shape':d[idx]=d[idx][:-1]
    if kind=='notarray':d[idx]=None
    with pytest.raises(ValueError):exclude(*d,**params(data))

@pytest.mark.parametrize('idx',[0,1,2,3])
def test_bound_arrays_cannot_change(data,idx):
    p=params(data);data[idx].flat[0]=not bool(data[idx].flat[0])
    with pytest.raises(ValueError,match='binding'):exclude(*data,**p)

def test_high_alpha_background_refuses_entire_request(data):
    data[1][1,1]=33
    with pytest.raises(ValueError,match='ceiling'):run(data)
    assert data[1][4,4]==12

def test_background_protected_overlap(data):
    data[3][1,1]=True
    with pytest.raises(ValueError,match='crosses protected'):run(data)

def test_protected_without_foreground(data):
    data[3][:]=False;data[3][-2:,-2:]=True
    with pytest.raises(ValueError,match='retain some'):run(data)

def test_no_high_anchor(data):
    data[1][data[1]>=128]=20
    with pytest.raises(ValueError,match='anchor'):run(data)

def test_thin_fractional_detail_clearance_refuses(data):
    # Delicate detail is low alpha and far from the opaque object.
    data[1][9,10]=1;data[3][9,10]=True
    data[2][9,10]=False
    with pytest.raises(ValueError,match='clearance'):run(data)

def test_high_alpha_clearance_refuses_without_overlap(data):
    data[2][:]=False;data[2][15:18,40:45]=True
    with pytest.raises(ValueError,match='clearance'):run(data)

@pytest.mark.parametrize('kind',['rgbdtype','rgbshape','alphadtype','alphashape','large'])
def test_invalid_native(data,kind):
    rgb,a,b,p=data
    if kind=='rgbdtype':rgb=rgb.astype(float)
    if kind=='rgbshape':rgb=rgb[:,:,0]
    if kind=='alphadtype':a=a.astype(float)
    if kind=='alphashape':a=a[:-1]
    if kind=='large':
        rgb=np.zeros((1001,1001,3),np.uint8);a=np.zeros((1001,1001),np.uint8);b=np.ones(a.shape,bool);p=b.copy()
    with pytest.raises(ValueError):exclude(rgb,a,b,p,**params(data))

@pytest.mark.parametrize('ceiling,clearance',[(1,4),(32,64)])
def test_control_boundaries(data,ceiling,clearance):
    data[1][data[2]]=1
    data[2][:]=False;data[2][0,0]=True
    if clearance==64:
        # Existing shape cannot support 64px from its protected subject.
        with pytest.raises(ValueError,match='clearance'):run(data,ceiling=ceiling,clearance=clearance)
    else:assert run(data,ceiling=ceiling,clearance=clearance)[1][0,0]==0

@pytest.mark.parametrize('value',[0,5,True,2.,np.int64(2)])
def test_bad_seam_controls(data,value):
    with pytest.raises(ValueError,match='seam ceiling'):run(data,seam_ceiling=value)

def test_seam_refuses_whole_region_before_edits(data):
    data[1][9,4]=3;before=data[1].copy()
    with pytest.raises(ValueError,match='abrupt alpha seam'):run(data)
    assert np.array_equal(data[1],before)

def test_low_seam_exact_change_reported(data):
    data[1][9,4]=2;out_rgb,out,m=run(data)
    assert out[9,4]==0 and m['seam_peak_removed_alpha']==2 and m['reviewed_seam_pixels']>0

def test_frame_is_not_internal_seam(data):
    data[1][0,4]=12;out_rgb,out,m=run(data)
    assert out[0,4]==0 and m['seam_peak_removed_alpha']==0
