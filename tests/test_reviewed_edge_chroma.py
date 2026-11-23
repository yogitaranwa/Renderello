import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.reviewed_edge_chroma import transfer_reviewed_edge_chroma as transfer


@pytest.fixture
def sample():
    rgb=np.full((32,40,3),40,np.uint8)
    a=np.zeros((32,40),np.uint8);a[6:26,8:32]=255
    rgb[a==255]=(170,135,70)
    target=np.zeros(a.shape,bool);target[10:22,8:11]=True
    rgb[target]=(80,170,55)
    donor=np.zeros(a.shape,bool);donor[10:22,14:16]=True
    protected=np.zeros(a.shape,bool);protected[12:18,20:26]=True
    return rgb,a,target,donor,protected


def call(sample,**kw):
    rgb,a,*_=sample
    options=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),review_note='Inspected same cloth. Test fixture, not a photograph.',reviewed=True)
    options.update(kw)
    return transfer(*sample,**options)


def test_edited_edge_never_global(sample):
    before=[x.copy() for x in sample]
    out,m=call(sample)
    rgb,a,t,d,p=sample
    assert np.array_equal(out[~t],rgb[~t])
    assert np.array_equal(out[p],rgb[p])
    assert all(np.array_equal(x,y) for x,y in zip(sample,before))
    assert m['changed_pixels']==36 and m['changed_opaque_pixels']==36
    assert not m['fully_source_rgb_exact'] and not m['automatic_acceptance']
    assert not m['physical_recovery'] and not m['alpha_changed']
    assert m['protected_rgb_exact']
    w=np.array([.2126,.7152,.0722])
    assert np.max(np.abs(out[t]@w-rgb[t]@w))<.51
    assert m['output_sha256']==digest(out)


@pytest.mark.parametrize('kw',[{'reviewed':False},{'reviewed':1},{'review_note':''},{'review_note':None}, {'source_sha256':'0'*64},{'alpha_sha256':'0'*64},
    {'edge_width':True},{'edge_width':np.nan},{'edge_width':25},{'edge_width':0},
    {'donor_distance':np.inf},{'donor_distance':0},{'donor_distance':97},{'donor_distance':True},
    {'strength':0},{'strength':1.01},{'strength':-.1},{'strength':np.nan},{'strength':True}, {'strength':'0.5'}])
def test_invalid_controls(sample,kw):
    with pytest.raises(ValueError):call(sample,**kw)


@pytest.mark.parametrize('which',[2,3,4])
@pytest.mark.parametrize('kind',['empty','wrong_dtype','shape'])
def test_bad_masks(sample,which,kind):
    v=list(sample)
    if kind=='empty':v[which]=np.zeros(sample[1].shape,bool)
    if kind=='wrong_dtype':v[which]=v[which].astype(np.uint8)
    if kind=='shape':v[which]=v[which][:-1]
    with pytest.raises(ValueError):call(v)


@pytest.mark.parametrize('kind',['target_donor','target_protected','donor_protected','target_bg','donor_fractional','protected_fractional','interior','too_far','no_bg'])
def test_annotation_conflicts(sample,kind):
    rgb,a,t,d,p=[v.copy() for v in sample];opts={}
    if kind=='target_donor':d[10,8]=True
    elif kind=='target_protected':p[10,8]=True
    elif kind=='donor_protected':p[10,14]=True
    elif kind=='target_bg':t[0,0]=True
    elif kind=='donor_fractional':a[d]=254
    elif kind=='protected_fractional':a[p]=254
    elif kind=='interior':t[15,19]=True
    elif kind=='too_far':opts['donor_distance']=2
    elif kind=='no_bg':a[:]=255
    with pytest.raises(ValueError):call([rgb,a,t,d,p],**opts)


def test_full_strength_and_fractional_target(sample):
    rgb,a,t,d,p=[v.copy() for v in sample];a[t]=200
    out,m=call([rgb,a,t,d,p],strength=1)
    assert m['changed_pixels'] and m['changed_opaque_pixels']==0
    assert np.array_equal(out[~t],rgb[~t])


def test_gamut_clip_is_reported(sample):
    rgb,a,t,d,p=[v.copy() for v in sample];rgb[d]=(255,0,0);rgb[t]=(250,250,250)
    out,m=call([rgb,a,t,d,p],strength=1)
    assert m['clipped_pixels']==t.sum()
    assert out.dtype==np.uint8 and out.max()==255


def test_copy_when_equal_chroma(sample):
    rgb,a,t,d,p=sample;rgb[t]=rgb[d][0]
    out,m=call(sample)
    assert np.array_equal(out,rgb) and m['fully_source_rgb_exact']
    out[:]=0
    assert rgb.any()


@pytest.mark.parametrize('kw',[{'strength':np.complex64(.5)},{'edge_width':8+0j},{'donor_distance':complex(32,1)},{'review_note':'x'*2001}])
def test_nonreal_controls_and_oversize_note_refused(sample,kw):
    with pytest.raises(ValueError):call(sample,**kw)
