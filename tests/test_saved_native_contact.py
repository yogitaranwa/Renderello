from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.contact_spans import ContactPolicy,ContactSpan
from renderello.saved_native_contact import compare_saved_result_contact

def compare(*args, **kwargs):
    kwargs.setdefault('support_floor_row',31)
    return compare_saved_result_contact(*args, **kwargs)
from test_native_result_import import saved


def setup(saved):
    rgb,initial,_,_,root=saved
    from renderello.native_result_import import load_verified_result
    alpha,_=load_verified_result(root,rgb,initial)
    bg=np.full(rgb.shape,230,np.uint8)
    p=ContactPolicy('front','opaque','complete','front',digest(rgb),digest(alpha),reviewed=True,length=5)
    spans=[ContactSpan('base',16,32,29,31)]
    return root,rgb,initial,bg,spans,p


def test_saved_to_scene_exact_opaque_and_attached(saved):
    args=setup(saved);before=[v.copy() for v in args[1:4]]
    prior,final,shade,meta=compare(*args)
    assert meta['opaque_source_pixels_exact'] and meta['import_replay']['files_verified']==9
    assert not meta['accepted'] and not meta['independent_review']
    assert meta['attachments'][0]['all_attached']
    assert meta['shadow_on_hard_support_pixels']==0
    assert meta['contact']['recovered_3d_geometry'] is False
    assert np.any(prior!=final) and shade.max()>0
    assert all(np.array_equal(x,y) for x,y in zip(args[1:4],before))


@pytest.mark.parametrize('kw',[
 {'material':'transparent'},{'material':'reflective'},{'material':'unknown'},
 {'framing':'cropped'},{'framing':'unknown'},{'source_pose':'suspended'},
 {'source_pose':'elevated'},{'camera':'elevated'},{'reviewed':False},
 {'source_sha256':'0'*64},{'alpha_sha256':'0'*64}])
def test_rejects_unmatched_declarations(saved,kw):
    args=list(setup(saved));args[-1]=replace(args[-1],**kw)
    with pytest.raises(ValueError):compare(*args)


@pytest.mark.parametrize('kind',['source','initial','missing_output','bg_shape','spans_empty','spans_wrong_bottom','span_outside','policy_untyped'])
def test_saved_bridge_does_not_bypass_checks(saved,kind):
    args=list(setup(saved))
    if kind=='source':args[1]=args[1].copy();args[1][0,0]=0
    elif kind=='initial':args[2]=args[2].copy();args[2][0,0]=255
    elif kind=='missing_output':(args[0]/'alpha.png').unlink()
    elif kind=='bg_shape':args[3]=args[3][:-1]
    elif kind=='spans_empty':args[4]=[]
    elif kind=='spans_wrong_bottom':args[4]=[ContactSpan('wrong',16,32,0,2)]
    elif kind=='span_outside':args[4]=[ContactSpan('wrong',40,60,29,31)]
    elif kind=='policy_untyped':args[-1]={}
    with pytest.raises((ValueError,OSError)):compare(*args)


@pytest.mark.parametrize('kw',[
 {'support_floor_row':None},{'support_floor_row':True},{'support_floor_row':-1},
 {'support_floor_row':40},{'support_floor_row':1},{'support_tolerance':True},
 {'support_tolerance':-1},{'support_tolerance':5},{'support_tolerance':1.5}])
def test_floor_declaration_cannot_be_guessed(saved,kw):
    with pytest.raises(ValueError):compare(*setup(saved),**kw)


def test_missing_floor_refused(saved):
    with pytest.raises(ValueError):compare_saved_result_contact(*setup(saved))


@pytest.mark.parametrize('translation',[(0,0),(10,12),[8,5]])
def test_larger_canvas_native_translation(saved,translation):
    args=list(setup(saved));args[3]=np.full((70,80,3),220,np.uint8)
    before=args[1].copy()
    _,out,_,m=compare(*args,translation=translation)
    dx,dy=translation
    from renderello.native_result_import import load_verified_result
    a,_=load_verified_result(args[0],args[1],args[2]);h,w=a.shape
    assert np.array_equal(out[dy:dy+h,dx:dx+w][a==255],before[a==255])
    assert m['source_to_canvas_translation']==list(translation)
    assert m['native_support_floor_row']==31 and m['support_floor_row']==31+dy
    assert m['import_replay']['source_rgb_exact'] and not m['source_cropped']
    assert m['attachments'][0]['all_attached']
    assert m['source_policy']['source_sha256']==digest(before)


@pytest.mark.parametrize('translation',[(-1,0),(0,-1),(True,0),(0.,0),(0,),None,'1,2',(100,0),(0,100)])
def test_translation_cannot_clip_or_resample(saved,translation):
    with pytest.raises(ValueError):compare(*setup(saved),translation=translation)


@pytest.mark.parametrize('translation',[(0,0),(10,12)])
def test_flatlay_uses_overhead_pose_without_upright_floor(saved,translation):
    args=list(setup(saved));args[3]=np.full((70,80,3),225,np.uint8);args[4]=[]
    args[-1]=replace(args[-1],camera='flatlay',source_pose='overhead')
    prior,out,shade,m=compare_saved_result_contact(*args,translation=translation)
    from renderello.native_result_import import load_verified_result
    a,_=load_verified_result(args[0],args[1],args[2]);dx,dy=translation;h,w=a.shape
    assert np.array_equal(out[dy:dy+h,dx:dx+w][a==255],args[1][a==255])
    assert np.any(prior!=out) and shade.max()>0
    assert m['support_floor_row'] is None and m['native_support_floor_row'] is None
    assert m['attachments']==[] and m['contact']['method']=='flatlay_finite_outside_distance'
    assert m['support_plane']=='REVIEWED_FLATLAY_SILHOUETTE' and not m['accepted']


@pytest.mark.parametrize('kind',['floor','span','front_pose','camera_mismatch'])
def test_flatlay_never_guesses_upright_support(saved,kind):
    args=list(setup(saved));args[4]=[];args[-1]=replace(args[-1],camera='flatlay',source_pose='overhead');kw={}
    if kind=='floor':kw['support_floor_row']=31
    if kind=='span':args[4]=[ContactSpan('base',16,32,29,31)]
    if kind=='front_pose':args[-1]=replace(args[-1],source_pose='front')
    if kind=='camera_mismatch':args[-1]=replace(args[-1],camera='front')
    with pytest.raises(ValueError):compare_saved_result_contact(*args,**kw)
