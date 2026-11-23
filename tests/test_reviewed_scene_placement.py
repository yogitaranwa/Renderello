from dataclasses import replace
import numpy as np
import pytest
from renderello.candidate_diagnostics import digest
from renderello.contact_spans import ContactPolicy,ContactSpan
from renderello.reviewed_scene_placement import ScenePlacementReview,plan_reviewed_placement,compose_reviewed_placement,PlacementRefused


def data(overhead=True):
 s=np.full((40,32,3),180,np.uint8);a=np.zeros((40,32),np.uint8);a[5:30,8:24]=255;a[4,8:24]=100
 scene=np.full((80,96,3),90,np.uint8);plane=np.ones((80,96),bool);exclude=np.zeros_like(plane)
 pose='overhead'if overhead else'front'
 r=ScenePlacementReview(digest(s),digest(a),digest(scene),digest(plane),digest(exclude),pose,pose,'soft_left','soft_left','Source inspection hypothesis','Actual target raster inspection hypothesis',True,True)
 p=ContactPolicy(camera='flatlay'if overhead else'front',source_pose=pose,material='opaque',framing='complete',source_sha256=digest(s),alpha_sha256=digest(a),reviewed=True,length=5)
 return s,a,scene,plane,exclude,r,p,[]if overhead else[ContactSpan('base',8,24,29,29)]


def plan(d,**kw):return plan_reviewed_placement(*d,translation=kw.get('translation',(10,20)))


@pytest.mark.parametrize('overhead',[True,False])
def test_exact_composition_and_target_clip(overhead):
 d=list(data(overhead));d[3][60:]=False;d[5]=replace(d[5],surface_sha256=digest(d[3]));out,shadow,r=compose_reviewed_placement(*d,translation=(10,20));assert np.array_equal(out[20:60,10:42][d[1]==255],d[0][d[1]==255]);assert not shadow[~d[3]].any()and not r['accepted']and not r['alpha_changed']


@pytest.mark.parametrize('which',['source_reviewed','target_reviewed'])
def test_unreviewed_refuses(which):
 d=list(data());d[5]=replace(d[5],**{which:False});r=plan(d);assert r['blockers']
 with pytest.raises(PlacementRefused)as e:compose_reviewed_placement(*d,translation=(10,20))
 assert e.value.report==r


@pytest.mark.parametrize('field,value,block',[('target_pose','front','ACTUAL_TARGET_CAMERA_MISMATCH'),('target_light','soft_right','ACTUAL_TARGET_LIGHT_MISMATCH'),('source_light','unknown','SOURCE_TARGET_LIGHT_NOT_REVIEWED'),('source_pose','suspended','SUPPORTED_SOURCE_TARGET_POSE_NOT_ESTABLISHED')])
def test_actual_scene_mismatch(field,value,block):
 d=list(data());d[5]=replace(d[5],**{field:value});assert any(x.startswith(block)for x in plan(d)['blockers'])


@pytest.mark.parametrize('array_index,field',[(0,'source_sha256'),(1,'alpha_sha256'),(2,'scene_sha256'),(3,'surface_sha256'),(4,'exclusion_sha256')])
def test_bound_input_mutation(array_index,field):
 d=list(data());v=d[array_index];v.flat[0]=not bool(v.flat[0])if v.dtype==bool else int(v.flat[0])+1
 with pytest.raises(ValueError,match='binding'):plan(d)


def test_occluder_overlap_and_flatlay_surface():
 d=list(data());d[4][25,20]=True;d[3][26,20]=False;d[5]=replace(d[5],exclusion_sha256=digest(d[4]),surface_sha256=digest(d[3]));r=plan(d);assert r['product_exclusion_overlap_pixels']==1 and r['off_surface_footprint_pixels']==1 and len(r['blockers'])==2


def test_entire_source_fit_and_no_boolean_coordinate():
 d=data();assert'WHOLE_NATIVE_SOURCE_DOES_NOT_FIT_NO_RESIZE_OR_CROP'in plan(d,translation=(90,60))['blockers']
 with pytest.raises(ValueError):plan(d,translation=(True,0))


def test_native_span_and_first_contact_row():
 d=list(data(False));d[3][50,18:34]=False;d[5]=replace(d[5],surface_sha256=digest(d[3]));assert'CONTACT_FOOTPRINT_OUTSIDE_REVIEWED_SURFACE'in plan(d)['blockers']
 d=list(data(False));d[-1]=[ContactSpan('outside',33,35,29,29)]
 with pytest.raises(ValueError,match='original source'):plan(d)


def test_material_and_policy_mismatch():
 d=list(data());d[6]=replace(d[6],material='transparent');assert'OPAQUE_ONLY_NO_GLASS_OR_REFLECTION_RECOVERY'in plan(d)['blockers']
 d=list(data());d[6]=replace(d[6],camera='front');assert'CONTACT_POLICY_CONTRADICTS_ACTUAL_SCENE_REVIEW'in plan(d)['blockers']


def test_masks_and_review_bounds():
 d=list(data());d[3]=d[3].astype('uint8')
 with pytest.raises(ValueError,match='boolean'):plan(d)
 with pytest.raises(ValueError):replace(data()[5],source_reviewed=1)
 with pytest.raises(ValueError):replace(data()[5],target_note='')


def test_planner_refuses_insufficient_finite_shadow_room():
 d=list(data(False));d[6]=replace(d[6],length=40)
 assert 'INSUFFICIENT_FINITE_SHADOW_ROOM'in plan(d)['blockers']
 with pytest.raises(PlacementRefused):compose_reviewed_placement(*d,translation=(10,20))


def test_zero_opacity_upright_is_planned_refusal_not_late_error():
 d=list(data(False));d[6]=replace(d[6],opacity=0)
 assert 'UPRIGHT_VISIBLE_CONTACT_REQUIRED'in plan(d)['blockers']
 with pytest.raises(PlacementRefused):compose_reviewed_placement(*d,translation=(10,20))
