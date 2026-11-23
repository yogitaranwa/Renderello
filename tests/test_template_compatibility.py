from dataclasses import replace
import json
import numpy as np
import pytest
from PIL import Image
from renderello.template_compatibility import SourceSceneReview,NEEDS,plan_template
from renderello.templates import TEMPLATES
from renderello.schemas import Aspect
from renderello.candidate_diagnostics import digest
from renderello.template_locked_preview import preview_reviewed


def case():
 rgb=np.full((24,32,3),127,np.uint8);a=np.zeros((24,32),np.uint8);a[4:20,5:27]=255
 r=SourceSceneReview(digest(rgb),digest(a),'front','opaque','complete','floor','front_left_soft','Synthetic source contract, no photograph claim.',True)
 return rgb,a,r


def test_profiles_cover_exactly_draft_library():assert set(NEEDS)==set(TEMPLATES)and 8<=len(NEEDS)<=12


def test_matching_declarations_never_mean_accepted_or_target_verified():
 rgb,a,r=case();snap=rgb.copy();before=a.copy();p=plan_template(rgb,a,'clean_studio',r)
 assert p['eligible_for_reviewed_preview']and not p['photo_accepted']and not p['actual_target_scene_verified']
 assert not p['template_polish_verified']and not p['product_light_changed']
 assert p['template_state']=='DRAFT_NOT_PHOTOGRAPHICALLY_ACCEPTED'
 assert np.array_equal(rgb,snap)and np.array_equal(a,before)


@pytest.mark.parametrize('key',list(TEMPLATES))
def test_profiles_do_not_silently_adapt_template_requirements(key):
 rgb,a,r=case();n=NEEDS[key];r=replace(r,pose=n.camera,support=n.support,light=n.light)
 p=plan_template(rgb,a,key,r)
 assert p['needs']['camera']==n.camera
 if n.reflection:assert 'REQUESTED_REFLECTION_NOT_RECOVERED'in p['blockers']
 if n.rotated_product:assert 'REQUESTED_ROTATION_NOT_SOURCE_LOCKED'in p['blockers']
 assert p['eligible_for_reviewed_preview']==(not n.reflection and not n.rotated_product and n.camera!='unknown')


@pytest.mark.parametrize('pose',['low','elevated','overhead','suspended'])
def test_viewpoint_mismatch_refused(pose):
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',replace(r,pose=pose));assert 'CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE'in p['blockers']


@pytest.mark.parametrize('material',['unknown','transparent','reflective'])
def test_glass_no_opaque_fallback(material):
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',replace(r,material=material));assert not p['eligible_for_reviewed_preview']and not p['glass_recovered']


@pytest.mark.parametrize('light',['unknown','top_rim','left_soft','open_shade','overhead_soft'])
def test_frozen_source_light_is_not_claimed_recovered(light):
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',replace(r,light=light));assert not p['eligible_for_reviewed_preview']and not p['product_light_changed']


@pytest.mark.parametrize('support',['unknown','platform','flatlay','none'])
def test_support_mismatch_requires_separate_target_review(support):
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',replace(r,support=support));assert 'SUPPORT_MISMATCH_REQUIRES_NEW_SURFACE_REVIEW'in p['blockers']


@pytest.mark.parametrize('aspect',list(Aspect))
def test_native_aspect_no_resize(aspect):
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',r,aspect=aspect);w,h=aspect.size;assert p['canvas']==[w,h]and p['whole_source_translation']==[(w-32)//2,(h-24)//2]


def test_large_native_source_not_squeezed_into_landscape():
 rgb=np.zeros((900,700,3),np.uint8);a=np.zeros((900,700),np.uint8);a[5:895,5:695]=255;_,_,r=case();r=replace(r,source_sha256=digest(rgb),alpha_sha256=digest(a));p=plan_template(rgb,a,'clean_studio',r,aspect=Aspect.LANDSCAPE)
 assert 'NATIVE_SOURCE_DOES_NOT_FIT_ASPECT_NO_RESIZE'in p['blockers']and p['whole_source_translation']is None


@pytest.mark.parametrize('name',['source_sha256','alpha_sha256'])
def test_stale_review_refused(name):
 rgb,a,r=case()
 with pytest.raises(ValueError,match='different'):plan_template(rgb,a,'clean_studio',replace(r,**{name:'0'*64}))


@pytest.mark.parametrize('field,value',[('pose','sideways'),('material','glass'),('light','fake'),('framing','maybe'),('support','air'),('reviewed',1),('note',''),('note','a'*2001),('source_sha256','bad')])
def test_invalid_declarations(field,value):
 _,_,r=case()
 with pytest.raises(ValueError):replace(r,**{field:value})


def test_frame_alpha_and_empty_are_not_complete_source_evidence():
 rgb,a,r=case();a[0,0]=1;r=replace(r,alpha_sha256=digest(a));p=plan_template(rgb,a,'clean_studio',r);assert 'FRAME_TOUCHING_ALPHA_REQUIRES_REVIEW'in p['blockers']
 a[:]=0;r=replace(r,alpha_sha256=digest(a));p=plan_template(rgb,a,'clean_studio',r);assert 'NO_HARD_FOREGROUND'in p['blockers']


def test_reviewed_preview_opt_in_refuses_before_output_creation(tmp_path):
 rgb,a,r=case();s=tmp_path/'s.png';m=tmp_path/'a.png';Image.fromarray(rgb).save(s);Image.fromarray(a).save(m)
 with pytest.raises(ValueError,match='compatibility'):preview_reviewed(s,m,{'core':[10,10,20,15]},tmp_path/'out',reviewed=True,material='opaque',template='flat_lay',scene_review=r)
 assert not(tmp_path/'out').exists()


def test_reviewed_preview_records_opt_in_plan(tmp_path):
 rgb,a,r=case();s=tmp_path/'s.png';m=tmp_path/'a.png';Image.fromarray(rgb).save(s);Image.fromarray(a).save(m)
 out=preview_reviewed(s,m,{'core':[10,10,20,15]},tmp_path/'out',reviewed=True,material='opaque',template='clean_studio',scene_review=r)
 p=json.loads((tmp_path/'out/manifest.json').read_text())['source_template_compatibility'];assert p['eligible_for_reviewed_preview']and not p['photo_accepted']


def test_opt_in_translation_cannot_crop_native_frame(tmp_path):
 rgb,a,r=case();s=tmp_path/'s.png';m=tmp_path/'a.png';Image.fromarray(rgb).save(s);Image.fromarray(a).save(m)
 with pytest.raises(ValueError,match='clips source frame'):
  preview_reviewed(s,m,{'core':[10,10,20,15]},tmp_path/'out',reviewed=True,material='opaque',template='clean_studio',scene_review=r,offset=[-1,0])
 assert not(tmp_path/'out').exists()


def test_opt_in_execution_translation_is_recorded(tmp_path):
 rgb,a,r=case();s=tmp_path/'s.png';m=tmp_path/'a.png';Image.fromarray(rgb).save(s);Image.fromarray(a).save(m)
 preview_reviewed(s,m,{'core':[10,10,20,15]},tmp_path/'out',reviewed=True,material='opaque',template='clean_studio',scene_review=r,offset=[5,6])
 p=json.loads((tmp_path/'out/manifest.json').read_text())['source_template_compatibility'];assert p['executed_source_translation']==[5,6]


def test_unknown_source_camera_is_not_reported_as_known_mismatch():
 rgb,a,r=case();p=plan_template(rgb,a,'clean_studio',replace(r,pose='unknown'))
 assert 'SOURCE_CAMERA_ELEVATION_UNREVIEWED' in p['blockers']
 assert 'CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE' not in p['blockers']


@pytest.mark.parametrize('pose',['unknown','front','low','elevated','overhead','suspended'])
def test_editorial_three_quarter_is_not_an_elevation_claim(pose):
 rgb,a,r=case();p=plan_template(rgb,a,'editorial_beauty',replace(r,pose=pose,support='platform',light='left_soft'))
 assert NEEDS['editorial_beauty'].camera=='unknown'
 assert 'TEMPLATE_CAMERA_ELEVATION_NOT_DISAMBIGUATED' in p['blockers']
 assert 'CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE' not in p['blockers']
 assert not p['eligible_for_reviewed_preview']
 assert p['camera_requirement_text']==TEMPLATES['editorial_beauty'].camera
 assert 'NOT_AZIMUTH' in p['camera_semantics']


@pytest.mark.parametrize('key',['natural_outdoor','reflective_surface'])
def test_surface_level_text_requires_low_camera_review(key):
 rgb,a,r=case();p=plan_template(rgb,a,key,r)
 assert NEEDS[key].camera=='low'
 assert 'CAMERA_MISMATCH_NO_VIEWPOINT_CHANGE' in p['blockers']
