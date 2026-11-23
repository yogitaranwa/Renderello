import hashlib
import numpy as np
from PIL import Image
import pytest
from renderello.empty_scene_prompt import build_empty_scene_prompt, UnsupportedEmptySceneStaging, EmptyScenePrompt
from renderello.templates import get_template
from renderello.schemas import Aspect,CutoutResult
from renderello.compose import composite_with_scene

@pytest.mark.parametrize('view',[None,'front','unknown','elevated'])
def test_frontal_source_is_not_silently_turned_into_overhead(view):
    with pytest.raises(UnsupportedEmptySceneStaging):
        build_empty_scene_prompt(get_template('flat_lay'),reviewed_source_view=view)

def test_overhead_has_uniform_guide_and_no_floor_anchor(tmp_path):
    a=np.zeros((100,140,4),np.uint8);a[20:80,25:115,:3]=(210,30,65);a[20:80,25:115,3]=255
    p=tmp_path/'fixture.png';Image.fromarray(a).save(p)
    co=CutoutResult(p,p,(25,20,90,60),'synthetic_overhead_fixture',.38)
    prompt=build_empty_scene_prompt(get_template('flat_lay'),reviewed_source_view='overhead',reviewed_alpha_sha256=hashlib.sha256(a[...,3].tobytes()).hexdigest())
    seen=[]
    def scene(canvas,text):
        seen.append((canvas.copy(),text));return np.full_like(canvas,210)
    out,d=composite_with_scene(co,get_template('flat_lay'),Aspect.SQUARE,0,scene,prompt=prompt,conditioning='empty_guide')
    assert len(seen)==1 and set(np.unique(seen[0][0]))=={176}
    assert seen[0][1] is prompt
    assert 'floor-wall junction' not in prompt
    assert 'no wall, no horizon' in prompt
    assert d['scene_mode']=='flatlay_surface_experimental'
    assert d['foreground_view_change']=='none'
    assert 'support_plane_horizon' not in d
    assert d['grounding_shadow'] is None
    assert d['paste_bit_exact']
    assert d['publication_status']=='NOT_EVALUATED'
    rgb=np.array(out);ys,xs=np.where((rgb==[210,30,65]).all(axis=2))
    assert len(xs)==90*60
    assert abs((xs.min()+xs.max())/2-rgb.shape[1]/2)<2
    assert abs((ys.min()+ys.max())/2-rgb.shape[0]/2)<2
    Image.fromarray(seen[0][0]).save(tmp_path/'overhead-guide.png')
    Image.fromarray(rgb).save(tmp_path/'overhead-fixture-result.png')

def test_overhead_cannot_be_reused_on_floor_template_before_callback(tmp_path):
    a=np.ones((20,20,4),np.uint8)*255;p=tmp_path/'p.png';Image.fromarray(a).save(p)
    co=CutoutResult(p,p,(0,0,20,20),'fixture',1)
    prompt=build_empty_scene_prompt(get_template('flat_lay'),reviewed_source_view='overhead',reviewed_alpha_sha256=hashlib.sha256(a[...,3].tobytes()).hexdigest())
    def forbidden(*args):raise AssertionError('callback reached')
    with pytest.raises(ValueError,match='overhead staging'):
        composite_with_scene(co,get_template('clean_studio'),Aspect.SQUARE,0,forbidden,prompt=prompt,conditioning='empty_guide')

def test_missing_semantic_alpha_review_refused():
    with pytest.raises(ValueError,match='pixel-reviewed complete native alpha'):
        build_empty_scene_prompt(get_template('flat_lay'),reviewed_source_view='overhead')

def test_changed_alpha_refused_before_callback(tmp_path):
    a=np.ones((20,20,4),np.uint8)*255;p=tmp_path/'p.png';Image.fromarray(a).save(p)
    co=CutoutResult(p,p,(0,0,20,20),'fixture',1)
    prompt=build_empty_scene_prompt(get_template('flat_lay'),reviewed_source_view='overhead',reviewed_alpha_sha256='0'*64)
    def forbidden(*args):raise AssertionError('callback reached')
    with pytest.raises(ValueError,match='alpha differs'):
        composite_with_scene(co,get_template('flat_lay'),Aspect.SQUARE,0,forbidden,prompt=prompt,conditioning='empty_guide')
