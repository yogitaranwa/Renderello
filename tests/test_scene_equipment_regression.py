"""Real returned background plus synthetic controls. No provider calls."""
from pathlib import Path
import cv2, numpy as np, pytest
from PIL import Image
from renderello.scene_acceptance import long_straight_edges
from renderello.empty_scene_prompt import build_empty_scene_prompt
from renderello.templates import get_template

@pytest.mark.parametrize('side',[640,1024])
def test_recorded_softbox_edge_warns_at_native_and_aligned_size(side):
    p=Path(__file__).parent/'fixtures'/'equipment-scene-regression.png'
    a=np.array(Image.open(p).convert('RGB'))
    a=cv2.resize(a,(side,side),interpolation=cv2.INTER_LANCZOS4)
    r=long_straight_edges(a)
    assert r['intrusion'] and r['screen_only'] and r['needs_eye_review']
    assert not r['semantic_equipment_detection']
    assert any(l['p1'][0]>.9*side and l['p2'][0]>.9*side for l in r['lines'])

def test_smooth_gradient_is_not_cleared_as_quality():
    a=np.repeat(np.linspace(120,210,640).astype(np.uint8)[:,None,None],640,1)
    a=np.repeat(a,3,2);r=long_straight_edges(a)
    assert not r['intrusion'] and r['needs_eye_review'] and r['screen_only']

def test_sweep_curve_not_long_straight_equipment_edge():
    a=np.full((640,640,3),200,np.uint8)
    cv2.ellipse(a,(320,200),(280,250),0,0,180,(90,90,90),2)
    assert not long_straight_edges(a)['intrusion']

def test_prompt_keeps_equipment_outside_edges():
    p=build_empty_scene_prompt(get_template('clean_studio'))
    assert 'entirely outside the frame, including the image edges' in p
    assert 'finished empty environment' in p and 'no product' in p

@pytest.mark.parametrize('bad',['show the reflector','visible fill card','light stand on right','tripod on the left','lighting rig on stage'])
def test_equipment_adding_instruction_rejected(bad):
    with pytest.raises(ValueError):build_empty_scene_prompt(get_template('clean_studio'),instruction=bad)

@pytest.mark.parametrize('a',[np.zeros((0,1,3),np.uint8),np.zeros((10,10),np.uint8),np.zeros((10,10,3),np.float32)])
def test_invalid_frame_refused(a):
    with pytest.raises(ValueError):long_straight_edges(a)

def test_recorded_scene_replay_flags_intrusion_without_extra_call(tmp_path):
    from renderello.cutout import supplied_mask_cutout
    from renderello.compose import composite_with_scene
    from renderello.schemas import Aspect
    src=Image.new('RGB',(120,160),(40,100,150));a=np.zeros((160,120),np.uint8);a[10:150,30:90]=255
    mp=tmp_path/'alpha.png';Image.fromarray(a).save(mp)
    co=supplied_mask_cutout(src,mp,tmp_path/'co')
    bg=np.array(Image.open(Path(__file__).parent/'fixtures'/'equipment-scene-regression.png').convert('RGB'))
    calls=[]
    def replay(canvas,prompt):calls.append(prompt);return bg.copy()
    out,diag=composite_with_scene(co,get_template('clean_studio'),Aspect.SQUARE,0,replay,
        prompt=build_empty_scene_prompt(get_template('clean_studio')),conditioning='empty_guide',
        scene_accept=long_straight_edges,scene_max_attempts=1)
    assert len(calls)==1 and diag['scene_attempts']==1
    assert diag['scene_acceptance']['intrusion']
    assert diag['needs_eye_review'] and diag['publication_status']=='NOT_EVALUATED'
    assert diag['paste_bit_exact']
