import json
import numpy as np
import pytest
from PIL import Image
from renderello.source_contract import SourceContract,source_mask_preflight
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.pipeline import run_pipeline
from renderello.config import RunConfig


def arrays():
    rgb=np.full((80,100,3),150,np.uint8);a=np.zeros((80,100),np.uint8);a[10:70,20:80]=255
    return rgb,a,(20,10,60,60)


def test_unknown_never_claims_verified():
    d=source_mask_preflight(*arrays());assert d['passed']
    assert d['semantic_mask_accuracy']=='NOT_VERIFIED' and len(d['warnings'])>=3


@pytest.mark.parametrize('pose',['front','elevated','suspended'])
def test_nonoverhead_cannot_become_overhead(pose):
    d=source_mask_preflight(*arrays(),SourceContract(pose,'opaque','complete'),overhead_scene=True)
    assert not d['passed'] and 'camera angle' in d['blockers'][0]


def test_overhead_compatible_not_publication_accepted():
    d=source_mask_preflight(*arrays(),SourceContract('overhead','opaque','complete'),overhead_scene=True)
    assert d['passed'] and d['publication_status']=='NOT_EVALUATED'


def test_transparent_opaque_paste_blocked():
    d=source_mask_preflight(*arrays(),SourceContract(material='transparent'));assert not d['passed']
    assert source_mask_preflight(*arrays(),SourceContract(material='transparent'),glass_mode=True)['passed']


def test_crop_requirement():
    rgb,a,b=arrays();a[0:20,20:80]=255
    d=source_mask_preflight(rgb,a,(20,0,60,70),SourceContract(framing='complete'),require_complete=True)
    assert not d['passed'] and d['frame_contact']['needs_review']


def test_multiple_parts_retained():
    rgb,a,b=arrays();a[5:8,5:8]=255;a[72:77,85:90]=255;old=a.copy()
    d=source_mask_preflight(rgb,a,(5,5,85,72));assert len(d['mask_components'])==3
    assert np.array_equal(old,a) and d['semantic_mask_accuracy']=='NOT_VERIFIED'


@pytest.mark.parametrize('kw',[dict(pose='auto'),dict(material='glass'),dict(framing='auto')])
def test_bad_declarations(kw):
    with pytest.raises(ValueError):SourceContract(**kw)


def test_gpu_callback_not_called_for_declared_mismatch(tmp_path):
    rgb,a,b=arrays();src=tmp_path/'s.png';mask=tmp_path/'a.png';Image.fromarray(rgb).save(src);Image.fromarray(a).save(mask)
    class Expensive:
        flatlay_surface=True
        def __call__(self,*a):raise AssertionError('GPU must not run')
    result=run_pipeline(EditRequest(src,get_template('flat_lay'),mask_path=mask),[Route.SCENE_COMPOSE],
       RunConfig(variants=1,out_root=tmp_path/'runs'),scene_fn=Expensive(),source_contract=SourceContract('front','opaque','complete'))
    m=json.loads(result.manifest_path.read_text());assert not m['variants'][0]['success']
    assert 'incompatible' in str(m['variants'][0]['qa']['critical_errors'])
    assert not m['source_mask_preflight']['passed']


def test_camera_matched_strips_source_rgb_and_records_pose(tmp_path):
    from renderello.compose import composite_with_scene
    from renderello.schemas import CutoutResult,Aspect
    from renderello.iclight_gpu import CameraMatchedCallback
    rgba=np.zeros((100,100,4),np.uint8);rgba[20:80,20:80]=[201,1,13,255]
    path=tmp_path/'p.png';Image.fromarray(rgba).save(path)
    class Model:
        def relight(self,a,p,**kw):
            assert np.unique(a).tolist()==[176]
            self.prompt=p;self.kw=kw;return np.full_like(a,180),0.
    m=Model();cb=CameraMatchedCallback(m,5,'elevated')
    _,d=composite_with_scene(CutoutResult(path,path,(20,20,60,60),'test',.36),
       get_template('clean_studio'),Aspect.SQUARE,0,cb)
    assert d['paste_bit_exact'] and d['declared_camera_pose']=='elevated'
    assert '20 degrees' in m.prompt and m.kw['seed']==5
    assert d['needs_eye_review'] and d['publication_status']=='NOT_EVALUATED'
    with pytest.raises(ValueError):CameraMatchedCallback(m,0,'overhead')


def test_camera_matched_styles_keep_geometry_explicit_and_distinct():
    from renderello.iclight_gpu import CameraMatchedCallback
    import numpy as np
    class Fake:
        def relight(self,canvas,prompt,**kwargs):
            self.prompt=prompt
            return [canvas]
    model=Fake();seen=[]
    for style in ('clean_studio','luxury_hero','editorial_beauty'):
        fn=CameraMatchedCallback(model,pose='front',style=style)
        fn(np.full((24,24,3),176,np.uint8),'arbitrary original scene prompt')
        assert 'low horizon' in model.prompt and 'floor occupying lower two thirds' in model.prompt
        seen.append(model.prompt)
    assert len(set(seen))==3

@pytest.mark.parametrize('backdrop',['dirty','dark'])
def test_invalid_glass_source_blocks_before_scene_call(tmp_path, backdrop):
    rgb=np.full((80,100,3),50 if backdrop=='dark' else 240,np.uint8)
    if backdrop=='dirty':rgb[:,::2]=[2,90,20]
    a=np.zeros((80,100),np.uint8);a[10:70,20:80]=255
    src=tmp_path/'g.png';mask=tmp_path/'m.png';Image.fromarray(rgb).save(src);Image.fromarray(a).save(mask)
    def expensive(*a):raise AssertionError('invalid source must not call model')
    r=run_pipeline(EditRequest(src,get_template('clean_studio'),mask_path=mask),[Route.SCENE_COMPOSE],
      RunConfig(variants=1,out_root=tmp_path/'runs'),scene_fn=expensive,glass_mode=True,
      source_contract=SourceContract('front','transparent','complete'))
    m=json.loads(r.manifest_path.read_text());assert not m['variants'][0]['success']
    assert not m['source_mask_preflight']['passed']
    assert 'backdrop' in str(m['source_mask_preflight']['blockers'])
    assert not r.variants[0].output_path.exists()

@pytest.mark.parametrize('route',[Route.CUTOUT_COMPOSE,Route.HYBRID])
def test_glass_flag_does_not_authorize_opaque_route(tmp_path,route):
    rgb=np.full((80,100,3),240,np.uint8);rgb[10:70,20:80]=100
    a=np.zeros((80,100),np.uint8);a[10:70,20:80]=255
    src=tmp_path/'g.png';mask=tmp_path/'m.png';Image.fromarray(rgb).save(src);Image.fromarray(a).save(mask)
    r=run_pipeline(EditRequest(src,get_template('clean_studio'),mask_path=mask),[route],
      RunConfig(variants=1,out_root=tmp_path/'runs'),glass_mode=True,
      source_contract=SourceContract('front','transparent','complete'))
    m=json.loads(r.manifest_path.read_text());assert not m['variants'][0]['success']
    assert 'opaque composite route' in str(m['variants'][0]['qa']['critical_errors'])


@pytest.mark.parametrize('route',[Route.CUTOUT_COMPOSE,Route.SCENE_COMPOSE])
@pytest.mark.parametrize('pose',['front','elevated','suspended'])
def test_flatlay_template_checks_pose_without_callback_marker(tmp_path,route,pose):
    rgb,a,_=arrays();src=tmp_path/'s.png';mask=tmp_path/'a.png';Image.fromarray(rgb).save(src);Image.fromarray(a).save(mask)
    result=run_pipeline(EditRequest(src,get_template('flat_lay'),mask_path=mask),[route],
        RunConfig(variants=1,out_root=tmp_path/'runs'),source_contract=SourceContract(pose,'opaque','complete'))
    m=json.loads(result.manifest_path.read_text())
    assert not m['source_mask_preflight']['passed'] and not m['variants'][0]['success']
    assert 'camera angle' in str(m['source_mask_preflight']['blockers'])
    assert not result.variants[0].output_path.exists()


def test_unknown_flatlay_pose_still_needs_review():
    d=source_mask_preflight(*arrays(),overhead_scene=True)
    assert d['passed'] and any('unknown source pose' in w for w in d['warnings'])


def test_flatlay_overhead_source_allowed_but_not_accepted(tmp_path):
    rgb,a,_=arrays();src=tmp_path/'s.png';mask=tmp_path/'a.png';Image.fromarray(rgb).save(src);Image.fromarray(a).save(mask)
    result=run_pipeline(EditRequest(src,get_template('flat_lay'),mask_path=mask),[Route.CUTOUT_COMPOSE],
        RunConfig(variants=1,out_root=tmp_path/'runs'),source_contract=SourceContract('overhead','opaque','complete'))
    m=json.loads(result.manifest_path.read_text())
    assert m['source_mask_preflight']['passed'] and m['variants'][0]['success']
    assert m['source_mask_preflight']['publication_status']=='NOT_EVALUATED'
