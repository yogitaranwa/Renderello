import json,hashlib
from dataclasses import asdict
import numpy as np
import pytest
from PIL import Image
from renderello import scene_placement_cli as m
from renderello.reviewed_scene_placement import ScenePlacementReview
from renderello.contact_spans import ContactPolicy
from renderello.candidate_diagnostics import digest


def fixture(tmp_path):
    s=np.full((30,30,3),160,np.uint8);a=np.zeros((30,30),np.uint8);a[7:20,8:22]=255
    bg=np.full((60,70,3),90,np.uint8);pl=np.full((60,70),255,np.uint8);ex=np.zeros_like(pl)
    arrays=dict(source=s,alpha=a,scene=bg,surface=pl,exclusion=ex);paths={}
    for n,v in arrays.items():
        paths[n]=tmp_path/(n+'.png');Image.fromarray(v).save(paths[n])
    r=ScenePlacementReview(digest(s),digest(a),digest(bg),digest(pl>0),digest(ex>0),'overhead','overhead','diffuse','diffuse','Synthetic source fixture','Synthetic plane fixture',True,True)
    p=ContactPolicy('flatlay','opaque','complete','overhead',digest(s),digest(a),True,length=3)
    spec=dict(schema='renderello-actual-scene-placement-review.v1',files={n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in paths.items()},review=asdict(r),contact_policy=asdict(p),spans=[],translation=[10,10])
    paths['review']=tmp_path/'review.json';paths['review'].write_text(json.dumps(spec));return paths,spec,s,a


def save(paths,spec):paths['review'].write_text(json.dumps(spec))


def test_snapshot_export_and_float_shadow(tmp_path):
    paths,spec,s,a=fixture(tmp_path);out=tmp_path/'out';r=m.run(**paths,out=out)
    assert r['status']=='EXECUTED_REVIEW_REQUIRED' and not r['photo_accepted'] and r['models_called']==0
    assert not(out/'INCOMPLETE').exists()
    for n,p in paths.items():assert p.read_bytes()==(out/('review.json'if n=='review'else n+'.png')).read_bytes()
    im=np.array(Image.open(out/'designed-preview.png'));assert np.array_equal(im[10:40,10:40][a==255],s[a==255])
    shadow=np.load(out/'shadow.npy',allow_pickle=False);assert shadow.dtype.kind=='f'and shadow.shape==(60,70)
    assert len(r['outputs'])==3


@pytest.mark.parametrize('change',[{'target_pose':'front'},{'source_light':'unknown'},{'source_reviewed':False},{'target_light':'hard_left'}])
def test_expected_refusal_manifest_no_candidate(tmp_path,change):
    paths,spec,*_=fixture(tmp_path);spec['review'].update(change);save(paths,spec);out=tmp_path/'out';r=m.run(**paths,out=out)
    assert r['status']=='REFUSED' and r['report']['blockers'] and not r['outputs']
    assert not(out/'designed-preview.png').exists() and not(out/'INCOMPLETE').exists()
    assert json.loads((out/'manifest.json').read_text())==r


@pytest.mark.parametrize('what',['file_hash','pixel_hash','schema','extra_field','span_type','translation'])
def test_invalid_input_does_not_create_output(tmp_path,what):
    paths,spec,*_=fixture(tmp_path)
    if what=='file_hash':spec['files']['source']='0'*64
    if what=='pixel_hash':spec['review']['scene_sha256']='0'*64
    if what=='schema':spec['schema']='wrong'
    if what=='extra_field':spec['unexpected']=True
    if what=='span_type':spec['spans']={}
    if what=='translation':spec['translation']=[True,0]
    save(paths,spec);out=tmp_path/'out'
    with pytest.raises((ValueError,TypeError)):m.run(**paths,out=out)
    assert not out.exists()


@pytest.mark.parametrize('what',['RGBA','mask','orientation'])
def test_no_silent_normalization(tmp_path,what):
    paths,spec,*_=fixture(tmp_path)
    if what=='RGBA':Image.open(paths['source']).convert('RGBA').save(paths['source'])
    if what=='mask':
        im=Image.open(paths['surface']);v=np.array(im);v[0,0]=127;Image.fromarray(v).save(paths['surface'])
    if what=='orientation':
        im=Image.open(paths['source']);e=im.getexif();e[274]=6;im.save(paths['source'],exif=e)
    spec['files']={n:hashlib.sha256(paths[n].read_bytes()).hexdigest()for n in m.NAMES};save(paths,spec)
    with pytest.raises(ValueError):m.run(**paths,out=tmp_path/'out')
    assert not(tmp_path/'out').exists()


def test_output_preserved(tmp_path):
    paths,*_=fixture(tmp_path);out=tmp_path/'out';out.mkdir();(out/'keep').write_text('untouched')
    with pytest.raises(FileExistsError):m.run(**paths,out=out)
    assert(out/'keep').read_text()=='untouched'


def test_unexpected_composer_error_leaves_incomplete(tmp_path,monkeypatch):
    paths,*_=fixture(tmp_path);out=tmp_path/'out'
    def fail(*a,**kw):raise RuntimeError('injected unexpected error')
    monkeypatch.setattr(m,'compose_reviewed_placement',fail)
    with pytest.raises(RuntimeError):m.run(**paths,out=out)
    assert(out/'INCOMPLETE').exists() and not(out/'manifest.json').exists()


def test_cli_entry(tmp_path,capsys):
    paths,*_=fixture(tmp_path);out=tmp_path/'out';argv=[]
    for n,p in dict(paths,out=out).items():argv+=['--'+n,str(p)]
    assert m.main(argv)==0
    assert json.loads(capsys.readouterr().out)['status']=='EXECUTED_REVIEW_REQUIRED'


def test_snapshot_limit(tmp_path):
    p=tmp_path/'bytes';p.write_bytes(b'1234')
    with pytest.raises(ValueError):m._snapshot(p,3)


def test_dimension_budget_before_decode(tmp_path):
    paths,*_=fixture(tmp_path)
    with pytest.raises(ValueError,match='budget'):m._decode(paths['source'].read_bytes(),'RGB',10)
