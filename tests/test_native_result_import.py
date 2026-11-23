from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from renderello.checked_research.native_matting import run as make_result, sha256
from renderello.native_result_import import load_verified_result
from renderello.candidate_studio import run as studio, main


class TestPredictor:
    __test__ = False
    info = {'test_fixture_not_model_inference': True}
    def __call__(self, rgb, tri):
        return np.full(tri.shape, .5, np.float32), {'inference_seconds': 0}


@pytest.fixture
def saved(tmp_path):
    rgb = np.full((40, 48, 3), 220, np.uint8)
    alpha = np.zeros((40, 48), np.uint8)
    alpha[10:30, 12:36] = 255
    rgb[alpha == 255] = (20, 70, 120)
    s, a, result = tmp_path/'source.png', tmp_path/'mask.png', tmp_path/'result'
    Image.fromarray(rgb).save(s); Image.fromarray(alpha).save(a)
    make_result(s, a, result, source_sha256=sha256(s), mask_sha256=sha256(a),
                radius=2, predictor=TestPredictor())
    return rgb, alpha, s, a, result


def update_manifest(result, filename):
    p=result/'manifest.json';j=json.loads(p.read_text());data=(result/filename).read_bytes()
    j['outputs'][filename]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    p.write_text(json.dumps(j))


def test_replay_and_source_pairing(saved):
    rgb, initial, _, _, result = saved
    alpha, evidence=load_verified_result(result,rgb,initial)
    assert evidence['files_verified']==9 and evidence['source_rgb_exact']
    assert evidence['replay_exact'] and evidence['no_new_inference']
    assert not evidence['automatic_acceptance'] and not evidence['glass_optics_verified']
    assert evidence['model_execution_provenance']=='MANIFEST_CLAIM_NOT_AUTHENTICATED_BY_REPLAY'
    assert np.array_equal(alpha,np.array(Image.open(result/'alpha.png')))
    alpha[:]=0
    assert np.array(Image.open(result/'alpha.png')).max()==255


@pytest.mark.parametrize('kind',['wrong_source','wrong_initial','missing','symlink','bad_hash',
    'extra_output','missing_output_entry','wrong_shape','accepted','bad_manifest','oversize_manifest'])
def test_refuses_inconsistent_result(saved,tmp_path,kind):
    rgb, initial, _, _, result=saved
    if kind=='wrong_source':rgb=rgb.copy();rgb[0,0]=0
    elif kind=='wrong_initial':initial=initial.copy();initial[0,0]=255
    elif kind=='missing':(result/'alpha.png').unlink()
    elif kind=='symlink':p=result/'alpha.png';data=p.read_bytes();p.unlink();q=tmp_path/'other.png';q.write_bytes(data);p.symlink_to(q)
    elif kind=='bad_hash':(result/'alpha.png').write_bytes(b'changed')
    elif kind=='bad_manifest':(result/'manifest.json').write_text('invalid')
    elif kind=='oversize_manifest':(result/'manifest.json').write_bytes(b' '*65_537)
    else:
        p=result/'manifest.json';j=json.loads(p.read_text())
        if kind=='extra_output':j['outputs']['../escape.png']={}
        elif kind=='missing_output_entry':del j['outputs']['alpha.png']
        elif kind=='wrong_shape':j['inputs']['native_shape']=[48,40,3]
        elif kind=='accepted':j['status']='ACCEPTED'
        p.write_text(json.dumps(j))
    with pytest.raises((ValueError,OSError)):load_verified_result(result,rgb,initial)


@pytest.mark.parametrize('filename', ['source-native.png','component-rgba.png','alpha.png',
    'derived-trimap.png','prior-support-mask.png','dark.png','light.png'])
def test_semantic_replay_even_when_hash_updated(saved,filename):
    rgb, initial, _, _, result=saved
    p=result/filename;im=Image.open(p);array=np.array(im);array[0,0]=0 if array[0,0].sum() else 255
    Image.fromarray(array).save(p);update_manifest(result,filename)
    with pytest.raises(ValueError):load_verified_result(result,rgb,initial)


@pytest.mark.parametrize('kind',['nan','range','shape','object','enforced'])
def test_invalid_numpy_replay(saved,kind):
    rgb, initial, _, _, result=saved
    filename='raw-alpha.npy';raw=np.load(result/filename)
    if kind=='nan':raw[0,0]=np.nan
    elif kind=='range':raw[0,0]=2
    elif kind=='shape':raw=raw[:2]
    elif kind=='object':raw=np.array([{'not':'allowed'}],dtype=object)
    else:filename='enforced-alpha.npy';raw=np.zeros_like(raw)
    np.save(result/filename,raw);update_manifest(result,filename)
    with pytest.raises(ValueError):load_verified_result(result,rgb,initial)


def test_studio_integration_keeps_not_accepted(saved,tmp_path):
    rgb, initial, source, mask, result=saved
    out=tmp_path/'studio'
    report=studio(source,out,{'prior':mask},native_results={'gpu':{'initial':'prior','directory':result}})
    assert report['accepted_count']==0 and report['automatically_selected'] is None
    assert report['native_result_imports']['gpu']['replay_exact']
    assert 'SOURCE_BOUND_REVIEW_SPEC_REQUIRED' in report['candidates']['gpu']['render_blockers']
    assert any(flag.startswith('REFINEMENT_CHANGES_') for flag in report['candidates']['gpu']['render_blockers'])
    assert (out/'native-import-gpu.json').exists()
    assert (out/'DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()
    assert np.array_equal(np.array(Image.open(out/'candidates/gpu/cutout.png'))[:,:,:3],rgb)


@pytest.mark.parametrize('import_request', [{'wrong':1},{'initial':'missing','directory':'x'}])
def test_studio_bad_import_preserves_failure(saved,tmp_path,import_request):
    _, _, source, mask, _=saved;out=tmp_path/'failed'
    with pytest.raises(ValueError):studio(source,out,{'prior':mask},native_results={'gpu':import_request})
    assert (out/'failure.json').exists() and (out/'INCOMPLETE').exists()
    assert not (out/'DIAGNOSTICS_COMPLETE_NOT_ACCEPTED').exists()


@pytest.mark.parametrize('argument',['wrong','a==c','=b=c','a=b='])
def test_cli_rejects_bad_native_result(saved,tmp_path,argument):
    _, _, source, mask, _=saved
    with pytest.raises(SystemExit) as exc:
        main(['--source',str(source),'--out',str(tmp_path/'cli'),'--alpha','prior='+str(mask),'--native-result',argument])
    assert exc.value.code==2
