import csv,json
from pathlib import Path
import pytest
from PIL import Image
from renderello.completion import saved_variant_failed
from renderello.gallery import build_gallery
from renderello.benchmark import summarize_runs

@pytest.mark.parametrize('success',[None,True,False])
@pytest.mark.parametrize('marker',['variant failed: ModuleNotFoundError: skimage','cutout failed: RuntimeError: model'])
def test_saved_failures_override_legacy_file_success(success,marker,tmp_path):
    rd=tmp_path/'run';rd.mkdir();p=rd/'partial.png';Image.new('RGB',(16,16)).save(p)
    v=dict(route='pure_prompt',seed=1,output=str(p),qa=dict(template='clean_studio',score_fidelity=1,critical_errors=[marker]))
    if success is not None:v['success']=success
    m=dict(source=str(p),template=dict(name="Clean studio",key="clean_studio"),variants=[v]);mp=rd/'manifest.json';mp.write_text(json.dumps(m));original=mp.read_bytes()
    assert saved_variant_failed(v)
    h=build_gallery([rd],tmp_path/'gallery.html').read_text();assert 'FAILED / PARTIAL ARTIFACT' in h and 'auto fidelity' not in h
    cp=summarize_runs([rd],tmp_path/'summary');row=list(csv.DictReader(cp.open()))[0]
    assert row['success']=='False' and not row['score_fidelity']
    assert mp.read_bytes()==original

@pytest.mark.parametrize('warnings',[[],['product shape drift'],['label text unverified'],['nonempty matte reaches frame boundary']])
def test_quality_warnings_not_execution_failure(warnings):
    assert not saved_variant_failed(dict(success=True,qa=dict(critical_errors=warnings)))
