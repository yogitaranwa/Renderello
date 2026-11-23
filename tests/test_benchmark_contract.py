import csv,json
from pathlib import Path
import pytest
from renderello.benchmark import run_benchmark,load_manifest
from renderello.config import RunConfig
from tests.conftest import make_product

def manifest(tmp_path,**extra):
    make_product().save(tmp_path/'s.png')
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(sources=[dict(path='s.png',label='fixture',license='synthetic',source_contract=dict(pose='front',material='opaque',framing='complete'),templates=['clean_studio'])],templates=['clean_studio','luxury_hero'],routes=['scene_compose'],variants=1,**extra)));return p

def test_manifest_review_contract_reaches_preflight_and_preserves_failure(tmp_path):
    p=manifest(tmp_path)
    def forbidden(*a):raise AssertionError('incompatible camera must block before model')
    forbidden.camera_pose='elevated';forbidden.camera_matched=True
    out=run_benchmark(p,RunConfig(variants=4,out_root=tmp_path/'runs'),scene_fn=forbidden)
    rows=list(csv.DictReader(out.open()));assert len(rows)==1
    assert rows[0]['success']=='False'
    assert 'camera poses disagree' in rows[0]['critical_errors']
    assert json.loads(rows[0]['source_contract'])['pose']=='front'
    summary=json.loads((out.parent/'benchmark_report.json').read_text())
    assert summary['variants']==1 and summary['gates']['no_critical_alteration']['denominator']==1
    assert summary['templates_represented']==['clean_studio']

@pytest.mark.parametrize('bad',[0,5,True,'2'])
def test_invalid_repeat_count_fails_before_run(tmp_path,bad):
    p=manifest(tmp_path);m=json.loads(p.read_text());m['variants']=bad;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):load_manifest(p)
