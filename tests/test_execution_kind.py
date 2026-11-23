import pytest
from renderello.execution_kind import describe_execution


@pytest.mark.parametrize('route,backend,kwargs,expected',[
 ('pure_prompt','stub',{},'TEST_DOUBLE_NOT_GENERATIVE_MODEL'),
 ('hybrid','stub',{},'TEST_DOUBLE_NOT_GENERATIVE_MODEL'),
 ('scene_compose','supplied_mask',{'scene_mock':True},'PROCEDURAL_MOCK_SCENE'),
 ('scene_compose','supplied_mask',{'reference':True},'SAVED_REFERENCE_BACKDROP'),
 ('cutout_compose','grabcut',{},'DETERMINISTIC_COMPOSITE'),
 ('pure_prompt','magichour',{},'EXECUTOR_NOT_CLASSIFIED'),
 ('scene_compose','grabcut',{},'EXECUTOR_NOT_CLASSIFIED')])
def test_executor_class_does_not_create_quality_claim(route,backend,kwargs,expected):
    d=describe_execution(route,backend,**kwargs)
    assert d['kind']==expected and not d['quality_evidence'] and not d['credits_verified']


def test_pipeline_records_stub_and_gallery_recovers_legacy(product_path,tmp_path):
    import json
    from renderello.pipeline import run_pipeline
    from renderello.config import RunConfig
    from renderello.schemas import EditRequest,Route
    from renderello.templates import get_template
    from renderello.backends.base import StubEditor
    from renderello.gallery import build_gallery
    r=run_pipeline(EditRequest(product_path,get_template('clean_studio')),[Route.PURE_PROMPT],RunConfig(variants=1,out_root=tmp_path/'r'),editor=StubEditor())
    m=json.loads(r.manifest_path.read_text());assert m['variants'][0]['execution_kind']['kind']=='TEST_DOUBLE_NOT_GENERATIVE_MODEL'
    m['variants'][0].pop('execution_kind');r.manifest_path.write_text(json.dumps(m))
    assert 'TEST DOUBLE: deterministic test editor' in build_gallery([r.run_dir],tmp_path/'g.html').read_text()
