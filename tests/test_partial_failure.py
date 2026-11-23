import json
import pytest
from renderello.pipeline import run_pipeline
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.config import RunConfig
from renderello.backends.base import StubEditor
from renderello.gallery import build_gallery


@pytest.mark.parametrize('route',[Route.CUTOUT_COMPOSE,Route.PURE_PROMPT,Route.HYBRID])
def test_saved_png_after_qa_exception_is_failed_not_success(monkeypatch,product_path,tmp_path,route):
    from renderello import qa
    def fail(*a,**k):raise RuntimeError('controlled QA failure after export')
    monkeypatch.setattr(qa,'evaluate',fail)
    r=run_pipeline(EditRequest(product_path,get_template('clean_studio')),[route],RunConfig(variants=1,out_root=tmp_path/'runs'),editor=StubEditor())
    v=r.variants[0];m=json.loads(r.manifest_path.read_text())
    assert v.output_path.is_file() and not v.success and v.qa.score_fidelity is None
    assert not m['variants'][0]['success'] and m['variants'][0]['artifact_exists']
    assert m['variants'][0]['fidelity_review']['status']=='FAILED_PARTIAL_ARTIFACT'
    assert not m['variants'][0]['fidelity_review']['required']
    assert r.best_by_qa() is None
    html=build_gallery([r.run_dir],tmp_path/'gallery.html').read_text()
    assert 'FAILED / PARTIAL ARTIFACT' in html and 'No quality rating' in html
    assert 'auto fidelity' not in html


def test_webui_partial_result_has_no_normal_download(monkeypatch,product_path,tmp_path):
    from renderello import qa
    from renderello.webui import _Handler
    def fail(*a,**k):raise RuntimeError('controlled QA failure after export')
    monkeypatch.setattr(qa,'evaluate',fail)
    r=run_pipeline(EditRequest(product_path,get_template('clean_studio')),[Route.PURE_PROMPT],RunConfig(variants=1,out_root=tmp_path/'job'),editor=StubEditor())
    (tmp_path/'job'/'source-review.json').write_text(json.dumps(dict(pose='unknown',material='unknown',framing='unknown',require_complete=False)))
    handler=object.__new__(_Handler);handler.out_root=tmp_path
    page=handler._results_page(tmp_path/'job',product_path,r,'clean_studio','',None).decode()
    assert 'FAILED / PARTIAL ARTIFACT' in page and 'Download PNG' not in page
    assert 'auto fidelity' not in page
