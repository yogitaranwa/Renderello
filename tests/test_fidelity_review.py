import json
import pytest
from renderello.fidelity_review import review_requirement,review_caption
from renderello.pipeline import run_pipeline
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.config import RunConfig
from renderello.backends.base import StubEditor
from renderello.gallery import build_gallery


@pytest.mark.parametrize('route',['pure_prompt','hybrid'])
def test_quiet_generative_route_still_requires_exact_fidelity_review(route):
    state=review_requirement(route,[])
    assert state['required'] and not state['product_text_verified'] and not state['photo_accepted']
    assert 'Zero automated warnings' in review_caption(state)


@pytest.mark.parametrize('route',['cutout_compose','scene_compose'])
def test_quiet_composite_is_not_a_quality_pass(route):
    state=review_requirement(route,[])
    assert not state['required'] and state['status']=='NO_TRIPWIRE_NOT_A_QUALITY_PASS'
    assert 'unverified' in review_caption(state)


def test_failed_output_not_queued_as_existing_image():
    state=review_requirement('hybrid',['error'],output_exists=False)
    assert state['status']=='NOT_GENERATED' and not state['required']


def test_warning_on_any_route_needs_review():
    assert review_requirement('cutout_compose',['mask slab'])['required']


def test_pipeline_and_gallery_do_not_hide_a_quiet_hybrid(product_path,tmp_path):
    result=run_pipeline(EditRequest(source_path=product_path,template=get_template('clean_studio')),
                        [Route.CUTOUT_COMPOSE,Route.HYBRID],RunConfig(variants=1,out_root=tmp_path/'runs'),editor=StubEditor())
    m=json.loads(result.manifest_path.read_text())
    hybrid=next(v for v in m['variants']if v['route']=='hybrid')
    assert hybrid['fidelity_review']['required']
    # A legacy manifest lacks the new metadata. Gallery must recompute based
    # on route, not trust absent/stale pass-like fields or overwrite outputs.
    for v in m['variants']:
        v.pop('fidelity_review',None)
        if v['route']=='hybrid':v['qa']['critical_errors']=[]
    result.manifest_path.write_text(json.dumps(m))
    out=build_gallery([result.run_dir],tmp_path/'gallery.html').read_text()
    assert 'FIDELITY REVIEW REQUIRED: this route generates product pixels.' in out
    assert 'Zero automated warnings' in out
    assert result.variants[1].output_path.exists()


def test_benchmark_queue_includes_quiet_generated_route():
    from renderello.benchmark import _summarize
    common=dict(source='s',source_sha256='a'*64,template='clean_studio',seed=0,
                score_fidelity=5,critical_errors='',seconds=1.,credits_estimated=0)
    rows=[dict(common,route='hybrid',output='h.png'),dict(common,route='cutout_compose',output='c.png')]
    summary=_summarize(rows,1)
    queue=summary['gates']['human_review_queue']
    assert [r['output']for r in queue]==['h.png']
    assert queue[0]['flags']==''
    assert summary['acceptance_pass'] is None
