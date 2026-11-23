from dataclasses import replace
import pytest
from renderello.brief_acceptance import PlannedOutput, Attempt, Rating, evaluate
from renderello.review import DIMENSIONS


def fixture(n=20,good=20,usable=20):
    p=[PlannedOutput(str(i),'source'+str(i),'template'+str(i%8),'method',0,source_sha256=f'{i:064x}') for i in range(n)]
    a=[Attempt(str(i),'attempt'+str(i),'success',float(i),1.0,f'{i+10000:064x}') for i in range(n)]
    r=[Rating(str(i),f'{i+10000:064x}',{d:4 if d!='publish_ready_usefulness' or i<usable else 3 for d in DIMENSIONS},i<good,() if i<good else ('changed text',),'assistant_judgment','review'+str(i)) for i in range(n)]
    return p,a,r


def test_exact_original_gate_boundaries():
    p,a,r=fixture(good=16,usable=14);s=evaluate(p,a,r)
    assert s['fidelity_gate'] and s['usefulness_gate'] and s['coverage_gate']
    assert not s['production_go'] and s['blind_competitive_acceptance']=='NOT_ESTABLISHED'
    assert s['no_critical_fraction']==.8 and s['publish_usable_fraction']==.7
    assert s['rating_provenance']==['assistant_judgment']


def test_one_below_either_gate_fails():
    p,a,r=fixture(good=15,usable=14);assert not evaluate(p,a,r)['fidelity_gate']
    p,a,r=fixture(good=20,usable=13);assert not evaluate(p,a,r)['usefulness_gate']


def test_unrated_failure_remains_denominator():
    p,a,r=fixture();a[-1]=Attempt('19','attempt19','failed',5,None,failure='failed');r=r[:-1]
    s=evaluate(p,a,r)
    assert s['planned_outputs']==20 and s['rated_outputs']==19
    assert s['no_critical_fraction']==.95 and s['unrated_outputs']==1
    assert not s['all_reviews_complete'] and not s['charged_credits_complete']
    assert s['credits_per_generated_output'] is None


def test_no_attempts_or_ratings_not_a_zero_cost_success():
    p,_,_=fixture();s=evaluate(p,[],[])
    assert s['missing_attempt_outputs']==20 and s['generated_outputs']==0
    assert s['median_attempt_seconds'] is None
    assert s['credits_per_publish_usable_output'] is None
    assert not s['charged_credits_complete']


def test_retry_not_inferred_from_failure_count():
    p,a,r=fixture();a.append(Attempt('0','extra','failed',3,2,failure='timeout'))
    s=evaluate(p,a,r);assert s['retry_attempts']==1 and s['failed_attempts']==1
    assert s['charged_credits_known']==22 and s['credits_per_generated_output']==1.1


def test_first_try_cannot_be_replaced_by_retry():
    p,a,r=fixture();a.insert(0,Attempt('0','first-failed','failed',2,1,failure='failed first try'))
    with pytest.raises(ValueError,match='first-try'):evaluate(p,a,r)
    p[0]=replace(p[0],first_try=False)
    assert evaluate(p,a,r)['retry_attempts']==1


def test_one_method_per_summary_and_unique_plan():
    p,a,r=fixture()
    with pytest.raises(ValueError):evaluate(p+[p[0]],a,r)
    p[0]=replace(p[0],method='other')
    with pytest.raises(ValueError):evaluate(p,a,r)


def test_foreign_duplicate_attempts_and_ratings():
    p,a,r=fixture()
    for aa,rr in [(a+[a[0]],r),(a,r+[r[0]]),(a+[replace(a[0],output_id='foreign',attempt_id='new')],r),(a,[replace(r[0],output_sha256='b'*64)]+r[1:])]:
        with pytest.raises(ValueError):evaluate(p,aa,rr)


def test_named_competitors_do_not_create_blind_acceptance():
    p,a,r=fixture();s=evaluate(p,a,r,competing_specialists=('toolA','toolB'))
    assert not s['production_go']
    with pytest.raises(ValueError):evaluate(p,a,r,competing_specialists=('toolA','toolA'))


@pytest.mark.parametrize('kw',[dict(seconds=True),dict(seconds=-1),dict(seconds=float('nan')),
 dict(credits_charged=float('inf')),dict(status='done'),dict(output_sha256='z'*64),dict(output_sha256=None),dict(failure='not success')])
def test_attempt_invalid(kw):
    args=dict(output_id='o',attempt_id='a',status='success',seconds=1,credits_charged=0,output_sha256='a'*64);args.update(kw)
    with pytest.raises(ValueError):Attempt(**args)


@pytest.mark.parametrize('kw',[dict(alternative=True),dict(alternative=-1),dict(first_try=1),dict(source_id=''),dict(method=' ')])
def test_invalid_plan(kw):
    d=dict(output_id='o',source_id='s',template_id='t',method='m',alternative=0);d.update(kw)
    with pytest.raises(ValueError):PlannedOutput(**d)


def test_invalid_scores_and_claimed_human_label():
    _,_,r=fixture();r=r[0]
    for kw in [dict(scores={}),dict(scores={d:True for d in DIMENSIONS}),dict(no_critical=1),dict(no_critical=False),dict(critical_errors=('error',)),dict(provenance='independent_human'),dict(review_reference='')]:
        with pytest.raises(ValueError):replace(r,**kw)


def test_latency_and_zero_credit_means_recorded_free_not_projected():
    p,a,r=fixture();a=[replace(x,credits_charged=0) for x in a];s=evaluate(p,a,r)
    assert s['charged_credits_complete'] and s['charged_credits_known']==0
    assert s['credits_per_publish_usable_output']==0
    assert s['median_attempt_seconds']==9.5


@pytest.mark.parametrize('kw',[dict(publish_threshold=True),dict(publish_threshold=6),dict(minimum_sources=0),dict(minimum_templates=1.1)])
def test_invalid_evaluation_policy(kw):
    with pytest.raises(ValueError):evaluate(*fixture(),**kw)


def test_scores_are_snapshot_not_mutable_external_mapping():
    p,a,r=fixture();scores={d:4 for d in DIMENSIONS}
    r[0]=replace(r[0],scores=scores);scores['publish_ready_usefulness']=1
    assert r[0].scores['publish_ready_usefulness']==4
    with pytest.raises(TypeError):r[0].scores['publish_ready_usefulness']=1


def test_source_names_cannot_inflate_distinct_content_coverage():
    p,a,r=fixture();p=[replace(x,source_sha256='a'*64) for x in p]
    s=evaluate(p,a,r);assert not s['coverage_gate'] and s['bound_distinct_source_hashes']==1
    p=[replace(x,source_sha256=None) for x in p];s=evaluate(p,a,r)
    assert not s['coverage_gate'] and s['unbound_source_ids']==20


def test_duplicate_logical_slots_and_conflicting_hash_refused():
    p,a,r=fixture();p.append(replace(p[0],output_id='duplicate'))
    with pytest.raises(ValueError,match='logical'):evaluate(p,a,r)
    p,a,r=fixture();p[1]=replace(p[1],source_id=p[0].source_id)
    with pytest.raises(ValueError,match='conflicting'):evaluate(p,a,r)


def test_missing_latency_is_not_reported_as_complete():
    p,a,r=fixture();a[0]=replace(a[0],seconds=None)
    s=evaluate(p,a,r);assert not s['latency_complete'] and s['latency_samples']==19


def test_failed_attempt_cannot_carry_success_hash():
    with pytest.raises(ValueError):Attempt('o','a','failed',1,0,'a'*64,'failed')


def test_plan_only_does_not_establish_coverage():
    p,_,_=fixture();s=evaluate(p,[],[])
    assert s['planned_coverage_gate'] and not s['coverage_gate']
    assert s['generated_artifact_bound_sources']==0
    assert s['reviewed_artifact_bound_templates']==0


def test_generated_unreviewed_coverage_is_separate():
    p,a,_=fixture();s=evaluate(p,a,[])
    assert s['generated_artifact_bound_sources']==20
    assert s['generated_artifact_bound_templates']==8
    assert s['reviewed_artifact_bound_sources']==0 and not s['coverage_gate']


def test_same_artifact_renamed_under_twenty_sources_does_not_pad_coverage():
    p,a,r=fixture();a=[replace(x,output_sha256='a'*64)for x in a];r=[replace(x,output_sha256='a'*64)for x in r]
    s=evaluate(p,a,r)
    assert s['reviewed_bound_source_hashes']==20 and s['reviewed_template_ids']==8
    assert s['reviewed_artifact_bound_sources']==1 and s['reviewed_artifact_bound_templates']==1
    assert s['unique_successful_artifact_hashes']==1 and s['repeated_successful_artifact_occurrences']==19
    assert not s['coverage_gate']
    # Rates stay per planned slot; duplicate diagnostics do not silently change denominators.
    assert s['planned_outputs']==20 and s['rated_outputs']==20


def test_failed_first_try_retry_not_generated_first_try_coverage():
    p,a,r=fixture();a.insert(0,Attempt('0','failed-first','failed',1,0,failure='first failed'))
    s=evaluate(p,a,r[1:])
    assert s['generated_outputs']==20 and s['eligible_generated_output_slots']==19
    assert s['generated_artifact_bound_sources']==19
    assert not s['coverage_gate']


def test_failed_only_template_not_counted_as_generated_or_reviewed():
    p,a,r=fixture(n=8);a[-1]=Attempt('7','attempt7','failed',1,0,failure='failed')
    s=evaluate(p,a,r[:-1],minimum_sources=7,minimum_templates=8)
    assert s['distinct_templates']==8 and s['generated_template_ids']==7 and not s['coverage_gate']


def test_hash_alias_sources_still_do_not_inflate_reviewed_source_coverage():
    p,a,r=fixture();p=[replace(x,source_sha256='c'*64)for x in p]
    s=evaluate(p,a,r)
    assert s['reviewed_bound_source_hashes']==1 and s['reviewed_artifact_bound_sources']==1
    assert s['reviewed_artifact_bound_templates']==8 and not s['coverage_gate']


def test_matching_reassigns_shared_artifact_to_cover_another_identity():
    from renderello.brief_acceptance import _artifact_matching_count
    assert _artifact_matching_count({'A':{'x','y'},'B':{'x'}})==2
    assert _artifact_matching_count({'A':{'x','y'},'B':{'x','y'},'C':{'x','y'}})==2
    assert _artifact_matching_count({})==0


def test_iterative_matching_agrees_with_exhaustive_small_graphs():
    from renderello.brief_acceptance import _artifact_matching_count
    from itertools import product
    import random
    rng=random.Random(20261113)
    for _ in range(100):
        graph={str(i):{str(j)for j in range(4)if rng.randrange(2)}for i in range(4)}
        best=0
        for choices in product(*[sorted(v)+[None]for v in graph.values()]):
            nonempty=[v for v in choices if v is not None]
            if len(set(nonempty))==len(nonempty):best=max(best,len(nonempty))
        assert _artifact_matching_count(graph)==best


def test_visual_template_distinctness_not_inferred_from_artifact_diversity():
    s=evaluate(*fixture())
    assert s['coverage_gate'] and s['template_visual_distinctness']=='NOT_ESTABLISHED'
    assert 'TEMPLATE_VISUAL_DISTINCTNESS_REQUIRES_REVIEW'in s['blockers']
    assert not s['production_go']
