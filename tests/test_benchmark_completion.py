from renderello.benchmark import _summarize

def row(success,output='x.png',score=5):
 return dict(source='source',route='pure_prompt',template='clean_studio',output=output,critical_errors=[],success=success,score_fidelity=score,seconds=1.0,credits_estimated=50)

def test_executor_failure_cannot_silently_count_as_no_critical_or_publication_proxy():
 s=_summarize([row(False)],1);g=s['gates'];r=s['routes']['pure_prompt']
 assert g['no_critical_alteration']['numerator']==0
 assert g['publish_ready']['numerator']==0
 assert not g['human_review_queue']
 assert len(g['execution_failure_queue'])==1
 assert r['mean_fidelity'] is None and r['completed_outputs']==0
 assert r['credits_per_publishable_output'] is None
 assert 'estimated' in r['credits_basis']

def test_mixed_attempt_denominators_and_cost_labels():
 s=_summarize([row(True,'ok.png'),row(False,'bad.png')],1);g=s['gates'];r=s['routes']['pure_prompt']
 assert g['no_critical_alteration']['rate']==0.5
 assert g['publish_ready']['rate']==0.5
 assert len(g['human_review_queue'])==1 and len(g['execution_failure_queue'])==1
 assert r['completed_outputs']==1 and r['failed_attempts']==1
 assert r['credits_per_successful_output']==100
 assert 'executor completion' in r['success_rate_basis']
 assert r['credits_per_publishable_output'] is None


def test_human_rating_cannot_promote_an_executor_failure():
 s=_summarize([row(False)],1,human={'x.png':dict(no_critical=True,publishable=5,rater='owner')})
 assert s['gates']['no_critical_alteration']['numerator']==0
 assert s['gates']['publish_ready']['numerator']==0
