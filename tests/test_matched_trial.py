from pathlib import Path
import copy,json,hashlib
import pytest
from PIL import Image
from renderello.matched_trial import freeze_plan,verify_plan,audit_results,read_json,pixel_identity
from renderello.review import DIMENSIONS

@pytest.fixture
def setup(tmp_path):
 source=tmp_path/'source.png';Image.new('RGB',(32,24),(21,45,96)).save(source)
 spec=dict(methods=[dict(id='a',kind='magic_hour',selection_complete=False,settings={'model':'pending'}),dict(id='b',kind='specialist',selection_complete=False,settings={'model':'unexposed'})],cases=[dict(id='c1',source=str(source),template='studio',prompt='retain exact product',aspect='1:1',tags=['reflective'],origin='photograph_claimed',rights_record='private research; not clearance')])
 root=tmp_path/'plan';ev=tmp_path/'evidence';ev.mkdir();return spec,root,ev

def freeze(setup):
 spec,root,ev=setup;freeze_plan(spec,root);return root,ev

def event(ev,**kw):
 p=ev/'output.png';Image.new('RGB',(32,24),'red').save(p)
 e=dict(case='c1',method='a',attempt=1,state='output',seconds=0,credits_charged=10,output='output.png',output_sha256=hashlib.sha256(p.read_bytes()).hexdigest());e.update(kw);return e

def rating(**kw):
 r=dict(dimensions={d:4 for d in DIMENSIONS},critical_errors=[],no_critical=True,rater='owner');r.update(kw);return r

def test_blank_plan(setup):
 root,ev=freeze(setup);p=verify_plan(root);r=audit_results(root,ev,[])
 assert len(p['tasks'])==2 and r['planned_tasks']==2
 assert r['methods']['a']['not_started']==1
 assert r['methods']['a']['claimed_targets_met'] is None
 assert r['acceptance_pass'] is None

def test_output_not_quality(setup):
 root,ev=freeze(setup);r=audit_results(root,ev,[event(ev)])['methods']['a']
 assert r['first_outputs']==1 and r['claimed_publishable'] is None
 assert r['actual_credits_total']==10
 assert r['first_attempt_median_seconds']==0
 assert r['actual_credits_per_claimed_publishable'] is None

def test_claim_is_not_authentication(setup):
 root,ev=freeze(setup);r=audit_results(root,ev,[event(ev,rating=rating())])
 assert r['methods']['a']['claimed_targets_met'] is True
 assert r['methods']['a']['acceptance_pass'] is None
 assert r['quality_status']=='NOT_EVALUATED'

@pytest.mark.parametrize('rater',['agent_eye','independent_human'])
def test_rater_labels(setup,rater):
 root,ev=freeze(setup);r=audit_results(root,ev,[event(ev,rating=rating(rater=rater))])['methods']['a']
 assert r['acceptance_pass'] is None
 assert r['claimed_publishable']==(None if rater=='agent_eye' else 1)

@pytest.mark.parametrize('key,value',[('seconds',-1),('seconds',True),('seconds',float('nan')),('credits_charged',float('inf')),('credits_charged',-1),('attempt',True),('attempt',0),('attempt',101),('state','success'),('method','unplanned')])
def test_bad_events(setup,key,value):
 root,ev=freeze(setup);e=event(ev);e[key]=value
 with pytest.raises(ValueError):audit_results(root,ev,[e])

@pytest.mark.parametrize('change',[{'dimensions':{}},{'no_critical':False},{'critical_errors':['changed_logo']},{'rater':'robot'},{'dimensions':{d:True for d in DIMENSIONS}}])
def test_bad_ratings(setup,change):
 root,ev=freeze(setup);r=rating();r.update(change)
 with pytest.raises(ValueError):audit_results(root,ev,[event(ev,rating=r)])

def test_failure_denominator_retry_cost(setup):
 root,ev=freeze(setup)
 first=dict(case='c1',method='a',attempt=1,state='failed',seconds=12,credits_charged=5,failure='provider refused')
 second=event(ev,attempt=2,rating=rating())
 r=audit_results(root,ev,[second,first])['methods']['a']
 assert r['first_failures']==1 and r['first_outputs']==0 and r['retries_recorded']==1
 assert r['actual_credits_total']==15 and r['claimed_publishable']==0
 assert r['actual_credits_per_claimed_publishable'] is None

def test_missing_first_refused(setup):
 root,ev=freeze(setup)
 with pytest.raises(ValueError,match='missing earlier'):audit_results(root,ev,[event(ev,attempt=2)])

def test_duplicate_attempt_refused(setup):
 root,ev=freeze(setup);e=event(ev)
 with pytest.raises(ValueError,match='duplicate attempt'):audit_results(root,ev,[e,e])

def test_unknown_charge_not_zero(setup):
 root,ev=freeze(setup);r=audit_results(root,ev,[event(ev,credits_charged=None,credits_estimated=50,rating=rating())])['methods']['a']
 assert r['actual_credits_total'] is None and r['unknown_charge_attempts']==1
 assert r['actual_credits_per_claimed_publishable'] is None

def test_no_overwrite(setup):
 spec,root,ev=setup;freeze_plan(spec,root)
 with pytest.raises(FileExistsError):freeze_plan(spec,root)

@pytest.mark.parametrize('mutation',['extra','changed','missing'])
def test_frozen_tamper(setup,mutation):
 root,ev=freeze(setup)
 if mutation=='extra':(root/'extra').write_text('extra')
 elif mutation=='changed':(root/'plan.json').write_text('{}')
 else:(root/'inputs/c1.png').unlink()
 with pytest.raises(ValueError):verify_plan(root)

def test_duplicate_decoded_source_cell(setup):
 spec,root,ev=setup;source=Path(spec['cases'][0]['source']);other=source.with_name('other.png')
 Image.open(source).save(other,compress_level=0)
 assert other.read_bytes()!=source.read_bytes()
 c=copy.deepcopy(spec['cases'][0]);c.update(id='c2',source=str(other));spec['cases'].append(c)
 with pytest.raises(ValueError,match='duplicate normalized'):freeze_plan(spec,root)
 assert not root.exists()

def test_one_source_different_templates_counts_once(setup):
 spec,root,ev=setup;c=copy.deepcopy(spec['cases'][0]);c.update(id='c2',template='outdoor');spec['cases'].append(c);freeze_plan(spec,root)
 r=audit_results(root,ev,[]);assert r['unique_normalized_sources']==1 and r['planned_tasks']==4

@pytest.mark.parametrize('key,value',[('id','../a'),('tags',['invented']),('tags',['reflective','reflective']),('aspect','4:3'),('origin','verified_photo'),('rights_record',''),('prompt','')])
def test_bad_cases(setup,key,value):
 spec,root,ev=setup;spec['cases'][0][key]=value
 with pytest.raises(ValueError):freeze_plan(spec,root)
 assert not root.exists()

@pytest.mark.parametrize('path',['../output.png','/tmp/output.png','C:/output.png','foo\\bar.png','./output.png'])
def test_output_paths(setup,path):
 root,ev=freeze(setup)
 with pytest.raises(ValueError):audit_results(root,ev,[event(ev,output=path)])

def test_output_mutation(setup):
 root,ev=freeze(setup);e=event(ev);(ev/'output.png').write_bytes(b'x')
 with pytest.raises(ValueError,match='bytes changed'):audit_results(root,ev,[e])

def test_output_symlink(setup):
 root,ev=freeze(setup);e=event(ev);(ev/'link.png').symlink_to(ev/'output.png');e['output']='link.png'
 with pytest.raises(ValueError,match='symlink'):audit_results(root,ev,[e])

@pytest.mark.parametrize('payload',['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'])
def test_strict_json(tmp_path,payload):
 p=tmp_path/'a.json';p.write_text(payload)
 with pytest.raises(ValueError):read_json(p)

def test_repeated_output_not_hidden(setup):
 root,ev=freeze(setup);a=event(ev);b=dict(a,method='b');r=audit_results(root,ev,[a,b])
 assert len(r['repeated_output_pixels'])==1
 assert r['planned_tasks']==2

def test_explicit_conversion_rate(setup):
 root,ev=freeze(setup);r=audit_results(root,ev,[event(ev)],usd_per_credit=.01)
 assert r['methods']['a']['usd_cost_total']==.1

def test_alpha_identity(tmp_path):
 a=Image.new('RGBA',(3,4),(1,2,3,0));b=Image.new('RGBA',(3,4),(9,2,3,0))
 assert pixel_identity(a)!=pixel_identity(b)

@pytest.mark.parametrize('state',['failed','unconfirmed'])
def test_failure_cannot_carry_ratings(setup,state):
 root,ev=freeze(setup)
 with pytest.raises(ValueError):audit_results(root,ev,[event(ev,state=state,failure='no',rating=rating())])
