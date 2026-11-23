from dataclasses import replace
from decimal import Decimal
import json
import subprocess
import sys
import pytest
from renderello.brief_acceptance import PlannedOutput, Attempt, Rating
from renderello.brief_economics import PriceEvidence, build_report, from_payload, amount
from renderello.review import DIMENSIONS


def fixture():
    p=[PlannedOutput(str(i),'s'+str(i),'t'+str(i),'m',0,source_sha256=f'{i:064x}') for i in range(2)]
    a=[Attempt(str(i),'a'+str(i),'success',10+i,50,f'{i+5:064x}') for i in range(2)]
    r=[Rating(str(i),f'{i+5:064x}',{d:4 for d in DIMENSIONS},True,(),'assistant_judgment','review'+str(i)) for i in range(2)]
    return p,a,r


def price():
    return PriceEvidence('USD','0.002','included_subscription_allocation','test fixture, not vendor quote','2026-11-14','2026-11-14','fixture-provider-account')


def report(*args,**kw):
    opts=dict(charge_credit_system='fixture-provider-account',as_of='2026-11-14',price=price(),sale_price='1',sale_reference='hypothetical test price',other_cost_per_usable='0.05',other_cost_reference='hypothetical full other costs')
    opts.update(kw);return build_report(*(args or fixture()),**opts)


def test_decimal_cost_and_margin_scenario_not_actual_profit_or_acceptance():
    r=report()
    assert Decimal(r['total_attempt_credit_cost'])==Decimal('.2')
    assert Decimal(r['cost_per_distinct_reviewed_usable_artifact'])==Decimal('.1')
    assert Decimal(r['contribution_per_usable_scenario'])==Decimal('.85')
    assert Decimal(r['contribution_margin_fraction_scenario'])==Decimal('.85')
    assert not r['production_go'] and r['best_quality_model'] is None and r['value_fallback'] is None
    assert not r['acceptance']['coverage_gate'] and r['live_calls']==0


def test_failed_attempt_cost_and_explicit_retry_are_not_lost():
    p,a,r=fixture();p[0]=replace(p[0],first_try=False)
    a.insert(0,Attempt('0','failed-first','failed',5,10,failure='provider failed'))
    s=report(p,a,r)
    assert Decimal(s['total_attempt_credit_cost'])==Decimal('.22')
    assert Decimal(s['cost_per_distinct_reviewed_usable_artifact'])==Decimal('.11')
    assert s['acceptance']['retry_attempts']==1 and s['acceptance']['median_attempt_seconds']==10


def test_standalone_failure_is_not_called_retry():
    p,a,r=fixture();a[1]=Attempt('1','a1','failed',5,10,failure='fail')
    s=report(p,a,r[:1]);assert s['acceptance']['retry_attempts']==0
    assert Decimal(s['cost_per_distinct_reviewed_usable_artifact'])==Decimal('.12')
    assert s['all_successful_artifacts_reviewed']
    assert not s['acceptance']['all_reviews_complete']
    assert Decimal(s['total_attempt_credit_cost'])==Decimal('.12')


@pytest.mark.parametrize('kind',['missing_charge','missing_attempt','unknown'])
def test_incomplete_history_refuses_total_and_margin(kind):
    p,a,r=fixture()
    if kind=='missing_charge':a[0]=replace(a[0],credits_charged=None)
    elif kind=='missing_attempt':a=a[1:];r=r[1:]
    else:a[0]=Attempt('0','a0','unknown',None,0,failure='pending');r=r[1:]
    s=report(p,a,r)
    assert not s['charge_evidence_complete']
    assert s['total_attempt_credit_cost'] is None and s['contribution_per_usable_scenario'] is None


@pytest.mark.parametrize('day',['2026-11-13','2026-11-15'])
def test_outdated_or_future_rate_is_not_current(day):
    s=report(as_of=day)
    assert not s['price_in_declared_validity'] and s['total_attempt_credit_cost'] is None


def test_no_rate_not_catalog_fallback():
    s=report(price=None)
    assert s['total_attempt_credit_cost'] is None and 'PRICE_EVIDENCE_MISSING' in s['blockers']


def test_no_sales_or_other_cost_defaults():
    s=report(sale_price=None,sale_reference=None,other_cost_per_usable=None,other_cost_reference=None)
    assert s['contribution_per_usable_scenario'] is None
    assert s['cost_per_distinct_reviewed_usable_artifact'] is not None


def test_explicit_zero_other_cost_is_scenario_not_omission():
    s=report(other_cost_per_usable='0');assert Decimal(s['contribution_per_usable_scenario'])==Decimal('.9')


def test_zero_recorded_credit_cost_not_null_or_future_free_claim():
    p,a,r=fixture();a=[replace(x,credits_charged=0) for x in a]
    s=report(p,a,r);assert Decimal(s['total_attempt_credit_cost'])==0
    assert Decimal(s['cost_per_distinct_generated_artifact'])==0
    assert 'not_proof' in s['zero_recorded_credits_semantics']


def test_duplicate_artifacts_cannot_lower_unit_cost():
    p,a,r=fixture();a[1]=replace(a[1],output_sha256=a[0].output_sha256);r[1]=replace(r[1],output_sha256=a[0].output_sha256)
    s=report(p,a,r);assert s['distinct_usable_artifacts']==1
    assert Decimal(s['cost_per_distinct_reviewed_usable_artifact'])==Decimal('.2')


def test_critical_or_unusable_output_cannot_create_usable_denominator():
    p,a,r=fixture();r=[replace(x,no_critical=False,critical_errors=('invented text',)) for x in r]
    s=report(p,a,r);assert s['distinct_usable_artifacts']==0 and s['contribution_per_usable_scenario'] is None


def test_first_try_cannot_hide_failed_first_attempt():
    p,a,r=fixture();a.insert(0,Attempt('0','failed-first','failed',5,0,failure='fail'))
    with pytest.raises(ValueError,match='first-try'):report(p,a,r)


@pytest.mark.parametrize('v',[True,-1,'NaN','Infinity','-Infinity','bad',[],{},'1e16','0.0000000000001'])
def test_bad_amounts(v):
    with pytest.raises(ValueError):amount(v,'test')


@pytest.mark.parametrize('kw',[dict(currency='usd'),dict(currency='US'),dict(per_credit='0'),dict(per_credit=True),dict(rate_basis='free_forever'),dict(source_reference=''),dict(observed_on='tomorrow'),dict(valid_through='2026-11-13')])
def test_bad_price_evidence(kw):
    args=dict(currency='USD',per_credit='.1',rate_basis='usage_invoice',source_reference='invoice',observed_on='2026-11-14',valid_through='2026-11-14',credit_system='fixture-provider-account');args.update(kw)
    with pytest.raises(ValueError):PriceEvidence(**args)


@pytest.mark.parametrize('kw',[dict(sale_reference=None),dict(other_cost_reference=None),dict(as_of='today'),dict(sale_price=0),dict(other_cost_per_usable=-1),dict(price={})])
def test_missing_or_invalid_explicit_policy(kw):
    with pytest.raises(ValueError):report(**kw)


def payload():
    p,a,r=fixture()
    return dict(plan=[vars(x) for x in p],attempts=[vars(x) for x in a],ratings=[dict(vars(x),scores=dict(x.scores)) for x in r],price=vars(price()),charge_credit_system='fixture-provider-account',as_of='2026-11-14')


def test_json_adapter_and_unknown_fields():
    x=payload();assert from_payload(json.loads(json.dumps(x)))['charge_evidence_complete']
    x['live']=True
    with pytest.raises(ValueError):from_payload(x)


def test_cli_no_overwrite_offline_roundtrip(tmp_path):
    src=tmp_path/'in.json';out=tmp_path/'out.json';src.write_text(json.dumps(payload()))
    cmd=[sys.executable,'-m','renderello.brief_economics','--input',str(src),'--output',str(out)]
    first=subprocess.run(cmd,capture_output=True,text=True);assert first.returncode==0,first.stderr
    before=out.read_bytes();assert json.loads(before)['live_calls']==0
    assert subprocess.run(cmd,capture_output=True).returncode==2
    assert out.read_bytes()==before


def test_duplicate_artifact_conflicting_review_is_not_usable():
    p,a,r=fixture();a[1]=replace(a[1],output_sha256=a[0].output_sha256)
    r[1]=replace(r[1],output_sha256=a[0].output_sha256,no_critical=False,critical_errors=('changed logo',))
    s=report(p,a,r)
    assert s['distinct_usable_artifacts']==0
    assert s['cost_per_distinct_reviewed_usable_artifact'] is None
    assert 'CONFLICTING_DUPLICATE_ARTIFACT_REVIEWS' in s['blockers']


@pytest.mark.parametrize('scope',[None,'different-provider','same-provider-different-account'])
def test_credit_system_must_explicitly_match(scope):
    with pytest.raises(ValueError,match='credit systems'): report(charge_credit_system=scope)


def test_unreviewed_successful_retry_artifact_keeps_usable_economics_unknown():
    p,a,r=fixture();p[0]=replace(p[0],first_try=False)
    a.append(Attempt('0','another-generated-attempt','success',20,50,'f'*64))
    s=report(p,a,r)
    assert s['acceptance']['all_reviews_complete']
    assert not s['all_successful_artifacts_reviewed']
    assert s['cost_per_distinct_reviewed_usable_artifact'] is None
    assert 'INCOMPLETE_SUCCESSFUL_ARTIFACT_REVIEWS' in s['blockers']
