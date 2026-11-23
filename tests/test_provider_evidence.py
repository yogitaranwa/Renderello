from pathlib import Path
import io,json,sys,subprocess
from dataclasses import asdict
import pytest,requests
from PIL import Image
from renderello.backends.magichour import MagicHourBackend,MagicHourError
from renderello.schemas import Aspect,EditRequest,Route
from renderello.config import RunConfig
from renderello.pipeline import run_pipeline
from renderello.templates import get_template
from renderello.brief_acceptance import PlannedOutput
from renderello.provider_attempts import import_attempts


class Reply:
    status_code=200
    def __init__(self,data=None,content=b''):self.data=data;self.content=content
    def json(self):return self.data
    def raise_for_status(self):pass


def backend(tmp,live=True):return MagicHourBackend(live=live,receipt_path=tmp/'receipts.jsonl',cost_log=tmp/'cost.jsonl',poll_interval=0)


def http(monkeypatch,*,detail=None,submission=None,fail_download=False):
    monkeypatch.setenv('MAGIC_HOUR_API_KEY','test-fixture-only')
    def post(url,**kw):
        if url.endswith('upload-urls'):return Reply({'items':[{'upload_url':'https://upload.invalid/a','file_path':'a.png'}]})
        return Reply(submission or {'id':'p1','credits_charged':150})
    def get(url,**kw):
        if 'image-projects'in url:return Reply(detail or {'status':'complete','credits_charged':50,'downloads':[{'url':'https://download.invalid/a'}]})
        if fail_download:raise RuntimeError('fixture download failure')
        out=io.BytesIO();Image.new('RGB',(40,40),'teal').save(out,'PNG');return Reply(content=out.getvalue())
    monkeypatch.setattr(requests,'post',post);monkeypatch.setattr(requests,'put',lambda *a,**kw:Reply());monkeypatch.setattr(requests,'get',get)


def pipeline(tmp,b):
    src=tmp/'source.png';mask=tmp/'mask.png'
    Image.new('RGB',(40,40),'teal').save(src)
    from PIL import ImageDraw
    a=Image.new('L',(40,40),0);ImageDraw.Draw(a).rectangle((8,8,31,31),fill=255);a.save(mask)
    return run_pipeline(EditRequest(src,get_template('clean_studio'),mask_path=mask),[Route.PURE_PROMPT],RunConfig(variants=1,out_root=tmp/'runs'),editor=b)


def plan(result):
    m=json.loads(result.manifest_path.read_text())
    return [PlannedOutput('o1','s1','clean_studio','qwen-edit',0,source_sha256=m['image_intake']['source_sha256'])]


def test_actual_final_charge_and_hash_joined_not_catalog(monkeypatch,tmp_path):
    http(monkeypatch);b=backend(tmp_path);r=pipeline(tmp_path,b);v=json.loads(r.manifest_path.read_text())['variants'][0];e=v['provider_evidence']
    assert e['credits_charged']==50 and e['submission_credits_observed']==150 and e['project_id']=='p1'
    assert e['pipeline_completed'] and e['output_sha256'] and not e['mock']
    assert v['credits_estimated']==0 and 'not_actual_charge'in v['credits_estimated_semantics']
    attempts,refs=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})]);assert attempts[0].credits_charged==50 and attempts[0].status=='success'
    assert refs[0]['project_id']=='p1'


def test_failure_cannot_reuse_previous_call_receipt(monkeypatch,tmp_path):
    http(monkeypatch);b=backend(tmp_path);b.edit(Image.new('RGB',(10,10)),'','',0,Aspect.SQUARE);prior=b.attempt_evidence()['attempt_id']
    monkeypatch.delenv('MAGIC_HOUR_API_KEY');r=pipeline(tmp_path,b);e=json.loads(r.manifest_path.read_text())['variants'][0]['provider_evidence']
    assert e['attempt_id']!=prior and e['project_id'] is None and e['credits_charged'] is None
    attempts,_=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})]);assert attempts[0].status=='unknown' and attempts[0].credits_charged is None


@pytest.mark.parametrize('mode',['provider_error','download_error','missing_final_charge','pending_timeout'])
def test_failed_unknown_and_partial_charges_remain_recorded(monkeypatch,tmp_path,mode):
    detail={'status':'complete','credits_charged':50,'downloads':[{'url':'https://download.invalid/a'}]}
    if mode=='provider_error':detail={'status':'error','credits_charged':25}
    if mode=='missing_final_charge':detail.pop('credits_charged')
    if mode=='pending_timeout':detail={'status':'rendering','credits_charged':75}
    http(monkeypatch,detail=detail,fail_download=mode=='download_error');b=backend(tmp_path);b.poll_timeout=0;r=pipeline(tmp_path,b)
    e=json.loads(r.manifest_path.read_text())['variants'][0]['provider_evidence'];a,_=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})])
    if mode=='provider_error':assert e['credits_charged']==25 and a[0].status=='failed'
    elif mode=='download_error':assert e['credits_charged']==50 and a[0].status=='failed'
    elif mode=='missing_final_charge':assert e['credits_charged'] is None and a[0].status=='success'
    else:assert e['credits_charged'] is None and e['latest_credits_observed']==75 and a[0].status=='unknown'


def test_mock_zero_is_explicit_and_adapter_does_not_make_price(tmp_path):
    b=backend(tmp_path,False);r=pipeline(tmp_path,b);e=json.loads(r.manifest_path.read_text())['variants'][0]['provider_evidence']
    assert e['mock'] and e['credits_charged']==0 and e['project_id'] is None
    a,_=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})]);assert a[0].credits_charged==0
    from renderello.brief_economics import build_report
    assert build_report(plan(r),a,[],as_of='2026-11-14')['total_attempt_credit_cost'] is None


@pytest.mark.parametrize('charge',[True,-1,float('nan'),float('inf'),'50',None])
def test_bad_or_missing_charge_never_defaults_zero(tmp_path,charge):
    b=backend(tmp_path);b.last_result={'id':'p','final':{'status':'complete','credits_charged':charge}}
    assert b.attempt_evidence()['credits_charged'] is None


def test_manifest_duplicate_recovery_legacy_or_changed_artifact_refused(monkeypatch,tmp_path):
    http(monkeypatch);r=pipeline(tmp_path,backend(tmp_path));p=plan(r)
    with pytest.raises(ValueError):import_attempts(p,[(r.manifest_path,{0:'o1'}),(r.manifest_path,{0:'o1'})])
    m=json.loads(r.manifest_path.read_text());m['variants'][0]['provider_evidence']=None;r.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='legacy'):import_attempts(p,[(r.manifest_path,{0:'o1'})])


def test_cli_outputs_existing_economics_payload(tmp_path):
    r=pipeline(tmp_path,backend(tmp_path,False));input=tmp_path/'input.json';output=tmp_path/'output.json'
    input.write_text(json.dumps(dict(plan=[asdict(p) for p in plan(r)],as_of='2026-11-14',runs=[dict(manifest=str(r.manifest_path),mapping=[dict(variant_index=0,output_id='o1')])])))
    cmd=[sys.executable,'-m','renderello.cli','provider-attempts','--input',str(input),'--output',str(output)]
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
    from renderello.brief_economics import from_payload
    e=from_payload(json.loads(output.read_text()), evidence_path=output);assert e['known_charged_credits']=='0' and e['total_attempt_credit_cost'] is None
    assert subprocess.run(cmd,capture_output=True).returncode==2
    from renderello.evidence_pair import validate_pair
    validate_pair(output)
    assert subprocess.run([sys.executable,'-m','renderello.cli','brief-economics','--input',str(output),'--output',str(tmp_path/'report.json')],capture_output=True).returncode==0
    with pytest.raises(ValueError,match='evidence_path'):from_payload(json.loads(output.read_text()))


def test_qa_failure_preserves_actual_charge_and_partial_hash(monkeypatch,tmp_path):
    http(monkeypatch)
    from renderello import qa
    monkeypatch.setattr(qa,'evaluate',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('fixtureQAerror')))
    r=pipeline(tmp_path,backend(tmp_path));e=json.loads(r.manifest_path.read_text())['variants'][0]['provider_evidence']
    assert e['credits_charged']==50 and e['pipeline_artifact_exists'] and e['output_sha256'] and not e['pipeline_completed']
    a,_=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})]);assert a[0].status=='failed' and a[0].credits_charged==50 and a[0].output_sha256 is None


def test_transport_unknown_never_reuses_id_or_cost(monkeypatch,tmp_path):
    http(monkeypatch);b=backend(tmp_path);b.edit(Image.new('RGB',(10,10)),'','',0,Aspect.SQUARE)
    def bad(url,**kw):
        if url.endswith('upload-urls'):return Reply({'items':[{'upload_url':'https://upload.invalid/a','file_path':'a.png'}]})
        raise requests.Timeout('fixture transport ambiguous')
    monkeypatch.setattr(requests,'post',bad);r=pipeline(tmp_path,b);e=json.loads(r.manifest_path.read_text())['variants'][0]['provider_evidence']
    assert e['stage']=='submission_started' and e['project_id'] is None and e['credits_charged'] is None
    a,_=import_attempts(plan(r),[(r.manifest_path,{0:'o1'})]);assert a[0].status=='unknown'


def test_changed_artifact_or_mismatched_plan_refused(monkeypatch,tmp_path):
    from dataclasses import replace
    http(monkeypatch);r=pipeline(tmp_path,backend(tmp_path));p=plan(r)
    with pytest.raises(ValueError,match='mismatch'):import_attempts([replace(p[0],method='wrong')],[(r.manifest_path,{0:'o1'})])
    r.variants[0].output_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact'):import_attempts(p,[(r.manifest_path,{0:'o1'})])


def test_repeat_project_id_on_different_attempts_is_recovery_not_double_bill(monkeypatch,tmp_path):
    from dataclasses import replace
    http(monkeypatch);b=backend(tmp_path);r1=pipeline(tmp_path,b);r2=pipeline(tmp_path,b);p=[replace(plan(r1)[0],first_try=False)]
    with pytest.raises(ValueError,match='repeated provider project'):import_attempts(p,[(r1.manifest_path,{0:'o1'}),(r2.manifest_path,{0:'o1'})])


def test_all_recorded_variants_must_be_mapped(tmp_path):
    r=pipeline(tmp_path,backend(tmp_path,False));m=json.loads(r.manifest_path.read_text());m['variants'].append(dict(m['variants'][0]));r.manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='every recorded'):import_attempts(plan(r),[(r.manifest_path,{0:'o1'})])
