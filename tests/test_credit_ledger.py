import json
import pytest
from renderello.credit_ledger import CreditLedger

@pytest.fixture
def ledger(tmp_path):return CreditLedger(tmp_path/'ledger',cap=5000,authority_reference='authenticated-owner-message-reference')

def reserve(l,job='job1',quote=150,**kw):
 d=dict(cost_source='observed current quote URL',account_id='known-project-account',request_sha256='a'*64);d.update(kw);return l.reserve(job,quote,**d)

def test_budget_holds_unknown(ledger):
 r=reserve(ledger);assert r['remaining']==4850
 ledger.submitted('job1','provider-id')
 with pytest.raises(ValueError,match='unsettled'):reserve(ledger,'job2')
 assert ledger.snapshot()['remaining']==4850

def test_settlement(ledger):
 reserve(ledger);ledger.submitted('job1','id');r=ledger.settle('job1',100,receipt_reference='actual-details.json')
 assert r['remaining']==4900
 reserve(ledger,'job2',4900);assert ledger.snapshot()['remaining']==0

def test_cap_stop(ledger):
 with pytest.raises(ValueError,match='exceeded'):reserve(ledger,quote=5001)
 assert ledger.snapshot()['remaining']==5000

def test_exact_cap(ledger):assert reserve(ledger,quote=5000)['remaining']==0

def test_never_repeat_job(ledger):
 reserve(ledger);ledger.settle('job1',0,receipt_reference='verified no submission')
 with pytest.raises(ValueError,match='recover'):reserve(ledger)

def test_overcharge_recorded_stops(ledger):
 reserve(ledger);r=ledger.settle('job1',5001,receipt_reference='actual overcharge receipt')
 assert r['over_cap'] is True
 with pytest.raises(ValueError):reserve(ledger,'job2',1)

def test_no_implicit_zero(ledger):
 reserve(ledger)
 with pytest.raises(ValueError):ledger.settle('job1',None,receipt_reference='unknown')
 assert ledger.snapshot()['remaining']==4850

@pytest.mark.parametrize('n',[True,-1,float('nan'),float('inf'),'50'])
def test_invalid_quote(ledger,n):
 with pytest.raises(ValueError):reserve(ledger,quote=n)

@pytest.mark.parametrize('change',[{'cost_source':''},{'account_id':''},{'request_sha256':'x'},{'request_sha256':'Z'*64}])
def test_quote_evidence(ledger,change):
 with pytest.raises(ValueError):reserve(ledger,**change)

def test_account_cannot_change(ledger):
 reserve(ledger);ledger.settle('job1',10,receipt_reference='r')
 with pytest.raises(ValueError,match='account'):reserve(ledger,'job2',account_id='other')

def test_no_silent_cap_change(ledger):
 with pytest.raises(ValueError):CreditLedger(ledger.root,cap=9000)

def test_tampering(ledger):
 reserve(ledger);p=ledger.root/'event-000001.json';d=json.loads(p.read_text());d['quote']=0;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='hash chain'):ledger.snapshot()

def test_missing_event(ledger):
 reserve(ledger);ledger.submitted('job1','id');(ledger.root/'event-000001.json').unlink()
 with pytest.raises(ValueError,match='missing'):ledger.snapshot()

def test_write_lock(ledger):
 (ledger.root/'.writer.lock').write_text('lock')
 with pytest.raises(FileExistsError):reserve(ledger)

def test_invalid_transition(ledger):
 with pytest.raises(ValueError):ledger.submitted('missing','id')
 reserve(ledger);ledger.settle('job1',10,receipt_reference='r')
 with pytest.raises(ValueError):ledger.submitted('job1','id')
 with pytest.raises(ValueError):ledger.settle('job1',0,receipt_reference='r')

def test_small_overcharge_blocks_later_jobs(ledger):
 reserve(ledger);ledger.settle('job1',151,receipt_reference='unexpected charge')
 with pytest.raises(ValueError,match='overcharge'):reserve(ledger,'job2',1)
