import json
import pytest
from renderello.credit_ledger import CreditLedger
from renderello.supervised_mh_pilot import build_request,submit_once,QUOTES_1K

class Response:
 def __init__(self,data,status=200):self.data=data;self.status_code=status;self.text=json.dumps(data)
 def json(self):return self.data
 def raise_for_status(self):
  if self.status_code>=400:raise RuntimeError('provider HTTP failure')
class Session:
 def __init__(self):self.posts=0;self.result={'id':'project1','credits_charged':150};self.account={'id':'account','credits':5000};self.error=None;self.status=200
 def get(self,*a,**kw):return Response(self.account)
 def post(self,*a,**kw):
  self.posts+=1
  if self.error:raise self.error
  return Response(self.result,self.status)
@pytest.fixture
def setup(tmp_path):
 return Session(),CreditLedger(tmp_path/'ledger',cap=5000,authority_reference='original owner message'),tmp_path/'evidence',build_request('case1','nano-banana-pro','Full exact original prompt','api-assets/id/source.jpg')
def run(s,l,p,r):return submit_once(s,'unreal-test-credential',r,l,'job1',p,'account')
def test_one_submission(setup):
 s,l,p,r=setup;body=run(*setup);assert s.posts==1;assert body['credits_charged']==150;assert l.snapshot()['remaining']==4850
 with pytest.raises(FileExistsError):run(*setup)
 assert s.posts==1

def test_timeout_retains_reservation(setup):
 s,l,p,r=setup;s.error=TimeoutError('unknown transmission')
 with pytest.raises(TimeoutError):run(*setup)
 assert l.snapshot()['remaining']==4850
 with pytest.raises(FileExistsError):run(*setup)
 assert s.posts==1

@pytest.mark.parametrize('receipt',[{'id':'project1'},{'credits_charged':150},{'id':'project1','credits_charged':None},{'id':'project1','credits_charged':True},{'id':'project1','credits_charged':-1}])
def test_unknown_receipt_fails_closed(setup,receipt):
 s,l,p,r=setup;s.result=receipt
 with pytest.raises(ValueError):run(*setup)
 assert l.snapshot()['remaining']==4850
 assert list(l.snapshot()['jobs'].values())[0]['status']!='settled'

def test_changed_price_stops_but_records(setup):
 s,l,p,r=setup;s.result['credits_charged']=200
 with pytest.raises(ValueError,match='differs'):run(*setup)
 assert l.snapshot()['remaining']==4800

@pytest.mark.parametrize('account',[{'id':'wrong','credits':5000},{'id':'account','credits':0},{'id':'account'},{'id':'account','credits':True}])
def test_account_before_post(setup,account):
 s,l,p,r=setup;s.account=account
 with pytest.raises(ValueError):run(*setup)
 assert s.posts==0

@pytest.mark.parametrize('change',[{'resolution':'4k'},{'aspect_ratio':'16:9'},{'image_count':4},{'image_count':True},{'model':'unknown'},{'extra':'field'}])
def test_unquoted_config(setup,change):
 s,l,p,r=setup;r.update(change)
 with pytest.raises(ValueError):run(*setup)
 assert s.posts==0

def test_400_never_assumes_zero(setup):
 s,l,p,r=setup;s.status=400
 with pytest.raises(RuntimeError):run(*setup)
 assert l.snapshot()['remaining']==4850
 assert s.posts==1

@pytest.mark.parametrize('model',QUOTES_1K)
def test_prompt_not_truncated(model):
 prompt='Exact visible fine print. '*500
 r=build_request('case1',model,prompt,'api-assets/id/x.jpg');assert r['style']['prompt']==prompt

@pytest.mark.parametrize('asset',['https://unexpected.test/a.jpg','api-assets/../x',''])
def test_unsupported_asset(asset):
 with pytest.raises(ValueError):build_request('case1','nano-banana-pro','prompt',asset)
