import io,json,time
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from renderello import colab_demo as d

class Response:
    def __init__(self, data=None, content=None):
        self.data=data; self.content=content; self.status_code=200
        self.text=json.dumps(data)
    def json(self): return self.data
    def raise_for_status(self): pass

class Session:
    def __init__(self): self.posts=[]; self.gets=[]; self.puts=[];self.fail_submit=False
    def get(self,url,**kw):
        self.gets.append(url)
        if url.endswith('/account'): return Response({'id':'account-one','credits':1000})
        if '/image-projects/' in url:
            return Response({'id':url.split('/')[-1],'status':'complete','credits_charged':50,'downloads':[{'url':'https://images.example/output.png'}]})
        buf=io.BytesIO();Image.new('RGB',(32,32),'red').save(buf,'PNG');return Response(content=buf.getvalue())
    def post(self,url,**kw):
        self.posts.append((url,kw))
        if url.endswith('upload-urls'): return Response({'items':[{'upload_url':'https://upload.example/asset','file_path':'api-assets/test/source.png'}]})
        if self.fail_submit: raise TimeoutError('ambiguous request')
        return Response({'id':'project-'+str(len(self.posts)),'credits_charged':50})
    def put(self,url,**kw): self.puts.append(url);return Response({})

@pytest.fixture
def source(tmp_path):
    p=tmp_path/'photo.jpg';Image.new('RGB',(64,64),'white').save(p);return p

def options(source,tmp_path,session):
    return dict(source=source,root=tmp_path/'batch',session=session,key='fake-runtime-key',
        expected_account='account-one',cap=100,authority_reference='owner confirmed this test',
        source_upload_approved=True,spending_approved=True,quote_confirmed=True)

@pytest.mark.parametrize('field',['source_upload_approved','spending_approved','quote_confirmed'])
def test_no_effect_without_scope(field,source,tmp_path):
    s=Session();kw=options(source,tmp_path,s);kw[field]=False
    with pytest.raises(ValueError):d.submit_live(**kw)
    assert not s.gets and not s.posts and not s.puts

def test_batch_and_recovery(source,tmp_path):
    s=Session();kw=options(source,tmp_path,s);result=d.submit_live(**kw)
    assert result['ledger']['committed_or_reserved']==100
    assert len(s.posts)==3 and len(s.puts)==1
    assert all(p[1]['json']['image_count']==1 for p in s.posts[1:])
    assert d.recover_live(kw['root'],session=s,key='fake-runtime-key')[0]['output']
    with pytest.raises(ValueError,match='hourly'):d.recover_live(kw['root'],session=s,key='fake-runtime-key')
    before=len(s.posts)
    with pytest.raises(FileExistsError):d.submit_live(**kw)
    assert len(s.posts)==before

def test_wrong_account_never_uploads(source,tmp_path):
    s=Session();kw=options(source,tmp_path,s);kw['expected_account']='different'
    with pytest.raises(ValueError,match='Account'):d.submit_live(**kw)
    assert not s.posts and not s.puts

def test_cap_blocks_before_network(source,tmp_path):
    s=Session();kw=options(source,tmp_path,s);kw['cap']=99
    with pytest.raises(ValueError):d.submit_live(**kw)
    assert not s.gets and not s.posts

def test_unknown_submission_reserves_and_stops(source,tmp_path):
    s=Session();s.fail_submit=True;kw=options(source,tmp_path,s)
    with pytest.raises(TimeoutError):d.submit_live(**kw)
    ledger=d.CreditLedger(kw['root']/'credit-ledger').snapshot()
    assert ledger['committed_or_reserved']==50 and len(s.posts)==2
    assert not (kw['root']/'jobs'/'alternative-2').exists()
    with pytest.raises(ValueError,match='unknown'):d.recover_live(kw['root'],session=s,key='fake-runtime-key')

@pytest.mark.parametrize('n',[0,1,5,True])
def test_invalid_alternatives(n):
    with pytest.raises(ValueError):d.validate_choices('clean_studio','',n)

def test_transparent_preview_refuses_before_cutout(source,tmp_path):
    with pytest.raises(ValueError,match='transparency'):
        d.preview(source,tmp_path/'free',material='transparent')
    assert not (tmp_path/'free').exists()

def test_free_actual_two_outputs(source,tmp_path):
    # Nontrivial silhouette; no network, CUDA or external model needed.
    im=Image.new('RGB',(160,180),'white')
    from PIL import ImageDraw
    ImageDraw.Draw(im).rectangle((45,30,115,160),fill='navy')
    im.save(source)
    result=d.preview(source,tmp_path/'free',material='opaque')
    assert len(result['outputs'])==2 and Path(result['gallery']).is_file()
