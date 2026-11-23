import io,zipfile,json,threading,urllib.request,urllib.error
from pathlib import Path
import pytest
from renderello.reviewed_pipeline import unpack_automatic,run_pipeline
from renderello.webui import _Handler
from http.server import ThreadingHTTPServer

def archive(names):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for n in names:z.writestr(n,b'{}')
 return b.getvalue()

@pytest.mark.parametrize('names',[
 ['../source.png'],['folder/source.png'],['source.png','source.png'],['.hidden'],['source.png','cutout.png','estimated-alpha.png'],
 ['source.png','cutout.png','estimated-alpha.png','bilateral-mask-manifest.json','pyramid-mask-manifest.json'],list(map(str,range(21)))])
def test_zip_rejects_unsafe_or_invalid(tmp_path,names):
 with pytest.raises(ValueError):unpack_automatic(archive(names),tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_zip_only_expected_files(tmp_path):
 names=['source.png','cutout.png','estimated-alpha.png','bilateral-mask-manifest.json','other.txt'];unpack_automatic(archive(names),tmp_path/'out');assert not (tmp_path/'out/other.txt').exists();assert len(list((tmp_path/'out').iterdir()))==4

def test_checkpoint_missing_no_work(tmp_path):
 with pytest.raises(ValueError,match='checkpoint'):run_pipeline(tmp_path,'prompt-0123456789ab',{},b'',None)
 assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('path',['/reviewed-run','/reviewed-confirm'])
def test_status_and_cross_origin_block(tmp_path,path):
 h=type('Bound',(_Handler,),dict(out_root=tmp_path,reviewed_checkpoint=None,reviewed_lock=threading.Lock()));s=ThreadingHTTPServer(('127.0.0.1',0),h);threading.Thread(target=s.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(s.server_port)
 try:
  assert json.load(urllib.request.urlopen(base+'/reviewed-status'))=={'configured':False}
  req=urllib.request.Request(base+path,data=b'',headers={'Origin':'http://outside.invalid'})
  with pytest.raises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
  assert e.value.code==400
 finally:s.shutdown();s.server_close()

def test_oversized_zip(tmp_path):
 with pytest.raises(ValueError,match='cap'):unpack_automatic(b'x'*20_000_001,tmp_path/'out')
