import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello import lucida_mask as lm

def source(tmp_path):
 p=tmp_path/'input.png';Image.fromarray(np.arange(9*12*3,dtype=np.uint8).reshape(9,12,3)).save(p);return p

def test_native_source_rgb_exact_and_no_quality_claim(tmp_path):
 p=source(tmp_path);alpha=np.full((9,12),135,dtype=np.uint8);alpha[1:4,2:6]=255
 r=lm.export_review(p,alpha,tmp_path/'out')
 a=np.array(Image.open(tmp_path/'out/cutout.png'))
 assert np.array_equal(a[...,:3],np.array(Image.open(p)));assert np.array_equal(a[...,3],alpha)
 assert r['quality']=='NOT_EVALUATED';assert not r['refraction_estimated'];assert not r['background_color_removed'];assert r['semantic_completeness']=='NOT_VERIFIED'
 assert len(r['output_sha256'])==6
 for name,sha in r['output_sha256'].items():assert lm.digest(tmp_path/'out'/name)==sha
 assert json.loads((tmp_path/'out/lucida-mask-manifest.json').read_text())==r

def test_existing_output_preserved(tmp_path):
 p=source(tmp_path);out=tmp_path/'out';out.mkdir();(out/'old').write_text('KEEP')
 with pytest.raises(ValueError,match='exists'):lm.export_review(p,np.full((9,12),1,dtype=np.uint8),out)
 assert (out/'old').read_text()=='KEEP'

@pytest.mark.parametrize('alpha',[np.zeros((9,12),dtype=np.uint8),np.ones((12,9),dtype=np.uint8),np.ones((9,12),dtype=float),np.ones((9,12,1),dtype=np.uint8)])
def test_invalid_alpha_leaves_no_output(tmp_path,alpha):
 with pytest.raises(ValueError):lm.export_review(source(tmp_path),alpha,tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_size_cap(tmp_path):
 p=tmp_path/'large.png';Image.new('RGB',(2001,2000)).save(p)
 with pytest.raises(ValueError,match='four million'):lm.export_review(p,np.ones((1,1),dtype=np.uint8),tmp_path/'out')

def test_missing_model_is_not_downloaded(tmp_path):
 with pytest.raises(ValueError,match='missing'):lm.verify_model(tmp_path)
 assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('kind',['badsize','badhash','symlink'])
def test_publisher_checks(tmp_path,monkeypatch,kind):
 monkeypatch.setattr(lm,'FILES',{'model':(hashlib.sha256(b'good').hexdigest(),4)})
 p=tmp_path/'model'
 if kind=='symlink':
  (tmp_path/'target').write_bytes(b'good')
  try:p.symlink_to(tmp_path/'target')
  except OSError as e:
   if getattr(e,'winerror',None)==1314:pytest.skip('Windows account lacks symlink creation privilege; symlink rejection not exercised')
   raise
 else:p.write_bytes(b'bad' if kind=='badsize' else b'evil')
 with pytest.raises(ValueError):lm.verify_model(tmp_path)

def test_publisher_checks_accept_exact(tmp_path,monkeypatch):
 monkeypatch.setattr(lm,'FILES',{'model':(hashlib.sha256(b'good').hexdigest(),4)})
 (tmp_path/'model').write_bytes(b'good');assert lm.verify_model(tmp_path)==tmp_path

def test_guarded_timeout(tmp_path,monkeypatch):
 import subprocess
 def fail(*args,**kw):assert kw['timeout']==180;raise subprocess.TimeoutExpired(args[0],180)
 monkeypatch.setattr(subprocess,'run',fail)
 with pytest.raises(RuntimeError,match='180 second'):lm.guarded_file('a','b','c')

def test_worker_failure(tmp_path,monkeypatch):
 import subprocess,types
 monkeypatch.setattr(subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=1,stderr='FAILED'))
 with pytest.raises(RuntimeError,match='FAILED'):lm.guarded_file('a','b','c')

def test_opaque_export_is_exact(tmp_path):
 p=source(tmp_path);lm.export_review(p,np.full((9,12),255,dtype=np.uint8),tmp_path/'o')
 for name in ['white','black','magenta']:assert np.array_equal(np.array(Image.open(tmp_path/'o'/(name+'-review.png'))),np.array(Image.open(p)))

def test_existing_output_prevents_model_import(tmp_path):
 p=source(tmp_path);(tmp_path/'o').mkdir()
 with pytest.raises(ValueError,match='exists'):lm.infer_file(p,tmp_path/'o',tmp_path/'missing')
