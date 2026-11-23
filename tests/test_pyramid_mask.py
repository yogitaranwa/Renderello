import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello import pyramid_mask as m

def case(tmp_path,monkeypatch):
 p=tmp_path/'source.png';Image.new('RGB',(30,20),(11,22,33)).save(p);c=tmp_path/'model.pth';c.write_bytes(b'test stub, not real weights');monkeypatch.setattr(m,'MODEL_SHA256',hashlib.sha256(c.read_bytes()).hexdigest());return p,c

def test_checkpoint_missing_or_wrong(tmp_path):
 s=tmp_path/'s.png';Image.new('RGB',(3,3)).save(s)
 with pytest.raises(ValueError,match='missing'):m.infer_file(s,tmp_path/'out',tmp_path/'absent')
 c=tmp_path/'c';c.write_bytes(b'wrong')
 with pytest.raises(ValueError,match='SHA256'):m.infer_file(s,tmp_path/'out',c)
 assert not (tmp_path/'out').exists()

def test_existing_output_preserved(tmp_path,monkeypatch):
 p,c=case(tmp_path,monkeypatch);o=tmp_path/'out';o.mkdir();(o/'old').write_text('keep')
 with pytest.raises(ValueError,match='exists'):m.infer_file(p,o,c)
 assert (o/'old').read_text()=='keep'

def test_pixel_cap_and_small(tmp_path):
 p=tmp_path/'s.png';Image.new('RGB',(2001,2000)).save(p)
 with pytest.raises(ValueError,match='million'):m.infer_file(p,tmp_path/'out',tmp_path/'c')
 Image.new('RGB',(2,2)).save(p)
 with pytest.raises(ValueError,match='small'):m.infer_file(p,tmp_path/'out',tmp_path/'c')

@pytest.mark.parametrize('alpha',[np.zeros((20,30),dtype=np.uint8),np.ones((20,29),dtype=np.uint8),np.ones((20,30),dtype=float)])
def test_invalid_alpha_no_output(tmp_path,monkeypatch,alpha):
 p,c=case(tmp_path,monkeypatch);monkeypatch.setattr(m,'_predict',lambda *a:alpha)
 with pytest.raises(ValueError,match='alpha'):m.infer_file(p,tmp_path/'out',c)
 assert not (tmp_path/'out').exists()

def test_rgb_exact_provenance_and_outputs_with_stub(tmp_path,monkeypatch):
 p,c=case(tmp_path,monkeypatch);alpha=np.full((20,30),125,dtype=np.uint8);monkeypatch.setattr(m,'_predict',lambda *a:alpha)
 d=m.infer_file(p,tmp_path/'out',c);out=tmp_path/'out';cut=np.array(Image.open(out/'cutout.png'))
 assert np.array_equal(cut[:,:,:3],np.array(Image.open(p)));assert np.array_equal(cut[:,:,3],alpha)
 assert d['quality']=='NOT_EVALUATED';assert d['mode']=='fast';assert d['source_rgb_unchanged'];assert 'not optical' in d['alpha_kind']
 for name,h in d['output_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
 assert len(d['output_sha256'])==6
