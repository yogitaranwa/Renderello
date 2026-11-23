from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import pytest
import renderello.reviewed_alpha_lock_cli as cli

@pytest.fixture
def data(tmp_path,monkeypatch):
 rgb=np.full((48,48,3),120,np.uint8);a=np.zeros((48,48),np.uint8);a[10:38,12:36]=255;a[38,12:36]=150
 f=np.zeros(a.shape,np.uint8);f[30:45]=255
 for name,v in [('source',rgb),('alpha',a),('region',f)]:Image.fromarray(v).save(tmp_path/(name+'.png'))
 kw={n:tmp_path/(n+'.png') for n in ['source','alpha','region']};kw.update({n+'_file_sha256':hashlib.sha256(kw[n].read_bytes()).hexdigest() for n in list(kw)})
 kw.update(out=tmp_path/'out',reviewed=True,material='opaque',review_note='Synthetic prior-alpha lock, not image truth.',radius=2)
 def worker(source,alpha,out,trimap_path,**kw):
  out.mkdir();tri=np.array(Image.open(trimap_path));solved=np.full(a.shape,100,np.uint8);solved[tri==0]=0;solved[tri==255]=255
  Image.fromarray(rgb).save(out/'foreground.png');Image.fromarray(solved).save(out/'refined-alpha.png');return {'synthetic_fixture':True}
 monkeypatch.setattr(cli,'guarded_file',worker);return kw

def test_saved_cli_bundle_and_fractional_lock(data):
 m=cli.execute(**data);D=data['out'];a=np.array(Image.open(data['alpha']));out=np.array(Image.open(D/'locked-alpha.png'))
 assert np.array_equal(out[30:45],a[30:45]) and m['all_source_rgb_exact'] and not m['accepted']
 assert m['worker']['synthetic_fixture'] and m['status']=='EXECUTED_NOT_ACCEPTED'
 for name,h in m['output_sha256'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h
 assert json.loads((D/'manifest.json').read_text())==m
 assert json.loads((D/'part-preservation.json').read_text())['all_named_parts_preserved']
 with pytest.raises(ValueError,match='preserve'):cli.execute(**data)

@pytest.mark.parametrize('kw',[{'source_file_sha256':'x'},{'alpha_file_sha256':'x'},{'region_file_sha256':'x'},
 {'reviewed':False},{'material':'glass'},{'radius':0},{'review_note':''}])
def test_bad_input_no_output(data,kw):
 data.update(kw)
 with pytest.raises(ValueError):cli.execute(**data)
 assert not data['out'].exists()

@pytest.mark.parametrize('kind',['source_mode','alpha_mode','region_mode','region_partial','region_shape','region_empty','missing'])
def test_bad_files_before_reserve(data,kind):
 target='source' if kind=='source_mode' else ('alpha' if kind=='alpha_mode' else 'region')
 if kind=='missing':data['source']=data['source'].with_name('missing.png')
 else:
  im=Image.open(data[target]);arr=np.array(im)
  if kind.endswith('mode'):im.convert('RGB' if im.mode=='L' else 'RGBA').save(data[target])
  elif kind=='region_partial':arr[0,0]=128;Image.fromarray(arr).save(data[target])
  elif kind=='region_shape':Image.fromarray(arr[:-1]).save(data[target])
  elif kind=='region_empty':Image.fromarray(np.zeros(arr.shape,np.uint8)).save(data[target])
  data[target+'_file_sha256']=hashlib.sha256(data[target].read_bytes()).hexdigest()
 with pytest.raises(ValueError):cli.execute(**data)
 assert not data['out'].exists()

def test_guard_refusal_retained(data,monkeypatch):
 def refused(*args,**kw):raise RuntimeError('synthetic guard refusal')
 monkeypatch.setattr(cli,'guarded_file',refused)
 with pytest.raises(RuntimeError):cli.execute(**data)
 assert json.loads((data['out']/'refused.json').read_text())['accepted'] is False
 assert not (data['out']/'manifest.json').exists()
