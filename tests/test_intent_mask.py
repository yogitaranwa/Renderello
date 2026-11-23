import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello import intent_mask as m
from renderello.prompted_mask import MODEL_SHA256

def setup(tmp):
 s=tmp/'s.png';rgb=np.full((5,7,3),[11,22,33],np.uint8);Image.fromarray(rgb).save(s);h=m.sha(s)
 a=np.full((5,7),127,np.uint8);b=np.zeros((5,7),np.uint8);b[1:4,2:5]=255
 folders=[]
 for n,alpha in [('auto',a),('review',b)]:
  p=tmp/n;p.mkdir();Image.fromarray(rgb).save(p/'source.png');Image.fromarray(alpha).save(p/'estimated-alpha.png');Image.fromarray(np.dstack([rgb,alpha])).save(p/'cutout.png')
  d={'source_sha256':h,'model':'stub','output_sha256':{f.name:m.sha(f) for f in p.iterdir()}}
  if n=='review':
   pp=p/'reviewed-prompts.json';pp.write_text(json.dumps(dict(source_sha256=h,image_size=[7,5],origin='reviewed_source_geometry',objects=[dict(name='one',box=[0,0,7,5],points=[],labels=[])])));d.update(prompt_sha256=m.sha(pp),prompt_origin='reviewed_source_geometry',objects=['one'],model_sha256=MODEL_SHA256)
  (p/('bilateral-mask-manifest.json' if n=='auto' else 'prompted-mask-manifest.json')).write_text(json.dumps(d));folders.append(p)
 return s,*folders,a,b

def test_exact_intersection_rgb_and_provenance(tmp_path):
 s,a,b,x,y=setup(tmp_path);out=tmp_path/'out';d=m.gate_file(s,a,b,out);rgba=np.array(Image.open(out/'cutout.png'))
 assert np.array_equal(rgba[:,:,:3],np.array(Image.open(s)));assert np.array_equal(rgba[:,:,3],np.minimum(x,y));assert d['removed_nonzero_pixels']==26;assert d['quality']=='NOT_EVALUATED';assert d['objects']==['one'];assert d['retained_alpha_values_exact']
 for n,h in d['output_sha256'].items():assert m.sha(out/n)==h

def test_existing_preserved(tmp_path):
 s,a,b,_,_=setup(tmp_path);o=tmp_path/'out';o.mkdir();(o/'keep').write_text('old')
 with pytest.raises(ValueError,match='exists'):m.gate_file(s,a,b,o)
 assert (o/'keep').read_text()=='old'

@pytest.mark.parametrize('what',['source','output','prompt','origin','binary','empty','model','cutout-alpha'])
def test_fail_closed(tmp_path,what):
 s,a,b,x,y=setup(tmp_path);manifest=a/'bilateral-mask-manifest.json';rmanifest=b/'prompted-mask-manifest.json';d=json.loads(manifest.read_text());e=json.loads(rmanifest.read_text())
 if what=='source':d['source_sha256']='wrong'
 if what=='output':(a/'estimated-alpha.png').write_bytes(b'bad')
 if what in ['prompt','origin']:
  p=b/'reviewed-prompts.json';j=json.loads(p.read_text());j['origin']='automatic';p.write_text(json.dumps(j))
  if what=='origin':e['prompt_sha256']=m.sha(p)
 if what in ['binary','empty']:
  v=np.ones((5,7),np.uint8) if what=='binary' else np.zeros((5,7),np.uint8);Image.fromarray(v).save(b/'estimated-alpha.png');e['output_sha256']['estimated-alpha.png']=m.sha(b/'estimated-alpha.png')
 if what=='model':e['model_sha256']='wrong'
 if what=='cutout-alpha':
  q=a/'cutout.png';v=np.array(Image.open(q));v[:,:,3]=255;Image.fromarray(v).save(q);d['output_sha256']['cutout.png']=m.sha(q)
 manifest.write_text(json.dumps(d));rmanifest.write_text(json.dumps(e))
 with pytest.raises(ValueError):m.gate_file(s,a,b,tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_ambiguous_manifest(tmp_path):
 s,a,b,x,y=setup(tmp_path);(a/'pyramid-mask-manifest.json').write_text('{}')
 with pytest.raises(ValueError,match='exactly one'):m.gate_file(s,a,b,tmp_path/'out')

def test_missing_automatic_pixels_not_recovered(tmp_path):
 s,a,b,x,y=setup(tmp_path);x[2,3]=0;Image.fromarray(x).save(a/'estimated-alpha.png');rgb=np.array(Image.open(s));Image.fromarray(np.dstack([rgb,x])).save(a/'cutout.png');p=a/'bilateral-mask-manifest.json';d=json.loads(p.read_text());d['output_sha256']={f:m.sha(a/f) for f in d['output_sha256']};p.write_text(json.dumps(d));m.gate_file(s,a,b,tmp_path/'out');assert np.array(Image.open(tmp_path/'out/estimated-alpha.png'))[2,3]==0
