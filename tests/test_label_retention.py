import json
import numpy as np
from PIL import Image
import pytest
from renderello import label_retention as m
from renderello.intent_mask import sha

def setup(tmp_path):
 s=tmp_path/'source.png';Image.new('RGB',(10,8),(11,22,33)).save(s);c=tmp_path/'cutout.png';Image.new('RGBA',(10,8),(11,22,33,255)).save(c);r=dict(source_sha256=sha(s),image_size=[10,8],regions=[dict(name='label',box=[2,2,8,6])]);return s,c,r

def test_exact_is_not_transcription(tmp_path):
 s,c,r=setup(tmp_path);d=m.check_regions(s,c,r);assert d['all_retention_checks_passed'];assert d['transcription']=='NOT_PERFORMED';assert d['quality']=='NOT_EVALUATED'

def test_rgb_and_alpha_independently_fail(tmp_path):
 s,c,r=setup(tmp_path);v=np.array(Image.open(c));v[2,2,:3]=0;v[3,3,3]=0;Image.fromarray(v).save(c);d=m.check_regions(s,c,r);x=d['regions'][0];assert x['rgb_changed_pixels']==1;assert x['below_required_alpha_pixels']==1;assert not d['all_retention_checks_passed']

@pytest.mark.parametrize('what',['source','size','empty','outside','float','reversed','duplicate','threshold','rgba','resized'])
def test_invalid_regions(tmp_path,what):
 s,c,r=setup(tmp_path)
 if what=='source':r['source_sha256']='wrong'
 if what=='size':r['image_size']=[8,10]
 if what=='empty':r['regions']=[]
 if what=='outside':r['regions'][0]['box']=[0,0,11,7]
 if what=='float':r['regions'][0]['box']=[0.,0,8,7]
 if what=='reversed':r['regions'][0]['box']=[8,0,2,7]
 if what=='duplicate':r['regions']*=2
 if what=='threshold':r['regions'][0]['minimum_alpha']=256
 if what=='rgba':Image.new('RGB',(10,8)).save(c)
 if what=='resized':Image.new('RGBA',(8,10)).save(c)
 with pytest.raises(ValueError):m.check_regions(s,c,r)

def test_preserve_prior_reports(tmp_path):
 s,c,r=setup(tmp_path);rp=tmp_path/'regions.json';rp.write_text(json.dumps(r));out=tmp_path/'report.json';out.write_text('old')
 with pytest.raises(ValueError):m.check_file(s,c,rp,out)
 assert out.read_text()=='old'
