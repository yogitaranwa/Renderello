import json,io,zipfile,shutil,hashlib
import pytest
from PIL import Image
from renderello import reviewed_pipeline as m
from test_intent_mask import setup

ACCEPT='I inspected this exact inferred silhouette'
def fixture(tmp_path,monkeypatch):
 s,a,b,x,y=setup(tmp_path);job='prompt-0123456789ab';p=tmp_path/job;p.mkdir();shutil.copy2(s,p/'original-source.bin');prompts=json.loads((b/'reviewed-prompts.json').read_text());prompts['objects'][0].update(points=[[3,2]],labels=[1]);bundle=io.BytesIO()
 with zipfile.ZipFile(bundle,'w') as z:
  for f in a.iterdir():z.write(f,f.name)
 checkpoint=tmp_path/'model';checkpoint.write_bytes(b'stub')
 def stub(source,out,pp,ckpt):
  shutil.copytree(b,out);shutil.copy2(pp,out/'reviewed-prompts.json');man=out/'prompted-mask-manifest.json';d=json.loads(man.read_text());d['prompt_sha256']=hashlib.sha256(pp.read_bytes()).hexdigest();man.write_text(json.dumps(d))
  rgb=Image.open(source).convert('RGB');mask=Image.open(out/'estimated-alpha.png')
  for color,value in [('white',(250,250,250)),('black',(20,20,20)),('magenta',(190,0,150))]:
   bg=Image.new('RGB',rgb.size,value);bg.paste(rgb,mask=mask);bg.save(out/(color+'-review.png'))
 monkeypatch.setattr(m,'guarded_file',stub);return job,prompts,bundle.getvalue(),checkpoint

def test_infer_does_not_combine_then_exact_review(tmp_path,monkeypatch):
 args=fixture(tmp_path,monkeypatch);d=m.preview_pipeline(tmp_path,*args);root=tmp_path/d['job'];assert not (root/'combined').exists();assert d['stage']=='REVIEW_SILHOUETTE_NOT_COMBINED';assert len(d['preview_urls'])==3
 result=m.finish_pipeline(tmp_path,d['job'],d['fingerprint'],ACCEPT);assert (root/'combined/cutout.png').is_file();assert result['report']['quality']=='NOT_EVALUATED';assert (root/'reviewed-mask-result.zip').is_file()
 with pytest.raises(ValueError,match='awaiting'):m.finish_pipeline(tmp_path,d['job'],d['fingerprint'],ACCEPT)

@pytest.mark.parametrize('what',['acceptance','fingerprint','source','mask','preview','auto','prompts'])
def test_confirmation_fail_closed(tmp_path,monkeypatch,what):
 d=m.preview_pipeline(tmp_path,*fixture(tmp_path,monkeypatch));root=tmp_path/d['job'];fp=d['fingerprint'];accept=ACCEPT
 if what=='acceptance':accept='yes'
 elif what=='fingerprint':fp='wrong'
 else:
  file={'source':'original-source.bin','mask':'reviewed/estimated-alpha.png','preview':'reviewed/black-review.png','auto':'automatic/estimated-alpha.png','prompts':'reviewed-prompts.json'}[what];(root/file).write_bytes(b'changed')
 with pytest.raises(ValueError):m.finish_pipeline(tmp_path,d['job'],fp,accept)
 assert not (root/'combined').exists()

def test_box_only_rejected_before_inference(tmp_path,monkeypatch):
 job,prompts,bundle,ckpt=fixture(tmp_path,monkeypatch);prompts['objects'][0].update(points=[],labels=[])
 with pytest.raises(ValueError,match='positive'):m.preview_pipeline(tmp_path,job,prompts,bundle,ckpt)
 assert not list(tmp_path.glob('intent-*'))

@pytest.mark.parametrize('job',['../intent-0123456789ab','intent-xyz','prompt-0123456789ab'])
def test_confirm_path_rejected(tmp_path,job):
 with pytest.raises(ValueError,match='invalid'):m.finish_pipeline(tmp_path,job,'x',ACCEPT)
