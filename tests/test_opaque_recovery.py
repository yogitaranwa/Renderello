import json
import pytest
from PIL import Image
from renderello import opaque_recovery as m
from test_intent_mask import setup
from renderello.intent_mask import sha

def ready(tmp_path):
 s,a,r,x,y=setup(tmp_path);p=r/'reviewed-prompts.json';d=json.loads(p.read_text());d['objects'][0].update(points=[[3,2]],labels=[1]);p.write_text(json.dumps(d));mp=r/'prompted-mask-manifest.json';d=json.loads(mp.read_text());d['prompt_sha256']=sha(p);mp.write_text(json.dumps(d));return s,r,sha(r/'estimated-alpha.png')

def test_wrapper_routes_reviewed_support_without_edge_rgb(tmp_path,monkeypatch):
 s,r,h=ready(tmp_path);calls=[]
 def stub(source,alpha,out,**kw):
  calls.append((alpha,kw));out.mkdir();return dict(warnings=[],source_rgb_unchanged=True)
 monkeypatch.setattr(m,'guarded_file',stub);out=tmp_path/'out';d=m.recover_file(s,r,out,reviewed_alpha_sha256=h,material='opaque',framing='complete',confirm_reviewed_silhouette=True)
 assert calls==[(r/'estimated-alpha.png',dict(method='cf',radius=3,edge_colors=False))];assert d['quality']=='NOT_EVALUATED';assert d['fine_print']=='NOT_INDEPENDENTLY_VERIFIED';assert (out/'opaque-recovery-manifest.json').exists()

@pytest.mark.parametrize('change',[{'material':'transparent'},{'material':'unknown'},{'framing':'cropped'},{'framing':'unknown'},{'confirm_reviewed_silhouette':False},{'reviewed_alpha_sha256':'wrong'},{'method':'guided'}])
def test_scope_rejected_before_work(tmp_path,change):
 s,r,h=ready(tmp_path);kw=dict(reviewed_alpha_sha256=h,material='opaque',framing='complete',confirm_reviewed_silhouette=True);kw.update(change)
 with pytest.raises(ValueError):m.recover_file(s,r,tmp_path/'out',**kw)
 assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('what',['source','cutout','model','prompt','points','binary','origin'])
def test_review_provenance_fail_closed(tmp_path,what):
 s,r,h=ready(tmp_path);mp=r/'prompted-mask-manifest.json';d=json.loads(mp.read_text());pp=r/'reviewed-prompts.json';p=json.loads(pp.read_text())
 if what=='source':d['source_sha256']='wrong'
 if what=='model':d['model_sha256']='wrong'
 if what=='prompt':pp.write_text('{}')
 if what=='points':p['objects'][0].update(points=[],labels=[]);pp.write_text(json.dumps(p));d['prompt_sha256']=sha(pp)
 if what=='origin':d['prompt_origin']='automatic'
 if what=='cutout':Image.new('RGBA',(7,5),(1,2,3,255)).save(r/'cutout.png');d['output_sha256']['cutout.png']=sha(r/'cutout.png')
 if what=='binary':Image.new('L',(7,5),127).save(r/'estimated-alpha.png');d['output_sha256']['estimated-alpha.png']=sha(r/'estimated-alpha.png');h=sha(r/'estimated-alpha.png')
 mp.write_text(json.dumps(d))
 with pytest.raises(ValueError):m.verify_reviewed(s,r,h)

def test_existing_output_preserved(tmp_path):
 s,r,h=ready(tmp_path);out=tmp_path/'out';out.mkdir();(out/'keep').write_text('old')
 with pytest.raises(ValueError,match='exists'):m.recover_file(s,r,out,reviewed_alpha_sha256=h,material='opaque',framing='complete',confirm_reviewed_silhouette=True)
 assert (out/'keep').read_text()=='old'
