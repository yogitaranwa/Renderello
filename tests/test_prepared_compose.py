from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw
import pytest
from renderello.reviewed_source import prepare_reviewed_source
from renderello.prepared_compose import load_prepared,compose_prepared
from renderello.image_intake import IntakeError


def fixture(tmp_path,orientation=6):
 source=Image.new('RGB',(80,60),'white');ImageDraw.Draw(source).rectangle((20,10,40,50),fill=(20,70,120))
 mask=Image.new('L',(80,60),0);ImageDraw.Draw(mask).rectangle((20,10,40,50),fill=255)
 exif=Image.Exif();exif[274]=orientation;source.save(tmp_path/'source.png',exif=exif);mask.save(tmp_path/'mask.png')
 out=tmp_path/'prepared';prepare_reviewed_source(tmp_path/'source.png',out,mask=tmp_path/'mask.png',mask_coordinates='encoded');return out

@pytest.mark.parametrize('orientation',range(1,9))
def test_all_orientation_preparations_are_loadable(tmp_path,orientation):
 out=fixture(tmp_path,orientation);before={p.name:p.read_bytes() for p in out.iterdir()};record,files=load_prepared(out)
 assert len(files)==2 and record['semantic_mask_accuracy']=='NOT_VERIFIED'
 assert before=={p.name:p.read_bytes() for p in out.iterdir()}

@pytest.mark.parametrize('name',['normalized-source.png','reviewed-alpha.png'])
def test_changed_prepared_bytes_refused_before_output(tmp_path,name):
 out=fixture(tmp_path);(out/name).write_bytes(b'changed');dest=tmp_path/'runs'
 with pytest.raises(IntakeError,match='changed'):compose_prepared(out,dest)
 assert not dest.exists()

@pytest.mark.parametrize('value',['[]','{}','{"intake":null}','{"intake":{"output_size":[true,60]}}','{"x":1,"x":2}','{"x":NaN}'])
def test_invalid_manifest_refused(tmp_path,value):
 out=fixture(tmp_path);(out/'prepared-source.json').write_text(value)
 with pytest.raises(IntakeError):load_prepared(out)

def test_incomplete_preparation_refused(tmp_path):
 out=fixture(tmp_path);(out/'PREPARATION_INCOMPLETE').touch()
 with pytest.raises(IntakeError,match='Incomplete'):load_prepared(out)

@pytest.mark.parametrize('change',['wrong-mode','wrong-size','hidden-exif','empty-alpha'])
def test_rehashed_but_bad_image_refused(tmp_path,change):
 out=fixture(tmp_path);p=out/'reviewed-alpha.png';im=Image.open(p).copy()
 if change=='wrong-mode':im=im.convert('RGB')
 elif change=='wrong-size':im=im.resize((10,10))
 elif change=='empty-alpha':im=Image.new('L',im.size,0)
 if change=='hidden-exif':
  exif=Image.Exif();exif[274]=6;im.save(p,exif=exif)
 else:im.save(p)
 record=json.loads((out/'prepared-source.json').read_text());record['alpha_png_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();(out/'prepared-source.json').write_text(json.dumps(record))
 with pytest.raises(IntakeError):load_prepared(out)

def test_symlink_prepared_file_refused(tmp_path):
 out=fixture(tmp_path);p=out/'reviewed-alpha.png';target=tmp_path/'held.png';p.rename(target)
 try:p.symlink_to(target)
 except OSError as e:
  if getattr(e,'winerror',None)==1314:pytest.skip('Windows lacks symlink privilege')
  raise
 with pytest.raises(IntakeError,match='symlink'):load_prepared(out)

def test_cpu_bridge_real_pipeline_and_strict_json(tmp_path):
 out=fixture(tmp_path);before={p.name:p.read_bytes() for p in out.iterdir()}
 r=compose_prepared(out,tmp_path/'runs',seed=7)
 assert len(r.variants)==1 and r.variants[0].output_path.is_file()
 m=json.loads(r.manifest_path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 assert m['variants'][0]['backend']=='supplied_mask' and m['variants'][0]['seed']==7
 audit=json.loads((r.run_dir.parent.parent/'prepared-compose-audit.json').read_text());assert audit['relighting'] is False and audit['paid_backend'] is False and audit['publication_status']=='NOT_EVALUATED'
 assert audit['composition_may_resize_source'] is True and audit['pixel_fidelity']=='SEE_SCOPED_NATIVE_PIXEL_CHECKS'
 assert audit['native_pixel_checks'][0]['status']=='MATCH'
 assert audit['native_pixel_checks'][0]['native_rgb_exact'] is True
 assert before=={p.name:p.read_bytes() for p in out.iterdir()}


def test_source_changed_after_validation_cannot_change_snapshot(tmp_path,monkeypatch):
 import renderello.pipeline as pipeline
 original=pipeline.run_pipeline
 out=fixture(tmp_path);expected=(out/'normalized-source.png').read_bytes()
 def inspect(request,*args,**kwargs):
  (out/'normalized-source.png').write_bytes(b'changed during composition')
  assert request.source_path.read_bytes()==expected and request.source_path.parent!=out
  return original(request,*args,**kwargs)
 monkeypatch.setattr(pipeline,'run_pipeline',inspect);compose_prepared(out,tmp_path/'runs')


def test_large_product_resize_is_reported_not_called_pixel_exact(tmp_path):
 from renderello.pipeline import _placed_region
 from renderello.schemas import CutoutResult,Aspect
 source=Image.new('RGB',(1500,1100),'white');mask=Image.new('L',source.size,0)
 ImageDraw.Draw(source).rectangle((100,100,1399,999),fill=(15,65,115));ImageDraw.Draw(mask).rectangle((100,100,1399,999),fill=255)
 source.save(tmp_path/'large.png');mask.save(tmp_path/'large-mask.png')
 prepared=tmp_path/'large-prepared';prepare_reviewed_source(tmp_path/'large.png',prepared,mask=tmp_path/'large-mask.png',mask_coordinates='display')
 result=compose_prepared(prepared,tmp_path/'large-runs')
 m=json.loads(result.manifest_path.read_text());c=m['cutout'];co=CutoutResult(Path(c['rgba_path']),Path(c['matte_path']),tuple(c['bbox']),c['backend'],c['coverage'])
 placed=_placed_region(co,Aspect.SQUARE,0,'clean_studio');assert placed[2]<co.bbox[2] and placed[3]<co.bbox[3]
 audit=json.loads((result.run_dir.parent.parent/'prepared-compose-audit.json').read_text());assert audit['pixel_fidelity']=='SEE_SCOPED_NATIVE_PIXEL_CHECKS'

 assert audit['native_pixel_checks'][0]['status']=='NATIVE_SIZE_CHANGED'
 assert audit['native_pixel_checks'][0]['native_rgb_exact'] is None


def test_saved_output_one_bit_corruption_is_detected_after_pipeline(tmp_path,monkeypatch):
 import numpy as np
 import renderello.pipeline as pipeline
 original=pipeline.run_pipeline
 out=fixture(tmp_path)
 def corrupt(*args,**kwargs):
  result=original(*args,**kwargs)
  for variant in result.variants:
   with Image.open(variant.output_path) as im:array=np.array(im)
   array^=1;Image.fromarray(array).save(variant.output_path)
  return result
 monkeypatch.setattr(pipeline,'run_pipeline',corrupt)
 with pytest.raises(IntakeError,match='changed fully opaque'):
  compose_prepared(out,tmp_path/'corrupt-runs')
 job=next((tmp_path/'corrupt-runs').iterdir());assert (job/'FAILED').is_file()
 audit=json.loads((job/'prepared-compose-audit.json').read_text())
 check=audit['native_pixel_checks'][0]
 assert check['status']=='MISMATCH' and check['native_rgb_exact'] is False and check['max_channel_error']==1
 assert check['changed_pixels']==check['fully_opaque_source_pixels']
 assert check['quality_acceptance']=='NOT_EVALUATED'
