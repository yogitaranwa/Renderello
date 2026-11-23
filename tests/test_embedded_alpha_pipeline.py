from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from renderello.pipeline import run_pipeline
from renderello.pipeline_intake import load_generation_image
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.config import RunConfig

def source(tmp_path,transparent=True):
 a=np.zeros((80,60,4),np.uint8);a[:,:,:3]=(20,100,180);a[12:68,15:45,3]=255;a[13:67,14,3]=80
 p=tmp_path/'source.png';Image.fromarray(a if transparent else a[:,:,:3]).save(p);return p,a

def test_opt_in_reuses_alpha_without_segmentation_or_rgb_change(tmp_path,monkeypatch):
 p,a=source(tmp_path);before=p.read_bytes();from renderello import pipeline
 def forbidden(*a,**kw):raise AssertionError('segmentation must not run')
 monkeypatch.setattr(pipeline,'cutout',forbidden)
 r=run_pipeline(EditRequest(p,get_template('clean_studio')),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'),use_embedded_alpha=True)
 m=json.loads(r.manifest_path.read_text());assert m['mask_provenance']=='embedded_source_alpha' and m['image_intake']['embedded_alpha_used']
 assert m['variants'][0]['success'] and p.read_bytes()==before
 with Image.open(m['cutout']['rgba_path'])as im:assert np.array_equal(np.array(im),a)
 with Image.open(m['mask_path'])as im:assert np.array_equal(np.array(im),a[:,:,3])
 assert not m['image_intake']['legacy_alpha_discarded']

@pytest.mark.parametrize('route',[Route.PURE_PROMPT,Route.HYBRID])
def test_no_silent_ignored_alpha_on_regenerative_route(tmp_path,route):
 p,a=source(tmp_path)
 with pytest.raises(ValueError,match='not product regeneration'):
  run_pipeline(EditRequest(p,get_template('clean_studio')),[route],RunConfig(out_root=tmp_path/'runs'),use_embedded_alpha=True)

def test_alpha_choice_requires_alpha_and_no_conflicting_mask(tmp_path):
 p,a=source(tmp_path,False)
 with pytest.raises(ValueError,match='requires a source with alpha'):load_generation_image(p,use_embedded_alpha=True)
 with pytest.raises(ValueError,match='not both'):
  run_pipeline(EditRequest(p,get_template('clean_studio'),mask_path=p),[Route.CUTOUT_COMPOSE],RunConfig(out_root=tmp_path/'runs'),use_embedded_alpha=True)


def test_orientation_moves_rgb_and_alpha_together(tmp_path):
 p,a=source(tmp_path);im=Image.open(p);ex=Image.Exif();ex[274]=6;im.save(tmp_path/'oriented.png',exif=ex)
 rgb,record,alpha=load_generation_image(tmp_path/'oriented.png',use_embedded_alpha=True)
 assert rgb.size==alpha.size==(80,60) and np.array_equal(np.array(alpha),np.rot90(a[:,:,3],-1))
 assert np.array_equal(np.array(rgb),np.rot90(a[:,:,:3],-1))


def test_default_discloses_discarded_alpha(tmp_path):
 p,a=source(tmp_path);rgb,r=load_generation_image(p)
 assert r['legacy_alpha_discarded'] and not r['embedded_alpha_used'] and r['embedded_alpha_sha256'] is None

@pytest.mark.parametrize('value',[0,255])
def test_empty_or_fully_opaque_alpha_refuses_without_segmentation_fallback(tmp_path,monkeypatch,value):
 p,a=source(tmp_path);a[:,:,3]=value;Image.fromarray(a).save(p)
 from renderello import pipeline
 def forbidden(*a,**kw):raise AssertionError('must not fallback')
 monkeypatch.setattr(pipeline,'cutout',forbidden)
 r=run_pipeline(EditRequest(p,get_template('clean_studio')),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'),use_embedded_alpha=True)
 assert not r.variants[0].success and not r.variants[0].output_path.exists()


def test_conflicting_web_upload_choice_refused_before_job_created(tmp_path):
 from renderello.webui import _Handler
 h=object.__new__(_Handler);h.out_root=tmp_path/'jobs'
 with pytest.raises(ValueError,match='not a combination'):
  h._run_job(tmp_path/'none.png','clean_studio','',None,2,Route.CUTOUT_COMPOSE,use_embedded_alpha=True,mask_upload=('m.png',b'x'))
 assert not h.out_root.exists()
