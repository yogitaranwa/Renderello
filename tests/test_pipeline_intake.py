import json,hashlib
from pathlib import Path
import pytest
from PIL import Image,ImageCms,ImageOps
from renderello.pipeline_intake import load_generation_image
from renderello.pipeline import run_pipeline
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.config import RunConfig
from renderello.backends.base import StubEditor


def profiled(tmp_path):
 p=tmp_path/'profiled.png';im=Image.new('RGB',(80,60),(180,80,30));im.save(p,icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes());return p

def test_untagged_pixels_unchanged_and_immutable(product_path):
 before=product_path.read_bytes();im,record=load_generation_image(product_path)
 with Image.open(product_path)as old:assert im.tobytes()==old.convert('RGB').tobytes()
 assert product_path.read_bytes()==before and not record['colour_accuracy_verified']
 assert record['normalized_rgb_sha256']==hashlib.sha256(im.tobytes()).hexdigest()


def test_profile_requires_explicit_policy_and_no_invalid_profile_fallback(tmp_path):
 p=profiled(tmp_path);before=p.read_bytes()
 with pytest.raises(ValueError,match='explicit colour-conversion'):load_generation_image(p)
 im,record=load_generation_image(p,allow_colour_conversion=True)
 assert 'icc_to_srgb' in record['transformations'] and p.read_bytes()==before
 bad=tmp_path/'bad.png';im.save(bad,icc_profile=b'not ICC')
 with pytest.raises(ValueError,match='ICC conversion failed'):load_generation_image(bad,allow_colour_conversion=True)


def test_generation_uses_managed_pixels_and_records_basis(tmp_path,monkeypatch):
 p=profiled(tmp_path);expected,record=load_generation_image(p,allow_colour_conversion=True);seen=[]
 from renderello import pipeline
 def cutout(image,*args,**kwargs):seen.append(image.tobytes());raise RuntimeError('controlled before model')
 monkeypatch.setattr(pipeline,'cutout',cutout)
 r=run_pipeline(EditRequest(p,get_template('clean_studio')),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'),allow_colour_conversion=True)
 m=json.loads(r.manifest_path.read_text());assert seen==[expected.tobytes()]
 assert m['image_intake']['normalized_rgb_sha256']==record['normalized_rgb_sha256']
 assert (r.run_dir/'normalized-source.png').is_file() and not m['variants'][0]['success']


def test_orientation_applied_and_external_mask_not_implicitly_rotated(tmp_path):
 p=tmp_path/'oriented.jpg';im=Image.new('RGB',(80,60));ex=Image.Exif();ex[274]=6;im.save(p,exif=ex)
 actual,record=load_generation_image(p);assert actual.size==(60,80) and 'exif_orientation_6' in record['transformations']
 mask=tmp_path/'mask.png';Image.new('L',(80,60),255).save(mask)
 with pytest.raises(ValueError,match='prior normalized source/mask pairing'):
  run_pipeline(EditRequest(p,get_template('clean_studio'),mask_path=mask),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'))

def test_reference_profile_not_silently_ignored(tmp_path,product_path):
 ref=profiled(tmp_path)
 with pytest.raises(ValueError,match='explicit colour-conversion'):
  run_pipeline(EditRequest(product_path,get_template('clean_studio'),reference_path=ref),[Route.SCENE_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'))


def test_high_bit_input_refused_before_editor(tmp_path):
 p=tmp_path/'hdr.tif';Image.new('I;16',(30,30)).save(p)
 with pytest.raises(ValueError,match='High-bit-depth|Unsupported bit depth'):
  run_pipeline(EditRequest(p,get_template('clean_studio')),[Route.PURE_PROMPT],RunConfig(variants=1,out_root=tmp_path/'runs'),editor=StubEditor())
