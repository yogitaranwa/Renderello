import json
from pathlib import Path
import pytest
from PIL import Image,ImageCms
from renderello.benchmark import run_benchmark
from renderello.config import RunConfig

def inputs(tmp_path):
 Image.new('RGB',(16,16),(120,30,20)).save(tmp_path/'plain.png')
 Image.new('RGB',(16,16),(120,30,20)).save(tmp_path/'profiled.png',icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes())
 p=tmp_path/'sources.json';p.write_text(json.dumps(dict(sources=[dict(path=n,label=n,license='synthetic fixture')for n in ['plain.png','profiled.png']],templates=['clean_studio'],routes=['pure_prompt'],variants=1)));return p

def test_late_profile_refused_before_any_generation(tmp_path,monkeypatch):
 from renderello import benchmark
 called=[];monkeypatch.setattr(benchmark,'run_pipeline',lambda *a,**kw:called.append(kw))
 with pytest.raises(ValueError,match='explicit colour-conversion'):
  run_benchmark(inputs(tmp_path),RunConfig(out_root=tmp_path/'runs'))
 assert not called

def test_explicit_policy_propagated_after_full_preflight(tmp_path,monkeypatch):
 from renderello import benchmark
 called=[]
 def pipeline(*a,**kw):called.append(kw);raise RuntimeError('stop before generation')
 monkeypatch.setattr(benchmark,'run_pipeline',pipeline)
 with pytest.raises(RuntimeError,match='stop before generation'):
  run_benchmark(inputs(tmp_path),RunConfig(out_root=tmp_path/'runs'),allow_colour_conversion=True)
 assert called[0]['allow_colour_conversion'] is True

def test_gallery_caption_records_normalized_basis():
 from renderello.gallery import _source_caption
 s=_source_caption(dict(image_intake=dict(transformations=['icc_to_srgb'])))
 assert 'icc_to_srgb' in s and 'normalized RGB' in s
