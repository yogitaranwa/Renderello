from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello.checked_source_lineage import verify_lossless_container_lineage as verify

@pytest.fixture
def pair(tmp_path):
 a=tmp_path/'a.jpg';b=tmp_path/'b.png'
 Image.fromarray(np.random.default_rng(42).integers(0,256,(21,31,3),dtype=np.uint8)).save(a)
 with Image.open(a) as im:im.save(b)
 return a,b

def test_native_rgb_normalization_explicit_only(pair):
 r=verify(*pair)
 assert r['decoded_rgb_exact'] and r['original_file_sha256']!=r['derived_file_sha256']
 assert r['parent_record'] is None
 for key in ['source_photo_origin_verified','rights_verified','publication_approved','default_source_lookup_changed']:assert r[key] is False

def test_repainting_refused(pair):
 a,b=pair;rgb=np.array(Image.open(b));rgb[0,0,0]^=1;Image.fromarray(rgb).save(b)
 with pytest.raises(ValueError,match='RGB changed'):verify(a,b)

def test_color_metadata_change_refused(pair):
 from PIL.PngImagePlugin import PngInfo
 a,b=pair;im=Image.open(b);im.load();p=PngInfo();p.add(b'sRGB',b'\0');im.save(b,pnginfo=p)
 with pytest.raises(ValueError,match='metadata changed'):verify(a,b)

def test_alpha_mode_not_converted(pair):
 a,b=pair;im=Image.open(b).convert('RGBA');im.save(b)
 with pytest.raises(ValueError,match='Native RGB'):verify(a,b)

def test_resize_refused(pair):
 a,b=pair;im=Image.open(b).resize((30,21));im.save(b)
 with pytest.raises(ValueError,match='RGB changed'):verify(a,b)

def test_renaming_cannot_establish_parent(tmp_path):
 a=tmp_path/'watch1.jpg';b=tmp_path/'source.png';Image.new('RGB',(3,3),'red').save(a)
 Image.open(a).save(b);assert verify(a,b)['parent_record']is None

def test_symlink_refused(pair,tmp_path):
 a,b=pair;p=tmp_path/'link.png';p.symlink_to(b)
 with pytest.raises(ValueError,match='Regular bounded'):verify(a,p)

def test_known_parent_record_without_clearance(tmp_path):
 root=Path(__file__).parents[1];a=root/'benchmark/sources/watch1.jpg';b=tmp_path/'renamed.png'
 with Image.open(a)as im:
  if im.width*im.height>1_000_000:pytest.skip('Fixture above bounded source budget')
  im.save(b,**({'icc_profile':im.info['icc_profile']}if im.info.get('icc_profile')else{}))
 r=verify(a,b);assert r['parent_record']['label']=='watch1'
 assert not r['rights_verified'] and not r['source_photo_origin_verified']

@pytest.mark.parametrize('kind,payload',[(b'cICP',b'\x01\x0d\x00\x01'),(b'cLLI',b'\0'*8),(b'tRNS',b'\0'*6),(b'acTL',b'\0'*8)])
def test_unsupported_interpretation_chunks_refused(pair,kind,payload):
 import zlib
 a,b=pair;data=b.read_bytes();chunk=len(payload).to_bytes(4,'big')+kind+payload+(zlib.crc32(kind+payload)&0xffffffff).to_bytes(4,'big');b.write_bytes(data[:33]+chunk+data[33:])
 with pytest.raises(ValueError,match='metadata'):verify(a,b)

def test_orientation_not_silently_rotated(pair):
 a,b=pair;im=Image.open(a);im.load();exif=Image.Exif();exif[274]=6;im.save(a,exif=exif)
 with pytest.raises(ValueError,match='Orientation'):verify(a,b)

def test_source_pixel_limit(pair,monkeypatch):
 from renderello import checked_source_lineage
 monkeypatch.setattr(checked_source_lineage,'MAX_PIXELS',2)
 with pytest.raises(ValueError,match='megapixel'):verify(*pair)

def test_source_byte_limit(pair,monkeypatch):
 from renderello import checked_source_lineage
 monkeypatch.setattr(checked_source_lineage,'MAX_BYTES',2)
 with pytest.raises(ValueError,match='bounded'):verify(*pair)
