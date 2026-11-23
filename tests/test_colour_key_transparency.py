"""Lossless intake checks for PNG colour-key transparency, not visual quality."""
import hashlib
import pytest
from PIL import Image, ImageCms, ImageOps
from renderello.image_intake import inspect_image, save_intake, IntakePolicy

@pytest.mark.parametrize('mode,key,values',[
 ('RGB',(10,20,30),[(10,20,30),(10,20,31),(0,0,0),(255,255,255)]),
 ('L',17,[17,18,0,255]),
])
@pytest.mark.parametrize('profiled',[False,True])
def test_colour_key_alpha_and_colour_exact(tmp_path,mode,key,values,profiled):
 if profiled and mode=='L':pytest.skip('sRGB profile is not a grayscale profile')
 p=tmp_path/'colour-key.png';im=Image.new(mode,(4,1));im.putdata(values)
 kw={'transparency':key}
 if profiled:kw['icc_profile']=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
 im.save(p,**kw);original=p.read_bytes();before=hashlib.sha256(original).hexdigest()
 with Image.open(p) as source:expected=source.convert('RGBA');source.load()
 r=inspect_image(p,IntakePolicy(allow_colour_conversion=profiled))
 assert r.image.mode=='RGBA'
 assert r.image.getchannel('A').tobytes()==expected.getchannel('A').tobytes()
 # Using an sRGB source/destination keeps these sample colours exact too.
 assert r.image.tobytes()==expected.tobytes()
 assert r.record['source_sha256']==before and p.read_bytes()==original
 assert r.record['input_mode']==mode
 assert 'Existing alpha retained; semantic mask correctness is not verified' in r.record['warnings']
 dest=tmp_path/'normalized';save_intake(r,dest)
 with Image.open(dest/'normalized.png') as actual:assert actual.tobytes()==expected.tobytes()
 r.close()

@pytest.mark.parametrize('mode,key,color',[('RGB',(10,20,30),(10,20,30)),('L',17,17)])
def test_all_transparent_colour_key_stays_transparent(tmp_path,mode,key,color):
 p=tmp_path/'empty.png';Image.new(mode,(5,5),color).save(p,transparency=key)
 r=inspect_image(p)
 assert r.image.mode=='RGBA' and r.image.getchannel('A').getextrema()==(0,0)
 assert not r.record['automatic_publication_pass']
 r.close()

@pytest.mark.parametrize('mode,color',[('RGB',(10,20,30)),('L',17)])
def test_no_colour_key_does_not_invent_alpha(tmp_path,mode,color):
 p=tmp_path/'opaque.png';Image.new(mode,(3,2),color).save(p)
 r=inspect_image(p)
 assert r.image.mode=='RGB' and r.image.convert('RGBA').getchannel('A').getextrema()==(255,255)
 r.close()

def test_colour_key_orientation_and_alpha_stay_aligned(tmp_path):
 p=tmp_path/'oriented.png';im=Image.new('RGB',(3,2));im.putdata([(10,20,30),(10,20,31),(0,0,0),(0,0,0),(10,20,30),(255,255,255)])
 exif=Image.Exif();exif[274]=6;im.save(p,transparency=(10,20,30),exif=exif)
 with Image.open(p) as src:expected=ImageOps.exif_transpose(src).convert('RGBA')
 r=inspect_image(p)
 assert r.image.size==(2,3) and r.image.tobytes()==expected.tobytes()
 r.close()
