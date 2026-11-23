import hashlib,json
from pathlib import Path
import pytest
from PIL import Image,ImageOps
from renderello.reviewed_source import prepare_reviewed_source,_TRANSPOSE
from renderello.image_intake import IntakeError

@pytest.mark.parametrize('orientation',range(1,9))
@pytest.mark.parametrize('coordinates',['encoded','display'])
def test_explicit_mask_coordinates_all_orientations(tmp_path,orientation,coordinates):
 p=tmp_path/'source.png';im=Image.new('RGB',(4,3));im.putdata([(i*17,i*9,i*3) for i in range(12)]);exif=Image.Exif();exif[274]=orientation;im.save(p,exif=exif)
 alpha=Image.new('L',im.size);alpha.putdata([255,0,0,0,0,150,0,0,0,0,1,0]);expected=alpha.transpose(_TRANSPOSE[orientation]) if orientation in _TRANSPOSE else alpha
 mp=tmp_path/'mask.png';(alpha if coordinates=='encoded' else expected).save(mp)
 old=p.read_bytes();mold=mp.read_bytes();out=tmp_path/'out';r=prepare_reviewed_source(p,out,mask=mp,mask_coordinates=coordinates)
 with Image.open(out/'reviewed-alpha.png') as actual:assert actual.size==expected.size and actual.tobytes()==expected.tobytes()
 with Image.open(p) as original:rgb=ImageOps.exif_transpose(original).convert('RGB')
 with Image.open(out/'normalized-source.png') as actual:assert actual.tobytes()==rgb.tobytes() and actual.size==rgb.size
 assert p.read_bytes()==old and mp.read_bytes()==mold
 assert r['external_mask_sha256']==hashlib.sha256(mold).hexdigest()
 assert r['semantic_mask_accuracy']=='NOT_VERIFIED' and not r['legacy_routes_changed']

@pytest.mark.parametrize('mode,key,values',[('RGB',(10,20,30),[(10,20,30),(0,0,0)]),('L',17,[17,200])])
def test_embedded_colour_key_becomes_exact_explicit_alpha(tmp_path,mode,key,values):
 p=tmp_path/'s.png';im=Image.new(mode,(2,1));im.putdata(values);im.save(p,transparency=key)
 out=tmp_path/'out';prepare_reviewed_source(p,out)
 with Image.open(out/'reviewed-alpha.png') as alpha:assert list(alpha.tobytes())==[0,255]

@pytest.mark.parametrize('kind',['missing','undeclared','wrong-size','rgb-mask','empty','mask-exif','mask-transparency','conflict'])
def test_ambiguous_sources_fail_before_destination(tmp_path,kind):
 p=tmp_path/'s.png';Image.new('RGBA' if kind=='conflict' else 'RGB',(4,3),(10,20,30,150) if kind=='conflict' else (10,20,30)).save(p)
 mask=tmp_path/'m.png';m=Image.new('RGB' if kind=='rgb-mask' else 'L',(3,3) if kind=='wrong-size' else (4,3),0 if kind=='empty' else 255);kw={}
 if kind=='mask-exif':e=Image.Exif();e[274]=2;kw['exif']=e
 if kind=='mask-transparency':kw['transparency']=255
 m.save(mask,**kw);out=tmp_path/'out';args={}
 if kind!='missing':args=dict(mask=mask,mask_coordinates=None if kind=='undeclared' else 'encoded')
 with pytest.raises((IntakeError,ValueError)):prepare_reviewed_source(p,out,**args)
 assert not out.exists()

def test_matching_embedded_and_external_alpha_allowed_and_no_overwrite(tmp_path):
 p=tmp_path/'s.png';Image.new('RGBA',(4,3),(10,20,30,150)).save(p);m=tmp_path/'m.png';Image.new('L',(4,3),150).save(m)
 out=tmp_path/'out';prepare_reviewed_source(p,out,mask=m,mask_coordinates='display');prior={f.name:f.read_bytes() for f in out.iterdir()}
 with pytest.raises(FileExistsError):prepare_reviewed_source(p,out,mask=m,mask_coordinates='encoded')
 assert {f.name:f.read_bytes() for f in out.iterdir()}==prior


def test_high_bit_mask_refused_before_destination(tmp_path):
 p=tmp_path/'s.png';Image.new('RGB',(4,3)).save(p)
 m=tmp_path/'16.png';Image.new('I;16',(4,3),65535).save(m)
 with pytest.raises(IntakeError,match='8-bit'):prepare_reviewed_source(p,tmp_path/'out',mask=m,mask_coordinates='encoded')
 assert not (tmp_path/'out').exists()
