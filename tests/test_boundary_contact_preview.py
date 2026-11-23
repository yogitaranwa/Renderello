from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello.boundary_contact_preview import refine_alpha,compose_contact

@pytest.fixture
def data():
 rgb=np.arange(60*70*3,dtype='uint16').reshape(60,70,3)%256
 a=np.zeros((60,70),dtype='uint8');a[8:50,10:60]=255
 return Image.fromarray(rgb.astype('uint8')),Image.fromarray(a),{'label':[20,20,40,35]}

def test_boundary_only_and_label(data):
 im,a,p=data;r=refine_alpha(a,p,trim=.7,feather=1.4);aa=np.array(a);rr=np.array(r)
 assert np.all(rr<=aa);assert np.all(rr[20:35,20:40]==255);assert np.all(rr[12:46,14:56]==255)

def test_contact_exact_source_and_integer_offset(data,tmp_path):
 im,a,p=data;r=compose_contact(im,a,tmp_path/'x',protected_regions=p,offset=[12,14],canvas=(100,100));o=np.array(Image.open(tmp_path/'x/output.png'));ys,xs=np.nonzero(np.array(a)==255)
 assert np.array_equal(o[ys+14,xs+12],np.array(im)[ys,xs]);assert r['opaque_changed_pixels']==0;assert r['integer_translation']==[12,14]

def test_contact_support_is_lowest_silhouette(data,tmp_path):
 im,a,p=data;r=compose_contact(im,a,tmp_path/'x',protected_regions=p,offset=[12,14],canvas=(100,100));assert r['support_centre'][1]==63

@pytest.mark.parametrize('offset',[[1.5,2],[1],[1,2,3]])
def test_bad_offset(data,tmp_path,offset):
 im,a,p=data
 with pytest.raises(ValueError):compose_contact(im,a,tmp_path/'x',protected_regions=p,offset=offset)

def test_no_overwrite(data,tmp_path):
 im,a,p=data;(tmp_path/'x').mkdir()
 with pytest.raises(ValueError):compose_contact(im,a,tmp_path/'x',protected_regions=p,offset=[0,0])

def test_clipping(data,tmp_path):
 im,a,p=data
 with pytest.raises(ValueError):compose_contact(im,a,tmp_path/'x',protected_regions=p,offset=[-20,0])

def test_empty_mask(data,tmp_path):
 im,a,p=data
 with pytest.raises(ValueError):compose_contact(im,Image.new('L',im.size),tmp_path/'x',protected_regions=p,offset=[0,0])

def test_missing_protection(data,tmp_path):
 im,a,p=data
 with pytest.raises(ValueError):compose_contact(im,a,tmp_path/'x',protected_regions={},offset=[0,0])

def test_nonopaque_label(data,tmp_path):
 im,a,p=data
 with pytest.raises(ValueError):refine_alpha(a,{'bad':[0,0,5,5]})

def test_contour_does_not_erase_protected_label(data):
 im,a,p=data
 with pytest.raises(ValueError):refine_alpha(a,p,contour=[(10,8),(15,8),(15,49),(10,49)])

@pytest.mark.parametrize('trim,feather',[(-1,1),(1,0)])
def test_invalid_band(data,trim,feather):
 im,a,p=data
 with pytest.raises(ValueError):refine_alpha(a,p,trim=trim,feather=feather)
