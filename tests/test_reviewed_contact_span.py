import numpy as np
import pytest
from PIL import Image,ImageDraw
from renderello.scene_shadow import reviewed_span_contact,projected_soft_shadow


def sloped():
 a=np.zeros((100,70),np.uint8)
 for col in range(10,60):a[10:90-(col-10)//3,col]=255
 return a


def test_span_follows_raised_end_and_keeps_alpha():
 a=sloped();before=a.copy();span=reviewed_span_contact(a,(200,200),(50,50),(15,55));default=projected_soft_shadow(a,(200,200),(50,50));row=50+np.flatnonzero(a[:,54]>127)[-1]+1
 assert span[row,104]>default[row,104]+.05
 assert np.array_equal(a,before) and np.isfinite(span).all() and span.max()<=.45


@pytest.mark.parametrize('span',[None,(1,),('2',8),(True,8),(-1,8),(5,5),(5,6),(0,80),(0,12),(65,70)])
def test_bad_span_rejected(span):
 with pytest.raises(ValueError):reviewed_span_contact(sloped(),(200,200),(50,50),span)


def test_zero_cast_uses_reviewed_contact_and_default_is_unchanged():
 a=sloped();assert np.array_equal(projected_soft_shadow(a,(200,200),(50,50),contact_span=(15,55)),reviewed_span_contact(a,(200,200),(50,50),(15,55)))
 from renderello.scene_plate import support_shadow
 assert np.array_equal(projected_soft_shadow(a,(200,200),(50,50)),support_shadow(a,(200,200),(50,50)))


def test_no_floor_room_refused():
 a=np.full((20,20),255,np.uint8)
 with pytest.raises(ValueError):reviewed_span_contact(a,(30,30),(5,10),(1,15))


def test_cli_only_forwards_explicit_span(monkeypatch,tmp_path):
 from renderello import cli,scene_plate
 captured=[];monkeypatch.setattr(scene_plate,'compose_plate',lambda *a,**k:captured.append(k) or {})
 assert cli.main(['scene-plate','--prepared','p','--background','b','--baseline','900','--out',str(tmp_path/'out'),'--shadow-model','height_soft_linear','--reviewed-contact-span','25','55'])==0
 assert captured[0]['reviewed_contact_span']==[25,55]


def test_real_entrypoint_source_coordinates_plane_and_no_partial_output(tmp_path):
 from renderello.reviewed_source import prepare_reviewed_source
 from renderello.scene_plate import compose_plate
 a=sloped();rgb=np.full((*a.shape,3),230,np.uint8);rgb[a>0]=(20,65,140)
 Image.fromarray(rgb).save(tmp_path/'src.png');Image.fromarray(a).save(tmp_path/'alpha.png');Image.new('RGB',(1024,1024),(210,200,190)).save(tmp_path/'plate.png')
 prepare_reviewed_source(tmp_path/'src.png',tmp_path/'prepared',mask=tmp_path/'alpha.png',mask_coordinates='display')
 opts=dict(baseline=900,shadow_model='height_soft_linear',reviewed_contact_span=(15,55))
 meta=compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'out',**opts)
 assert meta['native_pixel_audit']['status']=='MATCH' and meta['reviewed_contact_span_source_columns']==(15,55)
 assert meta['contact_span_geometry']=='EXPLICIT_REVIEW_NOT_INFERRED'
 plane=np.full((1024,1024),255,np.uint8);x,y=meta['placement'];plane[y+int(np.flatnonzero(a[:,54]>127)[-1])-10+1,x+54-10]=0
 Image.fromarray(plane).save(tmp_path/'plane.png')
 with pytest.raises(ValueError,match='support plane'):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/'bad-plane',support_plane_path=tmp_path/'plane.png',**opts)
 assert not (tmp_path/'bad-plane').exists()
 for n,kw in [('off',dict(contact=False)),('legacy',dict(shadow_model='legacy')),('bounds',dict(reviewed_contact_span=(0,60)))]:
  args=opts|kw
  with pytest.raises(ValueError):compose_plate(tmp_path/'prepared',tmp_path/'plate.png',tmp_path/n,**args)
  assert not (tmp_path/n).exists()


@pytest.mark.parametrize('opacity',[float('nan'),float('inf'),-.1,.36])
def test_reviewed_span_does_not_bypass_opacity_validation(opacity):
 with pytest.raises(ValueError):projected_soft_shadow(sloped(),(200,200),(50,50),contact_span=(15,55),opacity=opacity)
