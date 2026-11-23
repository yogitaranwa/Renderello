import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello import native_studio as m
from renderello.intent_mask import sha

def bundle(tmp_path,size=(140,180)):
 b=tmp_path/'bundle';b.mkdir();rgb=np.full((size[1],size[0],3),[31,41,59],np.uint8);alpha=np.zeros(rgb.shape[:2],np.uint8);alpha[10:-10,10:-10]=255;Image.fromarray(rgb).save(b/'source.png');Image.fromarray(alpha).save(b/'refined-alpha.png');Image.fromarray(np.dstack([rgb,alpha])).save(b/'refined-cutout.png');manifest=dict(source_rgb_unchanged=True,edge_rgb_estimated=False,source_sha256='fixture',output_sha256={f.name:sha(f) for f in b.iterdir()});(b/'matting-manifest.json').write_text(json.dumps(manifest));return b

@pytest.mark.parametrize('style',list(m.STYLES))
def test_native_core_exact_each_style(tmp_path,style):
 b=bundle(tmp_path);o=tmp_path/style;d=m.compose_native(b,style,o,source_pose='front',material='opaque',framing='complete');assert d['scale']==1;assert d['opaque_core_rgb_max_error']==0;assert d['quality']=='NOT_EVALUATED';im=np.array(Image.open(o/'candidate.png'));x,y=d['placement'];assert (im[y:y+160,x:x+120]==[31,41,59]).all();assert y+160==d['support_baseline']
 for f,h in d['output_sha256'].items():assert sha(o/f)==h

@pytest.mark.parametrize('kw',[dict(source_pose='overhead'),dict(source_pose='unknown'),dict(material='transparent'),dict(framing='cropped'),dict(material='unknown'),dict(framing='unknown')])
def test_scope_rejected(tmp_path,kw):
 b=bundle(tmp_path);args=dict(source_pose='front',material='opaque',framing='complete');args.update(kw)
 with pytest.raises(ValueError):m.compose_native(b,'porcelain_sweep',tmp_path/'out',**args)
 assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('what',['hash','estimated','rgb','alpha','border'])
def test_bundle_fail_closed(tmp_path,what):
 b=bundle(tmp_path);p=b/'matting-manifest.json';d=json.loads(p.read_text())
 if what=='estimated':d['edge_rgb_estimated']=True
 if what=='hash':d['output_sha256']['source.png']='wrong'
 if what in ('rgb','alpha','border'):
  f=b/'refined-cutout.png';a=np.array(Image.open(f))
  if what=='rgb':a[20,20,0]=99
  if what=='alpha':a[20,20,3]=0
  if what=='border':a[0,20,3]=255;Image.fromarray(a[:,:,3]).save(b/'refined-alpha.png');d['output_sha256']['refined-alpha.png']=sha(b/'refined-alpha.png')
  Image.fromarray(a).save(f);d['output_sha256'][f.name]=sha(f)
 p.write_text(json.dumps(d))
 with pytest.raises(ValueError):m.compose_native(b,'porcelain_sweep',tmp_path/'out',source_pose='front',material='opaque',framing='complete')

def test_oversize_no_silent_resize(tmp_path):
 b=bundle(tmp_path,size=(800,800))
 with pytest.raises(ValueError,match='760'):m.compose_native(b,'porcelain_sweep',tmp_path/'out',source_pose='front',material='opaque',framing='complete')

def test_deterministic_background_and_unknown_style():
 a=np.array(m.background('sage_arch',size=(512,512)));b=np.array(m.background('sage_arch',size=(512,512)));assert np.array_equal(a,b)
 with pytest.raises(ValueError):m.background('unknown')

def test_attempts_preserved(tmp_path):
 b=bundle(tmp_path);o=tmp_path/'out';o.mkdir();(o/'keep').write_text('old')
 with pytest.raises(ValueError,match='exists'):m.compose_native(b,'porcelain_sweep',o,source_pose='front',material='opaque',framing='complete')
 assert (o/'keep').read_text()=='old'

def test_final_native_region_pixels_checked(tmp_path):
 b=bundle(tmp_path);rp=tmp_path/'regions.json';rp.write_text(json.dumps(dict(source_sha256='fixture',image_size=[140,180],regions=[dict(name='small print',box=[20,20,60,40])])));d=m.compose_native(b,'porcelain_sweep',tmp_path/'out',source_pose='front',material='opaque',framing='complete',regions_path=rp);assert d['native_region_checks']['all_source_native_regions_retained'];assert d['native_region_checks']['regions'][0]['pixels']==800

def test_region_check_detects_invisible_or_changed_label():
 rgba=np.full((20,20,4),255,np.uint8);scene=rgba[:,:,:3].copy();r=dict(source_sha256='h',image_size=[20,20],regions=[dict(name='label',box=[2,2,10,10])]);rgba[3,3,3]=0;scene[4,4,:]=0;d=m.check_native_regions(rgba,scene,[0,0],r,'h');assert not d['all_source_native_regions_retained'];assert d['regions'][0]['final_rgb_changed_pixels']==1;assert d['regions'][0]['nonopaque_source_pixels']==1

@pytest.mark.parametrize('what',['source','bounds','duplicate','outside_scene'])
def test_native_region_binding(tmp_path,what):
 rgba=np.full((20,20,4),255,np.uint8);r=dict(source_sha256='h',image_size=[20,20],regions=[dict(name='label',box=[2,2,10,10])]);translation=[0,0]
 if what=='source':r['source_sha256']='wrong'
 if what=='bounds':r['regions'][0]['box']=[0,0,21,5]
 if what=='duplicate':r['regions']*=2
 if what=='outside_scene':translation=[15,0]
 with pytest.raises(ValueError):m.check_native_regions(rgba,rgba[:,:,:3],translation,r,'h')

@pytest.mark.parametrize('style',['clay_plinth','blush_steps'])
def test_platform_rejects_wide_group(tmp_path,style):
 b=bundle(tmp_path,size=(720,500))
 with pytest.raises(ValueError,match='too wide'):m.compose_native(b,style,tmp_path/'out',source_pose='elevated',material='opaque',framing='complete')
 assert not (tmp_path/'out').exists()
