from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello.label_retention import check_regions
from renderello.intent_mask import sha

def case(tmp_path,alpha=255,threshold=255,color=(100,150,200)):
 src=tmp_path/'src.png';cut=tmp_path/'cut.png'
 Image.new('RGB',(3,2),color).save(src)
 Image.new('RGBA',(3,2),(*color,alpha)).save(cut)
 r=dict(source_sha256=sha(src),image_size=[3,2],regions=[dict(name='print',box=[0,0,3,2],minimum_alpha=threshold)])
 return src,cut,r

def test_opaque_composites_exact(tmp_path):
 r=check_regions(*case(tmp_path))
 assert r['all_source_native_composites_exact'] is True
 assert all(x['changed_pixels']==0 for x in r['regions'][0]['composite_checks'].values())

@pytest.mark.parametrize('a',[0,128,252,254])
def test_partial_is_not_source_exact(tmp_path,a):
 r=check_regions(*case(tmp_path,alpha=a));item=r['regions'][0]
 assert item['source_rgb_exact'] and not item['source_native_composite_exact']
 assert not r['all_retention_checks_passed']
 assert all(x['changed_pixels']==6 for x in item['composite_checks'].values())

def test_threshold_pass_does_not_claim_composite_exact(tmp_path):
 r=check_regions(*case(tmp_path,alpha=252,threshold=200))
 assert r['all_retention_checks_passed'] and not r['all_source_native_composites_exact']
 assert r['quality']=='NOT_EVALUATED' and r['transcription']=='NOT_PERFORMED'

def test_white_can_match_white_but_fail_other_backgrounds(tmp_path):
 r=check_regions(*case(tmp_path,alpha=0,color=(255,255,255)))['regions'][0]
 assert r['composite_checks']['white']['changed_pixels']==0
 assert r['composite_checks']['black']['max_channel_error']==255
 assert not r['source_native_composite_exact']

def test_actual_pillow_matches_report(tmp_path):
 src,cut,regions=case(tmp_path,alpha=252)
 r=check_regions(src,cut,regions)['regions'][0]
 for name,color in [('white',(255,255,255)),('black',(0,0,0)),('magenta',(255,0,255))]:
  with Image.open(cut) as im:
   seen=np.array(Image.alpha_composite(Image.new('RGBA',im.size,(*color,255)),im).convert('RGB'))
  expected=np.abs(seen.astype('int16')-np.array(Image.open(src)).astype('int16'))
  assert r['composite_checks'][name]['max_channel_error']==int(expected.max())
