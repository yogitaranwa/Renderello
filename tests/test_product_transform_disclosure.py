from renderello.pipeline import _product_transform_record

def test_native_size_does_not_claim_quality_or_verified_rgb():
 r=_product_transform_record((30,40,80,200),(300,400,80,200))
 assert not r['product_resampled'] and not r['native_rgb_exactness_claimed']
 assert r['alpha_semantics']=='NOT_VERIFIED' and not r['photographic_acceptance']

def test_downsample_disclosed_both_dimensions():
 r=_product_transform_record((30,40,300,800),(120,150,180,480))
 assert r['product_resampled'] and r['source_crop_size']==[300,800]
 assert r['placed_product_size']==[180,480]
 assert not r['native_rgb_exactness_claimed']

def test_rounding_equal_dimensions_not_resampled():
 r=_product_transform_record((0,0,1,1),(2,3,1,1))
 assert not r['product_resampled']

def test_real_http_resampling_is_disclosed(tmp_path,monkeypatch):
 import json
 from tests.test_web_rgba_import import client,fields,post
 from renderello import compose
 from renderello.webui import serve
 from PIL import Image
 import numpy as np
 server=serve(0,tmp_path)
 try:
  rgb=np.full((64,80,3),180,np.uint8);rgb[10:54,20:60]=[40,80,130]
  a=np.zeros((64,80),np.uint8);a[10:54,20:60]=255
  monkeypatch.setattr(compose,'placement_params',lambda seed,key:(.02,0,.8))
  text=post('http://127.0.0.1:'+str(server.server_address[1]),fields(rgb,np.dstack([rgb,a])))
  assert 'PRODUCT RESAMPLING:' in text
  manifests=list(tmp_path.rglob('manifest.json'));assert manifests
  for p in manifests:
   for variant in json.loads(p.read_text())['variants']:
    assert variant['product_transform']['product_resampled']
    assert not variant['product_transform']['native_rgb_exactness_claimed']
    assert 'native pixel-exactness NOT claimed' in variant['qa']['notes']
 finally:server.shutdown();server.server_close()
