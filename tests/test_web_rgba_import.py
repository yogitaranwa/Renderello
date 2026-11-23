import io,json,re,urllib.request,urllib.error,urllib.parse
import numpy as np
import pytest
from PIL import Image
from renderello.webui import serve
from tests.test_webui import _multipart


def png(a,**kw):
 b=io.BytesIO();Image.fromarray(a).save(b,'PNG',**kw);return b.getvalue()


@pytest.fixture
def client(tmp_path):
 rgb=np.full((64,80,3),180,np.uint8);rgb[10:54,20:60]=[40,80,130];alpha=np.zeros((64,80),np.uint8);alpha[10:54,20:60]=255;rgba=np.dstack([rgb,alpha]);s=serve(0,tmp_path);url=f'http://127.0.0.1:{s.server_address[1]}'
 try:yield url,tmp_path,rgb,alpha,rgba
 finally:s.shutdown();s.server_close()


def post(url,fields):
 req=urllib.request.Request(url+'/run',data=_multipart(fields),headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
 return urllib.request.urlopen(req).read().decode()


def fields(rgb,rgba):return dict(photo=('source.png',png(rgb),'image/png'),rgba_result=('result.png',png(rgba),'image/png'),variants=('',b'2','text/plain'),route=('',b'cutout_compose','text/plain'))


def test_upload_and_rerun_reverify_same_rgba(client):
 url,root,rgb,a,rgba=client
 page=urllib.request.urlopen(url).read().decode();assert 'name="rgba_result"'in page
 text=post(url,fields(rgb,rgba));assert 'VERIFIED SOURCE-RGB RGBA IMPORT'in text;job=re.search(r'name="job" value="([a-f0-9]+)"',text).group(1)
 assert json.loads((root/job/'rgba-import.json').read_text())['source_rgb_exact'];assert np.array_equal(np.array(Image.open(root/job/'mask.png')),a)
 req=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode(dict(job=job,variants=2,route='cutout_compose')).encode());text=urllib.request.urlopen(req).read().decode();newjob=re.search(r'name="job" value="([a-f0-9]+)"',text).group(1)
 assert 'VERIFIED SOURCE-RGB RGBA IMPORT'in text and newjob!=job;assert (root/newjob/'external-rgba.png').read_bytes()==(root/job/'external-rgba.png').read_bytes()
 assert json.loads((root/newjob/'rgba-import.json').read_text())['automatic_acceptance']is False


@pytest.mark.parametrize('case',['repaint_hidden','repaint_core','wrong_size','color_metadata','both_masks','regeneration','source_jpeg'])
def test_refuse_before_pipeline_outputs(client,case):
 url,root,rgb,a,rgba=client;f=fields(rgb,rgba)
 if case in ['repaint_hidden','repaint_core']:
  rgba=rgba.copy();rgba[0 if case=='repaint_hidden'else 20,25,0]+=1;f['rgba_result']=('result.png',png(rgba),'image/png')
 elif case=='wrong_size':f['rgba_result']=('result.png',png(rgba[:-1]),'image/png')
 elif case=='color_metadata':
  from PIL import PngImagePlugin
  info=PngImagePlugin.PngInfo();info.add(b'sRGB',b'\0');f['rgba_result']=('result.png',png(rgba,pnginfo=info),'image/png')
 elif case=='both_masks':f['mask']=('alpha.png',png(a),'image/png')
 elif case=='regeneration':f['route']=('',b'hybrid','text/plain')
 elif case=='source_jpeg':
  b=io.BytesIO();Image.fromarray(rgb).save(b,'JPEG');f['photo']=('source.jpg',b.getvalue(),'image/jpeg')
 with pytest.raises(urllib.error.HTTPError)as e:post(url,f)
 assert e.value.code==400
 assert not list(root.glob('*/**/manifest.json'))


def test_rerun_detects_changed_saved_rgba(client):
 url,root,rgb,a,rgba=client;text=post(url,fields(rgb,rgba));job=re.search(r'name="job" value="([a-f0-9]+)"',text).group(1);changed=rgba.copy();changed[0,0,1]+=1;(root/job/'external-rgba.png').write_bytes(png(changed))
 req=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode(dict(job=job,variants=2)).encode())
 with pytest.raises(urllib.error.HTTPError)as e:urllib.request.urlopen(req)
 assert e.value.code==400
