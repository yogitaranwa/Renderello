import io,json,hashlib,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello.prompt_editor import prepare_source,PAGE
from renderello.webui import _Handler
from http.server import ThreadingHTTPServer


def sample():
 b=io.BytesIO();Image.new('RGB',(40,30),(20,30,50)).save(b,format='PNG');return b.getvalue()


def test_preview_is_source_bound_and_rgb_exact(tmp_path):
 data=sample();d=prepare_source(data,tmp_path);assert d['source_sha256']==hashlib.sha256(data).hexdigest();assert d['image_size']==[40,30]
 folder=tmp_path/d['job'];assert (folder/'original-source.bin').read_bytes()==data
 assert np.array_equal(np.array(Image.open(folder/'source-preview.png')),np.array(Image.open(io.BytesIO(data))))
 d2=prepare_source(data,tmp_path);assert d2['job']!=d['job'];assert (folder/'source-preview.png').exists()


@pytest.mark.parametrize('data',[b'',b'not an image',b'X'*25_000_001],ids=['empty','not_an_image','over_size_25mb'])
def test_invalid_source(tmp_path,data):
 with pytest.raises(ValueError):prepare_source(data,tmp_path)


def test_pixel_cap(tmp_path):
 b=io.BytesIO();Image.new('RGB',(2001,2000)).save(b,format='PNG')
 with pytest.raises(ValueError,match='million'):prepare_source(b.getvalue(),tmp_path)


def test_routes_use_local_prepare_no_model(tmp_path):
 handler=type('Bound',(_Handler,),{'out_root':tmp_path});server=ThreadingHTTPServer(('127.0.0.1',0),handler);threading.Thread(target=server.serve_forever,daemon=True).start();base='http://127.0.0.1:'+str(server.server_port)
 try:
  page=urllib.request.urlopen(base+'/mask-prompts').read().decode();assert 'Preparation only' in page;assert 'reviewed_source_geometry' in page
  data=sample();boundary='TEST';body=b'--TEST\r\nContent-Disposition: form-data; name="photo"; filename="s.png"\r\nContent-Type: image/png\r\n\r\n'+data+b'\r\n--TEST--\r\n'
  req=urllib.request.Request(base+'/prompt-prepare',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary});d=json.load(urllib.request.urlopen(req));assert d['image_size']==[40,30]
  preview=urllib.request.urlopen(base+d['preview_url']).read();assert np.array_equal(np.array(Image.open(io.BytesIO(preview))),np.array(Image.open(io.BytesIO(data))))
 finally:server.shutdown();server.server_close()
