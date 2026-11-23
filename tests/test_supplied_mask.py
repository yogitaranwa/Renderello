import json
import numpy as np
import pytest
from PIL import Image
from renderello.cutout import supplied_mask_cutout
from renderello.pipeline import run_pipeline
from renderello.schemas import EditRequest,Route
from renderello.templates import get_template
from renderello.config import RunConfig


def test_supplied_mask_bypasses_model_and_preserves_rgb(tmp_path,monkeypatch):
    source=Image.new('RGB',(120,100),(30,60,90));p=tmp_path/'source.png';source.save(p)
    a=np.zeros((100,120),np.uint8);a[20:80,40:90]=255;mask=tmp_path/'a.png';Image.fromarray(a).save(mask)
    def fail(*a,**k):raise AssertionError('automatic backend must not run')
    monkeypatch.setattr('renderello.pipeline.cutout',fail)
    result=run_pipeline(EditRequest(p,get_template('clean_studio'),mask_path=mask),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'))
    assert result.variants[0].backend=='supplied_mask'
    m=json.loads(result.manifest_path.read_text());cut=np.array(Image.open(m['cutout']['rgba_path']))
    assert np.array_equal(cut[...,:3],np.array(source)) and np.array_equal(cut[...,3],a)
    assert 'SUPPLIED MASK' in result.variants[0].qa.notes
    assert m['mask_path']==str(mask)


@pytest.mark.parametrize('kind',['wrong_size','rgb','empty','full'])
def test_bad_masks_refused_not_silently_fixed(tmp_path,kind):
    src=Image.new('RGB',(100,80));a=Image.new('L',(100,80),0)
    if kind=='wrong_size':a=Image.new('L',(20,20),100)
    if kind=='rgb':a=a.convert('RGB')
    if kind=='full':a=Image.new('L',(100,80),255)
    p=tmp_path/'mask.png';a.save(p)
    with pytest.raises(ValueError):supplied_mask_cutout(src,p,tmp_path/'out')


def test_web_mask_upload_and_regenerate(tmp_path,monkeypatch):
    import io,re,urllib.request,urllib.parse
    from renderello.webui import serve
    from tests.test_webui import _multipart
    def fail(*a,**k):raise AssertionError('auto segmentation must not run')
    monkeypatch.setattr('renderello.pipeline.cutout',fail)
    source=io.BytesIO();Image.new('RGB',(120,100),(30,60,90)).save(source,'PNG')
    mask=io.BytesIO();a=np.zeros((100,120),np.uint8);a[20:80,40:90]=255;Image.fromarray(a).save(mask,'PNG')
    server=serve(0,tmp_path);url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        body=_multipart({'photo':('p.png',source.getvalue(),'image/png'),'mask':('a.png',mask.getvalue(),'image/png'),'variants':('',b'2','text/plain')})
        request=urllib.request.Request(url+'/run',data=body,headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
        response=urllib.request.urlopen(request).read().decode();assert 'SUPPLIED MASK' in response
        job=re.search(r'name="job" value="([a-f0-9]+)"',response).group(1)
        request=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode({'job':job,'template':'clean_studio','variants':2,'route':'cutout_compose'}).encode())
        response=urllib.request.urlopen(request).read().decode();assert 'SUPPLIED MASK' in response
        job2=re.search(r'name="job" value="([a-f0-9]+)"',response).group(1)
        assert (tmp_path/job2/'mask.png').read_bytes()==mask.getvalue()
        for manifest in (tmp_path/job2).glob('*/manifest.json'):
            assert json.loads(manifest.read_text())['cutout']['backend']=='supplied_mask'
    finally:server.shutdown()


@pytest.mark.parametrize("model",["u2netp","u2net"])
def test_estimated_mask_sidecar_is_hash_bound(tmp_path,model):
    import hashlib
    src=tmp_path/'src.png';Image.new('RGB',(120,100),(30,60,90)).save(src)
    a=np.zeros((100,120),np.uint8);a[20:80,40:90]=255;mask=tmp_path/'estimated-alpha.png';Image.fromarray(a).save(mask)
    sidecar=tmp_path/'mask-manifest.json'
    meta={'backend':model+'_experimental','alpha_sha256':hashlib.sha256(mask.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()};sidecar.write_text(json.dumps(meta))
    def run():return run_pipeline(EditRequest(src,get_template('clean_studio'),mask_path=mask),[Route.CUTOUT_COMPOSE],RunConfig(variants=1,out_root=tmp_path/'runs'))
    result=run();assert json.loads(result.manifest_path.read_text())['mask_provenance']=='estimated_'+model+'_experimental'
    assert 'ESTIMATED '+model.upper()+'_EXPERIMENTAL MASK' in result.variants[0].qa.notes
    meta['alpha_sha256']='0'*64;sidecar.write_text(json.dumps(meta))
    assert json.loads(run().manifest_path.read_text())['mask_provenance']=='supplied_mask'


def test_real_cli_module_run_does_not_crash_after_saving(tmp_path):
    import subprocess,sys,os
    from pathlib import Path
    src=tmp_path/'src.png';Image.new('RGB',(120,100),(30,60,90)).save(src)
    a=np.zeros((100,120),np.uint8);a[20:80,40:90]=255;mask=tmp_path/'mask.png';Image.fromarray(a).save(mask)
    result=subprocess.run([sys.executable,'-m','renderello.cli','run','--source',str(src),'--mask',str(mask),'--templates','clean_studio','--routes','cutout_compose','--variants','1','--out',str(tmp_path/'runs')],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert 'fidelity' in result.stdout
