import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from renderello.checked_research import native_matting as n


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inputs(tmp):
    s, m = tmp / 's.png', tmp / 'm.png'
    rgb = np.arange(40*32*3, dtype=np.uint16).reshape(40,32,3).astype('uint8')
    mask = np.zeros((40,32), np.uint8); mask[8:32,8:24] = 255
    Image.fromarray(rgb).save(s); Image.fromarray(mask).save(m)
    return s,m,rgb,mask


def fake(rgb, tri):
    return np.full(tri.shape, .4, np.float32), {'inference_seconds':0}


def invoke(tmp, **kw):
    s,m,rgb,mask=inputs(tmp)
    return n.run(s,m,tmp/'out',source_sha256=digest(s),mask_sha256=digest(m),predictor=fake,**kw)


def test_actual_files_export_recomposes_exactly(tmp_path):
    result=invoke(tmp_path)
    out=tmp_path/'out'; rgb=np.array(Image.open(out/'source-native.png')); a=np.array(Image.open(out/'alpha.png'))
    tri=np.array(Image.open(out/'derived-trimap.png'))
    assert (a[tri==255]==255).all() and (a[tri==0]==0).all()
    rgba=np.array(Image.open(out/'component-rgba.png'))
    assert np.array_equal(rgba[:,:,:3],rgb) and np.array_equal(rgba[:,:,3],a)
    for name,bg in [('dark',35),('light',242)]:
        aa=a.astype(np.uint32)[:,:,None]
        expected=((rgb.astype(np.uint32)*aa+bg*(255-aa)+127)//255).astype('uint8')
        assert np.array_equal(np.array(Image.open(out/(name+'.png'))),expected)
    assert result['metrics']['boundary_accuracy'] is None and not result['metrics']['publication_accepted']
    assert result['predictor']['unverified_injected_predictor']
    assert result['composite_alpha']=='exported_quantized_uint8'
    for name,rec in result['outputs'].items(): assert digest(out/name)==rec['sha256']
    assert json.loads((out/'manifest.json').read_text())==result


def test_refuses_existing_directory_and_does_not_call_predictor(tmp_path):
    s,m,_,_=inputs(tmp_path);out=tmp_path/'out';out.mkdir();(out/'keep.txt').write_text('original')
    with pytest.raises(ValueError,match='exists'):
        n.run(s,m,out,source_sha256=digest(s),mask_sha256=digest(m),predictor=lambda *a:pytest.fail('must not run'))
    assert (out/'keep.txt').read_text()=='original'


@pytest.mark.parametrize('which',['source','mask'])
def test_input_hash_mismatch_refuses_before_write(tmp_path,which):
    s,m,_,_=inputs(tmp_path);h={'source_sha256':digest(s),'mask_sha256':digest(m)};h[which+'_sha256']='0'*64
    with pytest.raises(ValueError,match='hash mismatch'): n.run(s,m,tmp_path/'out',predictor=fake,**h)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('bad',['',None,'a'*63,'A'*64,'z'*64])
def test_malformed_hash_refused(tmp_path,bad):
    s,m,_,_=inputs(tmp_path)
    with pytest.raises(ValueError,match='SHA-256'):n.read_bound(s,m,bad,digest(m))


@pytest.mark.parametrize('kind',['rgba','mask_rgb','dimensions','fractional','empty','full','large'])
def test_source_contract_refused(tmp_path,kind):
    s,m,rgb,mask=inputs(tmp_path)
    if kind=='rgba':Image.open(s).convert('RGBA').save(s)
    if kind=='mask_rgb':Image.open(m).convert('RGB').save(m)
    if kind=='dimensions':Image.new('L',(31,40)).save(m)
    if kind=='fractional':mask[0,0]=128;Image.fromarray(mask).save(m)
    if kind=='empty':Image.new('L',(32,40),0).save(m)
    if kind=='full':Image.new('L',(32,40),255).save(m)
    if kind=='large':Image.new('RGB',(1001,1000)).save(s);Image.new('L',(1001,1000)).save(m)
    with pytest.raises(ValueError):n.run(s,m,tmp_path/'out',source_sha256=digest(s),mask_sha256=digest(m),predictor=fake)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('radius',[0,17,True,2.5,float('nan')])
def test_bad_radius(radius):
    with pytest.raises(ValueError):n.derive_trimap(np.zeros((10,10),np.uint8),radius)


@pytest.mark.parametrize('mask',[np.zeros((10,10),np.uint8),np.ones((10,10),np.uint8)*255,np.zeros((10,10),float),np.zeros((3,3,3),np.uint8)])
def test_no_known_region_or_invalid_mask(mask):
    with pytest.raises(ValueError):n.derive_trimap(mask,3)


@pytest.mark.parametrize('raw',[np.array([[np.nan]],np.float32),np.array([[np.inf]],np.float32),np.array([[-.001]],np.float32),np.array([[1.001]],np.float32),np.array([[128]],np.uint8),np.zeros((2,2),float)])
def test_invalid_prediction_refused(raw):
    with pytest.raises(ValueError):n.checked_alpha(raw,np.array([[128]],np.uint8))


@pytest.mark.parametrize('tri',[np.array([[1]],np.uint8),np.array([[128]],np.int16),np.zeros((1,1,1),np.uint8)])
def test_bad_trimap_refused(tri):
    with pytest.raises(ValueError):n.checked_alpha(np.zeros((1,1),np.float32),tri)


def test_known_regions_enforced_and_input_not_modified():
    raw=np.full((1,3),.5,np.float32);a,a8=n.checked_alpha(raw,np.array([[0,128,255]],np.uint8))
    assert np.array_equal(raw,np.full((1,3),.5,np.float32))
    assert a.tolist()==[[0,.5,1]] and a8.tolist()==[[0,128,255]]


@pytest.mark.parametrize('bg',[-1,256,True,1.1])
def test_bad_background(bg):
    with pytest.raises(ValueError):n.composite(np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint8),bg)


def test_failed_predictor_leaves_no_output(tmp_path):
    s,m,_,_=inputs(tmp_path)
    def bad(*a):return np.full((40,32),np.nan),{}
    with pytest.raises(ValueError):n.run(s,m,tmp_path/'out',source_sha256=digest(s),mask_sha256=digest(m),predictor=bad)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('threads',[0,5,True,1.5])
def test_bad_cpu_thread_count_before_loading(threads):
    with pytest.raises(ValueError):n.CpuPredictor(threads)
