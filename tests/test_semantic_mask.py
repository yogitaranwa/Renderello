import sys,types,hashlib,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello import semantic_mask as sm


def inputs(tmp_path):
    p=tmp_path/'s.png';Image.new('RGB',(24,20),(40,50,60)).save(p);return p


def test_requires_explicit_model_download(tmp_path,monkeypatch):
    monkeypatch.setattr(sm,'cache_dir',lambda:tmp_path/'cache')
    with pytest.raises(ValueError,match='allow-download'):sm.infer_file(inputs(tmp_path),tmp_path/'out')


def test_checksum_mismatch_never_overwrites(tmp_path,monkeypatch):
    c=tmp_path/'cache';c.mkdir();p=c/'isnet-general-use.onnx';p.write_bytes(b'bad')
    monkeypatch.setattr(sm,'cache_dir',lambda:c)
    with pytest.raises(ValueError,match='checksum'):sm.infer_file(inputs(tmp_path),tmp_path/'out',True)
    assert p.read_bytes()==b'bad'


def test_explicit_session_and_source_rgb_exact(tmp_path,monkeypatch):
    c=tmp_path/'cache';c.mkdir();(c/'isnet-general-use.onnx').write_bytes(b'test')
    monkeypatch.setattr(sm,'cache_dir',lambda:c);monkeypatch.setattr(sm,'MODEL_MD5',hashlib.md5(b'test').hexdigest())
    class Session:
        def predict(self,image):return [Image.new('L',image.size,190)]
    def make(name,providers):
        assert name=='isnet-general-use';assert providers==['CPUExecutionProvider'];return Session()
    monkeypatch.setitem(sys.modules,'rembg',types.SimpleNamespace(new_session=make))
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata,'version',lambda x:'test')
    source=inputs(tmp_path);m=sm.infer_file(source,tmp_path/'out');a=np.array(Image.open(tmp_path/'out/cutout.png'))
    assert np.array_equal(a[...,:3],np.array(Image.open(source)));assert m['quality']=='NOT_EVALUATED';assert m['semantic_completeness']=='NOT_VERIFIED';assert m['model_sha256'];assert (tmp_path/'out/white-review.png').exists()
    with pytest.raises(ValueError,match='exists'):sm.infer_file(source,tmp_path/'out')


def test_source_pixel_cap(tmp_path):
    p=tmp_path/'huge.png';Image.new('RGB',(2001,2000)).save(p)
    with pytest.raises(ValueError,match='million'):sm.infer_file(p,tmp_path/'out')
