import hashlib
import pytest
from renderello.cpu_mask import verified_file,estimate

def test_verified_file_hash(tmp_path):
    p=tmp_path/'code.py';p.write_bytes(b'x')
    assert verified_file(p,hashlib.sha256(b'x').hexdigest())
    with pytest.raises(ValueError,match='hash mismatch'):verified_file(p,'0'*64)

def test_bad_model_refused_before_import_or_execution(tmp_path):
    code=tmp_path/'evil.py';code.write_text("raise AssertionError('must not execute')")
    weights=tmp_path/'w.pth';weights.write_bytes(b'untrusted')
    with pytest.raises(ValueError,match='hash mismatch'):
        estimate(tmp_path/'missing.jpg',code,weights,tmp_path/'output')
    assert not (tmp_path/'output').exists()
