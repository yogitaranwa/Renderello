from pathlib import Path
import hashlib
import pytest
from renderello.shadow_source_patch import prepare_experimental_shadow_source,PINNED_SHADOW_SHA256

SOURCE=Path(__file__).resolve().parents[1]/'research/gps-shadow/publisher_shadow_generation.py'

def test_pinned_source_patch_preserves_original_and_compiles():
 b=SOURCE.read_bytes();out,d=prepare_experimental_shadow_source(b);assert SOURCE.read_bytes()==b;assert hashlib.sha256(b).hexdigest()==PINNED_SHADOW_SHA256;assert out!=b;compile(out,'experimental','exec');assert len(d['changes'])==4;assert not d['model_inference_tested'];assert not d['default_enabled'];assert b'base_mask_embeddings.clone()' in out;assert b'temp = pred_bbx[i,2].clone()' in out;assert b'torch.from_numpy(box_array)' in out

@pytest.mark.parametrize('b',[b'',b'fake',b'print(1)',None,'text'])
def test_unrecognized_source_fails_closed(b):
 with pytest.raises(ValueError):prepare_experimental_shadow_source(b)

def test_modified_original_cannot_be_repatched():
 out,_=prepare_experimental_shadow_source(SOURCE.read_bytes())
 with pytest.raises(ValueError):prepare_experimental_shadow_source(out)
