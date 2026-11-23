import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from renderello.prompted_mask import validate_prompts,infer_file


def case(tmp_path):
 p=tmp_path/'s.png';Image.new('RGB',(30,20),(1,2,3)).save(p);d=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),image_size=[30,20],origin='reviewed_source_geometry',objects=[dict(name='product',box=[1,2,28,19],points=[[4,5]],labels=[1])]);return p,d


def test_source_binding_and_dimension(tmp_path):
 p,d=case(tmp_path);assert validate_prompts(d,p,30,20)
 d['source_sha256']='bad'
 with pytest.raises(ValueError,match='SHA256'):validate_prompts(d,p,30,20)
 p,d=case(tmp_path)
 with pytest.raises(ValueError,match='image_size'):validate_prompts(d,p,31,20)


@pytest.mark.parametrize('change',[{'origin':'automatic'},{'objects':[]},{'objects':[dict(name='a',box=[3,3,2,5])]},{'objects':[dict(name='a',box=[0,0,30,20],points=[[31,5]],labels=[1])]},{'objects':[dict(name='a',box=[0,0,30,20],points=[[1,5]],labels=[2])]},{'objects':[dict(name='a',box=[0,0,30,20],points=[[float('nan'),5]],labels=[1])]}])
def test_reject_invalid_prompts(tmp_path,change):
 p,d=case(tmp_path);d.update(change)
 with pytest.raises(ValueError):validate_prompts(d,p,30,20)


def test_checkpoint_and_existing_output_fail_closed(tmp_path):
 p,d=case(tmp_path);prompt=tmp_path/'p.json';prompt.write_text(json.dumps(d));model=tmp_path/'m.pth';model.write_bytes(b'wrong')
 with pytest.raises(ValueError,match='checkpoint'):infer_file(p,tmp_path/'out',prompt,model)
 (tmp_path/'out').mkdir()
 with pytest.raises(ValueError,match='exists'):infer_file(p,tmp_path/'out',prompt,model)


def test_overlarge_source(tmp_path):
 p=tmp_path/'s.png';Image.new('RGB',(2001,2000)).save(p)
 with pytest.raises(ValueError,match='million'):infer_file(p,tmp_path/'out',tmp_path/'p',tmp_path/'c')
