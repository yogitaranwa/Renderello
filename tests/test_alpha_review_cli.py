import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from renderello.alpha_review_cli import export_bound_review


@pytest.fixture
def fixture(tmp_path):
    source = Image.fromarray(np.arange(40*50*3,dtype=np.uint16).reshape(40,50,3).astype('uint8'))
    alpha = Image.new('L',(50,40),255)
    source.save(tmp_path/'source.png');alpha.save(tmp_path/'alpha.png')
    request = dict(schema_version=1,native_size=[50,40],source_rgb_sha256=hashlib.sha256(source.tobytes()).hexdigest(),baseline_alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),protected_regions={'label':[20,20,30,30]},patches=[dict(operation='exclude',polygon=[[1,1],[8,1],[8,8],[1,8]],source_review_note='Source boundary inspected. Test annotation only.')])
    path=tmp_path/'annotation.json';path.write_text(json.dumps(request))
    return tmp_path,request,path


def run(f):
    root,request,path=f;path.write_text(json.dumps(request))
    return export_bound_review(root/'source.png',root/'alpha.png',path,root/'output')


def test_correct_binding_exports_review_only(fixture):
    m=run(fixture);root,_,_=fixture
    assert m['alpha_changed_pixels']==64
    assert not m['quality_accepted'] and not m['production_review_created']
    assert (root/'output/manifest.json').is_file()
    assert not (root/'output/reviewed-alpha.png').exists()


@pytest.mark.parametrize('kind',['source','alpha'])
def test_one_pixel_change_refused_before_output(fixture,kind):
    root,_,_=fixture;p=root/(kind+'.png');im=Image.open(p);a=np.array(im);a[0,0]=0;Image.fromarray(a).save(p)
    with pytest.raises(ValueError,match='hash mismatch'):run(fixture)
    assert not (root/'output').exists()


@pytest.mark.parametrize('field,value',[
 ('schema_version',True),('schema_version',2),('native_size',[40,50]),
 ('native_size',[True,40]),('native_size',[50.0,40]),
 ('native_size',[100000,100000]),('source_rgb_sha256','x'*64),
 ('source_rgb_sha256','A'*64),('baseline_alpha_sha256','short')])
def test_bad_binding_refused(fixture,field,value):
    root,request,_=fixture;request[field]=value
    with pytest.raises(ValueError):run(fixture)
    assert not (root/'output').exists()


@pytest.mark.parametrize('kind',['source','alpha'])
def test_silent_mode_conversion_forbidden(fixture,kind):
    root,_,_=fixture;p=root/(kind+'.png');Image.open(p).convert('RGBA' if kind=='source' else 'RGB').save(p)
    with pytest.raises(ValueError,match='mode|alpha'):run(fixture)
    assert not (root/'output').exists()


def test_existing_output_preserved(fixture):
    root,_,_=fixture;(root/'output').mkdir();(root/'output/keep.txt').write_text('unchanged')
    with pytest.raises(ValueError,match='preserve'):run(fixture)
    assert (root/'output/keep.txt').read_text()=='unchanged'


def test_broken_output_symlink_refused(fixture):
    root,_,_=fixture;(root/'output').symlink_to(root/'absent',target_is_directory=True)
    with pytest.raises(ValueError,match='preserve'):run(fixture)
    assert not (root/'absent').exists()


def test_request_cannot_add_acceptance_field(fixture):
    root,request,_=fixture;request['quality_accepted']=True
    with pytest.raises(ValueError,match='exact'):run(fixture)
    assert not (root/'output').exists()


def test_invalid_patch_refused(fixture):
    root,request,_=fixture;request['patches'][0]['polygon']=[[20,20],[25,20],[25,25],[20,25]]
    with pytest.raises(ValueError,match='protected'):run(fixture)
    assert not (root/'output').exists()


def test_command_line_refusal_and_success(fixture):
    root,request,path=fixture;env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
    cmd=[sys.executable,'-m','renderello.alpha_review_cli','--source',str(root/'source.png'),'--alpha',str(root/'alpha.png'),'--request',str(path),'--output',str(root/'output')]
    result=subprocess.run(cmd,capture_output=True,text=True,env=env)
    assert result.returncode==0;assert json.loads(result.stdout)['quality_accepted'] is False
    result=subprocess.run(cmd,capture_output=True,text=True,env=env)
    assert result.returncode==2 and 'preserve' in result.stderr
