from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from renderello import reviewed_source_floor_cli as d

@pytest.fixture
def setup(tmp_path):
    rgb=np.full((9,11,3),245,np.uint8);rgb[2:5,3:7]=[50,60,70]
    a=np.zeros((9,11),np.uint8);a[2:5,3:7]=255
    s=np.zeros_like(a);s[5:8,2:9]=255
    source,alpha,selection=[tmp_path/n for n in ['source.png','alpha.png','selection.png']]
    Image.fromarray(rgb).save(source,icc_profile=b'test-profile')
    Image.fromarray(a).save(alpha);Image.fromarray(s).save(selection)
    kw=dict(source_sha256=d.sha256_file(source),alpha_sha256=d.sha256_file(alpha),selection_sha256=d.sha256_file(selection),background_rgb=[235,228,216],reviewed=True,review_note='Reviewed empty white floor only, no physical-light claim.')
    return source,alpha,selection,tmp_path/'out',kw,rgb,a,s

def test_native_exports_preserve_opaque_profile_and_unselected(setup):
    source,alpha,selection,out,kw,rgb,a,s=setup
    r=d.run_files(source,alpha,selection,out,**kw)
    image=Image.open(out/'reviewed-floor.png');v=np.array(image);base=np.array(Image.open(out/'no-floor.png'))
    assert image.size==(11,9) and image.info['icc_profile']==b'test-profile'
    assert np.array_equal(v[a==255],rgb[a==255])
    assert np.array_equal(v[s==0],base[s==0])
    assert r['original_alpha_unchanged'] and not r['physical_relighting'] and not r['accepted']
    assert json.loads((out/'review.json').read_text())['paid_calls']==0

@pytest.mark.parametrize('key,value',[('reviewed',False),('review_note',''),('source_sha256','f'*64),('alpha_sha256','BAD'),('selection_sha256','f'*64),('background_rgb',[159,230,230]),('background_rgb',[True,230,230]),('background_rgb',[230,230])])
def test_invalid_inputs_produce_no_output(setup,key,value):
    source,alpha,selection,out,kw,*_=setup;kw[key]=value
    with pytest.raises(ValueError):d.run_files(source,alpha,selection,out,**kw)
    assert not out.exists()

def test_no_overwrite(setup):
    source,alpha,selection,out,kw,*_=setup;out.mkdir();(out/'keep').write_text('preserved')
    with pytest.raises(ValueError,match='preserve'):d.run_files(source,alpha,selection,out,**kw)
    assert (out/'keep').read_text()=='preserved'

@pytest.mark.parametrize('mode',['L','RGBA'])
def test_source_mode_not_converted(setup,mode):
    source,alpha,selection,out,kw,*_=setup;Image.open(source).convert(mode).save(source);kw['source_sha256']=d.sha256_file(source)
    with pytest.raises(ValueError,match='no conversion'):d.run_files(source,alpha,selection,out,**kw)

def test_native_shape_mismatch(setup):
    source,alpha,selection,out,kw,*_=setup;Image.new('L',(8,8)).save(alpha);kw['alpha_sha256']=d.sha256_file(alpha)
    with pytest.raises(ValueError,match='dimensions'):d.run_files(source,alpha,selection,out,**kw)

def test_non_neutral_floor_refuses(setup):
    source,alpha,selection,out,kw,rgb,a,s=setup;rgb[s>0]=[200,180,230];Image.fromarray(rgb).save(source);kw['source_sha256']=d.sha256_file(source)
    with pytest.raises(ValueError,match='not neutral'):d.run_files(source,alpha,selection,out,**kw)
    assert not out.exists()

def test_cli_requires_review_flag_before_run(monkeypatch):
    monkeypatch.setattr(d,'run_files',lambda *a,**kw:pytest.fail('must not run'))
    with pytest.raises(SystemExit):d.main(['--source','s','--alpha','a','--selection','r','--output','o','--source-sha256','a'*64,'--alpha-sha256','b'*64,'--selection-sha256','c'*64,'--background-rgb','230','230','230','--review-note','Review'])


@pytest.mark.parametrize('changed_index',[0,1,2])
def test_decode_is_hash_bound_even_if_path_changes(setup,monkeypatch,changed_index):
    source,alpha,selection,out,kw,rgb,a,s=setup
    original=d._snapshot; paths=[source,alpha,selection]
    def snapshot_and_replace(path):
        data=original(path)
        if path==paths[changed_index]:
            Image.new('RGB' if changed_index==0 else 'L',(11,9),0).save(path)
        return data
    monkeypatch.setattr(d,'_snapshot',snapshot_and_replace)
    r=d.run_files(source,alpha,selection,out,**kw)
    saved=np.array(Image.open(out/'reviewed-floor.png'))
    assert np.array_equal(saved[a==255],rgb[a==255])
    assert r['input_read_semantics']=='bounded_immutable_byte_snapshots_hashed_before_decode'
    assert r['file_bindings']==dict(source=kw['source_sha256'],alpha=kw['alpha_sha256'],selection=kw['selection_sha256'])

def test_input_snapshot_byte_bound_refuses_without_export(setup,monkeypatch):
    source,alpha,selection,out,kw,*_=setup
    monkeypatch.setattr(d,'MAX_INPUT_BYTES',8)
    with pytest.raises(ValueError,match='bounded snapshot'):
        d.run_files(source,alpha,selection,out,**kw)
    assert not out.exists()
