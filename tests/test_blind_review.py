from pathlib import Path
import json,csv
import pytest
from PIL import Image
from renderello.blind_review import Candidate,build_pack,verify_pack


def pictures(tmp_path):
    paths=[]
    for i,colour in enumerate(['red','green','blue']):
        p=tmp_path/f'private-method-{i}.png';Image.new('RGB',(48,32),colour).save(p);paths.append(p)
    return paths


def test_pack_blank_ratings_private_key_and_missing_denominator(tmp_path):
    s,a,b=pictures(tmp_path);out=build_pack(s,[Candidate('method secret A',a),Candidate('method secret B',b),Candidate('failed method',None,'no output')],tmp_path/'review',seed=42,regions={'label':(1,2,20,22)})
    record=verify_pack(out);assert record['candidates']==3 and record['outputs']==2
    rows=list(csv.DictReader((out/'REVIEW/scores.csv').open()));assert len(rows)==3
    assert all(r['product_label_fidelity']=='' and r['rater']=='' for r in rows)
    assert sum(r['status']=='NOT_GENERATED' for r in rows)==1
    assert 'method secret' not in (out/'REVIEW/READ_ME.txt').read_text()
    key=json.loads((out/'PRIVATE/key.json').read_text());assert len(key['candidates'])==3
    for p in (out/'REVIEW').iterdir():assert 'private-method' not in p.name
    with Image.open(out/'REVIEW/source-region1.png') as im:assert im.size==(19,20)


def test_reproducible_order_and_source_unchanged(tmp_path):
    s,a,b=pictures(tmp_path);old=s.read_bytes();entries=[Candidate('a',a),Candidate('b',b)]
    x=build_pack(s,entries,tmp_path/'x',seed=5);y=build_pack(s,entries,tmp_path/'y',seed=5)
    assert json.loads((x/'PRIVATE/key.json').read_text())==json.loads((y/'PRIVATE/key.json').read_text())
    assert s.read_bytes()==old


def test_duplicate_pixels_rejected(tmp_path):
    s,a,b=pictures(tmp_path);b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError,match='Duplicate output'):build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    assert not (tmp_path/'out').exists()


def test_shape_mismatch_not_silently_resized(tmp_path):
    s,a,b=pictures(tmp_path);Image.new('RGB',(49,32)).save(b)
    with pytest.raises(ValueError,match='dimensions differ'):build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)


@pytest.mark.parametrize('box',[(-1,0,2,2),(0,0,49,32),(2,2,1,3),(0.,0,2,2)])
def test_invalid_roi(tmp_path,box):
    s,a,b=pictures(tmp_path)
    with pytest.raises(ValueError):build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1,regions={'test':box})


def test_no_overwrite(tmp_path):
    s,a,b=pictures(tmp_path);out=build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    with pytest.raises(FileExistsError):build_pack(s,[Candidate('a',a),Candidate('b',b)],out,seed=1)
    verify_pack(out)


def test_changed_review_image_rejected(tmp_path):
    s,a,b=pictures(tmp_path);out=build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    (out/'REVIEW/C01.png').write_bytes(b'bad')
    with pytest.raises(ValueError):verify_pack(out)


@pytest.mark.parametrize('args',[('',None,'failed'),('method',None,''),('method',Path('x'),'failed')])
def test_invalid_candidate(args):
    with pytest.raises(ValueError):Candidate(*args)


def test_duplicate_methods_rejected(tmp_path):
    s,a,b=pictures(tmp_path)
    with pytest.raises(ValueError):build_pack(s,[Candidate('same',a),Candidate('same',b)],tmp_path/'out',seed=1)


def test_too_few_candidates(tmp_path):
    s,a,b=pictures(tmp_path)
    with pytest.raises(ValueError):build_pack(s,[Candidate('a',a)],tmp_path/'out',seed=1)


def test_alpha_remains_native(tmp_path):
    s,a,b=pictures(tmp_path);Image.new('RGBA',(48,32),(255,0,0,100)).save(a)
    out=build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    key=json.loads((out/'PRIVATE/key.json').read_text());token=next(x['token'] for x in key['candidates'] if x['method']=='a')
    with Image.open(out/'REVIEW'/f'{token}.png') as im:assert im.mode=='RGBA' and im.getchannel('A').tobytes()==bytes([100])*48*32


def test_manifest_cannot_read_outside_review(tmp_path):
    s,a,b=pictures(tmp_path);out=build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    p=out/'COMPLETE.json';record=json.loads(p.read_text());record['review_sha256']['../private.txt']='00';p.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='escapes'):verify_pack(out)


def test_extra_file_in_review_rejected(tmp_path):
    s,a,b=pictures(tmp_path);out=build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    (out/'REVIEW/method-identities.txt').write_text('not blinded')
    with pytest.raises(ValueError,match='Unexpected'):verify_pack(out)


def test_same_pixels_different_png_encoding_rejected(tmp_path):
    s,a,b=pictures(tmp_path)
    from PIL.PngImagePlugin import PngInfo
    info=PngInfo();info.add_text('different_encoding','same actual pixels')
    with Image.open(a) as im:im.save(b,pnginfo=info)
    assert a.read_bytes()!=b.read_bytes()
    with pytest.raises(ValueError,match='Duplicate output pixels'):build_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    assert not (tmp_path/'out').exists()
