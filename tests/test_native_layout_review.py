from pathlib import Path
import json
import pytest
from PIL import Image
from renderello.blind_review import Candidate
from renderello.native_layout_review import build_native_pack,verify_native_pack,digest


def inputs(tmp_path):
    paths=[]
    for index,size in enumerate([(40,30),(33,40),(26,45)]):
        p=tmp_path/f'i{index}.png';Image.new('RGB',size,(index*60,1,40)).save(p);paths.append(p)
    return paths


def test_native_sizes_and_encoded_originals_preserved(tmp_path):
    s,a,b=inputs(tmp_path);out=build_native_pack(s,[Candidate('one',a),Candidate('two',b)],tmp_path/'out',seed=8)
    record=verify_native_pack(out);assert record['outputs']==2
    manifest=json.loads((out/'REVIEW/manifest.json').read_text())
    assert manifest['source_size']==[40,30] and manifest['registration']=='NOT_ESTABLISHED'
    assert not manifest['pixelwise_comparison_permitted'] and manifest['ratings']=='BLANK'
    assert (out/'REVIEW/source.png').read_bytes()==s.read_bytes()
    assert {tuple(x['native_size']) for x in manifest['candidates']}=={(33,40),(26,45)}
    assert {digest(p) for p in (out/'REVIEW').glob('C*.png')}=={digest(a),digest(b)}
    assert 'one' not in (out/'REVIEW/manifest.json').read_text()
    assert 'one' in (out/'PRIVATE/key.json').read_text()


def test_failure_retained_and_seed_reproducible(tmp_path):
    s,a,_=inputs(tmp_path);c=[Candidate('one',a),Candidate('failed',None,'native download unavailable')]
    x=build_native_pack(s,c,tmp_path/'x',seed=4);y=build_native_pack(s,c,tmp_path/'y',seed=4)
    assert verify_native_pack(x)['planned']==2 and verify_native_pack(x)['outputs']==1
    assert (x/'REVIEW/manifest.json').read_bytes()==(y/'REVIEW/manifest.json').read_bytes()
    assert 'native download unavailable' in (x/'REVIEW/manifest.json').read_text()


@pytest.mark.parametrize('seed',[True,False,1.2,None,'1'])
def test_bad_seed_no_partial_write(tmp_path,seed):
    s,a,b=inputs(tmp_path)
    with pytest.raises(ValueError):build_native_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=seed)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['same_output','source_duplicate','method_duplicate','bad_type','one_candidate','symlink','corrupt'])
def test_bad_candidates_refuse_before_output(tmp_path,kind):
    s,a,b=inputs(tmp_path);c=[Candidate('a',a),Candidate('b',b)]
    if kind=='same_output':c[1]=Candidate('b',a)
    elif kind=='source_duplicate':c[1]=Candidate('b',s)
    elif kind=='method_duplicate':c[1]=Candidate('a',b)
    elif kind=='bad_type':c[1]='wrong'
    elif kind=='one_candidate':c=c[:1]
    elif kind=='symlink':q=tmp_path/'link.png';q.symlink_to(b);c[1]=Candidate('b',q)
    elif kind=='corrupt':b.write_text('invalid image')
    with pytest.raises(Exception):build_native_pack(s,c,tmp_path/'out',seed=1)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('kind',['edit','extra','missing','symlink','path_escape','empty_manifest'])
def test_verify_rejects_tampered_pack(tmp_path,kind):
    s,a,b=inputs(tmp_path);out=build_native_pack(s,[Candidate('a',a),Candidate('b',b)],tmp_path/'out',seed=1)
    q=out/'REVIEW/C01.png';complete=out/'COMPLETE.json'
    if kind=='edit':q.write_bytes(b'changed')
    elif kind=='extra':(out/'REVIEW/extra.txt').write_text('extra')
    elif kind=='missing':q.unlink()
    elif kind=='symlink':q.unlink();q.symlink_to(a)
    elif kind=='path_escape':j=json.loads(complete.read_text());j['review_sha256']['../a.png']='0'*64;complete.write_text(json.dumps(j))
    else:complete.write_text('{"review_sha256":{}}')
    with pytest.raises(ValueError):verify_native_pack(out)


def test_no_overwrite(tmp_path):
    s,a,b=inputs(tmp_path);out=tmp_path/'out';out.mkdir();(out/'keep.txt').write_text('keep')
    with pytest.raises(FileExistsError):build_native_pack(s,[Candidate('a',a),Candidate('b',b)],out,seed=1)
    assert (out/'keep.txt').read_text()=='keep'


@pytest.mark.parametrize('method',[None,4,[],True])
def test_candidate_method_must_be_text(method):
    with pytest.raises(ValueError):Candidate(method,Path('image.png'))


@pytest.mark.parametrize('failure',[None,4,[],False])
def test_candidate_failure_must_be_text(failure):
    with pytest.raises(ValueError):Candidate('one',None,failure)


def test_digest_byte_cap(tmp_path):
    p=tmp_path/'large.dat'
    with p.open('wb') as f:f.truncate(40_000_001)
    with pytest.raises(ValueError,match='byte budget'):digest(p)


def test_changed_copy_leaves_no_complete_marker(tmp_path,monkeypatch):
    import renderello.native_layout_review as module
    s,a,b=inputs(tmp_path);copy=module.shutil.copyfile
    def changed(src,dst):
        result=copy(src,dst);Path(dst).write_bytes(b'changed during copy');return result
    monkeypatch.setattr(module.shutil,'copyfile',changed)
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='changed while copying'):
        build_native_pack(s,[Candidate('a',a),Candidate('b',b)],out,seed=1)
    assert not (out/'COMPLETE.json').exists()
