from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
from PIL import Image
import pytest
from renderello.native_region_cli import run


def setup(tmp):
    rgb=np.full((80,100,3),[30,100,150],np.uint8);a=np.zeros((80,100),np.uint8);a[20:60,25:75]=255
    src=tmp/'source.png';alpha=tmp/'prior.png';spec=tmp/'spec.json'
    Image.fromarray(rgb).save(src);Image.fromarray(a).save(alpha)
    data=dict(source_file_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),alpha_file_sha256=hashlib.sha256(alpha.read_bytes()).hexdigest(),roi=[10,10,90,70],reviewed=True,review_note='synthetic test ROI',material='opaque',method='guided',edge_colors=False,feather=4)
    spec.write_text(json.dumps(data));return src,alpha,spec,data


def test_native_file_entrypoint_preserves_native_rgb_and_no_overwrite(tmp_path):
    src,alpha,spec,data=setup(tmp_path);out=tmp_path/'out';before=src.read_bytes()
    r=run(src,alpha,spec,out)
    assert r['source_rgb_exact'] and not r['resampled'] and r['live_calls']==0
    assert src.read_bytes()==before and (out/'original-source.bin').read_bytes()==before
    with Image.open(out/'candidate.png') as im:
        assert im.mode=='RGBA' and im.size==(100,80)
        with Image.open(src) as s:assert np.array_equal(np.array(im)[...,:3],np.array(s))
    with pytest.raises(FileExistsError):run(src,alpha,spec,out)


@pytest.mark.parametrize('change',['wronghash','unreviewed','unknownfield','edge_bool','rotated','wrongmask'])
def test_file_inputs_refused_before_output(tmp_path,change):
    src,alpha,spec,data=setup(tmp_path);out=tmp_path/'out'
    if change=='wronghash':data['source_file_sha256']='f'*64
    elif change=='unreviewed':data['reviewed']=False
    elif change=='unknownfield':data['automatic_accept']=True
    elif change=='edge_bool':data['edge_colors']='false'
    elif change=='rotated':
        with Image.open(src) as im:
            ex=Image.Exif();ex[274]=6;im.save(src,exif=ex)
        data['source_file_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
    elif change=='wrongmask':
        Image.new('RGB',(100,80)).save(alpha);data['alpha_file_sha256']=hashlib.sha256(alpha.read_bytes()).hexdigest()
    spec.write_text(json.dumps(data))
    with pytest.raises(ValueError):run(src,alpha,spec,out)
    assert not out.exists()


@pytest.mark.parametrize('command',['brief-economics','blind-review-return','blind-review-pack','native-region-matte'])
def test_main_cli_routes_discoverable(command):
    r=subprocess.run([sys.executable,'-m','renderello.cli',command,'--help'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert '--' in r.stdout


def test_main_cli_native_real_output(tmp_path):
    src,alpha,spec,_=setup(tmp_path);out=tmp_path/'out'
    cmd=[sys.executable,'-m','renderello.cli','native-region-matte','--source',str(src),'--alpha',str(alpha),'--spec',str(spec),'--out',str(out)]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['outside_roi_rgb_alpha_exact']
    assert subprocess.run(cmd,capture_output=True).returncode==2


def test_main_cli_build_pack_then_return_roundtrip(tmp_path):
    import csv
    paths=[]
    for i,color in enumerate(['red','green','blue']):
        p=tmp_path/f'{i}.png';Image.new('RGB',(20,20),color).save(p);paths.append(p)
    m=tmp_path/'manifest.json';m.write_text(json.dumps(dict(source='0.png',seed=4,candidates=[dict(method='a',path='1.png'),dict(method='b',path='2.png')])));out=tmp_path/'pack'
    r=subprocess.run([sys.executable,'-m','renderello.cli','blind-review-pack','--manifest',str(m),'--out',str(out)],capture_output=True,text=True);assert r.returncode==0,r.stderr
    returned=tmp_path/'returned.csv';returned.write_bytes((out/'REVIEW/scores.csv').read_bytes());report=tmp_path/'report.json'
    r=subprocess.run([sys.executable,'-m','renderello.cli','blind-review-return','--pack',str(out),'--returned-csv',str(returned),'--provenance','assistant_judgment','--review-reference','test','--output',str(report)],capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert json.loads(report.read_text())['complete_reviews']==0
