from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from PIL import Image
from renderello.reviewed_chromatic_floor_cli import run,main


def setup(tmp):
    rgb=np.full((20,20,3),255,np.uint8);rgb[10,8:12]=[170,130,120]
    a=np.zeros((20,20),np.uint8);a[2:9,5:15]=255
    s=np.zeros_like(a);s[9:13,5:15]=255
    paths=[]
    for n,x in [('source',rgb),('alpha',a),('selection',s)]:
        p=tmp/(n+'.png');Image.fromarray(x).save(p);paths.append(p)
    kw={n+'_sha256':hashlib.sha256(p.read_bytes()).hexdigest()for n,p in zip(['source','alpha','selection'],paths)}
    kw.update(background_rgb=[235,225,215],reviewed=True,review_note='Synthetic reviewed unit test')
    return paths,kw


def test_saved_result_and_inputs(tmp_path):
    paths,kw=setup(tmp_path);dest=tmp_path/'out';m=run(*paths,dest,**kw)
    assert not (dest/'INCOMPLETE').exists()
    assert json.loads((dest/'manifest.json').read_text())==m
    assert m['provider_calls']==0 and m['review']['accepted'] is False
    for n,p in zip(['source','alpha','selection'],paths):assert (dest/(n+'.png')).read_bytes()==p.read_bytes()
    with pytest.raises(FileExistsError):run(*paths,dest,**kw)


@pytest.mark.parametrize('key',['source_sha256','alpha_sha256','selection_sha256'])
def test_bad_binding_no_output(tmp_path,key):
    paths,kw=setup(tmp_path);kw[key]='0'*64;dest=tmp_path/'out'
    with pytest.raises(ValueError):run(*paths,dest,**kw)
    assert not dest.exists()


@pytest.mark.parametrize('key,value',[('reviewed',False),('review_note',''),('background_rgb',[150,150,150]),('background_rgb',[True,255,255])])
def test_rejected_review_no_output(tmp_path,key,value):
    paths,kw=setup(tmp_path);kw[key]=value;dest=tmp_path/'out'
    with pytest.raises(ValueError):run(*paths,dest,**kw)
    assert not dest.exists()


def test_symlink_input_refused(tmp_path):
    paths,kw=setup(tmp_path);link=tmp_path/'link.png';link.symlink_to(paths[0]);paths[0]=link
    with pytest.raises(OSError):run(*paths,tmp_path/'out',**kw)


def test_partial_write_keeps_incomplete(tmp_path,monkeypatch):
    paths,kw=setup(tmp_path);dest=tmp_path/'out'
    def fail(*args,**kwargs):raise OSError('simulated export failure')
    monkeypatch.setattr(Image.Image,'save',fail)
    with pytest.raises(OSError):run(*paths,dest,**kw)
    assert (dest/'INCOMPLETE').exists() and not (dest/'manifest.json').exists()


def test_cli(tmp_path,capsys):
    paths,kw=setup(tmp_path);args=[]
    for n,p in zip(['source','alpha','selection'],paths):args.extend(['--'+n,str(p)])
    args.extend(['--out',str(tmp_path/'out'),'--reviewed','--review-note',kw['review_note'],'--background-rgb','235','225','215'])
    for n in ['source_sha256','alpha_sha256','selection_sha256']:args.extend(['--'+n.replace('_','-'),kw[n]])
    assert main(args)==0 and json.loads(capsys.readouterr().out)['provider_calls']==0
