import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from renderello.checked_research.lighting_pair_cli import execute, parser


def setup_case(tmp_path):
    names = dict(source=tmp_path/'source.png', alpha=tmp_path/'alpha.png',
                 left_gain=tmp_path/'left.npy', right_gain=tmp_path/'right.npy')
    Image.fromarray(np.full((12, 16, 3), 128, np.uint8)).save(names['source'])
    Image.fromarray(np.full((12, 16), 255, np.uint8)).save(names['alpha'])
    left = np.ones((12,16), np.float32); left[:, :8] = 2
    right = np.ones((12,16), np.float32); right[:, 8:] = 2
    np.save(names['left_gain'], left); np.save(names['right_gain'], right)
    fields = {k:str(v) for k,v in names.items()}
    fields.update({k+'_sha256':hashlib.sha256(v.read_bytes()).hexdigest() for k,v in names.items()})
    fields.update(output_dir=str(tmp_path/'out'), aligned_pair_reviewed=True)
    return argparse.Namespace(**fields)


def test_direction_and_manifest(tmp_path):
    args=setup_case(tmp_path); m=execute(args)
    assert m['metrics']['contrast_change_right_minus_left'] == -2
    assert json.loads((tmp_path/'out/manifest.json').read_text()) == m
    assert m['quality_accuracy'] is None and m['publication_accepted'] is False
    assert m['source_identity_independently_verified'] is False
    assert m['alignment_review_is_caller_declaration'] is True
    assert len(m['inputs']['left_gain']['sha256'])==64


@pytest.mark.parametrize('name',['source','alpha','left_gain','right_gain'])
def test_hash_refused(tmp_path,name):
    args=setup_case(tmp_path);setattr(args,name+'_sha256','0'*64)
    with pytest.raises(ValueError,match='hash'):execute(args)
    assert not (tmp_path/'out').exists()


def test_requires_alignment_declaration(tmp_path):
    args=setup_case(tmp_path);args.aligned_pair_reviewed=False
    with pytest.raises(ValueError,match='aligned pair review'):execute(args)
    assert not (tmp_path/'out').exists()


def test_existing_output_preserved(tmp_path):
    args=setup_case(tmp_path);execute(args)
    before=(tmp_path/'out/manifest.json').read_bytes()
    with pytest.raises(ValueError,match='already exists'):execute(args)
    assert (tmp_path/'out/manifest.json').read_bytes()==before


@pytest.mark.parametrize('bad',['shape','integer','nan','negative','object','archive'])
def test_gain_refused_before_output(tmp_path,bad):
    args=setup_case(tmp_path)
    arr=np.ones((12,16),np.float32)
    if bad=='shape':arr=arr[:-1]
    if bad=='integer':arr=arr.astype(np.int32)
    if bad=='nan':arr[0,0]=np.nan
    if bad=='negative':arr[0,0]=-1
    if bad=='object':arr=arr.astype(object)
    if bad=='archive':
        with open(args.left_gain,'wb') as f:np.savez(f,array=arr)
    else:np.save(args.left_gain,arr)
    args.left_gain_sha256=hashlib.sha256(open(args.left_gain,'rb').read()).hexdigest()
    with pytest.raises(ValueError):execute(args)
    assert not (tmp_path/'out').exists()


def test_parser_and_successful_cli(tmp_path):
    args=setup_case(tmp_path);fields=[]
    for name in ('source','alpha','left_gain','right_gain'):
        fields += ['--'+name.replace('_','-'),getattr(args,name),
                   '--'+name.replace('_','-')+'-sha256',getattr(args,name+'_sha256')]
    fields += ['--output-dir', args.output_dir, '--aligned-pair-reviewed']
    parsed=parser().parse_args(fields);assert vars(parsed)==vars(args)
    proc=subprocess.run([sys.executable,'-m','renderello.checked_research.lighting_pair_cli',*fields],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)['metrics']['contrast_change_right_minus_left']==-2


def test_unreviewed_cli_refusal(tmp_path):
    args=setup_case(tmp_path);fields=[]
    for name in ('source','alpha','left_gain','right_gain'):
        fields += ['--'+name.replace('_','-'),getattr(args,name),'--'+name.replace('_','-')+'-sha256',getattr(args,name+'_sha256')]
    proc=subprocess.run([sys.executable,'-m','renderello.checked_research.lighting_pair_cli',*fields,'--output-dir',args.output_dir],capture_output=True,text=True)
    assert proc.returncode==2 and 'REFUSED:' in proc.stderr
    assert not (tmp_path/'out').exists()
