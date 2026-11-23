from dataclasses import asdict
import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,PngImagePlugin
from renderello.reviewed_candidate_gate_cli import execute,load_policy
from renderello.matte_change_review import ChangeBudget

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.fixture
def files(tmp_path):
 rgb=np.full((24,32,3),127,np.uint8);a=np.zeros((24,32),np.uint8);a[5:20,5:25]=255
 edit=np.zeros_like(a);edit[5:8,5:25]=255;protect=np.zeros_like(a);protect[10:15,10:20]=255
 arrays=dict(source=rgb,before=a,candidate=np.dstack([rgb,a]),allowed=edit,protected=protect)
 paths={}
 for n,v in arrays.items():
  p=tmp_path/(n+'.png');Image.fromarray(v).save(p);paths[n]=p
 spec=dict(schema='reviewed-candidate-files.v1',files={n:sha(p)for n,p in paths.items()},reviewed=True,material='opaque',framing='complete',review_note='Synthetic exact fixture.',budget=asdict(ChangeBudget()))
 policy=tmp_path/'policy.json';policy.write_text(json.dumps(spec))
 return paths,policy,spec,arrays


def call(files,tmp_path,changes=None):
 paths,policy,spec,_=files
 if changes:
  changes(spec);policy.write_text(json.dumps(spec))
 return execute(**paths,policy=policy,policy_sha256=sha(policy),output_dir=tmp_path/'out')


def test_identity_audit_has_no_promoted_candidate(files,tmp_path):
 r=call(files,tmp_path);out=tmp_path/'out'
 assert r['decision']=='INTERVENTION_CONSTRAINTS_ONLY' and not r['photo_accepted']
 assert r['audit_only'] and not r['candidate_image_exported']
 assert set(p.name for p in out.iterdir())=={'audit.json','alpha-change-visualization.png'}
 assert json.loads((out/'audit.json').read_text())==r


def test_valid_but_out_of_scope_candidate_keeps_refusal_evidence(files,tmp_path):
 paths,policy,spec,arrays=files;v=arrays['candidate'].copy();v[11,12,3]=254;Image.fromarray(v).save(paths['candidate']);spec['files']['candidate']=sha(paths['candidate']);policy.write_text(json.dumps(spec));r=call(files,tmp_path)
 assert r['decision']=='REFUSED_INTERVENTION_SCOPE' and r['changed_protected_alpha_pixels']==1
 assert (tmp_path/'out/audit.json').exists() and not r['candidate_selected']


@pytest.mark.parametrize('n',['source','before','candidate','allowed','protected'])
def test_changed_file_binding_is_not_regenerated(files,tmp_path,n):
 paths,_,_,arrays=files;v=arrays[n].copy();v.flat[0]=17;Image.fromarray(v).save(paths[n])
 with pytest.raises(ValueError,match='hash'):call(files,tmp_path)
 assert not(tmp_path/'out').exists()


def test_changed_policy_hash(files,tmp_path):
 paths,policy,_,_=files
 with pytest.raises(ValueError,match='hash'):execute(**paths,policy=policy,policy_sha256='0'*64,output_dir=tmp_path/'out')


@pytest.mark.parametrize('key',['schema','files','reviewed','material','framing','review_note','budget'])
def test_missing_fields(files,tmp_path,key):
 with pytest.raises(ValueError):call(files,tmp_path,lambda p:p.pop(key))
 assert not(tmp_path/'out').exists()


def test_unknown_fields(files,tmp_path):
 with pytest.raises(ValueError):call(files,tmp_path,lambda p:p.update(extra=True))


@pytest.mark.parametrize('value',['0'*63,'G'*64,12,None])
def test_bad_binding_types(files,tmp_path,value):
 with pytest.raises(ValueError):call(files,tmp_path,lambda p:p['files'].update(source=value))


@pytest.mark.parametrize('control',list(asdict(ChangeBudget())))
def test_controls_must_be_explicit(files,tmp_path,control):
 with pytest.raises(ValueError):call(files,tmp_path,lambda p:p['budget'].pop(control))


def test_nonfinite_policy(files,tmp_path):
 paths,p,spec,_=files;spec['budget']['maximum_added_fraction']=float('nan');p.write_text(json.dumps(spec))
 with pytest.raises(ValueError):call(files,tmp_path)


def test_duplicate_policy_field(files,tmp_path):
 paths,p,_,_=files;text=p.read_text();p.write_text(text[:-1]+',"reviewed":true}')
 with pytest.raises(ValueError,match='duplicate'):call(files,tmp_path)


@pytest.mark.parametrize('name',['source','before','candidate','allowed','protected'])
def test_wrong_png_modes_are_not_converted(files,tmp_path,name):
 paths,p,spec,_=files;Image.new('P',(32,24)).save(paths[name]);spec['files'][name]=sha(paths[name]);p.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='exact|8bit'):call(files,tmp_path)


@pytest.mark.parametrize('name',['allowed','protected'])
def test_soft_mask_refused(files,tmp_path,name):
 paths,p,spec,arrays=files;v=arrays[name].copy();v[0,0]=128;Image.fromarray(v).save(paths[name]);spec['files'][name]=sha(paths[name]);p.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='binary'):call(files,tmp_path)


def test_metadata_refused(files,tmp_path):
 paths,p,spec,arrays=files;Image.fromarray(arrays['source']).save(paths['source'],icc_profile=b'unverified-profile');spec['files']['source']=sha(paths['source']);p.write_text(json.dumps(spec))
 with pytest.raises(ValueError,match='metadata'):call(files,tmp_path)


def test_existing_output_retained(files,tmp_path):
 d=tmp_path/'out';d.mkdir();(d/'old').write_text('preserve')
 with pytest.raises(ValueError,match='exists'):call(files,tmp_path)
 assert(d/'old').read_text()=='preserve'


def test_failed_hash_leaves_no_staging_directory(files,tmp_path):
 paths,p,spec,_=files;spec['files']['source']='0'*64;p.write_text(json.dumps(spec))
 with pytest.raises(ValueError):call(files,tmp_path)
 assert not list(tmp_path.glob('.candidate-audit-stage-*'))


def test_cli_refused_candidate_exit3_not_success(files,tmp_path):
 paths,p,spec,arrays=files;v=arrays['candidate'].copy();v[11,12,3]=254;Image.fromarray(v).save(paths['candidate']);spec['files']['candidate']=sha(paths['candidate']);p.write_text(json.dumps(spec));args=[sys.executable,'-m','renderello.reviewed_candidate_gate_cli']
 for n,v in dict(**paths,policy=p,policy_sha256=sha(p),output_dir=tmp_path/'out').items():args.extend(['--'+n.replace('_','-'),str(v)])
 result=subprocess.run(args,capture_output=True,text=True);assert result.returncode==3;assert json.loads(result.stdout)['decision']=='REFUSED_INTERVENTION_SCOPE';assert not result.stderr


def test_cli_invalid_input_exit2(files,tmp_path):
 paths,p,_,_=files;args=[sys.executable,'-m','renderello.reviewed_candidate_gate_cli']
 for n,v in dict(**paths,policy=p,policy_sha256='0'*64,output_dir=tmp_path/'out').items():args.extend(['--'+n.replace('_','-'),str(v)])
 result=subprocess.run(args,capture_output=True,text=True);assert result.returncode==2;assert 'REFUSED INPUT'in result.stderr;assert not(tmp_path/'out').exists()
