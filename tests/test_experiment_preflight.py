from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest
from PIL import Image
from renderello.experiment_preflight import validate_plan,main

@pytest.fixture
def setup(tmp_path):
 p=tmp_path/'source.png';Image.new('RGB',(61,63),'red').save(p)
 record=dict(id='a',input_file='source.png',input_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),decoded_size=[61,63],output_canvas=[64,64],native_translation=[1,0],source_resampled=False,canvas_only_padding=True)
 plan=dict(cases=[dict(id='a-clean',source_id='a',source='source.png',width=64,height=64,translation=[1,0],prompt='Full prompt kept.',template=dict(key='clean',aspect='1:1'))],seeds=[0,1],planned_outputs=2,sources=[record]);return plan,tmp_path

def test_success_no_mutation(setup):
 p,r=setup;before=deepcopy(p);files={x.name:x.read_bytes()for x in r.iterdir()};result=validate_plan(p,r)
 assert result['planned_outputs']==2 and result['all_template_aspects_fulfilled']
 assert p==before and files=={x.name:x.read_bytes()for x in r.iterdir()}
 assert all(result[k]is False for k in ['generation_executed','quality_accepted','execution_authorized','source_rights_verified','cost_verified','files_modified'])

def test_aspect_mismatch_is_explicit(setup):
 p,r=setup;p['cases'][0]['template']['aspect']='4:5';x=validate_plan(p,r);assert x['all_template_aspects_fulfilled']is False;assert x['case_checks'][0]['requested_aspect_fulfilled']is False

def test_unused_provenance_not_counted(setup):
 p,r=setup;p['sources'].append(dict(id='unexecuted'));x=validate_plan(p,r);assert x['executed_source_ids']==['a'];assert x['unused_provenance_source_ids']==['unexecuted']

@pytest.mark.parametrize('bad',[None,True,0,16,2.0,'2',-1,float('nan')])
def test_stale_or_wrong_count_refused(setup,bad):
 p,r=setup;p['planned_outputs']=bad
 with pytest.raises(ValueError):validate_plan(p,r)

@pytest.mark.parametrize('bad',[[],[0,0],[True],[1.0],[-1],[2**63],list(range(17))])
def test_bad_seeds(setup,bad):
 p,r=setup;p['seeds']=bad
 with pytest.raises(ValueError):validate_plan(p,r)

@pytest.mark.parametrize('key,bad',[('width',True),('width',63),('width',8193),('height',0),('translation',[-1,0]),('translation',[1,2]),('prompt',''),('source','../source.png'),('source_id','missing')])
def test_case_contradictions(setup,key,bad):
 p,r=setup;p['cases'][0][key]=bad
 with pytest.raises(ValueError):validate_plan(p,r)

@pytest.mark.parametrize('key,bad',[('decoded_size',[60,63]),('output_canvas',[96,64]),('native_translation',[0,0]),('source_resampled',True),('canvas_only_padding',False),('input_file_sha256','0'*64)])
def test_record_contradictions(setup,key,bad):
 p,r=setup;p['sources'][0][key]=bad
 with pytest.raises(ValueError):validate_plan(p,r)

@pytest.mark.parametrize('aspect',[None,'0:1','1:0','101:1','x:1','1:1:1','1.5:1'])
def test_bad_aspect(setup,aspect):
 p,r=setup;p['cases'][0]['template']['aspect']=aspect
 with pytest.raises(ValueError):validate_plan(p,r)

def test_duplicate_cases_refused(setup):
 p,r=setup;p['cases']*=2;p['planned_outputs']=4
 with pytest.raises(ValueError):validate_plan(p,r)

def test_duplicate_logical_case_different_id_refused(setup):
 p,r=setup;c=deepcopy(p['cases'][0]);c['id']='other';p['cases'].append(c);p['planned_outputs']=4
 with pytest.raises(ValueError):validate_plan(p,r)

def test_symlink_refused(setup):
 p,r=setup;(r/'link.png').symlink_to(r/'source.png');p['cases'][0]['source']=p['sources'][0]['input_file']='link.png'
 with pytest.raises(ValueError):validate_plan(p,r)

def test_clipping_refused(setup):
 p,r=setup;p['cases'][0]['translation']=p['sources'][0]['native_translation']=[4,0]
 with pytest.raises(ValueError):validate_plan(p,r)

def test_cli_reject_preserves_frozen_plan(setup,capsys):
 p,r=setup;p['planned_outputs']=16;f=r/'plan.json';f.write_text(json.dumps(p));before=f.read_bytes();assert main([str(f),'--root',str(r)])==2;assert f.read_bytes()==before;assert json.loads(capsys.readouterr().out)['validation']=='REJECTED'

def test_cli_success(setup,capsys):
 p,r=setup;f=r/'plan.json';f.write_text(json.dumps(p));assert main([str(f),'--root',str(r)])==0;assert json.loads(capsys.readouterr().out)['planned_outputs']==2

def test_duplicate_json_keys_cli_rejected(setup,capsys):
 p,r=setup;f=r/'bad.json';f.write_text('{"cases": [], "cases": []}');assert main([str(f),'--root',str(r)])==2;assert 'duplicate JSON key' in capsys.readouterr().out

def test_missing_image_cli_rejected(setup,capsys):
 p,r=setup;(r/'source.png').unlink();f=r/'plan.json';f.write_text(json.dumps(p));assert main([str(f),'--root',str(r)])==2;assert json.loads(capsys.readouterr().out)['validation']=='REJECTED'

def test_corrupt_image_cli_rejected(setup,capsys):
 p,r=setup;(r/'source.png').write_bytes(b'not an image');p['sources'][0]['input_file_sha256']=hashlib.sha256(b'not an image').hexdigest();f=r/'plan.json';f.write_text(json.dumps(p));assert main([str(f),'--root',str(r)])==2;assert json.loads(capsys.readouterr().out)['validation']=='REJECTED'

def test_different_templates_can_share_source(setup):
 p,r=setup;c=deepcopy(p['cases'][0]);c['id']='second';c['template']['key']='outdoor';p['cases'].append(c);p['planned_outputs']=4;assert validate_plan(p,r)['planned_outputs']==4

def test_duplicate_source_record_refused(setup):
 p,r=setup;p['sources']*=2
 with pytest.raises(ValueError):validate_plan(p,r)
