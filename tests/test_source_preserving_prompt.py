import pytest
from pathlib import Path
from dataclasses import replace
from renderello.source_preserving_prompt import build_source_preserving_prompt,CONTEXTS
from renderello.schemas import EditRequest
from renderello.source_contract import SourceContract
from renderello.templates import get_template
from renderello.prompts import build_prompt

@pytest.mark.parametrize('key',list(CONTEXTS))
def test_explicit_constrained_variant_never_claims_original_template_fulfilled(key):
 req=EditRequest(Path('source.jpg'),get_template(key));before=build_prompt(req)
 r=build_source_preserving_prompt(req,SourceContract('elevated','opaque','complete'))
 assert r['base_template']['camera']==req.template.camera
 assert 'original_camera_specification'in r['original_requirements_not_fulfilled']
 assert 'Do not relight or repaint'in r['prompt'] and 'do not rotate, tilt'in r['prompt']
 assert req.template.camera not in r['prompt'] and req.template.lighting not in r['prompt']
 assert not any(r[k]for k in ['executed','paid_call_authorized','polished_template','photo_accepted','source_pixel_lock_guaranteed','default_pipeline_changed'])
 assert build_prompt(req)==before

@pytest.mark.parametrize('c',[SourceContract(),SourceContract('front','unknown','complete'),SourceContract('front','opaque','unknown'),SourceContract('front','transparent','complete'),SourceContract('front','opaque','cropped')])
def test_no_silent_assumption_or_unsupported_material(c):
 with pytest.raises(ValueError):build_source_preserving_prompt(EditRequest(Path('x'),get_template('clean_studio')),c)

def test_full_long_instruction_retained():
 instruction='Keep distant green background details. '*100
 r=build_source_preserving_prompt(EditRequest(Path('x'),get_template('clean_studio'),custom_instruction=instruction),SourceContract('front','opaque','complete'))
 assert instruction.strip()in r['prompt']

def test_cli_binds_actual_source_and_never_overwrites(tmp_path):
 import hashlib,json
 from PIL import Image
 from renderello.source_preserving_prompt import main
 p=tmp_path/'source.png';Image.new('RGB',(30,40),(120,60,90)).save(p)
 before=p.read_bytes();out=tmp_path/'prompt.json'
 args=['--source',str(p),'--template','luxury_hero','--source-pose','elevated','--material','opaque','--framing','complete','--out',str(out)]
 assert main(args)==0
 r=json.loads(out.read_text());assert r['source_file_sha256']==hashlib.sha256(before).hexdigest()
 assert 'CANNOT_REPLACE_ORIGINAL_TEMPLATE_CASE'in r['benchmark_role']
 assert p.read_bytes()==before and not r['executed']
 with pytest.raises(FileExistsError):main(args)
 assert not list(tmp_path.glob('*.png'))[1:]
