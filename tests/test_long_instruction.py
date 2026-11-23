"""User text must reach the application intact, including its final clauses."""
import io
from pathlib import Path
import urllib.request,urllib.parse,urllib.error
import pytest
from renderello import webui
from renderello.prompts import build_prompt,validate_instruction
from renderello.schemas import EditRequest
from renderello.templates import get_template
from tests.test_webui import _multipart

LONG=('Diffuse neutral window light on a broad plain surface. '*80)+'FINAL_CLAUSE_KEEP_EMPTY_RIGHT'

def test_assembly_retains_full_long_instruction():
    p=build_prompt(EditRequest(Path('x.png'),get_template('clean_studio'),custom_instruction=LONG))
    assert LONG in p
    assert 'FINAL_CLAUSE_KEEP_EMPTY_RIGHT' in p

@pytest.mark.parametrize('text',['x'*16001,None,42])
def test_instruction_rejected_explicitly_before_work(text):
    with pytest.raises(ValueError):validate_instruction(text)


def test_product_change_after_old_limit_still_rejected():
    with pytest.raises(ValueError,match='alter the product'):
        validate_instruction('soft light '*40+'change the label')

@pytest.mark.parametrize('route',['/run','/rerun'])
@pytest.mark.parametrize('oversize',[False,True])
def test_web_routes_never_silently_cut_text(tmp_path,monkeypatch,route,oversize):
    captured=[]
    def run(self,src,key,instruction,*args,**kwargs):
        captured.append(instruction);return tmp_path,tmp_path/'source.png',None
    monkeypatch.setattr(webui._Handler,'_run_job',run)
    monkeypatch.setattr(webui._Handler,'_results_page',lambda *a:b'accepted')
    server=webui.serve(0,tmp_path)
    try:
        text='x'*16001 if oversize else LONG
        if route=='/run':
            body=_multipart({'photo':('p.png',b'source','image/png'),'instruction':('',text.encode(),'text/plain')})
            ctype='multipart/form-data; boundary=XBOUND'
        else:
            folder=tmp_path/'123456abcdef';folder.mkdir();(folder/'source.png').write_bytes(b'source')
            body=urllib.parse.urlencode({'job':folder.name,'instruction':text}).encode();ctype='application/x-www-form-urlencoded'
        req=urllib.request.Request(f'http://127.0.0.1:{server.server_address[1]}'+route,data=body,headers={'Content-Type':ctype})
        if oversize:
            with pytest.raises(urllib.error.HTTPError) as exc:urllib.request.urlopen(req)
            assert exc.value.code==400
            assert '16000' in exc.value.read().decode();assert captured==[]
        else:
            assert urllib.request.urlopen(req).read()==b'accepted';assert captured==[LONG]
    finally:server.shutdown();server.server_close()


def test_ui_uses_multiline_larger_explicit_limit():
    for form in [webui._PAGE,webui._RERUN]:
        assert '<textarea' in form
        assert 'maxlength="16000"' in form
        assert '280' not in form


def test_harmonize_also_retains_full_instruction():
    from renderello.prompts import build_harmonize_prompt
    assert LONG in build_harmonize_prompt(EditRequest(Path('x.png'),get_template('clean_studio'),custom_instruction=LONG))


def test_cli_instruction_file_reads_full_utf8_before_work(tmp_path,monkeypatch):
    from renderello import cli
    from renderello.schemas import RunResult
    import json
    captured=[]
    def fake_run(request,*args,**kwargs):
        captured.append(request.custom_instruction)
        return RunResult(request,[],tmp_path,tmp_path/'manifest.json')
    monkeypatch.setattr(cli,'run_pipeline',fake_run)
    instruction=tmp_path/'prompt.txt';instruction.write_text(LONG+' café',encoding='utf-8')
    assert cli.main(['run','--source','missing.png','--instruction-file',str(instruction),'--out',str(tmp_path)])==0
    assert captured==[LONG+' café']


def test_cli_scene_plate_calls_declared_geometry(tmp_path,monkeypatch):
    from renderello import cli,scene_plate
    captured=[]
    def compose(*args,**kwargs):captured.append((args,kwargs));return {'photographic_acceptance':'NOT_EVALUATED'}
    monkeypatch.setattr(scene_plate,'compose_plate',compose)
    assert cli.main(['scene-plate','--prepared','prepared','--background','scene.png','--baseline','940','--cast-offset','70','18','--out',str(tmp_path/'candidate')])==0
    assert captured[0][1]==dict(baseline=940,center_x=512,cast_offset=[70.,18.],contact=True,support_plane_path=None,contact_strength=1.,scene_exposure_ev=0.,scene_white_balance=(1.,1.,1.),background_kind='unspecified_inspected_background')
