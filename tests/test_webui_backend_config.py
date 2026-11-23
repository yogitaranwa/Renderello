import io,json,urllib.request
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from renderello import webui
from renderello.schemas import Route,Aspect
from renderello.backends.magichour import MagicHourBackend


def minimal_previews(root):
    folder=root/'_previews';folder.mkdir(exist_ok=True)
    for t in webui.all_templates():Image.new('RGB',(4,4),'gray').save(folder/(t.key+'.png'))


def test_dashboard_backend_is_server_owned_default_mock(tmp_path,monkeypatch):
    monkeypatch.setattr(webui,'_render_previews',minimal_previews)
    server=webui.serve(0,tmp_path,editor_kind='magichour',mh_model='nano-banana-2.1')
    try:
        page=urllib.request.urlopen(f'http://127.0.0.1:{server.server_address[1]}/').read().decode()
        assert 'MOCK Magic Hour nano-banana-2.1' in page and 'zero-spend' in page
        backend=server.RequestHandlerClass.editor_factory(tmp_path)
        assert isinstance(backend,MagicHourBackend) and not backend.live
        assert backend.cost_log==tmp_path/'mh_cost_log.jsonl'
        assert backend.receipt_path==tmp_path/'mh-projects.jsonl'
    finally:server.shutdown();server.server_close()


def test_handler_uses_server_backend_and_keeps_mask_path(tmp_path,monkeypatch):
    monkeypatch.setattr(webui,'_render_previews',minimal_previews)
    captured={}
    def pipeline(req,routes,cfg,**kw):captured.update(req=req,routes=routes,kw=kw);return 'result'
    monkeypatch.setattr(webui,'run_pipeline',pipeline)
    server=webui.serve(0,tmp_path,editor_kind='magichour',mh_model='nano-banana-pro')
    try:
        handler=object.__new__(server.RequestHandlerClass)
        src=tmp_path/'src.png';Image.new('RGB',(12,12),'blue').save(src)
        mask=io.BytesIO();Image.new('L',(12,12),255).save(mask,'PNG')
        handler._run_job(src,'clean_studio','long scene instruction',Aspect.SQUARE,2,Route.PURE_PROMPT)
        assert isinstance(captured['kw']['editor'],MagicHourBackend) and not captured['kw']['editor'].live
        handler._run_job(src,'clean_studio','',Aspect.SQUARE,2,Route.CUTOUT_COMPOSE,mask_upload=('mask.png',mask.getvalue()))
        assert captured['kw']['editor'] is None and captured['req'].mask_path.read_bytes()==mask.getvalue()
    finally:server.shutdown();server.server_close()


def test_live_hybrid_is_refused_before_pipeline(tmp_path,monkeypatch):
    monkeypatch.setattr(webui,'_render_previews',minimal_previews)
    monkeypatch.setattr(webui,'run_pipeline',lambda *a,**k:pytest.fail('must not call pipeline'))
    server=webui.serve(0,tmp_path,editor_kind='magichour',live=True)
    try:
        h=object.__new__(server.RequestHandlerClass);src=tmp_path/'s.png';Image.new('RGB',(12,12),'blue').save(src)
        with pytest.raises(ValueError,match='unsupported'):h._run_job(src,'clean_studio','',Aspect.SQUARE,2,Route.HYBRID)
    finally:server.shutdown();server.server_close()


@pytest.mark.parametrize('kw',[dict(editor_kind='unknown'),dict(editor_kind='stub',live=True),dict(editor_kind='magichour',mh_model='unknown'),dict(live='false')])
def test_invalid_server_mode_refused_before_start(tmp_path,kw):
    with pytest.raises(ValueError):webui.serve(0,tmp_path,**kw)


def test_dashboard_actual_mock_generates_two_and_retains_backend(tmp_path,monkeypatch):
    monkeypatch.setattr(webui,'_render_previews',minimal_previews)
    import requests
    monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('mock must not issue network POST'))
    from tests.conftest import make_product
    src=tmp_path/'product.png';make_product(96,128).save(src)
    server=webui.serve(0,tmp_path,editor_kind='magichour',mh_model='nano-banana-2.1')
    try:
        handler=object.__new__(server.RequestHandlerClass)
        job,dst,result=handler._run_job(src,'clean_studio','studio light',Aspect.SQUARE,2,Route.PURE_PROMPT)
        report=json.loads((result.run_dir/'manifest.json').read_text())
        assert len(report['variants'])==2
        assert all(v['backend']=='magichour:nano-banana-2.1:mock' for v in report['variants'])
        assert all(v['execution_kind']['kind']=='TEST_DOUBLE_NOT_GENERATIVE_MODEL' for v in report['variants'])
        assert all(Path(v['output']).is_file() for v in report['variants'])
    finally:server.shutdown();server.server_close()
