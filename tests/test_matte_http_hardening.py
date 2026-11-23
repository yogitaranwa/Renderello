"""HTTP boundary tests with real application/studio and controlled mattes.
No provider or network call is permitted. Quote-only preflight is stubbed.
"""
import importlib
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from renderello.matte_integrity import matte_stats, assess
from renderello.schemas import CutoutResult

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'webapp'))
import auth

@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setenv('RENDERELLO_RUNS', str(tmp_path))
    monkeypatch.setenv('RENDERELLO_ACCESS_CODE', 'test-local-only')
    monkeypatch.setenv('RENDERELLO_COOKIE_SECURE', '0')
    for k in ('RENDERELLO_LOGIN_EMAIL', 'RENDERELLO_LOGIN_PASSWORD_HASH', 'MAGIC_HOUR_API_KEY'):
        monkeypatch.delenv(k, raising=False)
    import main
    main = importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    auth._attempts.clear()
    assert c.post('/api/login', json={'code': 'test-local-only'}).status_code == 200
    import renderello.studio as studio
    monkeypatch.setattr(studio, '_enforce_budget', lambda *a, **k: None)
    calls = []
    def forbidden(*a, **k):
        calls.append('forbidden')
        raise AssertionError('reservation/paid/network call forbidden')
    monkeypatch.setattr(studio, 'reserve_budget', forbidden)
    import requests
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    def fake_cutouts(paths, out_dir, **kw):
        out_dir.mkdir(parents=True, exist_ok=True)
        alpha = np.zeros((200, 200), np.uint8)
        for y in (10, 80, 150):
            for x in (10, 80, 150):
                alpha[y:y+20, x:x+20] = 255
        p = out_dir / 'matte.png'; Image.fromarray(alpha).save(p)
        rgba = out_dir / 'rgba.png'; Image.fromarray(np.dstack([np.zeros((200,200,3),np.uint8),alpha])).save(rgba)
        return [SimpleNamespace(path=paths[0], error=None, cutout=CutoutResult(rgba,p,(0,0,200,200),'controlled',.09))]
    monkeypatch.setattr(studio, 'cutout_products', fake_cutouts)
    yield main, c, studio, calls, fake_cutouts
    auth._attempts.clear()


def upload(c, **kw):
    b = io.BytesIO(); Image.new('RGB', (200,200), 'gray').save(b,'PNG')
    data = dict(preset='clean_studio', provider='magichour', mh_route='scene', variants='1')
    data.update(kw)
    return c.post('/api/generate', data=data, files={'products':('source.png', b.getvalue(), 'image/png')})


def terminal(c, response):
    assert response.status_code == 200, response.text
    for _ in range(200):
        j = c.get('/api/jobs/' + response.json()['job_id']).json()
        if j['status'] != 'running': return j
        time.sleep(.01)
    pytest.fail('job never became terminal')


def test_gate_http_preview_no_spend(harness):
    main,c,studio,calls,_ = harness
    j = terminal(c, upload(c))
    assert j['status'] == 'failed' and j['error'].startswith('matte review required')
    assert 'no credits' in j['error']
    review = j['matte_review']
    url = '/api/runs/' + review['run_dir'] + '/' + review['previews'][0]
    r = c.get(url)
    assert r.status_code == 200 and r.headers['content-type'] == 'image/png'
    assert Image.open(io.BytesIO(r.content)).size == (400,200)
    from fastapi.testclient import TestClient
    assert TestClient(main.app).get(url).status_code == 401
    assert calls == []


def test_override_explicit_and_whole_unaffected(harness, monkeypatch):
    main,c,studio,calls,_ = harness
    reached=[]
    def stop(*a, **kw):
        reached.append(1)
        raise RuntimeError('Magic Hour test stop before provider')
    monkeypatch.setattr(studio, '_run_magichour_scene', stop)
    for value in ('0', 'true', 'yes', ''):
        assert 'matte_review' in terminal(c, upload(c, knob_matte_reviewed=value))
    j = terminal(c,upload(c,knob_matte_reviewed='1'))
    assert j['error'] == 'Magic Hour test stop before provider' and reached == [1]
    monkeypatch.setattr(studio, '_run_magichour', stop)
    j = terminal(c,upload(c,mh_route='whole'))
    assert j['error'] == 'Magic Hour test stop before provider' and len(reached) == 2
    assert calls == []


def test_concurrent_job_slot_and_failure_cleanup(harness,monkeypatch):
    main,c,studio,calls,real = harness
    started=threading.Event(); release=threading.Event()
    def blocked(*a, **kw):
        started.set(); assert release.wait(5); return real(*a,**kw)
    monkeypatch.setattr(studio,'cutout_products',blocked)
    r=upload(c)
    assert started.wait(2)
    try: assert upload(c).status_code == 429
    finally: release.set()
    assert terminal(c,r)['status'] == 'failed'
    assert terminal(c,upload(c))['status'] == 'failed'
    assert calls == []


@pytest.mark.parametrize('data',[b'',b'not-an-image',b'\x89PNG\r\n\x1a\n'+b'0'*64,b'\xff\xd8\xff'+b'0'*64])
def test_corrupt_images_rejected(harness,data):
    _,c,_,calls,_ = harness
    r=c.post('/api/generate',data={'preset':'clean_studio','provider':'magichour'},files={'products':('bad.png',data,'image/png')})
    assert r.status_code == 415
    assert calls == []


@pytest.mark.parametrize('bad',[np.zeros((0,0),np.uint8),np.zeros((2,2,3),np.uint8),np.zeros((2,2),float)])
def test_bad_mask_contract(bad):
    with pytest.raises(ValueError): matte_stats(bad)


def test_empty_full_tiny_thin_and_false_negatives():
    assert assess(matte_stats(np.zeros((100,100),np.uint8)))
    assert assess(matte_stats(np.full((100,100),255,np.uint8)))
    tiny=np.zeros((100,100),np.uint8); tiny[50,50]=255
    assert assess(matte_stats(tiny)) is None  # tiny valid product not auto-rejected
    thin=np.zeros((100,100),np.uint8); thin[:,50]=255
    assert assess(matte_stats(thin)) is None
    # Known false negatives: a single retained object when others disappeared,
    # or a merged foreground/background region. Not a fidelity detector.
    omitted=np.zeros((100,100),np.uint8); omitted[30:70,30:70]=255
    assert assess(matte_stats(omitted)) is None
    merged=np.zeros((100,100),np.uint8); merged[10:90,10:90]=255
    assert assess(matte_stats(merged)) is None


def test_tiny_product_below_rounded_coverage_still_passes():
    tiny=np.zeros((1000,1000),np.uint8);tiny[500,500]=255
    st=matte_stats(tiny)
    assert st['coverage']==0 and st['foreground_pixels']==1
    assert assess(st) is None


def test_missing_cutout_rejected_before_reservation(harness,monkeypatch):
    main,c,studio,calls,_=harness
    monkeypatch.setattr(studio,'cutout_products',lambda paths,*a,**k:[SimpleNamespace(path=paths[0],error='test cutout failed',cutout=None)])
    j=terminal(c,upload(c))
    assert j['error'].startswith('no product cutout')
    assert calls==[]


def test_multiple_sources_rejected_before_reservation(harness):
    main,c,studio,calls,_=harness
    b=io.BytesIO();Image.new('RGB',(200,200),'gray').save(b,'PNG')
    r=c.post('/api/generate',data={'preset':'clean_studio','provider':'magichour','mh_route':'scene'},files=[('products',('a.png',b.getvalue(),'image/png')),('products',('b.png',b.getvalue(),'image/png'))])
    j=terminal(c,r)
    assert j['error'].startswith('scene-compose requires exactly one')
    assert calls==[]
