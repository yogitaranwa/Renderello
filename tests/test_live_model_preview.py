"""Non-spending preview must distinguish arithmetic budget from live model permission."""
import pytest
from fastapi.testclient import TestClient
import webapp.main as wm


@pytest.fixture
def client(monkeypatch):
    import renderello.studio as st
    monkeypatch.setattr(wm, '_require_auth', lambda request: None)
    monkeypatch.setattr(st, 'budget_status', lambda model: dict(allocated=1530,spent=1525,remaining=5,max_batch=120,credits_per_image=5 if model=='flux-2-klein' else 30))
    # Preview is not a live validation or generation endpoint.
    monkeypatch.setattr(st, 'reserve_budget', lambda *a,**kw: pytest.fail('Estimate reserved credits'))
    from renderello.backends.magichour import MagicHourBackend
    monkeypatch.setattr(MagicHourBackend, 'balance', lambda *a,**kw: pytest.fail('Estimate queried provider account'))
    return TestClient(wm.app)


def test_cheap_disallowed_model_not_mistaken_for_live_eligibility(client,monkeypatch):
    monkeypatch.setenv('RENDERELLO_MH_LIVE_MODELS','seedream-v5-flash')
    data=client.get('/api/estimate',params=dict(provider='magichour',mh_model='flux-2-klein',variants=1)).json()
    assert data['within_budget'] is True
    assert data['model_allowed'] is False
    assert data['live_checks_verified'] is False
    assert 'anchor' in data['estimate_scope']


def test_allowed_model_still_subject_to_budget(client,monkeypatch):
    monkeypatch.setenv('RENDERELLO_MH_LIVE_MODELS','seedream-v5-flash')
    data=client.get('/api/estimate',params=dict(provider='magichour',mh_model='seedream-v5-flash',variants=1)).json()
    assert data['model_allowed'] is True and data['within_budget'] is False
    assert data['live_checks_verified'] is False


def test_configured_empty_allowlist_means_no_live_model(client,monkeypatch):
    monkeypatch.setenv('RENDERELLO_MH_LIVE_MODELS','')
    assert client.get('/api/estimate',params=dict(provider='magichour',mh_model='seedream-v5-flash',variants=1)).json()['model_allowed'] is False


def test_provider_catalogue_retains_prices_but_exposes_live_models(client,monkeypatch):
    monkeypatch.setenv('RENDERELLO_MH_LIVE_MODELS','seedream-v5-flash,nano-banana-2.1')
    monkeypatch.setattr(wm,'provider_report',lambda:[dict(name='magichour',available=True,models={'flux-2-klein':5,'seedream-v5-flash':30})])
    row=client.get('/api/providers').json()['providers'][0]
    assert row['live_models']==['nano-banana-2.1','seedream-v5-flash']
    assert row['models']['flux-2-klein']==5


def test_default_allowlist_remains_flash(client,monkeypatch):
    monkeypatch.delenv('RENDERELLO_MH_LIVE_MODELS',raising=False)
    assert client.get('/api/estimate',params=dict(provider='magichour',mh_model='seedream-v5-flash',variants=1)).json()['model_allowed'] is True


def test_procedural_remains_free_without_model_permission(client,monkeypatch):
    monkeypatch.setenv('RENDERELLO_MH_LIVE_MODELS','')
    data=client.get('/api/estimate',params=dict(provider='procedural',variants=1)).json()
    assert data['free'] is True and data['credits']==0
