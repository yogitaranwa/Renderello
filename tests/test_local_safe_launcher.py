import importlib.util
from pathlib import Path

def test_local_launcher_disables_paid_flags_and_keys(monkeypatch):
    p=Path(__file__).resolve().parents[1]/'run_local_safe.py'
    spec=importlib.util.spec_from_file_location('safe_local',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for k in ['RENDERELLO_MH_LIVE','RENDERELLO_MH_DEV_MOCK']:monkeypatch.setenv(k,'1')
    for k in ['MAGICHOUR_API_KEY','MAGIC_HOUR_API_KEY']:monkeypatch.setenv(k,'fake')
    mod.configure_safe_local()
    import os
    assert os.environ['RENDERELLO_MH_LIVE']=='0' and os.environ['RENDERELLO_MH_DEV_MOCK']=='0'
    assert os.environ['RENDERELLO_MH_BUDGET_CREDITS']=='0'
    assert os.environ['RENDERELLO_COOKIE_SECURE']=='0'
    assert not any(os.environ.get(k)for k in ['MAGICHOUR_API_KEY','MAGIC_HOUR_API_KEY'])
