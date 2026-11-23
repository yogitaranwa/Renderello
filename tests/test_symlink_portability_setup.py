from pathlib import Path
import os
import pytest


def failure(code):
    def fail(*args, **kwargs):
        error=OSError('simulated privilege failure')
        error.winerror=code
        raise error
    return fail


def test_explicit_fixture_skips_only_windows_1314(monkeypatch,tmp_path):
    monkeypatch.setattr(getattr(Path,'_accessor',os),'symlink',failure(1314))
    with pytest.raises(pytest.skip.Exception):
        (tmp_path/'link').symlink_to(tmp_path/'target')


@pytest.mark.parametrize('name',['_pytest.pathlib','renderello.example'])
def test_non_test_callers_keep_original_oserror(monkeypatch,tmp_path,name):
    monkeypatch.setattr(getattr(Path,'_accessor',os),'symlink',failure(1314))
    with pytest.raises(OSError):
        exec("p.symlink_to(q)",{'__name__':name,'p':tmp_path/'link','q':tmp_path/'target'})


def test_other_oserrors_are_not_skipped(monkeypatch,tmp_path):
    monkeypatch.setattr(getattr(Path,'_accessor',os),'symlink',failure(5))
    with pytest.raises(OSError):
        (tmp_path/'link').symlink_to(tmp_path/'target')
