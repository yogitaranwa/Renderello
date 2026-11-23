"""Offline fault injection; never invokes a provider or downloads models."""
from pathlib import Path
import json,os,subprocess,sys
import pytest
from renderello import evidence_pair as ep


def test_success_is_committed_and_repeat_is_no_overwrite(tmp_path):
    out=tmp_path/'a.json';stage=ep.export_pair(out,{'attempts':[]},{'records':[]})
    assert ep.validate_pair(out)['schema']=='evidence-pair.v1'
    assert stage.is_dir() and json.loads(out.read_text())=={'attempts':[]}
    before={p:p.read_bytes() for p in ep._paths(out)}
    with pytest.raises(ValueError,match='preserve earlier'):ep.export_pair(out,{}, {})
    assert all(p.read_bytes()==v for p,v in before.items())
    ep.recover_pair(stage,out) # idempotent explicit recovery


@pytest.mark.parametrize('step',range(1,4))
def test_staging_write_failure_never_publishes_partial(tmp_path,monkeypatch,step):
    out=tmp_path/'a.json';real=ep._write;calls=0
    def fail(path,data):
        nonlocal calls
        calls+=1
        if calls==step:raise OSError('fixture ENOSPC')
        return real(path,data)
    monkeypatch.setattr(ep,'_write',fail)
    with pytest.raises(ValueError,match='retained staging'):ep.export_pair(out,{'a':1},{'r':1})
    assert not any(os.path.lexists(p) for p in ep._paths(out))
    stages=list(tmp_path.glob('.a.json.stage-*'));assert len(stages)==1
    with pytest.raises((ValueError,OSError)):ep.validate_pair(out)
    with pytest.raises((ValueError,OSError)):ep.recover_pair(stages[0],out)


@pytest.mark.parametrize('step',range(1,4))
def test_link_failure_keeps_evidence_and_recovers_exact_pair(tmp_path,monkeypatch,step):
    out=tmp_path/'a.json';real=ep.os.link;calls=0
    def fail(source,target):
        nonlocal calls
        calls+=1
        if calls==step:raise OSError('fixture permission denied')
        return real(source,target)
    monkeypatch.setattr(ep.os,'link',fail)
    with pytest.raises(ValueError,match='retained staging'):ep.export_pair(out,{'a':1},{'r':1})
    stage=next(tmp_path.glob('.a.json.stage-*'))
    assert len([p for p in ep._paths(out) if p.exists()])==step-1
    with pytest.raises((ValueError,OSError)):ep.validate_pair(out)
    kept={p:p.read_bytes() for p in ep._paths(out) if p.exists()}
    monkeypatch.setattr(ep.os,'link',real);ep.recover_pair(stage,out)
    assert all(p.read_bytes()==v for p,v in kept.items())
    ep.validate_pair(out)


@pytest.mark.parametrize('step',range(1,6))
def test_directory_fsync_failure_is_not_reported_success(tmp_path,monkeypatch,step):
    out=tmp_path/'a.json';real=ep._sync_dir;calls=0
    def fail(path):
        nonlocal calls
        calls+=1
        if calls==step:raise OSError('fixture sync failure')
        return real(path)
    monkeypatch.setattr(ep,'_sync_dir',fail)
    with pytest.raises(ValueError,match='incomplete export'):ep.export_pair(out,{'a':1},{'r':1})
    stage=next(tmp_path.glob('.a.json.stage-*'));monkeypatch.setattr(ep,'_sync_dir',real)
    ep.recover_pair(stage,out);ep.validate_pair(out)


@pytest.mark.parametrize('which',range(3))
def test_existing_files_even_identical_never_overwritten(tmp_path,which):
    out=tmp_path/'a.json';target=ep._paths(out)[which];target.write_bytes(b'keep')
    with pytest.raises(ValueError,match='preserve earlier'):ep.export_pair(out,{}, {})
    assert target.read_bytes()==b'keep' and not list(tmp_path.glob('.*stage-*'))


def test_dangling_symlink_is_not_free_destination(tmp_path):
    out=tmp_path/'a.json';out.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):ep.export_pair(out,{}, {})
    assert out.is_symlink() and not (tmp_path/'missing').exists()


def failed_stage(tmp_path,monkeypatch):
    out=tmp_path/'a.json';real=ep.os.link
    monkeypatch.setattr(ep.os,'link',lambda *a:(_ for _ in ()).throw(OSError('fixture')))
    with pytest.raises(ValueError):ep.export_pair(out,{'a':1},{'r':1})
    monkeypatch.setattr(ep.os,'link',real)
    return out,next(tmp_path.glob('.a.json.stage-*'))


@pytest.mark.parametrize('name',['payload.json','references.json','commit.json'])
def test_corrupt_staging_refused_before_any_publish(tmp_path,monkeypatch,name):
    out,stage=failed_stage(tmp_path,monkeypatch);(stage/name).write_text('{}')
    with pytest.raises(ValueError):ep.recover_pair(stage,out)
    assert not any(p.exists() for p in ep._paths(out))


def test_recovery_refuses_wrong_destination_or_conflict_without_changes(tmp_path,monkeypatch):
    out,stage=failed_stage(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='mismatch'):ep.recover_pair(stage,tmp_path/'other.json')
    ref=ep._paths(out)[1];ref.write_text('precious earlier evidence')
    with pytest.raises(ValueError,match='differs'):ep.recover_pair(stage,out)
    assert not out.exists() and ref.read_text()=='precious earlier evidence'


@pytest.mark.parametrize('which',range(3))
def test_concurrent_destination_conflict_cannot_clobber(tmp_path,monkeypatch,which):
    out=tmp_path/'a.json';real=ep.os.link;target=ep._paths(out)[which]
    def race(source,dest):
        if dest==target:dest.write_bytes(b'concurrent evidence')
        real(source,dest)
    monkeypatch.setattr(ep.os,'link',race)
    with pytest.raises(ValueError,match='concurrent evidence conflict'):ep.export_pair(out,{}, {})
    assert target.read_bytes()==b'concurrent evidence'


@pytest.mark.parametrize('which',range(3))
def test_completed_pair_corruption_detected(tmp_path,which):
    out=tmp_path/'a.json';ep.export_pair(out,{'a':1},{'r':1});ep._paths(out)[which].write_text('{}')
    with pytest.raises(ValueError):ep.validate_pair(out)


def test_serialization_failure_leaves_no_stage(tmp_path):
    with pytest.raises(ValueError):ep.export_pair(tmp_path/'a.json',{'a':float('nan')},{})
    assert not list(tmp_path.iterdir())


def test_cli_validate_recover_and_refusal(tmp_path,monkeypatch):
    out,stage=failed_stage(tmp_path,monkeypatch)
    cmd=[sys.executable,'-m','renderello.cli','provider-attempts','--output',str(out)]
    assert subprocess.run(cmd+['--validate-output'],capture_output=True).returncode==2
    r=subprocess.run(cmd+['--recover-stage',str(stage)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert subprocess.run(cmd+['--validate-output'],capture_output=True).returncode==0
    assert subprocess.run(cmd+['--input','unused','--recover-stage',str(stage)],capture_output=True).returncode==2


@pytest.mark.parametrize('step',range(1,4))
def test_actual_partial_staging_write_is_preserved_and_not_recoverable(tmp_path,monkeypatch,step):
    out=tmp_path/'a.json';real=ep._write;calls=0
    def partial(path,data):
        nonlocal calls
        calls+=1
        if calls==step:
            path.write_bytes(data[:1]);raise OSError('fixture partial write')
        return real(path,data)
    monkeypatch.setattr(ep,'_write',partial)
    with pytest.raises(ValueError):ep.export_pair(out,{'a':1},{'r':1})
    stage=next(tmp_path.glob('.a.json.stage-*'))
    assert len(list(stage.iterdir()))==step
    assert not any(p.exists() for p in ep._paths(out))
    with pytest.raises((ValueError,OSError)):ep.recover_pair(stage,out)


@pytest.mark.parametrize('step',range(1,4))
def test_file_fsync_failure_keeps_staged_bytes_without_publishing(tmp_path,monkeypatch,step):
    out=tmp_path/'a.json';real=ep.os.fsync;calls=0
    def fail(fd):
        nonlocal calls
        calls+=1
        if calls==step:raise OSError('fixture file sync failure')
        return real(fd)
    monkeypatch.setattr(ep.os,'fsync',fail)
    with pytest.raises(ValueError):ep.export_pair(out,{'a':1},{'r':1})
    assert next(tmp_path.glob('.a.json.stage-*')).is_dir()
    assert not any(p.exists() for p in ep._paths(out))


def test_economics_refuses_incomplete_pair_before_building_report(tmp_path,monkeypatch):
    from renderello.brief_economics import from_payload
    out=tmp_path/'economics-input.json'
    payload={'evidence_pair_required':True}
    # Deliberately incomplete publication with only the payload linked.
    real=ep.os.link;calls=0
    def fail(source,target):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('fixture interruption after payload')
        real(source,target)
    monkeypatch.setattr(ep.os,'link',fail)
    with pytest.raises(ValueError):ep.export_pair(out,payload,{'records':[]})
    with pytest.raises(ValueError,match='evidence_path'):from_payload(payload)
    with pytest.raises(ValueError,match='regular evidence'):from_payload(payload,evidence_path=out)
    cmd=[sys.executable,'-m','renderello.cli','brief-economics','--input',str(out),'--output',str(tmp_path/'report.json')]
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode==2 and 'REFUSED' in r.stderr and not (tmp_path/'report.json').exists()


def test_economics_cannot_validate_other_payload(tmp_path):
    from renderello.brief_economics import from_payload
    out=tmp_path/'a.json';ep.export_pair(out,{'evidence_pair_required':True},{})
    with pytest.raises(ValueError,match='differs'):from_payload({'evidence_pair_required':True,'extra':1},evidence_path=out)


def test_windows_sync_mode_never_opens_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, '_directory_sync_mode', lambda: 'file_fsync_only_windows')
    monkeypatch.setattr(ep.os, 'open', lambda *a: (_ for _ in ()).throw(AssertionError('directory open')))
    ep._sync_dir(tmp_path)


def test_posix_permission_error_is_not_suppressed(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, '_directory_sync_mode', lambda: 'file_and_directory_fsync')
    monkeypatch.setattr(ep.os, 'open', lambda *a: (_ for _ in ()).throw(PermissionError('real directory failure')))
    with pytest.raises(PermissionError, match='real directory failure'):
        ep._sync_dir(tmp_path)


def test_windows_pair_records_weaker_directory_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(ep, '_directory_sync_mode', lambda: 'file_fsync_only_windows')
    out = tmp_path/'windows.json'
    ep.export_pair(out, {'a': 1}, {'r': 1})
    assert ep.validate_pair(out)['persistence'] == 'file_fsync_only_windows'


def test_legacy_marker_without_persistence_still_validates(tmp_path):
    out = tmp_path/'legacy.json'
    ep.export_pair(out, {}, {})
    marker = ep._paths(out)[2]
    value = json.loads(marker.read_text()); value.pop('persistence')
    marker.write_text(json.dumps(value))
    assert 'persistence' not in ep.validate_pair(out)


@pytest.mark.parametrize('value', ['unknown', None, False, {}, []])
def test_invalid_persistence_declaration_refused(tmp_path, value):
    out = tmp_path/'invalid-mode.json'; stage = ep.export_pair(out, {}, {})
    for path in [ep._paths(out)[2], stage/'commit.json']:
        commit = json.loads(path.read_text()); commit['persistence'] = value
        path.write_text(json.dumps(commit))
    with pytest.raises(ValueError, match='invalid pair'):
        ep.validate_pair(out)
    with pytest.raises(ValueError, match='commit mismatch'):
        ep.recover_pair(stage, out)
