"""Recoverable, no-overwrite pair publication. A marker commits both hashes.

Three filenames cannot be atomically committed together. Consumers must validate
before use. Staging and partial destinations stay on failure for explicit recovery.
"""
from pathlib import Path
import hashlib,json,os,tempfile

SCHEMA='evidence-pair.v1'


def _bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def _hash(data):return hashlib.sha256(data).hexdigest()


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>10_000_000:
        raise ValueError('bounded regular evidence file required: '+str(path))
    return path.read_bytes()


def _directory_sync_mode():
    # Python's Windows CRT cannot open directories for fsync. Do not catch
    # arbitrary PermissionError: those still indicate real failures on POSIX.
    return 'file_fsync_only_windows' if os.name == 'nt' else 'file_and_directory_fsync'


def _sync_dir(path):
    if _directory_sync_mode() == 'file_fsync_only_windows':
        return
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _write(path,data):
    with path.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())


def _paths(output):
    return [output,Path(str(output)+'.references.json'),Path(str(output)+'.complete.json')]


def _valid_commit(commit, output):
    required = {'schema', 'output', 'payload_sha256', 'references_sha256'}
    return (isinstance(commit, dict) and set(commit) in (required, required | {'persistence'})
        and commit['schema'] == SCHEMA and commit['output'] == str(output)
        and ('persistence' not in commit or commit['persistence'] in
             ('file_fsync_only_windows', 'file_and_directory_fsync')))


def validate_pair(output):
    output=Path(output).absolute();payload,refs,marker=_paths(output)
    commit=json.loads(_read(marker))
    if not _valid_commit(commit, output):
        raise ValueError('invalid pair completion marker')
    if _hash(_read(payload))!=commit['payload_sha256'] or _hash(_read(refs))!=commit['references_sha256']:
        raise ValueError('pair hash mismatch; do not consume')
    return commit


def recover_pair(stage,output):
    stage=Path(stage).absolute();output=Path(output).absolute()
    if stage.is_symlink() or not stage.is_dir() or stage.parent!=output.parent:
        raise ValueError('same-directory regular staging directory required')
    commit=json.loads(_read(stage/'commit.json'))
    if not _valid_commit(commit, output):
        raise ValueError('staging destination/commit mismatch')
    sources=[stage/'payload.json',stage/'references.json',stage/'commit.json']
    data=[_read(s) for s in sources]
    if _hash(data[0])!=commit['payload_sha256'] or _hash(data[1])!=commit['references_sha256']:
        raise ValueError('staged evidence hash mismatch')
    # Preflight ALL conflicts before publishing any remaining file. Never replace.
    destinations=_paths(output)
    for target,expected in zip(destinations,data):
        if os.path.lexists(target) and _read(target)!=expected:
            raise ValueError('existing evidence differs; preserve it: '+str(target))
    for source,target,expected in zip(sources,destinations,data):
        try:os.link(source,target)
        except FileExistsError:
            if _read(target)!=expected:raise ValueError('concurrent evidence conflict; preserve it: '+str(target))
        _sync_dir(output.parent)
    return validate_pair(output)


def export_pair(output,payload,references):
    output=Path(output).absolute()
    if any(os.path.lexists(p) for p in _paths(output)):
        raise ValueError('preserve earlier output/reference/completion files')
    # Serialize both BEFORE staging, so invalid values cannot leave half a pair.
    p,r=_bytes(payload),_bytes(references)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.stage-',dir=output.parent))
    try:
        _write(stage/'payload.json',p);_write(stage/'references.json',r)
        _write(stage/'commit.json',_bytes(dict(schema=SCHEMA,output=str(output),payload_sha256=_hash(p),references_sha256=_hash(r),persistence=_directory_sync_mode())))
        _sync_dir(stage);_sync_dir(output.parent)
        recover_pair(stage,output)
    except (OSError,ValueError) as exc:
        raise ValueError('incomplete export; retained staging '+str(stage)+'; validate/recover explicitly: '+str(exc)) from exc
    return stage
