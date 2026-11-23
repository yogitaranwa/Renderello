"""Logout revocation survives a process restart on the same host: the s1
stateless-token denylist is mirrored to disk (hash-only ids, 0600) and
reloaded, expired entries pruned. Compaction is a trigger not a cap: live
revocations are never evicted, duplicate appends are deduped, rewrites are
interval-gated and skipped when nothing is removable, and expired ids are
swept from memory. Legacy server-side tokens are unaffected."""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

import auth  # noqa: E402


@pytest.fixture
def denied_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RENDERELLO_SESSION_SECRET", "s" * 64)
    monkeypatch.setenv("RENDERELLO_DENYLIST_PATH", str(tmp_path / "revoked.jsonl"))
    auth._denied.clear()
    auth._denied_loaded = False
    auth._last_compact = 0.0
    auth._last_sweep = 0.0
    yield tmp_path / "revoked.jsonl"
    auth._denied.clear()
    auth._denied_loaded = False
    auth._last_compact = 0.0
    auth._last_sweep = 0.0


def _restart():
    """Simulate a process restart: in-memory state gone, disk remains."""
    auth._denied.clear()
    auth._denied_loaded = False


def test_logout_revocation_survives_restart(denied_env):
    tok = auth.new_session()
    assert tok.startswith("s1.")
    assert auth.valid_session(tok)
    auth.drop_session(tok)
    assert not auth.valid_session(tok)
    assert denied_env.is_file(), "denylist was not persisted"
    _restart()
    assert not auth.valid_session(tok), "revoked token valid again after restart"


def test_other_sessions_unaffected_after_restart(denied_env):
    keep = auth.new_session()
    drop = auth.new_session()
    auth.drop_session(drop)
    _restart()
    assert auth.valid_session(keep)
    assert not auth.valid_session(drop)


def test_expired_revocations_pruned_on_load(denied_env):
    expired = int(time.time()) - 10
    live = int(time.time()) + 3600
    keep_id = auth._token_id("s1.%d.keep.sig" % live)
    denied_env.write_text(
        json.dumps({"id": auth._token_id("s1.%d.old.sig" % expired),
                    "exp": expired}) + "\n"
        + json.dumps({"id": keep_id, "exp": live}) + "\n"
        + "not-json\n"
        + json.dumps({"token": "s1.legacy-raw.sig", "exp": live}) + "\n")
    _restart()
    auth._load_denied()
    assert keep_id in auth._denied
    assert len(auth._denied) == 1  # expired, corrupt and legacy raw-token lines skipped


def test_no_raw_token_on_disk(denied_env):
    tok = auth.new_session()
    auth.drop_session(tok)
    content = denied_env.read_text()
    assert tok not in content
    assert "s1." not in content  # sha256 hex ids only, no token material
    assert auth._token_id(tok) in content
    rec = json.loads(content.strip().splitlines()[-1])
    assert set(rec) == {"id", "exp"}


def test_denylist_file_mode_owner_only(denied_env):
    import stat
    auth.drop_session(auth.new_session())
    mode = stat.S_IMODE(denied_env.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_trigger_is_not_a_cap_live_revocations_grow_and_survive(denied_env, monkeypatch):
    monkeypatch.setattr(auth, "_DENYLIST_COMPACT_TRIGGER_BYTES", 300)
    monkeypatch.setattr(auth, "_DENYLIST_MAINT_INTERVAL", 0)  # every append may compact
    live = [auth.new_session() for _ in range(12)]
    for tok in live:
        auth.drop_session(tok)
    size = denied_env.stat().st_size
    assert size > 300, size  # live revocations are never evicted to meet the trigger
    lines = denied_env.read_text().strip().splitlines()
    assert len(lines) == len(live)
    assert len({json.loads(line)["id"] for line in lines}) == len(live)
    _restart()
    for tok in live:
        assert not auth.valid_session(tok), "live revocation lost"


def test_compaction_removes_the_specific_expired_hash(denied_env, monkeypatch):
    monkeypatch.setattr(auth, "_DENYLIST_COMPACT_TRIGGER_BYTES", 100)
    monkeypatch.setattr(auth, "_DENYLIST_MAINT_INTERVAL", 0)
    dead_id = auth._token_id("s1.1.dead.sig")
    denied_env.write_text(json.dumps({"id": dead_id, "exp": 1}) + "\n" * 20)
    auth.drop_session(auth.new_session())  # crossing triggers compaction
    lines = denied_env.read_text().strip().splitlines()
    assert dead_id not in denied_env.read_text()  # the exact expired hash is gone
    assert len(lines) == 1  # only the live revocation remains


def test_no_rewrite_when_nothing_removable(denied_env, monkeypatch):
    monkeypatch.setattr(auth, "_DENYLIST_COMPACT_TRIGGER_BYTES", 100)
    monkeypatch.setattr(auth, "_DENYLIST_MAINT_INTERVAL", 0)
    calls = []
    real = auth._write_denylist
    monkeypatch.setattr(auth, "_write_denylist",
                        lambda p, r: (calls.append(1), real(p, r)))
    for _ in range(8):
        auth.drop_session(auth.new_session())
    assert denied_env.stat().st_size > 100  # trigger crossed on many appends
    assert calls == []  # all-live file: compaction never rewrites it


def test_repeat_revocation_deduped(denied_env):
    tok = auth.new_session()
    auth.drop_session(tok)
    auth.drop_session(tok)
    lines = denied_env.read_text().strip().splitlines()
    assert len(lines) == 1


def test_expired_ids_swept_from_memory(denied_env, monkeypatch):
    tok = auth.new_session()
    auth.drop_session(tok)
    tid = auth._token_id(tok)
    assert tid in auth._denied
    auth._load_denied()  # settle lazy loading before corrupting the entry
    auth._denied[tid] = time.time() - 5  # revocation entry past its expiry
    auth._last_sweep = 0.0
    monkeypatch.setattr(auth, "_DENYLIST_MAINT_INTERVAL", 0)
    assert auth.valid_session(auth.new_session())  # triggers the lazy sweep
    assert tid not in auth._denied


def test_concurrent_persist_threads_no_lost_or_torn_records(denied_env):
    import threading
    toks = [auth.new_session() for _ in range(32)]
    threads = [threading.Thread(target=auth.drop_session, args=(t,)) for t in toks]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    lines = denied_env.read_text().strip().splitlines()
    assert len(lines) == len(toks)
    for line in lines:
        rec = json.loads(line)  # every line intact JSON (no torn writes)
        assert set(rec) == {"id", "exp"}
    _restart()
    for tok in toks:
        assert not auth.valid_session(tok)


def test_write_failure_keeps_memory_revocation(denied_env, monkeypatch):
    tok = auth.new_session()
    real_open = os.open
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(auth.os, "open", boom)
    auth.drop_session(tok)  # must not raise
    monkeypatch.setattr(auth.os, "open", real_open)
    assert not auth.valid_session(tok), "memory revocation lost on write failure"
    assert not denied_env.exists()  # nothing persisted, honestly
    _restart()  # restart loses what the disk never got
    assert auth.valid_session(tok), "write failure must not fake durability"


def test_unwritable_path_on_load_is_fail_soft(denied_env, monkeypatch):
    denied_env.parent.mkdir(parents=True, exist_ok=True)
    denied_env.write_bytes(b"\x00\x01\xff not json at all\n")
    _restart()
    auth._load_denied()  # corrupt bytes must not raise
    assert auth.valid_session(auth.new_session())


def test_missing_denylist_file_is_not_an_error(denied_env, monkeypatch):
    monkeypatch.setenv("RENDERELLO_DENYLIST_PATH",
                       str(denied_env.parent / "absent" / "revoked.jsonl"))
    _restart()
    auth._load_denied()  # must not raise
    tok = auth.new_session()
    assert auth.valid_session(tok)


def test_default_path_follows_runs_root(monkeypatch):
    monkeypatch.delenv("RENDERELLO_DENYLIST_PATH", raising=False)
    monkeypatch.setenv("RENDERELLO_RUNS", "/tmp/rr-test")
    assert auth._denylist_path() == "/tmp/rr-test/revoked-sessions.jsonl"
