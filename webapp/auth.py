"""Renderello login/auth module - modular credential + session handling.

Credential model (owner rules, 2026-11-20):
- Email + password login. The password is NEVER stored or compared in
  plaintext: the server reads only a PBKDF2-HMAC-SHA256 hash from the
  RENDERELLO_LOGIN_PASSWORD_HASH environment variable
  (format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>).
- The login email lives in RENDERELLO_LOGIN_EMAIL (login identifier only;
  no mail is ever sent to it).
- Legacy single-access-code mode remains only as a fallback when the email
  credentials are not configured (RENDERELLO_ACCESS_CODE).
- Sessions are random-token cookies, 12 h, HttpOnly, SameSite=lax (top-level navigations from links keep the session; all mutations are POST JSON).
- Login attempts are rate-limited (10 per 10 minutes) across two buckets:
  a global circuit breaker bounding the total attempt rate, plus, in
  password mode only, a per-email bucket bounding per-account guessing.
  The global key is deliberate: per-client-IP attribution has not been
  verified under the deployed proxy configuration (see /api/login).
  Slots are reserved before the credential check; failures keep their
  slot, successes refund it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time

SESSION_HOURS = 12
PBKDF2_ITERATIONS = 600_000
_RATE_WINDOW_S = 600
_RATE_MAX = 10

_sessions: dict[str, float] = {}
_attempts: dict[str, list[float]] = {}
_lock = threading.Lock()


def hash_password(password: str, salt: bytes | None = None,
                  iterations: int = PBKDF2_ITERATIONS) -> str:
    """Return a pbkdf2_sha256$iterations$salt_hex$hash_hex string."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def _verify_hash(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, hash_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def mode() -> str:
    """'password' when email+hash env is configured, else legacy 'code'."""
    if os.environ.get("RENDERELLO_LOGIN_EMAIL") and \
            os.environ.get("RENDERELLO_LOGIN_PASSWORD_HASH"):
        return "password"
    return "code"


def check_rate(ip: str) -> float:
    """Reserve one attempt slot for `ip`; raise RuntimeError('rate') when full.

    The slot is taken BEFORE the credentials are checked, so concurrent
    guesses all count. Returns the slot's timestamp for refund_rate.
    """
    now = time.time()
    with _lock:
        tries = [t for t in _attempts.get(ip, []) if now - t < _RATE_WINDOW_S]
        if len(tries) >= _RATE_MAX:
            raise RuntimeError("rate")
        tries.append(now)
        _attempts[ip] = tries
    return now


def refund_rate(ip: str, slot: float) -> None:
    """Give back one slot after a SUCCESSFUL login, so good logins do not
    count toward the lockout. Failed attempts keep their slot."""
    with _lock:
        tries = _attempts.get(ip)
        if tries and slot in tries:
            tries.remove(slot)
            if not tries:
                del _attempts[ip]


def verify_password_login(email: str, password: str) -> bool:
    expected_email = os.environ.get("RENDERELLO_LOGIN_EMAIL", "")
    stored = os.environ.get("RENDERELLO_LOGIN_PASSWORD_HASH", "")
    if not expected_email or not stored:
        return False
    email_ok = hmac.compare_digest(email.strip().lower(),
                                   expected_email.strip().lower())
    # Always run the (slow) hash check so response time does not reveal
    # whether the email matched.
    pw_ok = _verify_hash(password, stored)
    return email_ok and pw_ok


def verify_code_login(code: str) -> bool:
    expected = os.environ.get("RENDERELLO_ACCESS_CODE", "")
    return bool(expected) and hmac.compare_digest(code, expected)


_denied: dict[str, float] = {}  # sha256 id -> token expiry; never raw tokens
_denied_loaded = False
_io_lock = threading.Lock()  # serializes append + compaction against each other
_last_compact = 0.0  # monotonic time of the last compaction attempt
_last_sweep = 0.0  # monotonic time of the last in-memory expiry sweep
# Size at which the on-disk list is CONSIDERED for compaction. A trigger,
# NOT a cap: live (unexpired) revocations are never evicted to meet a size
# target, so the file grows with the number of revocations issued within the
# 12 h token TTL (in practice: signouts). Compaction only removes expired,
# corrupt or duplicate lines, runs at most once per _DENYLIST_MAINT_INTERVAL,
# and skips the rewrite when nothing is removable, so appends stay amortized
# O(1) and an all-live oversized file is not rescanned on every append.
_DENYLIST_COMPACT_TRIGGER_BYTES = 64 * 1024
_DENYLIST_MAINT_INTERVAL = 300  # seconds; gates compaction and memory sweeps


def _denylist_path() -> str:
    """File backing for the logout denylist.

    Default lives under RENDERELLO_RUNS so a process restart or worker reload
    on the same host keeps logged-out tokens revoked. NOTE: on hosts with an
    ephemeral filesystem (e.g. Render free tier) an instance replacement still
    clears it; CROSS-REPLACEMENT revocation is NOT solved by this file - the
    remaining window is bounded by the 12 h token expiry. Override with
    RENDERELLO_DENYLIST_PATH."""
    p = os.environ.get("RENDERELLO_DENYLIST_PATH", "")
    if p:
        return p
    return os.path.join(os.environ.get("RENDERELLO_RUNS", "/tmp/renderello-runs"),
                        "revoked-sessions.jsonl")


def _token_id(token: str) -> str:
    """Hash-only revocation identifier; the raw bearer token never touches disk."""
    return hashlib.sha256(token.encode()).hexdigest()


def _read_denied_records(path: str) -> list[dict]:
    """Live {id, exp} records; expired and corrupt lines are skipped
    individually and duplicate ids collapse (last wins). A missing or
    unreadable file yields nothing and never raises."""
    now = time.time()
    live: dict[str, float] = {}
    try:
        # errors="replace": undecodable bytes corrupt only their own line,
        # which then fails JSON parsing and is skipped like any bad record.
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if (isinstance(rec.get("id"), str)
                            and float(rec.get("exp", 0)) > now):
                        live[rec["id"]] = float(rec["exp"])
                except (ValueError, TypeError, AttributeError):
                    continue
    except OSError:
        pass
    return [{"id": k, "exp": v} for k, v in live.items()]


def _write_denylist(path: str, records: list[dict]) -> None:
    """Atomic whole-file replace, mode 0600 (owner-only revocation ids)."""
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)  # os.replace keeps tmp's mode; enforce anyway
    except OSError:
        pass


def _load_denied() -> None:
    """Best-effort load of persisted revocations (expired already pruned).
    A missing/unreadable file never locks anyone out and never un-revokes an
    id already denied in this process."""
    global _denied_loaded
    with _lock:
        if _denied_loaded:
            return
        _denied_loaded = True
        for rec in _read_denied_records(_denylist_path()):
            _denied[rec["id"]] = rec["exp"]


def _sweep_expired() -> None:
    """Drop expired ids from the in-memory denylist. Entries carry the
    token's own expiry, and expired tokens fail expiry validation anyway, so
    this never un-revokes a usable token. Caller holds _lock."""
    global _last_sweep
    now = time.time()
    for tid in [k for k, exp in _denied.items() if exp <= now]:
        del _denied[tid]
    _last_sweep = time.monotonic()


def _compact_denylist(path: str) -> None:
    """Prune expired/corrupt/duplicate lines via atomic replace. Caller holds
    _io_lock: serialized against appends, so no earlier record is lost and
    later appends land on the compacted file. The attempt is recorded even
    when nothing is removable, so an all-live oversized file is not rescanned
    (and needlessly rewritten) on every append."""
    global _last_compact
    _last_compact = time.monotonic()
    records = _read_denied_records(path)
    if sum(len(json.dumps(r)) + 1 for r in records) >= os.path.getsize(path):
        return  # nothing removable; a rewrite would not shrink the file
    _write_denylist(path, records)


def _persist_denied(token: str) -> None:
    try:
        _, expiry_s, _nonce, _sig = token.split(".", 3)
        exp = int(expiry_s)
    except (ValueError, TypeError):
        return
    try:
        path = _denylist_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _io_lock:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a") as fh:
                fh.write(json.dumps({"id": _token_id(token), "exp": exp}) + "\n")
            try:
                if (os.path.getsize(path) > _DENYLIST_COMPACT_TRIGGER_BYTES
                        and time.monotonic() - _last_compact
                        > _DENYLIST_MAINT_INTERVAL):
                    _compact_denylist(path)
            except OSError:
                pass
    except OSError:
        pass  # memory copy still holds the revocation for this process


def _stateless_secret() -> str:
    # RENDERELLO_SESSION_SECRET signs stateless session tokens so a deploy or
    # restart (Render restarts on every env change) no longer ends sessions.
    return os.environ.get("RENDERELLO_SESSION_SECRET", "")


def new_session() -> str:
    # Stateless HMAC-signed token when the secret is configured; legacy
    # server-side random tokens otherwise (invalid after restart).
    secret = _stateless_secret()
    if secret:
        expiry = int(time.time()) + SESSION_HOURS * 3600
        payload = f"{expiry}.{secrets.token_urlsafe(12)}"
        sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return f"s1.{payload}.{sig}"
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = time.time() + SESSION_HOURS * 3600
    return token


def _valid_stateless(token: str) -> bool:
    secret = _stateless_secret()
    if not secret:
        return False
    try:
        _, expiry_s, nonce, sig = token.split(".", 3)
        expiry = int(expiry_s)
    except (ValueError, TypeError):
        return False
    payload = f"{expiry_s}.{nonce}"
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    if expiry <= time.time():
        return False
    _load_denied()
    with _lock:
        if time.monotonic() - _last_sweep > _DENYLIST_MAINT_INTERVAL:
            _sweep_expired()
        return _token_id(token) not in _denied


def valid_session(token: str | None) -> bool:
    if not token:
        return False
    if token.startswith("s1."):
        return _valid_stateless(token)
    with _lock:
        expiry = _sessions.get(token)
        if expiry is None:
            return False
        if expiry <= time.time():
            _sessions.pop(token, None)
            return False
        return True


def drop_session(token: str | None) -> None:
    # Stateless tokens are denied via a hash-id denylist mirrored to disk
    # (see _denylist_path): process restarts on the same host keep the
    # revocation, and only sha256 ids are stored, never raw tokens. A host
    # with an ephemeral filesystem can still lose it on instance
    # replacement: cross-replacement revocation is NOT solved here, the
    # remaining exposure is bounded by token expiry.
    if not (token and token.startswith("s1.")):
        with _lock:
            _sessions.pop(token, None)
        return
    tid = _token_id(token)
    with _lock:
        _sessions.pop(token, None)
        if tid in _denied:
            return  # already revoked and persisted; skip a duplicate append
        try:
            _denied[tid] = float(token.split(".", 3)[1])
        except (ValueError, IndexError):
            _denied[tid] = float("inf")  # malformed: deny for process lifetime
    _persist_denied(token)
