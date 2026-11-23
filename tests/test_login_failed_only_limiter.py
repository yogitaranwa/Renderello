"""Failed-only login limiter: good logins refund their slots; failures,
in-flight guesses and both buckets still count; spoofed XFF is ignored."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "webapp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auth  # noqa: E402

EMAIL, PW = "admin@magichour.ai", "pw-12345"


@pytest.fixture
def client(monkeypatch):
    auth._attempts.clear()
    monkeypatch.setenv("RENDERELLO_LOGIN_EMAIL", EMAIL)
    monkeypatch.setenv("RENDERELLO_LOGIN_PASSWORD_HASH", auth.hash_password(PW))
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    yield TestClient(main.app)
    auth._attempts.clear()


def _login(c, pw, **kw):
    return c.post("/api/login", json={"email": EMAIL, "password": pw}, **kw)


def test_many_good_logins_never_lock(client):
    for _ in range(30):
        assert _login(client, PW).status_code == 200
    assert auth._attempts == {}  # slots fully refunded


def test_ten_failures_lock_even_for_correct_password(client):
    for _ in range(10):
        assert _login(client, "bad").status_code == 401
    assert _login(client, "bad").status_code == 429
    assert _login(client, PW).status_code == 429  # brute-force guard intact


def test_good_logins_do_not_refund_earlier_failures(client):
    for _ in range(9):
        assert _login(client, "bad").status_code == 401
    for _ in range(5):
        assert _login(client, PW).status_code == 200
    assert _login(client, "bad").status_code == 401   # 10th failure
    assert _login(client, PW).status_code == 429


def test_inflight_slots_are_reserved_before_verification():
    # Concurrent guesses all reserve a slot first; none is refunded unless
    # that same request succeeds, so parallel guessing cannot exceed the cap.
    slots = [auth.check_rate("9.9.9.9") for _ in range(10)]
    with pytest.raises(RuntimeError):
        auth.check_rate("9.9.9.9")
    auth.refund_rate("9.9.9.9", slots[0])  # one request succeeded
    auth.check_rate("9.9.9.9")             # exactly one slot came back
    with pytest.raises(RuntimeError):
        auth.check_rate("9.9.9.9")
    auth._attempts.clear()


def test_refund_is_idempotent_and_scoped():
    s = auth.check_rate("a")
    auth.check_rate("b")
    auth.refund_rate("a", s)
    auth.refund_rate("a", s)         # no error, no extra refund
    auth.refund_rate("b", 12345.0)   # unknown slot: nothing removed
    assert len(auth._attempts["b"]) == 1
    auth._attempts.clear()


def test_email_bucket_still_enforced_separately(client):
    for _ in range(10):
        assert _login(client, "bad").status_code == 401
    # a different client IP key does not exist here; email bucket alone full:
    auth._attempts.pop(next(k for k in auth._attempts if not k.startswith("email:")))
    assert _login(client, PW).status_code == 429


def test_spoofed_forwarding_headers_ignored(client):
    for i in range(10):
        r = _login(client, "bad", headers={"X-Forwarded-For": f"10.0.0.{i}"})
        assert r.status_code == 401
    r = _login(client, "bad", headers={"X-Forwarded-For": "203.0.113.9"})
    assert r.status_code == 429
    ips = [k for k in auth._attempts if not k.startswith("email:")]
    assert len(ips) == 1 and not ips[0].startswith(("10.", "203."))


def test_concurrent_http_failures_never_exceed_cap(client):
    # Real HTTP layer, 24 threads racing bad logins from one client IP:
    # reservations happen before verification, so at most 10 can proceed
    # to a 401 and the rest must 429 - never more than 10 counted.
    import concurrent.futures as cf
    def one(_):
        return _login(client, "bad").status_code
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        codes = list(ex.map(one, range(24)))
    assert all(c in (401, 429) for c in codes), codes
    assert codes.count(401) == 10 and codes.count(429) == 14
    ipkey = next(k for k in auth._attempts if not k.startswith("email:"))
    assert len(auth._attempts[ipkey]) == 10
    assert len(auth._attempts["email:" + EMAIL]) == 10


def test_concurrent_http_mixed_success_refunds_only_own(client):
    # 12 good + 12 bad racing: successes refund exactly their own slots,
    # failures keep theirs; the failure count is exact afterward.
    import concurrent.futures as cf
    def one(i):
        return _login(client, PW if i % 2 == 0 else "bad").status_code
    with cf.ThreadPoolExecutor(max_workers=24) as ex:
        codes = list(ex.map(one, range(24)))
    ipkey = next(k for k in auth._attempts if not k.startswith("email:"))
    failures = codes.count(401)
    assert len(auth._attempts[ipkey]) == failures
    assert len(auth._attempts["email:" + EMAIL]) == failures


def test_attempts_keyed_global_not_connection_ip(client):
    # Behind managed ingress the peer is always the platform LB, so the
    # limiter must not pretend to key on a client IP: one shared "global"
    # bucket plus the email bucket, independent of any forwarded header.
    assert _login(client, "bad", headers={"X-Forwarded-For": "198.51.100.7"}).status_code == 401
    assert _login(client, "bad", headers={"X-Forwarded-For": "203.0.113.9"}).status_code == 401
    assert set(auth._attempts) == {"global", "email:" + EMAIL}
    assert len(auth._attempts["global"]) == 2
    assert len(auth._attempts["email:" + EMAIL]) == 2
