"""Web app auth: PBKDF2 password login, legacy code fallback, sessions, rate limits."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

import auth  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("RENDERELLO_LOGIN_EMAIL", "RENDERELLO_LOGIN_PASSWORD_HASH",
              "RENDERELLO_ACCESS_CODE"):
        monkeypatch.delenv(k, raising=False)
    auth._sessions.clear()
    auth._attempts.clear()
    yield


def test_hash_roundtrip_and_reject():
    h = auth.hash_password("s3cret")
    assert h.startswith("pbkdf2_sha256$600000$")
    assert "s3cret" not in h
    assert auth._verify_hash("s3cret", h)
    assert not auth._verify_hash("nope", h)
    assert not auth._verify_hash("s3cret", "malformed")


def test_mode_defaults_to_legacy_code():
    assert auth.mode() == "code"


def test_password_login_requires_both_parts(monkeypatch):
    monkeypatch.setenv("RENDERELLO_LOGIN_EMAIL", "admin@magichour.ai")
    monkeypatch.setenv("RENDERELLO_LOGIN_PASSWORD_HASH",
                       auth.hash_password("correct horse"))
    assert auth.mode() == "password"
    assert auth.verify_password_login("admin@magichour.ai", "correct horse")
    assert auth.verify_password_login("  ADMIN@magichour.ai ", "correct horse")
    assert not auth.verify_password_login("admin@magichour.ai", "wrong")
    assert not auth.verify_password_login("other@example.com", "correct horse")


def test_code_login_only_without_password_env(monkeypatch):
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "letmein")
    assert auth.verify_code_login("letmein")
    assert not auth.verify_code_login("nope")
    assert not auth.verify_password_login("a@b.c", "letmein")


def test_session_lifecycle():
    tok = auth.new_session()
    assert auth.valid_session(tok)
    auth.drop_session(tok)
    assert not auth.valid_session(tok)
    assert not auth.valid_session("forged-token")
    assert not auth.valid_session(None)
    assert not auth.valid_session("9999999999.deadbeef")  # forged HMAC
    # A restart invalidates old tokens, including previously logged-out tokens.
    tok2 = auth.new_session()
    auth._sessions.clear()
    assert not auth.valid_session(tok2)
    assert not auth.valid_session(tok)


def test_rate_limit():
    for _ in range(10):
        auth.check_rate("1.2.3.4")
    with pytest.raises(RuntimeError):
        auth.check_rate("1.2.3.4")
    auth.check_rate("5.6.7.8")  # other IP unaffected


def test_login_endpoint_password_mode(monkeypatch):
    monkeypatch.setenv("RENDERELLO_LOGIN_EMAIL", "admin@magichour.ai")
    monkeypatch.setenv("RENDERELLO_LOGIN_PASSWORD_HASH",
                       auth.hash_password("pw-12345"))
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    assert c.get("/api/login-mode").json() == {"mode": "password"}
    bad = c.post("/api/login", json={"email": "admin@magichour.ai",
                                     "password": "bad"})
    assert bad.status_code == 401
    ok = c.post("/api/login", json={"email": "admin@magichour.ai",
                                    "password": "pw-12345"})
    assert ok.status_code == 200
    assert "renderello_session" in ok.cookies
    assert c.get("/api/presets", cookies=ok.cookies).status_code == 200
    assert c.get("/static/js/login.js").status_code == 200
    assert c.get("/static/js/app.js", cookies=ok.cookies).status_code == 200


def test_logged_out_cookie_rejected_after_restart(monkeypatch):
    monkeypatch.setenv("RENDERELLO_LOGIN_EMAIL", "test@example.com")
    monkeypatch.setenv("RENDERELLO_LOGIN_PASSWORD_HASH", auth.hash_password("test-only"))
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    login = client.post("/api/login", json={"email": "test@example.com", "password": "test-only"})
    cookie = login.cookies.get("renderello_session")
    assert cookie
    assert client.get("/api/presets").status_code == 200
    logout = client.post("/api/logout")
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert client.get("/api/presets", headers={"Cookie": f"renderello_session={cookie}"}).status_code == 401
    importlib.reload(auth)
    assert client.get("/api/presets", headers={"Cookie": f"renderello_session={cookie}"}).status_code == 401
    renewed = client.post("/api/login", json={"email": "test@example.com", "password": "test-only"})
    assert renewed.status_code == 200
    assert client.get("/api/presets").status_code == 200
