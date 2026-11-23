"""Upload decode edge cases through /api/generate: every bad file gets a 4xx
(never a 500), leaves no temp dir behind and frees its job slot."""
import io
import importlib
import os
import struct
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))
import auth  # noqa: E402


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("RENDERELLO_ACCESS_CODE", "code-12345")
    monkeypatch.setenv("RENDERELLO_COOKIE_SECURE", "0")
    monkeypatch.setenv("RENDERELLO_RUNS", str(runs))
    for k in ("RENDERELLO_LOGIN_EMAIL", "RENDERELLO_LOGIN_PASSWORD_HASH"):
        monkeypatch.delenv(k, raising=False)
    auth._attempts.clear()
    import main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    c = TestClient(main.app)
    assert c.post("/api/login", json={"code": "code-12345"}).status_code == 200
    yield main, c, runs
    auth._attempts.clear()


def _png(w, h):
    buf = io.BytesIO()
    Image.new("L", (w, h), 0).save(buf, "PNG", compress_level=9)
    return buf.getvalue()


def _png_header_only(w, h):
    """Valid PNG signature + IHDR claiming w x h, tiny body (bomb-like)."""
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


GOOD = _png(16, 16)
CASES = {
    "zero_bytes": b"",
    "png_magic_only": b"\x89PNG\r\n\x1a\n",
    "truncated_png": GOOD[: len(GOOD) // 2],
    "png_garbage_tail": b"\x89PNG\r\n\x1a\n" + os.urandom(2000),
    "jpeg_magic_garbage": b"\xff\xd8\xff" + os.urandom(2000),
    "webp_magic_garbage": b"RIFF\x10\x00\x00\x00WEBP" + os.urandom(500),
    "html_polyglot": b"\x89PNG\r\n\x1a\n<script>alert(1)</script>" + GOOD[8:],
    "text_file": b"hello world, definitely not an image",
    "svg": b"<svg xmlns='http://www.w3.org/2000/svg'><script>1</script></svg>",
    "gif": b"GIF89a" + os.urandom(100),
    "pixel_bomb_100kx100k": _png_header_only(100_000, 100_000),
    "too_wide_9000x1": _png(9000, 1),
    "too_many_pixels_6000x6000": _png_header_only(6000, 6000),
    "over_12mb": b"\x89PNG\r\n\x1a\n" + b"0" * (12 * 1024 * 1024 + 1),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_bad_upload_is_4xx_and_cleans_up(ctx, name):
    main, c, runs = ctx
    r = c.post("/api/generate",
               files=[("products", ("../../evil.png", CASES[name], "image/png"))],
               data={"preset": "clean_studio", "provider": "procedural",
                     "variants": "1"})
    assert 400 <= r.status_code < 500, (name, r.status_code, r.text[:200])
    assert not list(runs.glob("up-*")), "temp upload dir left behind"
    assert not [j for j in main._jobs.values() if j["status"] == "running"]


def test_bad_second_file_aborts_whole_request(ctx):
    main, c, runs = ctx
    r = c.post("/api/generate",
               files=[("products", ("a.png", GOOD, "image/png")),
                      ("references", ("b.png", CASES["truncated_png"], "image/png"))],
               data={"preset": "clean_studio", "provider": "procedural"})
    assert 400 <= r.status_code < 500
    assert not list(runs.glob("up-*"))
    assert not main._jobs or all(j["status"] != "running" for j in main._jobs.values())


def test_too_many_files_is_400(ctx):
    main, c, runs = ctx
    files = [("products", (f"{i}.png", GOOD, "image/png")) for i in range(9)]
    r = c.post("/api/generate", files=files,
               data={"preset": "clean_studio", "provider": "procedural"})
    assert r.status_code == 400
    assert not list(runs.glob("up-*"))


def test_bad_form_values_are_4xx(ctx):
    main, c, runs = ctx
    f = [("products", ("a.png", GOOD, "image/png"))]
    for data in ({"preset": "clean_studio", "variants": "abc"},
                 {"preset": "clean_studio", "mh_route": "teleport"},
                 {"preset": "clean_studio", "instruction": "x" * 2001},
                 {"preset": "../../etc/passwd"},
                 {"preset": ",".join(["clean_studio"] * 3 + ["nope"])}):
        r = c.post("/api/generate", files=f, data=data)
        assert 400 <= r.status_code < 500, (data, r.status_code)
    assert not list(runs.glob("up-*"))
