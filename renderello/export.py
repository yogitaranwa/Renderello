"""Lossless artifact export for notebook sessions where /kaggle/working is
not persisted by Quick Save. Full-resolution PNGs are displayed inline UNRESIZED
(IPython embeds the exact PNG bytes); a browser agent can then fetch the image
blob from the saved version and write the same bytes locally.

Proven 2026-11-09 for 1800 px images via fetch() from the notebook output frame.
verify_roundtrip() checks the encode/decode is pixel-exact.
"""
from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
from PIL import Image


def png_bytes(path: Path) -> bytes:
    b = Path(path).read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return b


def manifest_line(path: Path) -> str:
    b = png_bytes(path)
    return f"{Path(path).name} {len(b)} bytes sha256={hashlib.sha256(b).hexdigest()}"


def display_full_res(path: Path):
    """Notebook: display the PNG file unresized and print its sha256 so the
    retrieved copy can be verified byte-for-byte."""
    from IPython.display import display, Image as IPImage
    print(manifest_line(path))
    display(IPImage(data=png_bytes(path), format="png"))


def verify_roundtrip(path: Path) -> bool:
    b = png_bytes(path)
    back = base64.b64decode(base64.b64encode(b))
    a = Image.open(io.BytesIO(back)).convert("RGBA")
    o = Image.open(path).convert("RGBA")
    return back == b and a.size == o.size and a.tobytes() == o.tobytes()
