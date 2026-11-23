"""Shared fixtures: a synthetic product photo with real fidelity signals -
hard edges, a label with tiny text, a logo mark, distinct colors - so QA
metrics have something honest to measure. No external assets."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


def make_product(width: int = 640, height: int = 640) -> Image.Image:
    """Studio-ish photo of a 'skincare bottle': colored cylinder + cap, white
    label with tiny text rows and a red logo mark, on a mid-gray backdrop."""
    img = Image.new("RGB", (width, height), (168, 170, 172))
    d = ImageDraw.Draw(img)
    # Backdrop gradient floor line.
    d.rectangle([0, int(height * 0.8), width, height], fill=(150, 152, 154))
    cx = width // 2
    bw, bh = int(width * 0.28), int(height * 0.52)
    x0, y0 = cx - bw // 2, int(height * 0.78) - bh
    # Bottle body (teal) with slight vertical shading.
    body = Image.new("RGB", (bw, bh), (36, 130, 140))
    ba = np.array(body).astype(np.float32)
    shade = np.linspace(0.82, 1.12, bw, dtype=np.float32)[None, :, None]
    body = Image.fromarray(np.clip(ba * shade, 0, 255).astype(np.uint8))
    img.paste(body, (x0, y0))
    # Cap.
    d.rectangle([cx - bw // 4, y0 - int(height * 0.07), cx + bw // 4, y0],
                fill=(28, 28, 30))
    # Label with tiny text (the plan's hardest fidelity case).
    lx0, ly0 = x0 + int(bw * 0.12), y0 + int(bh * 0.30)
    lx1, ly1 = x0 + int(bw * 0.88), y0 + int(bh * 0.78)
    d.rectangle([lx0, ly0, lx1, ly1], fill=(246, 246, 244),
                outline=(60, 60, 60), width=2)
    d.ellipse([cx - 12, ly0 + 8, cx + 12, ly0 + 32], fill=(200, 40, 40))  # logo
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, line in enumerate(["DERMA", "ACTIVE 2%", "50 ml"]):
        d.text((lx0 + 8, ly0 + 40 + i * 14), line, fill=(20, 20, 20), font=font)
    # Specular highlight strip.
    d.line([x0 + int(bw * 0.2), y0 + 8, x0 + int(bw * 0.2), y0 + bh - 8],
           fill=(210, 240, 244), width=4)
    return img


@pytest.fixture()
def product_image() -> Image.Image:
    return make_product()


@pytest.fixture()
def product_path(tmp_path) -> "Path":
    from pathlib import Path
    p = tmp_path / "product.png"
    make_product().save(p)
    return p


@pytest.fixture(autouse=True)
def unsupported_windows_symlink_setup(monkeypatch):
    """Skip only when Windows denies constructing the symlink test fixture.

    Does not skip for generic OSError or change any production path checks.
    """
    from pathlib import Path
    original = Path.symlink_to
    def checked(self, target, target_is_directory=False):
        try:
            return original(self, target, target_is_directory=target_is_directory)
        except OSError as error:
            import sys
            caller = sys._getframe(1).f_globals.get('__name__', '').rsplit('.', 1)[-1]
            if getattr(error, 'winerror', None) == 1314 and caller.startswith('test_'):
                pytest.skip('Windows account cannot create symlink fixture (WinError1314); guard not exercised here')
            raise
    monkeypatch.setattr(Path, 'symlink_to', checked)
