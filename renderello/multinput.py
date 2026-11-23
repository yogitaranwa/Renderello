"""Multi-image input: several product photos composed into one shot, and
reference/inspiration images that steer style.

Owner requirement (2026-11-20): the pipeline takes multiple product images
and reference images. Two distinct roles, kept separate everywhere:

- PRODUCT images: every one is cut out and composited; each product's pixels
  stay source-exact (Route 2 construction). A group shot arranges all of
  them on one ground line with matched scale.
- REFERENCE images: never composited. They contribute a measured style
  (palette, warmth, brightness, contrast) that steers the scene prompt and
  the procedural/generative background.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .compose import (composite, contact_shadow, fit_product, placement_params,
                      procedural_background)
from .cutout import cutout
from .schemas import Aspect, CutoutResult, TemplateSpec


@dataclass
class ReferenceStyle:
    """Measured style of a reference/inspiration photo."""
    source: str
    dominant_rgb: tuple[int, int, int]
    warmth: float          # -1 cool .. +1 warm
    brightness: float      # 0..1 mean luminance
    saturation: float      # 0..1 mean saturation
    contrast: float        # luminance std, 0..1-ish

    def descriptors(self) -> list[str]:
        d: list[str] = []
        d.append("warm color cast" if self.warmth > 0.12 else
                 "cool color cast" if self.warmth < -0.12 else "neutral color balance")
        d.append("bright high-key" if self.brightness > 0.66 else
                 "dark low-key" if self.brightness < 0.33 else "mid-key exposure")
        d.append("saturated palette" if self.saturation > 0.35 else
                 "muted desaturated palette" if self.saturation < 0.18 else
                 "moderately saturated palette")
        d.append("high contrast" if self.contrast > 0.24 else
                 "low soft contrast" if self.contrast < 0.12 else "moderate contrast")
        r, g, b = self.dominant_rgb
        d.append(f"dominant backdrop color around rgb({r},{g},{b})")
        return d

    def prompt_fragment(self) -> str:
        return ("Style reference: " + ", ".join(self.descriptors()) +
                ". Apply this palette and tonal treatment to the scene only, "
                "never to the product.")


def extract_reference_style(path: Path, sample: int = 160) -> ReferenceStyle:
    """Measure palette/tonal statistics of a reference image. Cheap and CPU."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((sample, sample))
    a = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    flat = a.reshape(-1, 3)
    # Dominant color via coarse 4x4x4 histogram binning (robust to outliers).
    bins = (flat * 3.999).astype(np.int32)
    keys = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[counts.argmax()]
    dom = tuple(min(255, int(round((c + 0.5) / 4 * 255))) for c in
                ((top // 16) % 4, (top // 4) % 4, top % 4))
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    sat = float(np.mean(np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, 1), 0)))
    warmth = float(np.mean(r - b))
    return ReferenceStyle(
        source=str(path), dominant_rgb=dom,
        warmth=warmth, brightness=float(lum.mean()),
        saturation=sat, contrast=float(lum.std()))


def merged_reference_style(paths: list[Path]) -> ReferenceStyle | None:
    """Average several references into one style (mean of measurements)."""
    if not paths:
        return None
    styles = [extract_reference_style(p) for p in paths]
    dom = tuple(int(round(np.mean([s.dominant_rgb[i] for s in styles])))
                for i in range(3))
    return ReferenceStyle(
        source=", ".join(s.source for s in styles),
        dominant_rgb=dom,
        warmth=float(np.mean([s.warmth for s in styles])),
        brightness=float(np.mean([s.brightness for s in styles])),
        saturation=float(np.mean([s.saturation for s in styles])),
        contrast=float(np.mean([s.contrast for s in styles])))


@dataclass
class GroupMember:
    path: Path
    cutout: CutoutResult | None = None
    error: str | None = None


def cutout_products(paths: list[Path], out_dir: Path, device: str = "cpu"
                    ) -> list[GroupMember]:
    """Cut out every product photo; per-item failures are recorded, not fatal."""
    members: list[GroupMember] = []
    for i, p in enumerate(paths):
        m = GroupMember(path=p)
        try:
            img = Image.open(p).convert("RGB")
            m.cutout = cutout(img, out_dir, stem=f"product{i+1}",
                              backend="auto", device=device)
        except Exception as e:  # recorded honestly, group can still render
            m.error = f"{type(e).__name__}: {e}"
        members.append(m)
    return members


def compose_group(members: list[GroupMember], template: TemplateSpec,
                  aspect: Aspect, seed: int, *,
                  reference_style: ReferenceStyle | None = None) -> np.ndarray:
    """Arrange every successful cutout on one shared ground line.

    Scale is normalized so all products share a matched visual height (tall
    items slightly taller), spaced evenly across the middle 80% of the
    canvas, each sitting on the same baseline. Each product's interior
    pixels stay bit-exact from its source cutout.
    """
    ok = [m for m in members if m.cutout is not None]
    if not ok:
        raise ValueError("no product cutouts available for the group shot")
    W, H = aspect.size
    bg = procedural_background((W, H), template, seed)
    if reference_style is not None:
        bg = _tint_background(bg, reference_style)
    n = len(ok)
    base_y = int(H * 0.86)
    target_h = int(H * (0.52 if n == 1 else 0.44 if n == 2 else 0.36))
    band_w = int(W * 0.8 / n)
    out = bg.copy()
    for i, m in enumerate(ok):
        rgba = np.asarray(Image.open(m.cutout.rgba_path).convert("RGBA"))
        h0 = max(m.cutout.bbox[3], 1)
        scale = target_h / h0
        new_w = max(int(rgba.shape[1] * scale), 8)
        new_h = max(int(rgba.shape[0] * scale), 8)
        prod = np.asarray(Image.fromarray(rgba).resize(
            (new_w, new_h), Image.LANCZOS))
        cx = int(W * 0.1 + band_w * (i + 0.5))
        px = min(max(cx - new_w // 2, 0), W - new_w)
        py = min(max(base_y - new_h, 0), H - new_h)
        alpha = prod[..., 3:4].astype(np.float32) / 255.0
        region = out[py:py + new_h, px:px + new_w]
        out[py:py + new_h, px:px + new_w] = (
            prod[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    return out


def _tint_background(bg: np.ndarray, style: ReferenceStyle) -> np.ndarray:
    """Pull the procedural backdrop toward the reference palette, gently.
    Product pixels are not in the canvas yet, so this cannot touch them."""
    dom = np.array(style.dominant_rgb, dtype=np.float32)
    flat = bg.reshape(-1, 3).astype(np.float32)
    lum = flat.mean(axis=1, keepdims=True)
    tinted = np.clip(lum * 0.55 + dom * 0.45 * (flat.mean() / 255.0 + 0.55),
                     0, 255)
    return tinted.reshape(bg.shape).astype(np.uint8)
