"""Boundary review helper: crops the product edge band at full resolution.

Produces a side-by-side strip (final | raw scene | mask band) per sample so a
human can check halos, leftover silhouettes and contact shadows. The numbers
are only screens; the strips are for the eye.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import cv2


def edge_band(alpha: np.ndarray, inner: int = 3, outer: int = 12) -> np.ndarray:
    a = (alpha > 127).astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return (cv2.dilate(a, k(outer)) > 0) & ~(cv2.erode(a, k(inner)) > 0) & ~(a > 0) | \
           ((a > 0) & ~(cv2.erode(a, k(inner)) > 0))


def band_stats(final: np.ndarray, raw: np.ndarray, alpha: np.ndarray) -> dict:
    band = edge_band(alpha)
    outside = band & ~(alpha > 127)
    if outside.sum() == 0:
        return {"band_px": 0}
    d = np.abs(final[..., :3].astype(np.int16) - raw[..., :3].astype(np.int16)).mean(-1)
    return {"band_px": int(band.sum()), "outside_px": int(outside.sum()),
            "mean_abs_final_vs_raw_outside": round(float(d[outside].mean()), 2)}


def boundary_strip(final: Image.Image, raw: Image.Image, alpha: np.ndarray,
                   bbox: tuple[int, int, int, int], zoom: int = 3, pad: int = 24) -> Image.Image:
    x0, y0, x1, y1 = bbox
    H, W = alpha.shape
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    band = (edge_band(alpha) * 255).astype(np.uint8)
    panels = []
    for im in (np.array(final.convert("RGB")), np.array(raw.convert("RGB")),
               np.stack([band] * 3, -1)):
        c = im[y0:y1, x0:x1]
        panels.append(cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
    return Image.fromarray(np.concatenate(panels, axis=1))


def frame_edge_contact(matte, margin_px: int = 2, min_run_frac: float = 0.04) -> dict:
    """Detect a product cut off by the photo frame.

    The v0.4.9 GPU matrix (watch1) produced hard rectangular strap ends because the
    product touches the source frame: the cut IS the photo edge, not a matting error.
    Returns {side: fraction_of_that_side_covered} for sides where foreground (>127)
    covers at least min_run_frac of the side within margin_px of it, plus
    'touches' (list) and 'needs_review' (True when any side touches).
    """
    import numpy as np
    m = np.asarray(matte) > 127
    h, w = m.shape[:2]
    k = max(1, int(margin_px))
    sides = {"top": m[:k, :].any(axis=0).mean(), "bottom": m[-k:, :].any(axis=0).mean(),
             "left": m[:, :k].any(axis=1).mean(), "right": m[:, -k:].any(axis=1).mean()}
    sides = {s: round(float(v), 4) for s, v in sides.items()}
    touches = [s for s, v in sides.items() if v >= min_run_frac]
    return {**sides, "touches": touches, "needs_review": bool(touches)}
