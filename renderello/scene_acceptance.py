"""Acceptance screen for generated background scenes, before compositing.

Catches frames where the generator draws set equipment or frame-like straight
edges intruding into the scene (observed: a black softbox frame crossing the
right side of a studio sweep). The scene handed here is the EMPTY generated
background (no product yet), so long straight high-contrast lines are almost
always equipment or frame artifacts, not scene content; the wall/floor sweep
curve is not straight and does not trigger the detector.

The screen is a heuristic gate for regeneration or fallback, never a quality
certificate: an accepted scene can still be wrong in other ways, and the
composite still goes through the same-coordinate QA and eye review.
"""
from __future__ import annotations

import cv2
import numpy as np

# A straight line segment at least this fraction of the short canvas side is
# treated as an equipment/frame intrusion.
MIN_LINE_FRAC = 0.30
# Canny thresholds tuned for soft studio gradients: below these, sweep shading
# does not edge-detect; hard equipment edges do.
CANNY_LO = 50
CANNY_HI = 150


def _luma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def long_straight_edges(scene_rgb: np.ndarray) -> dict:
    """Detect long straight high-contrast edges (equipment frames, set walls'
    hard corners) in a generated empty background. Returns diagnostics with
    'intrusion' True when any straight line exceeds MIN_LINE_FRAC of the short
    side."""
    h, w = scene_rgb.shape[:2]
    short = min(h, w)
    lu = _luma(scene_rgb).astype(np.uint8)
    edges = cv2.Canny(lu, CANNY_LO, CANNY_HI)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(40, int(0.12 * short)),
        minLineLength=max(60, int(MIN_LINE_FRAC * short)),
        maxLineGap=max(8, int(0.012 * short)),
    )
    found = []
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in l)
            length = round(float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 1)
            found.append({"p1": [x1, y1], "p2": [x2, y2], "length": length})
    diag = {
        "lines": found,
        "min_line_px": int(MIN_LINE_FRAC * short),
        "intrusion": bool(found),
    }
    return diag
