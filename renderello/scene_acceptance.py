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


def long_straight_edges(scene_rgb: np.ndarray) -> dict:
    """Detect long straight high-contrast edges (equipment frames, set walls'
    hard corners) in a generated empty background. Returns diagnostics with
    'intrusion' True when any straight line exceeds MIN_LINE_FRAC of the short
    side."""
    if (not isinstance(scene_rgb, np.ndarray) or scene_rgb.dtype != np.uint8
            or scene_rgb.ndim != 3 or scene_rgb.shape[2] != 3
            or min(scene_rgb.shape[:2]) < 1):
        raise ValueError("scene must be nonempty uint8 RGB")
    h, w = scene_rgb.shape[:2]
    short = min(h, w)
    # Use OpenCV's rounded 8-bit RGB luminance. Truncating floating luminance
    # can break thin anti-aliased equipment edges into sub-threshold segments.
    lu = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(lu, CANNY_LO, CANNY_HI)
    # A one-degree/0.12-side-vote pass missed the slightly tilted edge of a
    # visible softbox in the 21 Nov paid regression. Keep that coarse pass,
    # then add bounded finer-angle/lower-vote passes over the SAME full frame.
    # Line length is never relaxed. This is a warning screen, not semantic
    # equipment detection, and cannot certify that a scene is empty.
    found = []
    passes = ((1.0, .12), (.5, .08), (.5, .06))
    for theta_degrees, vote_fraction in passes:
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180 * theta_degrees,
            threshold=max(30, int(vote_fraction * short)),
            minLineLength=max(60, int(MIN_LINE_FRAC * short)),
            maxLineGap=max(8, int(0.016 * short)),
        )
        if lines is None:
            continue
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(v) for v in l)
            length = round(float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 1)
            row = {"p1": [x1, y1], "p2": [x2, y2], "length": length}
            if row not in found:
                found.append(row)
            if len(found) >= 64:
                break
        if len(found) >= 64:
            break
    diag = {
        "lines": found,
        "min_line_px": int(MIN_LINE_FRAC * short),
        "intrusion": bool(found),
        "scope": "FULL_FRAME_STRAIGHT_EDGE_SCREEN",
        "needs_eye_review": True,
        "screen_only": True,
        "semantic_equipment_detection": False,
        "passes": len(passes),
    }
    return diag
