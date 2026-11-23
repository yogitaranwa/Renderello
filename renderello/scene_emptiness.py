"""Pre-paste emptiness screen for generated background scenes.

Flags a scene that appears to contain an object where the product will be
pasted - the 20 Nov 2026 duplicate-product failure mode. This is a SCREEN:
every verdict feeds the caller's scene-regeneration budget and sets
needs_eye_review=True unconditionally. It never certifies a scene clean and
never substitutes for human eye review.

Heuristics (all conservative, flag-only):
- object-like contrast blob inside the future placement rectangle
  (an empty guide scene is low-contrast there);
- long straight edges inside the placement rectangle (scene_acceptance
  covers the whole frame; this focuses where a duplicate would stand).
"""
import numpy as np


def _placement_rect(placement):
    px, py, pw, ph = placement
    return int(px), int(py), int(pw), int(ph)


def screen_scene_emptiness(scene_rgb, placement) -> dict:
    """scene_rgb: HxWx3 uint8 generated scene BEFORE the product paste.
    placement: (px, py, pw, ph) where the product cutout will land.
    Returns a flags dict; needs_eye_review is always True."""
    import cv2
    if not isinstance(scene_rgb, np.ndarray) or scene_rgb.ndim != 3 \
            or scene_rgb.shape[2] != 3 or scene_rgb.dtype != np.uint8:
        raise ValueError("scene must be a nonempty uint8 RGB array")
    h, w = scene_rgb.shape[:2]
    px, py, pw, ph = _placement_rect(placement)
    if min(pw, ph) < 1 or px < 0 or py < 0 or px + pw > w or py + ph > h:
        raise ValueError("placement outside the scene")
    roi = scene_rgb[py:py + ph, px:px + pw]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    full = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Contrast blob: an empty two-tone guide renders smoothly; a rendered
    # object raises local contrast well above the scene's median texture.
    roi_std = float(gray.std())
    scene_std = float(full.std())
    blob_suspect = roi_std > max(18.0, 1.6 * scene_std)

    # Straight-edge intrusion restricted to the placement area.
    edges = cv2.Canny(roi.astype(np.uint8), 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(20, min(pw, ph) // 4),
                            minLineLength=max(16, min(pw, ph) // 3),
                            maxLineGap=4)
    long_lines = 0 if lines is None else int(len(lines))
    edge_suspect = long_lines >= 3

    suspect_object = bool(blob_suspect or edge_suspect)
    return {
        "scope": "PLACEMENT_RECTANGLE_ONLY",
        "suspect_object_in_scene": suspect_object,
        "contrast_blob_suspect": blob_suspect,
        "roi_luma_std": round(roi_std, 2),
        "scene_luma_std": round(scene_std, 2),
        "placement_long_straight_edges": long_lines,
        "straight_edge_suspect": edge_suspect,
        "screen_only": True,
        "needs_eye_review": True,
    }
