"""Matte integrity assessment - pre-spend honesty gate for paid routes.

Why: on 21 Nov 2026 a cluttered multi-object source (multiproduct-barber1)
produced fragmented composites on all three Magic Hour models. The root
cause was upstream of the model: the u2netp_onnx salient-object matte kept
only a subset of the items (coverage 0.1161, 9 disconnected regions,
largest region 20% of the matte) and nothing in the pipeline flagged it.
Each paid call then faithfully composited a broken matte.

What this module does: measure, never guess. matte_stats() reports
coverage and connected-component structure of a cutout matte.
assess() turns those numbers into an optional review warning using
thresholds derived from the measured table below (21 Nov 2026, u2netp_onnx,
pinned weights sha256 309c8469...):

    source             coverage  comps>=0.2%  largest_share
    can2               0.2879    1            1.00
    glass1             0.3800    1            1.00
    deodorant          0.4682    1            1.00
    skincare           0.0616    1            1.00
    loose crop A       0.1452    3            0.59
    loose crop B       0.3635    3            0.45
    loose crop C        0.1568    2            0.76
    multip (cluttered) 0.1161    9            0.20   <- the failure

Warn band: components > 3 OR largest_share < 0.40. The four measured single-product sources pass; the measured cluttered failure warns. This
is a REVIEW gate, not an object detector: it cannot prove items are
missing, and a warning never blocks a confirmed run. Empty or nearly full masks also require review. Low nonzero coverage alone
is not a warning (skincare at 0.06 is a valid small product).
"""
from __future__ import annotations

import numpy as np

_MIN_COMPONENT_FRAC = 0.002   # ignore specks below 0.2% of the image
_MAX_COMPONENTS = 3           # measured valid products stay at <= 3
_MIN_LARGEST_SHARE = 0.40     # measured valid products stay >= 0.45


def matte_stats(alpha: np.ndarray) -> dict:
    """Component structure of an 8-bit matte (255 = product)."""
    import cv2
    alpha = np.asarray(alpha)
    if alpha.ndim != 2 or not alpha.size or alpha.dtype != np.uint8:
        raise ValueError("matte must be a nonempty two-dimensional uint8 array")
    solid = (alpha >= 128).astype(np.uint8)
    h, w = solid.shape
    n, _labels, st, _cent = cv2.connectedComponentsWithStats(solid, 8)
    areas = sorted((int(a) for a in st[1:, cv2.CC_STAT_AREA]), reverse=True)
    comps = [a for a in areas if a / (h * w) >= _MIN_COMPONENT_FRAC]
    total = int(solid.sum())
    return {
        "foreground_pixels": total,
        "coverage": round(float(solid.mean()), 4),
        "components": len(comps),
        "largest_share": round(areas[0] / total, 3) if total else 0.0,
        "component_fracs": [round(a / (h * w), 4) for a in comps[:8]],
    }


def _fragmented(stats: dict) -> bool:
    return (stats.get("foreground_pixels", stats["coverage"]) == 0 or stats["coverage"] >= 0.98
            or stats["components"] > _MAX_COMPONENTS
            or stats["largest_share"] < _MIN_LARGEST_SHARE)


def assess_compact(stats: dict) -> str | None:
    """Short fragment description for size-capped error channels (the webapp
    passes RuntimeErrors through at 300 chars)."""
    if not _fragmented(stats):
        return None
    return (f"matte needs review ({stats['components']} regions, "
            f"largest {stats['largest_share']:.0%}, "
            f"coverage {stats['coverage']:.1%})")


def assess(stats: dict) -> str | None:
    """Full review warning for manifests/results, else None."""
    c = assess_compact(stats)
    if c is None:
        return None
    return (f"cutout {c}. This pattern matched the 21 Nov multi-object "
            "composite failure. Review the saved matte before spending; "
            "re-run with matte_reviewed=1 to proceed anyway.")


class MatteReviewRequired(RuntimeError):
    """Carries only server-generated preview names, never an arbitrary URL."""
    def __init__(self, message, run_dir, previews):
        super().__init__(message)
        self.review_run_dir = run_dir
        self.review_previews = previews
