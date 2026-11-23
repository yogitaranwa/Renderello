"""Bounded native mask diagnostics and source-bound spatial intent constraints.

OpenCV connected components are descriptive, not semantic object recognition.
Nothing here deletes small components, fills holes, merges model candidates or
turns model agreement into accuracy. All coordinates refer to normalized pixels.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
import cv2
import numpy as np

MAX_PIXELS = 4_000_000


def digest(array):
    """Bind shape and dtype as well as bytes; raw bytes alone are ambiguous."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.shape).encode('ascii'))
    h.update(a.dtype.str.encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def native_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('native uint8 RGB required')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 3:
        raise ValueError('RGB dimensions must be at least 3 by 3')
    if rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise ValueError('4MP allocation budget exceeded; no silent downscale')
    return rgb


def native_alpha(alpha, shape):
    if not isinstance(alpha, np.ndarray) or alpha.dtype != np.uint8:
        raise ValueError('native uint8 alpha required; no floating range guessing')
    if alpha.ndim != 2 or alpha.shape != shape:
        raise ValueError('alpha must match normalized source; no resize')
    if alpha.size > MAX_PIXELS or min(alpha.shape) < 3:
        raise ValueError('alpha dimensions outside allocation budget')
    return alpha


@dataclass(frozen=True)
class RegionConstraint:
    """A reviewer-specified region, not an automatically discovered object.

    A small interior box is often safer than an object's full bounding rectangle:
    object bounding boxes also contain background. Negative regions can preserve
    handle holes or forbid a prop. Names describe intent, not identity proof.
    """
    name: str
    box: tuple[int, int, int, int]
    kind: str
    minimum_fraction: float = .95
    threshold: int = 128

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 120:
            raise ValueError('short region name required')
        if self.kind not in ('include', 'exclude'):
            raise ValueError('region kind must be include or exclude')
        if len(self.box) != 4 or any(type(v) is not int for v in self.box):
            raise ValueError('integer x0,y0,x1,y1 region required')
        x0, y0, x1, y1 = self.box
        if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError('nonempty positive region required')
        if isinstance(self.minimum_fraction, bool) or not math.isfinite(self.minimum_fraction) or not 0 < self.minimum_fraction <= 1:
            raise ValueError('minimum fraction must be in (0,1]')
        if type(self.threshold) is not int or not 1 <= self.threshold <= 255:
            raise ValueError('threshold must be integer 1..255')


def check_regions(alpha, constraints):
    native_alpha(alpha, alpha.shape)
    if len(constraints) > 256:
        raise ValueError('too many regions')
    names = set()
    result = []
    for c in constraints:
        if not isinstance(c, RegionConstraint) or c.name in names:
            raise ValueError('typed, uniquely named region constraints required')
        names.add(c.name)
        x0, y0, x1, y1 = c.box
        if x1 > alpha.shape[1] or y1 > alpha.shape[0]:
            raise ValueError('region outside native source')
        roi = alpha[y0:y1, x0:x1]
        good = roi >= c.threshold if c.kind == 'include' else roi < c.threshold
        fraction = float(good.mean())
        result.append(dict(**asdict(c), observed_fraction=fraction,
                           passed=fraction >= c.minimum_fraction))
    return result


def mask_diagnostics(rgb, alpha):
    native_rgb(rgb)
    native_alpha(alpha, rgb.shape[:2])
    hard = alpha >= 128
    support = alpha > 0
    flags = []
    if not support.any():
        flags.append('EMPTY_MASK')
    if not hard.any():
        flags.append('NO_HIGH_ALPHA_SUPPORT')
    if hard.all():
        flags.append('FULL_FRAME_FOREGROUND')
    if min(alpha.shape) < 64:
        flags.append('TINY_SOURCE_NO_DETAIL_RECOVERY')
    if max(alpha.shape) / min(alpha.shape) > 8:
        flags.append('ODD_ASPECT_REVIEW')
    if support.any():
        yy, xx = np.where(support)
        box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    else:
        box = None
    sides = {name: int(np.count_nonzero(edge)) for name, edge in (
        ('top', hard[0]), ('bottom', hard[-1]),
        ('left', hard[:, 0]), ('right', hard[:, -1]))}
    if any(sides.values()):
        flags.append('FRAME_TOUCH_NOT_PROOF_OF_COMPLETE_PRODUCT')
    count, labels, stats, centers = cv2.connectedComponentsWithStats(hard.astype(np.uint8), connectivity=8)
    # Bound JSON size independently of native image size. Keep total count exact.
    order = sorted(range(1, count), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
    components = []
    for i in order[:256]:
        x, y, w, h, area = map(int, stats[i])
        components.append(dict(label=i, box=[x, y, x+w, y+h], area=area,
                               centroid=[float(v) for v in centers[i]],
                               touches_frame=x == 0 or y == 0 or x+w == alpha.shape[1] or y+h == alpha.shape[0]))
    if count - 1 > 1:
        flags.append('MULTICOMPONENT_REVIEW_EVERY_PART')
    if count - 1 > 256:
        flags.append('COMPONENT_LIST_TRUNCATED_NOT_MASK')
    # Background components not touching the frame are holes. Never fill them:
    # real handle/scissor gaps and a false internal cut look identical topologically.
    nb, _, bs, _ = cv2.connectedComponentsWithStats((~hard).astype(np.uint8), connectivity=8)
    holes = []
    for x, y, w, h, area in bs[1:]:
        if x > 0 and y > 0 and x+w < alpha.shape[1] and y+h < alpha.shape[0]:
            holes.append(int(area))
    if holes:
        flags.append('INTERNAL_HOLES_REVIEW_NO_AUTOFILL')
    soft = int(((alpha > 0) & (alpha < 255)).sum())
    if soft > max(16, int(hard.sum()) * .3):
        flags.append('LARGE_SOFT_BAND_NOT_OPTICAL_TRANSMISSION')
    pixels = rgb[hard]
    luminance = pixels.astype(float) @ np.array([.2126, .7152, .0722]) if len(pixels) else np.empty(0)
    dark = float((luminance < 24).mean()) if len(pixels) else None
    white = float((luminance > 230).mean()) if len(pixels) else None
    if dark is not None and dark > .25:
        flags.append('DARK_FOREGROUND_EDGE_REVIEW')
    if white is not None and white > .25:
        flags.append('WHITE_FOREGROUND_EDGE_REVIEW')
    return dict(source_pixels_sha256=digest(rgb), alpha_pixels_sha256=digest(alpha),
                shape=list(alpha.shape), support_box=box,
                support_pixels=int(support.sum()), hard_pixels=int(hard.sum()),
                soft_pixels=soft, opaque_pixels=int((alpha == 255).sum()),
                frame_contact=sides, component_count=count-1,
                components=components, component_list_truncated=count-1 > 256,
                hole_count=len(holes), hole_areas=sorted(holes, reverse=True)[:256],
                dark_fraction=dark, white_fraction=white, flags=flags,
                semantic_completeness='NOT_EVALUATED', optics_recovered=False,
                automatic_acceptance=False)


def compare_masks(rgb, candidates):
    native_rgb(rgb)
    if not 2 <= len(candidates) <= 8:
        raise ValueError('compare 2..8 separately named candidates')
    for name, a in candidates.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError('short candidate name required')
        native_alpha(a, rgb.shape[:2])
    result = []
    names = list(candidates)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = candidates[left], candidates[right]
            ah, bh = a >= 128, b >= 128
            union = int((ah | bh).sum())
            intersection = int((ah & bh).sum())
            result.append(dict(left=left, right=right,
                left_sha256=digest(a), right_sha256=digest(b),
                changed_alpha_pixels=int((a != b).sum()),
                mean_absolute_alpha_delta=float(np.abs(a.astype(np.int16)-b.astype(np.int16)).mean()),
                hard_iou=intersection/union if union else None,
                hard_disagreement_pixels=int((ah ^ bh).sum()),
                left_only_pixels=int((ah & ~bh).sum()),
                right_only_pixels=int((bh & ~ah).sum()),
                agreement_is_accuracy=False, automatically_selected=None))
    return result


def disagreement_image(left, right):
    native_alpha(left, left.shape)
    native_alpha(right, left.shape)
    result = np.zeros((*left.shape, 3), dtype=np.uint8)
    a, b = left >= 128, right >= 128
    result[a & b] = (110, 110, 110)
    result[a & ~b] = (255, 90, 40)
    result[b & ~a] = (30, 170, 255)
    # Color key: orange left-only, blue right-only, grey hard agreement.
    return result
