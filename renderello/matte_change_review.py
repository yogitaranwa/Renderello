"""Native trimap and matte-change review, never an accuracy score.

OpenCV topology is used to expose dropped parts, filled holes and new bridges.
A baseline can be wrong. Preserving it does not establish semantic correctness.
No small component is discarded, no hole is filled, and no candidate is selected.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import cv2
import numpy as np
from .candidate_diagnostics import native_alpha, digest


@dataclass(frozen=True)
class ChangeBudget:
    threshold: int = 128
    minimum_part_retained: float = .9
    maximum_added_fraction: float = .1
    maximum_removed_fraction: float = .1
    maximum_hole_fill: float = .1
    maximum_alpha_mean_delta: float = .05
    detail_limit: int = 256

    def __post_init__(self):
        if type(self.threshold) is not int or not 1 <= self.threshold <= 255:
            raise ValueError('threshold must be integer 1..255')
        for key in ('minimum_part_retained', 'maximum_added_fraction',
                    'maximum_removed_fraction', 'maximum_hole_fill',
                    'maximum_alpha_mean_delta'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key+' must be finite fraction')
        if type(self.detail_limit) is not int or not 1 <= self.detail_limit <= 256:
            raise ValueError('detail limit must be integer 1..256')


def _components(binary):
    return cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)


def _require_pair(before, after):
    if not isinstance(before, np.ndarray) or before.ndim != 2:
        raise ValueError('native two dimensional alpha required')
    native_alpha(before, before.shape)
    native_alpha(after, before.shape)


def audit_trimap(alpha, trimap, *, unknown_limit=200_000):
    """Audit explicit seeds against an estimated alpha, without trusting either.

    Contradictions are review blockers, not proof the reviewer or model is wrong.
    Values 0/255 fix solver seeds;128 means unknown. No implicit threshold cast.
    """
    _require_pair(alpha, trimap)
    if type(unknown_limit) is not int or not 1 <= unknown_limit <= 4_000_000:
        raise ValueError('positive bounded integer unknown limit required')
    if not np.isin(trimap, [0, 128, 255]).all():
        raise ValueError('trimap needs only 0,128,255')
    fg, bg, unknown = trimap == 255, trimap == 0, trimap == 128
    blockers = []
    if not fg.any(): blockers.append('NO_FOREGROUND_SEED')
    if not bg.any(): blockers.append('NO_BACKGROUND_SEED')
    if int(unknown.sum()) > unknown_limit: blockers.append('UNKNOWN_BAND_EXCEEDS_BUDGET')
    contradictions = (fg & (alpha < 128)) | (bg & (alpha >= 128))
    if contradictions.any(): blockers.append('SEED_DISAGREES_WITH_INITIAL_ALPHA_REVIEW_BOTH')
    # Thin disconnected parts can lose all definite foreground during erosion.
    count, labels, stats, _ = _components(alpha >= 128)
    seeded = np.bincount(labels[fg].ravel(), minlength=count)
    unseeded = [i for i in range(1, count) if seeded[i] == 0]
    if unseeded: blockers.append('INITIAL_PART_WITHOUT_FOREGROUND_SEED')
    report = dict(alpha_sha256=digest(alpha), trimap_sha256=digest(trimap),
        foreground_seed_pixels=int(fg.sum()), background_seed_pixels=int(bg.sum()),
        unknown_pixels=int(unknown.sum()), contradictory_seed_pixels=int(contradictions.sum()),
        initial_parts=count-1, unseeded_part_count=len(unseeded),
        unseeded_part_labels=unseeded[:256], detail_truncated=len(unseeded)>256,
        blockers=blockers, seed_semantics='NOT_VERIFIED', automatic_acceptance=False)
    return report, contradictions


def review_change(before, after, budget=None):
    """Compare alpha revisions at identical native coordinates.

    Fractions of added/removed support use initial hard foreground area, not the
    whole canvas. Per-part retention prevents a large object hiding a lost small
    accessory. Hole fill uses each original enclosed background component. New
    connected components spanning multiple original parts are flagged as bridges.
    This does not certify that separated parts should remain separate in reality.
    """
    _require_pair(before, after)
    budget = ChangeBudget() if budget is None else budget
    if not isinstance(budget, ChangeBudget):
        raise ValueError('typed change budget required')
    b, a = before >= budget.threshold, after >= budget.threshold
    nb, lb, sb, _ = _components(b)
    na, la, sa, _ = _components(a)
    blockers = []
    area = int(b.sum())
    if not area: blockers.append('EMPTY_INITIAL_HARD_MASK')
    if not a.any(): blockers.append('EMPTY_REVISED_HARD_MASK')
    added, removed = a & ~b, b & ~a
    denominator = max(1, area)
    added_fraction, removed_fraction = float(added.sum())/denominator, float(removed.sum())/denominator
    if added_fraction > budget.maximum_added_fraction: blockers.append('ADDED_SUPPORT_EXCEEDS_BUDGET')
    if removed_fraction > budget.maximum_removed_fraction: blockers.append('REMOVED_SUPPORT_EXCEEDS_BUDGET')
    # Bincount is linear in pixels, rather than scanning the canvas per component.
    retained = np.bincount(lb[a].ravel(), minlength=nb)
    parts = []
    damaged = []
    for i in range(1, nb):
        part_area = int(sb[i, cv2.CC_STAT_AREA])
        fraction = int(retained[i])/part_area
        if fraction < budget.minimum_part_retained: damaged.append(i)
        if len(parts) < budget.detail_limit:
            x, y, w, h, _ = map(int, sb[i])
            parts.append(dict(label=i, box=[x,y,x+w,y+h], area=part_area,
                retained_pixels=int(retained[i]), retained_fraction=fraction))
    if damaged: blockers.append('INITIAL_PART_DAMAGED_OR_REMOVED')
    # Count unique (revised,initial) overlap pairs using a compact integer key.
    pairs = np.unique(la[b & a].astype(np.int64)*nb+lb[b & a])
    overlap_counts = np.bincount(pairs//nb, minlength=na)
    bridges = np.flatnonzero(overlap_counts > 1).tolist()
    isolated = [i for i in range(1,na) if overlap_counts[i] == 0]
    if bridges: blockers.append('REVISED_COMPONENT_BRIDGES_INITIAL_PARTS')
    if isolated: blockers.append('NEW_DISCONNECTED_FOREGROUND')
    nh, lh, sh, _ = _components(~b)
    filled = np.bincount(lh[a].ravel(), minlength=nh)
    holes, changed_holes = [], []
    h, w = b.shape
    hole_count = 0
    for i in range(1, nh):
        x, y, cw, ch, hole_area = map(int, sh[i])
        if x == 0 or y == 0 or x+cw == w or y+ch == h: continue
        hole_count += 1
        fraction = int(filled[i])/hole_area
        if fraction > budget.maximum_hole_fill: changed_holes.append(i)
        if len(holes) < budget.detail_limit:
            holes.append(dict(label=i, box=[x,y,x+cw,y+ch], area=hole_area,
                filled_pixels=int(filled[i]), filled_fraction=fraction))
    if changed_holes: blockers.append('INITIAL_HOLE_FILLED')
    delta = after.astype(np.int16)-before.astype(np.int16)
    mean_delta = float(np.abs(delta).mean()/255)
    if mean_delta > budget.maximum_alpha_mean_delta: blockers.append('ALPHA_CHANGE_EXCEEDS_BUDGET')
    truncated = nb-1 > budget.detail_limit or na-1 > budget.detail_limit or hole_count > budget.detail_limit
    if truncated: blockers.append('TOPOLOGY_DETAIL_LIMIT_REQUIRES_REVIEW')
    report = dict(before_sha256=digest(before), after_sha256=digest(after),
        budget=asdict(budget), initial_hard_pixels=area, revised_hard_pixels=int(a.sum()),
        initial_parts=nb-1, revised_parts=na-1, original_holes=hole_count,
        added_hard_pixels=int(added.sum()), removed_hard_pixels=int(removed.sum()),
        added_fraction=added_fraction, removed_fraction=removed_fraction,
        mean_normalized_alpha_delta=mean_delta, maximum_alpha_delta=int(np.abs(delta).max()),
        changed_alpha_pixels=int(np.count_nonzero(delta)), parts=parts, holes=holes,
        damaged_part_count=len(damaged), damaged_part_labels=damaged[:budget.detail_limit],
        bridge_count=len(bridges), bridge_labels=bridges[:budget.detail_limit],
        new_component_count=len(isolated), new_component_labels=isolated[:budget.detail_limit],
        filled_hole_count=len(changed_holes), filled_hole_labels=changed_holes[:budget.detail_limit],
        detail_truncated=truncated, blockers=blockers,
        within_configured_change_budget=not blockers, baseline_accuracy='NOT_VERIFIED',
        automatic_acceptance=False)
    # Red means alpha increased, blue means decreased. Magnitudes are byte deltas,
    # not an error map. The original/revised alpha arrays are never mutated.
    view = np.zeros((*before.shape,3),np.uint8)
    view[:,:,0] = np.clip(delta,0,255).astype(np.uint8)
    view[:,:,2] = np.clip(-delta,0,255).astype(np.uint8)
    return report, view
