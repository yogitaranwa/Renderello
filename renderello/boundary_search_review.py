"""Opt-in diagnostics for explicit local boundary searches. Never accepts a matte."""
import numpy as np


def boundary_search_review(baseline, traced, *, corridor_radius):
    """Flag search-limit saturation. No flags is not semantic quality acceptance.

    Inputs are same-column native y coordinates. This report does not compute
    accuracy or confidence and must not select a production alpha.
    """
    if type(corridor_radius) is not int or not 1 <= corridor_radius <= 128:
        raise ValueError('integer corridor radius required')
    b, t = np.asarray(baseline), np.asarray(traced)
    if (b.ndim != 1 or t.shape != b.shape or b.size < 2 or b.size > 10000
            or b.dtype.kind not in 'iu' or t.dtype.kind not in 'iu'
            or b.dtype.kind == 'b' or t.dtype.kind == 'b'):
        raise ValueError('matching integer native path coordinates required')
    # Bound before subtraction to avoid integer overflow and meaningless geometry.
    if np.any(b < 0) or np.any(t < 0) or np.any(b > 1_000_000) or np.any(t > 1_000_000):
        raise ValueError('path coordinates out of bounds')
    shift = np.abs(t.astype(np.int64)-b.astype(np.int64))
    if np.any(shift > corridor_radius):
        raise ValueError('traced path leaves explicit corridor')
    hits = np.flatnonzero(shift == corridor_radius).tolist()
    reasons = ['SEARCH_LIMIT_REACHED'] if hits else []
    return {'kind':'boundary-search-diagnostic','columns':int(b.size),
            'moved_columns':int(np.count_nonzero(shift)),
            'corridor_limit_column_indices':hits,'refusal_flags':reasons,
            'quality_accepted':False,'production_mask_selected':False,
            'semantic_review_required':True,
            'limitation':'No flags is not confidence, accuracy, or semantic boundary acceptance.'}
