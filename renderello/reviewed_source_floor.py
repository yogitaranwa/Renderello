"""Reviewed source-white floor tint illustration, not shadow recovery or relighting.

Only for a manually reviewed neutral-white source floor and a light, flat target.
Retains measured source-floor darkening instead of inventing an ellipse. Fractional
alpha and source-floor classification remain estimates. No default integration.
"""
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest


def tint_reviewed_source_floor(rgb, alpha, background, selection, *,
        source_sha256, alpha_sha256, background_sha256, selection_sha256,
        reviewed=False, source_floor=None, review_note=''):
    native_rgb(rgb); native_alpha(alpha, rgb.shape[:2]); native_rgb(background)
    if rgb.shape != background.shape or alpha.size > 1_000_000:
        raise ValueError('native matching target and 1MP budget required')
    if reviewed is not True or source_floor != 'neutral_white_opaque':
        raise ValueError('explicit neutral-white opaque source-floor review required')
    if not isinstance(review_note, str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('bounded source-floor review note required')
    native_alpha(selection, alpha.shape)
    if not selection.any() or not np.any((selection>0)&(alpha==0)):
        raise ValueError('nonempty reviewed source-floor selection required')
    if not np.any(alpha==255):
        raise ValueError('opaque source anchor required')
    for arr,h in [(rgb,source_sha256),(alpha,alpha_sha256),
                  (background,background_sha256),(selection,selection_sha256)]:
        if digest(arr)!=h: raise ValueError('source/alpha/background/selection binding mismatch')
    # A light flat target is an explicit limit, not evidence the physical light matches.
    if background.min()<160 or not np.all(background==background[0,0]):
        raise ValueError('light flat target only; no dark/material scene extrapolation')
    observed=(selection>0)&(alpha==0)
    spread=rgb.max(axis=2).astype(int)-rgb.min(axis=2).astype(int)
    if np.any(spread[observed]>12):
        raise ValueError('selected observed source floor is not neutral')
    # sRGB-domain tint is deliberately labelled, not a radiometric shadow estimate.
    a=alpha[:,:,None].astype(np.float64)/255
    m=selection[:,:,None].astype(np.float64)/255
    base=rgb*a+background*(1-a)
    floor=background*((1-m)+m*(rgb.astype(np.float64)/255))
    out=np.rint(rgb*a+floor*(1-a)).clip(0,255).astype(np.uint8)
    baseline=np.rint(base).astype(np.uint8)
    opaque=alpha==255
    if not np.array_equal(out[opaque],rgb[opaque]) or not np.array_equal(out[selection==0],baseline[selection==0]):
        raise RuntimeError('source/outside-selection invariants failed')
    return out,dict(schema='reviewed-source-floor-tint.v1',
        source_sha256=digest(rgb),alpha_sha256=digest(alpha),
        background_sha256=digest(background),selection_sha256=digest(selection),
        output_sha256=digest(out),review_note=review_note,
        opaque_source_rgb_exact=True,outside_selection_baseline_exact=True,
        source_resampled=False,source_lighting_unchanged=True,
        physical_relighting=False,recovered_shadow=False,accepted=False,
        default_pipeline_changed=False,changed_pixels=int(np.any(out!=baseline,axis=2).sum()),
        limitations=['Reviewer source-floor selection is not semantic truth.',
          'Source background must actually be neutral white; a near-white source is not calibrated.',
          'sRGB tint approximation, not a physical shadow or material model.',
          'Fractional mask errors may double-darken source boundaries.',
          'Wrong source-floor annotation can carry source objects or marks into the new background.',
          'Flat light backgrounds only. No photographic acceptance or production default.'])
