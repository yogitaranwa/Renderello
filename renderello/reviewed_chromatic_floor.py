"""Opt-in illustrated transfer of reviewed color-bearing white-floor reflections.

Not a shadow, light, emission or material recovery model. A separate experiment:
never weakens the neutral-only source-floor tool and never changes default routes.
"""
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest


def tint_reviewed_chromatic_floor(rgb, alpha, background, selection, *,
        source_sha256, alpha_sha256, background_sha256, selection_sha256,
        reviewed=False, source_floor=None, review_note=''):
    native_rgb(rgb); native_rgb(background); native_alpha(alpha, rgb.shape[:2])
    native_alpha(selection, alpha.shape)
    if rgb.shape != background.shape or alpha.size > 1_000_000:
        raise ValueError('matching native target and 1MP budget required')
    if reviewed is not True or source_floor != 'white_opaque_with_color_reflection':
        raise ValueError('explicit color-bearing white opaque floor review required')
    if not isinstance(review_note, str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('bounded review note required')
    for value, expected in [(rgb, source_sha256), (alpha, alpha_sha256),
                            (background, background_sha256), (selection, selection_sha256)]:
        if digest(value) != expected:
            raise ValueError('source/alpha/background/selection binding mismatch')
    if background.min()<160 or not np.all(background==background[0,0]):
        raise ValueError('light flat target only')
    observed=(selection>0)&(alpha==0)
    if not observed.any() or not np.any(alpha==255) or (selection>0).sum()>alpha.size*.20:
        raise ValueError('bounded local selection, observed floor and opaque anchor required')
    spread=rgb.max(axis=2).astype(int)-rgb.min(axis=2).astype(int)
    white=(rgb.min(axis=2)>=235)&(spread<=12)
    if np.mean(white[observed])<.10:
        raise ValueError('insufficient near-white floor context; no dark-floor extrapolation')
    a=alpha[:,:,None].astype(float)/255
    m=selection[:,:,None].astype(float)/255
    baseline=np.rint(rgb*a+background*(1-a)).astype(np.uint8)
    floor=background*((1-m)+m*(rgb.astype(float)/255))
    output=np.rint(rgb*a+floor*(1-a)).clip(0,255).astype(np.uint8)
    if not np.array_equal(output[alpha==255], rgb[alpha==255]) or not np.array_equal(output[selection==0],baseline[selection==0]):
        raise RuntimeError('native RGB or outside-selection invariant failed')
    return output,dict(schema='reviewed-chromatic-floor-illustration.v1',
        source_sha256=digest(rgb),alpha_sha256=digest(alpha),background_sha256=digest(background),
        selection_sha256=digest(selection),output_sha256=digest(output),review_note=review_note,
        changed_pixels=int(np.any(output!=baseline,axis=2).sum()),opaque_source_rgb_exact=True,
        outside_selection_baseline_exact=True,source_resampled=False,alpha_changed=False,
        default_pipeline_changed=False,accepted=False,physical_relighting=False,
        recovered_shadow=False,recovered_emission=False,near_white_observed_fraction=float(np.mean(white[observed])),
        limitations=['Manual floor selection is not semantic truth.',
          'Uncalibrated sRGB channel-wise tint is an illustration, not physical light or emission recovery.',
          'Observed source floor may carry unwanted texture, artifacts, objects or color casts.',
          'Fractional-alpha boundaries may double-darken; inspect each native output.',
          'Light flat targets only; no automatic acceptance or default integration.'])
