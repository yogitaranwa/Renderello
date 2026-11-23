"""Reviewed native low-alpha background exclusion, not automatic segmentation.

An explicit reviewer mask declares known background. Nothing is selected from
alpha values alone. The ceiling and distance checks are safety refusals, not
proof that low coverage is unimportant: real fine details can have low alpha.
Use on reviewed opaque sources only. No default pipeline integration.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from .candidate_diagnostics import native_rgb, native_alpha, digest


def exclude_reviewed_background(rgb, alpha, background, protected, *,
        source_sha256, alpha_sha256, background_sha256, protected_sha256,
        reviewed=False, material=None, review_note='', ceiling=32, clearance=12, seam_ceiling=2):
    """Zero exactly the reviewed background mask, or refuse the entire request.

    Source RGB and alpha outside the mask remain exact. Protected fractional
    alpha is supported. Inputs are never modified. No RGB decontamination,
    foreground recovery, model-quality ranking, or publication acceptance.
    """
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP native allocation budget; no resize')
    if reviewed is not True or material!='opaque':
        raise ValueError('explicit opaque source/background review required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('nonempty review note up to2000characters required')
    for mask in (background,protected):
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=alpha.shape or not mask.any():
            raise ValueError('nonempty native boolean background/protected masks required')
    for array,expected in ((rgb,source_sha256),(alpha,alpha_sha256),
                           (background,background_sha256),(protected,protected_sha256)):
        if digest(array)!=expected:raise ValueError('source/alpha/region binding mismatch')
    if type(ceiling) is not int or not 1<=ceiling<=32:
        raise ValueError('integer ceiling1..32 required')
    if type(clearance) is not int or not 4<=clearance<=64:
        raise ValueError('integer clearance4..64 native pixels required')
    if type(seam_ceiling) is not int or not 1<=seam_ceiling<=4:
        raise ValueError('integer seam ceiling1..4 required; no high-jump override')
    if np.any(background&protected):raise ValueError('background crosses protected region')
    if not np.any(protected&(alpha>0)):
        raise ValueError('protected mask must retain some prior foreground')
    if np.any(alpha[background]>ceiling):
        raise ValueError('background exceeds low-alpha ceiling; no partial silent exclusion')
    high=alpha>=128
    if not high.any():raise ValueError('high-alpha foreground anchor required')
    # Include protected low-alpha details as clearance anchors, not just opacity.
    anchors=high|protected
    distances=distance_transform_edt(~anchors)
    if np.any(distances[background]<clearance):
        raise ValueError('background violates foreground/detail clearance')
    # Only internal selection borders create an edit seam. The image frame
    # has no untouched neighbor and is deliberately not treated as a seam.
    seam=background&binary_dilation(~background,structure=np.ones((3,3),bool),border_value=0)
    seam_peak=int(alpha[seam].max()) if seam.any() else 0
    if seam_peak>seam_ceiling:
        raise ValueError('reviewed background creates an abrupt alpha seam; no high-jump override')
    out=alpha.copy();out[background]=0
    changed=out!=alpha
    if np.any(changed&~background) or not np.array_equal(out[protected],alpha[protected]):
        raise RuntimeError('background exclusion invariant failed')
    return rgb.copy(),out,dict(schema='reviewed-background-exclusion.v1',
        source_sha256=digest(rgb),alpha_sha256=digest(alpha),
        background_sha256=digest(background),protected_sha256=digest(protected),
        output_alpha_sha256=digest(out),review_note=review_note,material=material,
        ceiling=ceiling,clearance=clearance,seam_ceiling=seam_ceiling,
        seam_peak_removed_alpha=seam_peak,reviewed_seam_pixels=int(seam.sum()),
        reviewed_background_pixels=int(background.sum()),
        minimum_background_clearance=float(distances[background].min()),
        alpha_changed_pixels=int(changed.sum()),removed_alpha_units=int(alpha[changed].astype(np.int64).sum()),
        maximum_removed_alpha=int(alpha[changed].max()) if changed.any() else 0,
        protected_alpha_exact=True,outside_reviewed_background_exact=True,all_source_rgb_exact=True,
        source_resampled=False,accepted=False,default_pipeline_changed=False,
        independent_ground_truth=False,physical_recovery=False,
        limitations=['Reviewer-declared background is not segmentation truth.',
         'Fine low-alpha details can be erased by a wrong region; ceiling is not a semantic safeguard.',
         'Near-edge halos and source color contamination remain unchanged.',
         'Explicit zeroing can reveal a region seam. Inspect native light/dark outputs.',
         'Not for glass, transmission, soft shadows, smoke or inferred background.'])
