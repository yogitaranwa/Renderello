"""Opt-in annotated edge chroma transfer, not foreground/lighting recovery.

Targets and same-material donors are supplied by a reviewer in native source
coordinates. Original luminance proxy is retained before gamut clipping. Opaque
edge RGB CAN change; alpha and protected source pixels never do. No default use.
"""
import math
from numbers import Real
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb, native_alpha, digest


def transfer_reviewed_edge_chroma(rgb, alpha, target, donor, protected, *,
        source_sha256, alpha_sha256, review_note, reviewed=False,
        edge_width=8., donor_distance=32., strength=.75):
    native_rgb(rgb)
    native_alpha(alpha, rgb.shape[:2])
    if alpha.size > 1_000_000:
        raise ValueError('1MP research allocation budget; no resize')
    if reviewed is not True or not isinstance(review_note, str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('explicit source/material review note required')
    if digest(rgb) != source_sha256 or digest(alpha) != alpha_sha256:
        raise ValueError('annotation source/alpha binding mismatch')
    for mask in (target, donor, protected):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != alpha.shape:
            raise ValueError('native boolean target, donor and protected masks required')
        if not mask.any():
            raise ValueError('nonempty target, donor and protected masks required')
    if np.any(target & (donor | protected)) or np.any(donor & protected):
        raise ValueError('annotation masks must be disjoint')
    if np.any(alpha[target] < 128) or np.any(alpha[donor] != 255):
        raise ValueError('target needs high-alpha support and donors need opaque support')
    if np.any(alpha[protected] != 255):
        raise ValueError('protected source regions must be opaque')
    for v, lo, hi in ((edge_width, 1, 24), (donor_distance, 1, 96), (strength, 0, 1)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) or not lo < v <= hi:
            raise ValueError('finite bounded controls required')
    hard = alpha >= 128
    if hard.all():
        raise ValueError('explicit background required for edge distance')
    boundary_distance = distance_transform_edt(hard)
    if np.any(boundary_distance[target] > edge_width):
        raise ValueError('target extends beyond declared edge band; no interior repaint')
    distances, indices = distance_transform_edt(~donor, return_indices=True)
    if np.any(distances[target] > donor_distance):
        raise ValueError('target lacks nearby reviewed donors; no partial silent correction')
    # Same-material chroma is a review hypothesis. A weighted sRGB-code luminance
    # proxy is NOT calibrated scene luminance. Preserve it before gamut clipping.
    x = rgb[target].astype(np.float64)
    near = rgb[indices[0][target], indices[1][target]].astype(np.float64)
    weights = np.array([.2126, .7152, .0722])
    candidate = near + ((x @ weights) - (near @ weights))[:, None]
    mixed = x * (1 - strength) + candidate * strength
    clipped = np.any((mixed < 0) | (mixed > 255), axis=1)
    out = rgb.copy()
    out[target] = np.rint(np.clip(mixed, 0, 255)).astype(np.uint8)
    changed = np.any(out != rgb, axis=2)
    if np.any(changed & ~target) or not np.array_equal(out[protected], rgb[protected]):
        raise RuntimeError('source protection invariant failed')
    return out, dict(schema='reviewed-edge-chroma.v1', source_sha256=digest(rgb),
        alpha_sha256=digest(alpha), target_sha256=digest(target), donor_sha256=digest(donor),
        protected_sha256=digest(protected), output_sha256=digest(out), review_note=review_note,
        edge_width=float(edge_width), donor_distance=float(donor_distance), strength=float(strength),
        target_pixels=int(target.sum()), changed_pixels=int(changed.sum()),
        changed_opaque_pixels=int((changed & (alpha == 255)).sum()),
        clipped_pixels=int(clipped.sum()), protected_rgb_exact=True, alpha_changed=False,
        source_resampled=False, fully_source_rgb_exact=not bool(changed.any()),
        physical_recovery=False, automatic_acceptance=False,
        limitation='Annotated same-material chroma hypothesis. Can erase genuine edge color and retain wrong geometry. Opaque edge RGB is edited; no physical relighting.')
