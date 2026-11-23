"""Opt-in native, source-bound intent/coverage guard. No object detection/repair.

Named core masks and thresholds are explicit reviewer hypotheses. This checks
coverage of those pixels only, independently of a possibly incorrect prior.
Passing never verifies full-object completeness, opacity, optics or quality.
"""
import math
import re
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest

def review_instance_presence(rgb, rgba, regions, *, bindings, reviewed=False, review_note=''):
    native_rgb(rgb)
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (*rgb.shape[:2], 4):
        raise ValueError('exact native uint8 RGBA required; no resize')
    native_alpha(rgba[:, :, 3], rgb.shape[:2])
    if reviewed is not True or not isinstance(review_note, str) or (not review_note.strip()) or (len(review_note) > 2000):
        raise ValueError('explicit reviewed intent and bounded note required')
    if not isinstance(bindings, dict) or set(bindings) != {'source', 'candidate'} or bindings['source'] != digest(rgb) or (bindings['candidate'] != digest(rgba)):
        raise ValueError('exact source/candidate bindings required')
    if not isinstance(regions, list) or not 1 <= len(regions) <= 32:
        raise ValueError('one to32 reviewed regions required')
    names = set()
    rows = []
    blockers = []
    changed = int(np.any(rgb != rgba[:, :, :3], 2).sum())
    if changed:
        blockers.append('CANDIDATE_SOURCE_RGB_CHANGED')
    fields = {'name', 'mask', 'mask_sha256', 'intent', 'material', 'alpha_threshold', 'covered_fraction', 'mean_alpha'}
    for region in regions:
        if not isinstance(region, dict) or set(region) != fields:
            raise ValueError('exact region declaration fields required')
        name = region['name']
        mask = region['mask']
        intent = region['intent']
        material = region['material']
        if not isinstance(name, str) or re.fullmatch('[a-z][a-z0-9_-]{0,79}', name) is None or name in names:
            raise ValueError('unique lowercase bounded instance slug required')
        names.add(name)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != rgb.shape[:2] or (not mask.any()):
            raise ValueError('nonempty native boolean core required')
        if region['mask_sha256'] != digest(mask):
            raise ValueError('region binding mismatch')
        if intent not in ('retain', 'exclude') or material not in ('opaque', 'transparent', 'mixed', 'unknown'):
            raise ValueError('explicit region intent/material required')
        threshold = region['alpha_threshold']
        fraction = region['covered_fraction']
        mean = region['mean_alpha']
        if type(threshold) is not int or not 1 <= threshold <= 255:
            raise ValueError('alpha threshold1..255 required')
        for x in [fraction, mean]:
            if type(x) not in (float, int) or not math.isfinite(x) or (not 0 <= x <= 1):
                raise ValueError('finite coverage/mean limits0..1 required')
        if intent == 'retain' and (fraction <= 0 or mean <= 0):
            raise ValueError('nonzero retain limits required')
        for previous in regions[:len(rows)]:
            if previous['intent'] != intent and np.any(previous['mask'] & mask):
                raise ValueError('contradictory retain/exclude cores overlap')
        a = rgba[:, :, 3][mask]
        count = int((a >= threshold).sum())
        n = int(mask.sum())
        mass = int(a.astype(np.uint64).sum())
        covered_ok = count >= fraction * n if intent == 'retain' else count <= fraction * n
        mean_ok = mass >= mean * 255 * n if intent == 'retain' else mass <= mean * 255 * n
        reasons = []
        if not covered_ok:
            reasons.append('RETAIN_COVERAGE_LOW' if intent == 'retain' else 'EXCLUDED_COVERAGE_HIGH')
        if not mean_ok:
            reasons.append('RETAIN_MEAN_ALPHA_LOW' if intent == 'retain' else 'EXCLUDED_MEAN_ALPHA_HIGH')
        blockers.extend((name + ':' + x for x in reasons))
        rows.append(dict(name=name, mask_sha256=digest(mask), intent=intent, material=material, pixels=n, alpha_threshold=threshold, required_covered_fraction=fraction, required_mean_alpha=mean, covered_pixels=count, observed_covered_fraction=count / n, observed_mean_alpha=mass / (255 * n), constraint_satisfied=not reasons, blockers=reasons))
    return dict(schema='reviewed-instance-presence.v1', bindings=bindings.copy(), review_note=review_note, regions=rows, blockers=blockers, declared_coverage_satisfied=not blockers, source_rgb_changed_pixels=changed, photo_accepted=False, object_completeness_verified=False, instance_detection=False, independent_accuracy=None, physical_opacity_verified=False, transparent_optics_verified=False, candidate_selected=False, output_repaired=False, source_resampled=False, default_pipeline_changed=False, limitations=['Names, core masks, material and thresholds are reviewer declarations, not detected instances or truth.', 'Unreviewed objects, boundaries, labels and full-object completeness are not checked.', 'A correct low-alpha transparent object can fail a mistaken retention declaration.', 'Coverage above a threshold is not measured optical opacity or scene appearance.', 'No prior-mask agreement, inpainting, alpha clamping, repair, compositing or publication acceptance.'])

def require_instance_presence(rgb, rgba, regions, **kwargs):
    """Refuse the entire candidate on a failed declared-coverage audit.

    No altered/partly repaired image is returned. On pass the caller still owns
    other importer/model/material/quality checks. Arrays are never modified.
    """
    report = review_instance_presence(rgb, rgba, regions, **kwargs)
    if not report['declared_coverage_satisfied']:
        error = ValueError('reviewed instance coverage refused: ' + ','.join(report['blockers']))
        error.review_report = report
        raise error
    return report
