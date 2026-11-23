"""Exact source-bound matte intervention gate. Never repairs or selects a matte.

Caller-reviewed edit/protection regions are declarations, not semantic truth.
The whole candidate is refused rather than silently clamped to the prior.
Passing means only the declared intervention boundary was respected.
"""
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .matte_change_review import ChangeBudget, review_change


def review_candidate(rgb, before, rgba, allowed, protected, *, bindings,
                     reviewed=False, material=None, framing=None,
                     review_note='', budget=None):
    native_rgb(rgb); native_alpha(before, rgb.shape[:2])
    if before.size > 1_000_000:
        raise ValueError('native1MP review limit; no resize')
    if reviewed is not True or material != 'opaque' or framing != 'complete':
        raise ValueError('explicit complete opaque source/region review required')
    if not isinstance(review_note, str) or not review_note.strip() or len(review_note) > 2000:
        raise ValueError('nonempty bounded source/region review note required')
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (*before.shape, 4):
        raise ValueError('exact native uint8 RGBA candidate required')
    for mask in (allowed, protected):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != before.shape or not mask.any():
            raise ValueError('nonempty native boolean edit/protected regions required')
    if np.any(allowed & protected):
        raise ValueError('edit and protected declarations overlap')
    if not np.any(protected & (before > 0)):
        raise ValueError('protected declaration must retain prior foreground')
    arrays = dict(source=rgb, before=before, candidate=rgba, allowed=allowed, protected=protected)
    if not isinstance(bindings, dict) or set(bindings) != set(arrays):
        raise ValueError('exact five source/candidate/region bindings required')
    if any(not isinstance(bindings[n], str) or bindings[n] != digest(a) for n, a in arrays.items()):
        raise ValueError('source/candidate/region binding mismatch')
    if budget is not None and not isinstance(budget, ChangeBudget):
        raise ValueError('typed topology change budget required')
    after = rgba[:, :, 3]
    signed_delta = after.astype(np.int16) - before.astype(np.int16)
    changed = after != before
    rgb_changed = np.any(rgba[:, :, :3] != rgb, axis=2)
    outside = changed & ~allowed
    locked = changed & protected
    topology, delta = review_change(before, after, budget)
    blockers = []
    if rgb_changed.any(): blockers.append('CANDIDATE_SOURCE_RGB_CHANGED')
    if locked.any(): blockers.append('PROTECTED_ALPHA_CHANGED')
    if outside.any(): blockers.append('ALPHA_CHANGED_OUTSIDE_REVIEWED_EDIT_REGION')
    blockers.extend('TOPOLOGY_' + b for b in topology['blockers'])
    return dict(schema='reviewed-candidate-intervention-gate.v1', bindings=bindings.copy(),
        material=material, framing=framing, review_note=review_note,
        review_is_caller_declaration=True, edit_pixels=int(allowed.sum()),
        protected_pixels=int(protected.sum()), changed_alpha_pixels=int(changed.sum()),
        changed_protected_alpha_pixels=int(locked.sum()),
        maximum_protected_alpha_decrease=int(np.maximum(-signed_delta[protected], 0).max()),
        maximum_protected_alpha_increase=int(np.maximum(signed_delta[protected], 0).max()),
        protected_before_fully_opaque_pixels=int(((before == 255) & protected).sum()),
        protected_candidate_fully_opaque_pixels=int(((after == 255) & protected).sum()),
        changed_outside_edit_pixels=int(outside.sum()),
        changed_source_rgb_pixels=int(rgb_changed.sum()),
        originally_opaque_pixels_removed_to_zero=int(((before == 255) & (after == 0)).sum()),
        topology=topology, blockers=blockers,
        intervention_constraints_satisfied=not blockers,
        baseline_accuracy='NOT_VERIFIED', candidate_accuracy='NOT_VERIFIED',
        photo_accepted=False, candidate_selected=False, default_changed=False,
        source_resampled=False, output_repaired=False,
        limitations=['Region intent and source material are caller declarations, not semantic detection.',
            'A wrong prior can remain wrong. Passing this gate is not a quality score.',
            'No candidate is selected, clamped, composited or silently repaired.',
            'No lighting, scene/contact, fine-print readability or glass optics validation.']), delta
