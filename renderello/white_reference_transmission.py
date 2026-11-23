"""Research-only transmission hypotheses, never recovered glass optics.

For assumed white illumination/plate P=1 in linear RGB, C=R+T admits
0<=T<=min(C) for achromatic transmission and nonnegative additive R.
Every T in that interval reconstructs C, so reconstruction cannot select T.
The input alpha is estimated spatial coverage, not measured transparency.
"""
import math
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .candidate_harmonization import linear_over
from .refractive_transfer import decode, encode


def white_reference_hypothesis(source, alpha, target, white_samples, *,
                               bindings, transmission_fraction,
                               assume_white_reference=False, reviewed=False,
                               review_note=''):
    native_rgb(source)
    native_alpha(alpha, source.shape[:2])
    native_rgb(target)
    if target.shape != source.shape:
        raise ValueError('native matching target required; no resize')
    if (not isinstance(white_samples, np.ndarray)
            or white_samples.dtype != np.bool_
            or white_samples.shape != alpha.shape
            or int(white_samples.sum()) < 64):
        raise ValueError('at least 64 explicitly reviewed native white samples required')
    if np.any(alpha[white_samples] != 0):
        raise ValueError('white-reference samples must be outside estimated object coverage')
    expected = dict(source=digest(source), alpha=digest(alpha),
                    target=digest(target), white_samples=digest(white_samples))
    if not isinstance(bindings, dict) or bindings != expected:
        raise ValueError('exact source/alpha/target/white-sample bindings required')
    if (reviewed is not True or assume_white_reference is not True
            or not isinstance(review_note, str) or not review_note.strip()
            or len(review_note) > 2000):
        raise ValueError('explicit research white-reference assumption and review required')
    k = transmission_fraction
    if type(k) not in (float, int) or not math.isfinite(k) or not 0 <= k <= 1:
        raise ValueError('finite transmission fraction in 0..1 required')
    white_fraction = float(np.mean(np.all(source[white_samples] >= 250, axis=1)))
    if white_fraction < .95:
        raise ValueError('reviewed exterior samples do not support white-reference shortcut')
    if not np.any(alpha):
        raise ValueError('nonempty estimated object coverage required')
    observed = decode(source)
    upper = np.min(observed, axis=2)
    transmission = k * upper
    residual = observed - transmission[:, :, None]
    prediction = residual + transmission[:, :, None] * decode(target)
    if np.any(prediction < 0) or np.any(prediction > 1):
        raise ValueError('hypothesis exceeds linear gamut; no clipping')
    transformed = encode(prediction)
    candidate = linear_over(transformed, alpha, target)
    reconstructed = encode(residual + transmission[:, :, None])
    report = dict(
        schema='white-reference-transmission-hypothesis.v1', bindings=expected,
        transmission_fraction=k, white_sample_pixels=int(white_samples.sum()),
        white_sample_near_white_fraction=white_fraction,
        layer_reconstruction_changed_pixels=int(np.any(reconstructed != source, axis=2).sum()),
        candidate_changed_source_pixels=int(np.any(candidate != source, axis=2).sum()),
        alpha_unchanged=True, source_resampled=False,
        reference_origin='ASSUMED_UNIFORM_WHITE_NOT_CAPTURED_CLEAN_PLATE',
        transmission_origin='FEASIBLE_PHOTOMETRIC_HYPOTHESIS_NOT_MEASURED',
        flow_origin='IDENTITY_HYPOTHESIS_NOT_REFRACTION',
        real_glass_validated=False, photo_accepted=False, candidate_selected=False,
        source_rgb_locked=False, default_pipeline_changed=False, review_note=review_note,
        limitations=[
            'White exterior samples cannot prove the unseen plate behind the object.',
            'Exact layer reconstruction is an algebraic identity, not optical validation.',
            'Transmission and additive residual cannot be identified from this single source.',
            'Highlights, reflections and dark edges can be removed or retained incorrectly.',
            'No refraction, caustics, geometry, new illumination or missing framing is recovered.',
            'Estimated alpha adds spatial-coverage uncertainty, especially at boundaries.',
            'Changed product RGB is expected; this is not the source-RGB-locked route.'
        ])
    return candidate, transmission, report
