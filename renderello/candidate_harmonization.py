"""Explicit open-source color harmonization candidates, never physical relighting.

The default route changes BACKGROUND ONLY with scikit-image histogram matching.
The optional OpenCV Poisson route changes foreground color, requires explicit
opt-in and returns a changed-pixel map. Neither route recovers source lighting,
material, print accuracy, reflections or refraction. No automatic fallback.
"""
from __future__ import annotations
import importlib.metadata
import math
import cv2
import numpy as np
from skimage.exposure import match_histograms
from .candidate_diagnostics import native_rgb, native_alpha, digest

SOURCES = [
    'https://scikit-image.org/docs/stable/api/skimage.exposure.html',
    'https://scikit-image.org/docs/stable/license.html',
    'https://docs.opencv.org/4.10.0/df/da0/group__photo__clone.html',
    'https://opencv.org/license/',
]


def _fraction(value, upper, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= upper:
        raise ValueError(name + ' outside explicit bounds')


def _binary(mask, shape, name, nonempty=True):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(name + ' must be native boolean mask')
    if nonempty and not mask.any():
        raise ValueError(name + ' must not be empty')


def background_match(background, source, alpha, reference_mask, *, amount=.2,
                     maximum_delta=40, reference_reviewed=False):
    """Adjust a generated background toward reviewed source BACKGROUND samples.

    This preserves source RGB by changing the destination scene before paste.
    The reviewer must supply genuine backdrop samples, not merely assume every
    zero-alpha pixel is background. Sample errors can be caused by bad masks.
    A limited strength and hard change budget prevent silent extreme grading.
    """
    native_rgb(background)
    native_rgb(source)
    native_alpha(alpha, source.shape[:2])
    _binary(reference_mask, alpha.shape, 'reference mask')
    _fraction(amount, .5, 'amount')
    if type(maximum_delta) is not int or not 0 <= maximum_delta <= 80:
        raise ValueError('maximum delta must be integer 0..80')
    if reference_reviewed is not True:
        raise ValueError('reviewed source backdrop samples required')
    if np.any(reference_mask & (alpha != 0)):
        raise ValueError('backdrop samples intersect candidate product support')
    n = int(reference_mask.sum())
    if not 64 <= n <= 1_000_000:
        raise ValueError('64..1000000 background sample pixels required')
    reference = source[reference_mask][:, None, :]
    matched = match_histograms(background, reference, channel_axis=-1)
    if matched.shape != background.shape or not np.isfinite(matched).all():
        raise ValueError('invalid histogram matching result')
    candidate = np.rint(background.astype(float)*(1-amount)+matched*amount)
    if np.any((candidate < 0) | (candidate > 255)):
        raise ValueError('harmonization exceeds RGB gamut')
    candidate = candidate.astype(np.uint8)
    delta = np.abs(candidate.astype(np.int16)-background.astype(np.int16))
    if int(delta.max()) > maximum_delta:
        raise ValueError('harmonization exceeds configured color-change budget')
    report = dict(method='skimage_background_histogram_matching',
        runtime_version=importlib.metadata.version('scikit-image'),
        background_sha256=digest(background), source_sha256=digest(source),
        alpha_sha256=digest(alpha), reference_mask_sha256=digest(reference_mask),
        reference_pixels=n, strength=amount, maximum_observed_delta=int(delta.max()),
        changed_background_pixels=int(np.any(delta, axis=2).sum()),
        source_rgb_changed=False, physical_relighting=False,
        reference_review='caller_declaration_not_automatic_segmentation',
        automatic_acceptance=False, sources=SOURCES[:2])
    return candidate, report


def linear_over(source, alpha, background):
    native_rgb(source)
    native_rgb(background)
    native_alpha(alpha, source.shape[:2])
    if source.shape != background.shape:
        raise ValueError('native same-canvas background required')
    from .checked_research.contact_camera import linear, srgb
    a = alpha[..., None].astype(float)/255
    result = srgb(linear(source)*a+linear(background)*(1-a))
    result[alpha == 255] = source[alpha == 255]
    result[alpha == 0] = background[alpha == 0]
    return result


def poisson_candidate(source, alpha, background, *, protected=None,
                      color_change_permitted=False, mode='normal', strength=.25):
    """Bounded same-coordinate OpenCV cloning, explicitly not source-exact.

    The alpha-over baseline is always retained by the caller. Protected pixels
    (e.g. independently reviewed label interiors) are restored byte-for-byte.
    This restoration can itself create a seam and does not certify print or
    reflectance accuracy. Frame-touching sources are refused, not padded/cropped.
    """
    native_rgb(source)
    native_rgb(background)
    native_alpha(alpha, source.shape[:2])
    if source.shape != background.shape:
        raise ValueError('same native canvas required')
    if color_change_permitted is not True:
        raise ValueError('foreground-color-changing research requires explicit opt-in')
    if mode not in ('normal', 'mixed', 'monochrome'):
        raise ValueError('unsupported Poisson mode')
    _fraction(strength, .5, 'strength')
    hard = alpha >= 128
    if not hard.any() or hard.all():
        raise ValueError('nontrivial foreground required')
    if hard[0].any() or hard[-1].any() or hard[:, 0].any() or hard[:, -1].any():
        raise ValueError('frame-touching mask not eligible for same-canvas Poisson route')
    if protected is None:
        protected = np.zeros(alpha.shape, bool)
    _binary(protected, alpha.shape, 'protected pixels', nonempty=False)
    if np.any(protected & (alpha != 255)):
        raise ValueError('protected pixels must be fully opaque')
    y, x = np.where(hard)
    if x.max()-x.min() < 3 or y.max()-y.min() < 3:
        raise ValueError('foreground too thin for Poisson integration')
    mask = hard.astype(np.uint8)*255
    center = (int((x.min()+x.max()+1)//2), int((y.min()+y.max()+1)//2))
    flags = {'normal': cv2.NORMAL_CLONE, 'mixed': cv2.MIXED_CLONE,
             'monochrome': cv2.MONOCHROME_TRANSFER}
    try:
        cloned = cv2.seamlessClone(cv2.cvtColor(source, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(background, cv2.COLOR_RGB2BGR), mask, center, flags[mode])
    except cv2.error as exc:
        raise RuntimeError('OpenCV Poisson solver failed; no fallback promoted') from exc
    cloned = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
    if cloned.shape != source.shape or cloned.dtype != np.uint8:
        raise ValueError('unexpected Poisson output')
    baseline = linear_over(source, alpha, background)
    result = baseline.copy()
    # Never modify outside hard support or invent semi-transparent optical RGB.
    mixed = np.rint(baseline.astype(float)*(1-strength)+cloned*strength).astype(np.uint8)
    result[hard] = mixed[hard]
    result[protected] = source[protected]
    changed = np.any(result != baseline, axis=2)
    delta = np.abs(result.astype(np.int16)-baseline.astype(np.int16))
    report = dict(method='opencv_'+mode+'_poisson_color_candidate',
        opencv_version=cv2.__version__, strength=strength,
        source_sha256=digest(source), alpha_sha256=digest(alpha),
        background_sha256=digest(background), protected_sha256=digest(protected),
        changed_pixels=int(changed.sum()), maximum_channel_delta=int(delta.max()),
        protected_pixels=int(protected.sum()), protected_rgb_exact=bool(np.array_equal(result[protected], source[protected])),
        entire_product_source_exact=False,
        unchanged_from_alpha_over=not bool(changed.any()),
        physical_relighting=False, optics_recovered=False,
        automatic_acceptance=False, sources=SOURCES[2:])
    return result, changed, report
