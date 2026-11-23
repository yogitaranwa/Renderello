"""Native source-locked compositor with explicit per-object contact spans.

Contact spans are supplied in source coordinates and bound to the mask hash.
They are not estimated 3D support, cannot fix floating source poses, and never
silently bridge separate products onto one ground line. OpenCV distance/blur
and NumPy implement a bounded 2D illustration, not a physical renderer.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import cv2
import numpy as np
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear, srgb


@dataclass(frozen=True)
class ContactSpan:
    name: str
    x0: int
    x1: int
    bottom_min: int
    bottom_max: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 80:
            raise ValueError('short contact name required')
        if any(type(getattr(self, v)) is not int for v in ('x0', 'x1', 'bottom_min', 'bottom_max')):
            raise ValueError('integer contact coordinates required')
        if self.x0 < 0 or self.x1 <= self.x0 or self.bottom_min < 0 or self.bottom_max < self.bottom_min:
            raise ValueError('nonempty contact interval and ordered row range required')


@dataclass(frozen=True)
class ContactPolicy:
    camera: str
    material: str
    framing: str
    source_pose: str
    source_sha256: str
    alpha_sha256: str
    reviewed: bool = False
    opacity: float = .25
    decay: float = 3.0
    length: int = 20
    lateral_sigma: float = 2.0
    cast_dx: int = 5
    flatlay_cast_offset: tuple[int, int] = (0, 0)
    span_taper_px: int = 0

    def __post_init__(self):
        if self.camera not in ('front', 'elevated', 'flatlay'):
            raise ValueError('unsupported camera')
        if self.material not in ('opaque', 'transparent', 'reflective', 'unknown'):
            raise ValueError('unsupported material')
        if self.framing not in ('complete', 'cropped', 'unknown'):
            raise ValueError('unsupported framing')
        if self.source_pose not in ('front', 'elevated', 'overhead', 'suspended', 'unknown'):
            raise ValueError('unsupported source pose')
        if type(self.reviewed) is not bool:
            raise ValueError('reviewed must be boolean')
        for name, lo, hi in [('opacity', 0, .5), ('decay', .5, 12), ('lateral_sigma', 0, 12)]:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(name+' outside configured bounds')
        if type(self.length) is not int or not 1 <= self.length <= 96:
            raise ValueError('length must be 1..96')
        if type(self.cast_dx) is not int or not -64 <= self.cast_dx <= 64:
            raise ValueError('cast_dx must be -64..64')
        if type(self.span_taper_px) is not int or not 0 <= self.span_taper_px <= 64:
            raise ValueError('span taper must be integer 0..64')
        if self.camera == 'flatlay' and self.span_taper_px:
            raise ValueError('span taper applies only to upright contact spans')
        offset = self.flatlay_cast_offset
        if not isinstance(offset, (tuple, list)) or len(offset) != 2 or any(type(v) is not int or not -16 <= v <= 16 for v in offset):
            raise ValueError('flatlay cast offset requires two integer channels -16..16')
        if self.camera != 'flatlay' and any(offset):
            raise ValueError('flatlay cast offset is incompatible with upright camera')
        for h in (self.source_sha256, self.alpha_sha256):
            if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('source and alpha pixel hashes required')


def eligible(rgb, alpha, policy):
    native_rgb(rgb)
    native_alpha(alpha, rgb.shape[:2])
    if not isinstance(policy, ContactPolicy):
        raise ValueError('typed contact policy required')
    if digest(rgb) != policy.source_sha256 or digest(alpha) != policy.alpha_sha256:
        raise ValueError('contact declaration belongs to different source or alpha')
    if not policy.reviewed:
        raise ValueError('explicit source-bound contact review required')
    if policy.material != 'opaque':
        raise ValueError('source-locked contact route only accepts declared opaque materials')
    if policy.framing != 'complete':
        raise ValueError('cannot make complete-product contact claim for cropped/unreviewed source')
    if policy.source_pose in ('suspended', 'unknown'):
        raise ValueError('unknown or suspended source pose needs a different concept')
    if (policy.camera == 'flatlay') != (policy.source_pose == 'overhead'):
        raise ValueError('source and target camera incompatible')
    hard = alpha >= 128
    if not hard.any() or hard.all():
        raise ValueError('nontrivial native mask required')
    if any(edge.any() for edge in (hard[0], hard[-1], hard[:, 0], hard[:, -1])):
        raise ValueError('frame-touching support conflicts with complete contact route')


def contact_map(rgb, alpha, spans, policy):
    eligible(rgb, alpha, policy)
    h, w = alpha.shape
    if len(spans) > 64:
        raise ValueError('too many contact spans')
    if policy.camera == 'flatlay':
        if spans:
            raise ValueError('flatlay uses the whole silhouette, no upright spans')
        # A finite outside-distance field attaches at the silhouette, including
        # real holes, without applying guessed upright geometry to a flatlay.
        hard = alpha >= 128
        distance = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        shade = policy.opacity*np.exp(-distance/policy.decay)
        shade[distance > policy.length] = 0
        offset = policy.flatlay_cast_offset
        method = 'flatlay_finite_outside_distance'
        if any(offset):
            # Explicit 2D art direction, not recovered light or height. Keep a
            # weak attached rim, plus a translated silhouette outside hard RGB.
            dx, dy = offset
            shifted = cv2.warpAffine(hard.astype(np.uint8),
                np.array([[1, 0, dx], [0, 1, dy]], dtype=float), (w, h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
            if int(shifted.sum()) != int(hard.sum()):
                raise ValueError('flatlay cast extends beyond canvas; supply explicit room')
            cast_distance = cv2.distanceTransform((~shifted).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            cast = .8*policy.opacity*np.exp(-cast_distance/policy.decay)
            cast[cast_distance > policy.length] = 0
            shade = np.maximum(.3*shade, cast)
            method = 'authored_flatlay_offset_silhouette'
        shade[hard] = 0
        return shade, dict(method=method, spans=[], flatlay_cast_offset=list(offset),
            light_inferred=False, source_height_inferred=False)
    if not spans:
        raise ValueError('upright route needs explicit per-object contact spans')
    if any(not isinstance(s, ContactSpan) for s in spans):
        raise ValueError('typed contact spans required')
    if len({s.name for s in spans}) != len(spans):
        raise ValueError('unique contact names required')
    # Native lower contour per column. Empty columns remain -1 and never bridge.
    hard = alpha >= 128
    lower = np.where(hard.any(axis=0), h-1-np.argmax(hard[::-1], axis=0), -1)
    shade = np.zeros((h, w), np.float64)
    used_columns = np.zeros(w, bool)
    records = []
    for span in spans:
        if span.x1 > w or span.bottom_max >= h:
            raise ValueError('contact span outside native image')
        cols = np.arange(span.x0, span.x1)
        rows = lower[cols]
        if np.any(rows < span.bottom_min) or np.any(rows > span.bottom_max):
            raise ValueError('contact range contradicts native lower contour; no automatic projection')
        if used_columns[cols].any():
            raise ValueError('overlapping contact spans would double-count support')
        if np.any(rows + policy.length >= h):
            raise ValueError('native canvas too short for finite shadow; supply a larger explicit canvas')
        used_columns[cols] = True
        taper = np.ones(len(cols), np.float64)
        if policy.span_taper_px:
            # Explicit art direction of contact endpoints, not inferred support.
            # Smoothstep has zero slope at either end; no product pixels move.
            edge = np.minimum(np.arange(len(cols)) + .5, np.arange(len(cols))[::-1] + .5)
            t = np.minimum(1., edge / policy.span_taper_px)
            taper = t*t*(3-2*t)
        patch = np.zeros((h, w), np.float64)
        # The first row below each declared lower contour gets the unblurred
        # contact core. Lateral blur only; no vertical shift detaches it.
        for step in range(1, policy.length+1):
            weight = policy.opacity*math.exp(-(step-1)/policy.decay)
            patch[rows+step, cols] = weight*taper
        if policy.lateral_sigma:
            patch = cv2.GaussianBlur(patch, (0, 1), sigmaX=policy.lateral_sigma, sigmaY=0,
                                     borderType=cv2.BORDER_CONSTANT)
        # A deliberately weak same-attached cast, separate from the finite core.
        cast = np.zeros_like(patch)
        for step in range(1, policy.length+1):
            dx = round(policy.cast_dx*(step-1)/max(1, policy.length-1))
            target = cols + dx
            valid = (target >= 0) & (target < w)
            cast[rows[valid]+step, target[valid]] = .28*policy.opacity*math.exp(-(step-1)/(policy.decay*2))*taper[valid]
        patch = np.maximum(patch, cast)
        shade = np.maximum(shade, patch)
        records.append(dict(**asdict(span), columns=len(cols),
                            observed_bottom_range=[int(rows.min()), int(rows.max())], span_taper_px=policy.span_taper_px))
    # Soft edges still receive the background shadow underneath them.
    # Suppressing every nonzero alpha creates a gap on noisy model masks.
    shade[hard] = 0
    if not np.isfinite(shade).all() or np.any((shade < 0) | (shade > .5)):
        raise ValueError('invalid contact attenuation')
    return shade, dict(method='reviewed_native_contact_spans', spans=records)


def render_contact(source, alpha, background, spans, policy):
    native_rgb(background)
    if background.shape != source.shape:
        raise ValueError('explicit native same-canvas background required')
    shade, meta = contact_map(source, alpha, spans, policy)
    shaded = srgb(linear(background)*(1-shade[..., None]))
    result = linear_over(source, alpha, shaded)
    meta.update(policy=asdict(policy), source_resampled=False, source_cropped=False,
                source_rgb_sha256=digest(source), alpha_sha256=digest(alpha),
                background_sha256=digest(background), result_sha256=digest(result),
                opaque_rgb_exact=bool(np.array_equal(result[alpha == 255], source[alpha == 255])),
                changed_background_pixels=int((shade > 0).sum()),
                recovered_3d_geometry=False, automatic_acceptance=False,
                limitation='Reviewed 2D support illustration. Cannot repair pose, source reflection, missing parts or material lighting.')
    return result, shade, meta
