"""Conservative local-detail tripwire, not OCR or a product-identity verifier.

Whole-product averages can hide a small relabeled area. Compare high-frequency
luminance in overlapping, well-supported interior patches. Uniform brightness
changes are largely removed; resampling, misalignment and genuine texture edits
can still trigger review. Thresholds are provisional, not benchmark-calibrated.
"""
from __future__ import annotations
import cv2
import numpy as np


def detail_change_review(source, output, matte):
    for image in (source, output):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('aligned native uint8 RGB arrays required')
    if source.shape != output.shape:
        raise ValueError('aligned image dimensions must match; no implicit resampling')
    if not isinstance(matte, np.ndarray) or matte.dtype != np.uint8 or matte.shape != source.shape[:2]:
        raise ValueError('aligned uint8 matte required')
    height, width = matte.shape
    if height * width > 4_000_000:
        raise ValueError('detail review limited to four million aligned pixels')
    report = dict(schema='local-detail-review.v1', evaluated_patches=0,
                  flagged_patches=[], flagged_patch_count=0, status='INSUFFICIENT_DETAIL_EVIDENCE',
                  text_verified=False, publication_accepted=False,
                  thresholds_provisional=True,
                  limitations=['Local high-frequency change is not proof of changed text or logo.',
                               'Misalignment, blur and resampling may trigger review.',
                               'No flag does not establish label fidelity or publication readiness.'])
    if min(height, width) < 32:
        return report
    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
            for image in (source, output)]
    high = [g - cv2.GaussianBlur(g, (0, 0), 1.2) for g in gray]
    # Do not treat fractional alpha or source-background boundaries as label detail.
    interior = cv2.erode((matte == 255).astype(np.uint8),
                        np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT,
                        borderValue=0).astype(bool)
    def positions(length):
        return sorted(set(range(0, length - 31, 16)) | {length - 32})
    flagged = []
    for y in positions(height):
        for x in positions(width):
            section = np.s_[y:y + 32, x:x + 32]
            valid = interior[section]
            if valid.mean() < .9:
                continue
            energy = float(np.abs(high[0][section][valid]).mean())
            if energy < 2:
                continue
            report['evaluated_patches'] += 1
            difference = np.abs(high[0][section] - high[1][section])[valid]
            error = float(difference.mean())
            changed = float((difference > 12).mean())
            if error > 10 and changed > .25:
                flagged.append(dict(box=[x, y, 32, 32], highpass_mae=round(error, 4),
                                    changed_detail_fraction=round(changed, 4),
                                    source_detail_energy=round(energy, 4)))
    # Store bounded evidence, retaining the strongest changes rather than an
    # unbounded set of near-duplicate overlapping boxes.
    flagged.sort(key=lambda item: (-item['highpass_mae'], item['box']))
    report['flagged_patch_count'] = len(flagged)
    report['flagged_patches'] = flagged[:24]
    if report['evaluated_patches']:
        report['status'] = 'LOCAL_DETAIL_CHANGE_REVIEW_REQUIRED' if flagged else 'NO_LOCAL_TRIPWIRE_NOT_A_PASS'
    return report
