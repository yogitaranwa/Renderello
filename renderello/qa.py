"""Fidelity QA. The problem statement's scoring rubric (1-5, fidelity primary)
plus automated drift metrics. Metrics are computed inside the aligned product
region only; background differences are expected and never penalized.

Alignment: Route 2 outputs embed the (possibly scaled) source crop at a known
offset recorded by the pipeline, so we compare crop-vs-crop. Generative routes
re-segment the output (same cutout backend) and align by bbox before comparing -
approximate, and flagged as such in notes, because generative pixels can shift.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .schemas import QAReport


def _rgb(img: Image.Image | np.ndarray) -> np.ndarray:
    return np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img[..., :3]


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    m = mask > 127
    side = min(a.shape[:2])
    if m.sum() < 64 or side < 3:
        return float("nan")
    ssim_map = structural_similarity(a, b, channel_axis=2, full=True,
                                     data_range=255, win_size=min(7, side if side % 2 else side - 1))[1]
    return float(ssim_map[m].mean())


def masked_l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 127
    if m.sum() == 0:
        return float("nan")
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32))[m].mean())


def color_drift(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Bhattacharyya distance between per-channel histograms in the product
    region. 0 = identical color distribution; label recolors push this up."""
    m = (mask > 127).astype(np.uint8)
    if m.sum() == 0:
        return float("nan")
    total = 0.0
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], m, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], m, [64], [0, 256])
        total += cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
    return float(total / 3.0)


def edge_retention(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of source product edges still present in the output. Catches
    'melted' label text and deformed packaging better than L1 alone."""
    m = mask > 127
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 60, 160) > 0
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 60, 160) > 0
    eb_d = cv2.dilate(eb.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    src = ea & m
    if src.sum() == 0:
        return float("nan")
    return float((src & eb_d).sum() / src.sum())


def critical_error_checks(src: np.ndarray, out: np.ndarray,
                          mask: np.ndarray) -> list[str]:
    """Heuristic tripwires for the plan's critical-error classes. These FLAG
    for human eye review; they do not auto-fail, per the problem statement's
    'record critical errors separately' instruction."""
    errors: list[str] = []
    m = mask > 127
    if m.sum() == 0:
        return ["product matte empty in output alignment - manual review required"]
    l1 = masked_l1(src, out, mask)
    if l1 > 28:
        errors.append(f"large product-region pixel change (L1={l1:.1f}) - possible "
                      "redrawn label/logo; eye-check before use")
    # Localized repaint detector: a mean over the whole matte dilutes small but
    # fatal changes (a relabeled region). Flag when a meaningful fraction of
    # product pixels changed substantially.
    diff = np.abs(src.astype(np.float32) - out.astype(np.float32)).mean(axis=2)
    pct_changed = float(((diff > 60) & m).sum() / m.sum())
    if pct_changed > 0.08:
        errors.append(f"{pct_changed:.0%} of product pixels changed substantially - "
                      "localized repaint (label/logo area?) likely; eye-check")
    drift = color_drift(src, out, mask)
    if drift > 0.35:
        errors.append(f"strong product color shift (Bhattacharyya={drift:.2f}) - "
                      "packaging color may have changed")
    er = edge_retention(src, out, mask)
    if er == er and er < 0.55:  # NaN-safe
        errors.append(f"only {er:.0%} of product edges retained - label text or "
                      "packaging geometry may be deformed")
    # A low global average can conceal a small but important label repaint.
    # This is a review warning, never an OCR or identity verdict.
    from .detail_qa import detail_change_review
    detail = detail_change_review(src, out, mask)
    if detail.get("flagged_patch_count", 0):
        patch = detail["flagged_patches"][0]
        errors.append(f"local product-detail change in {detail['flagged_patch_count']} interior patches "
                      f"(strongest crop-relative box={patch['box']}, high-pass MAE={patch['highpass_mae']:.1f}) - "
                      "possible fine-label/texture change; compare original at full size; not verified OCR")
    return errors


def matte_sanity(matte: np.ndarray, bbox: tuple[int, int, int, int],
                 flag_frame_cut: bool = False) -> list[str]:
    """Cutout-quality tripwires. Route 2's pixel comparisons cannot see a bad
    matte (a kept background slab is source-exact by construction), so the
    matte itself must be scored. Flags feed critical_errors (fidelity cap 2).
    Heuristics tuned on the 21-photo licensed CPU stress set: isolated
    products sit well under these bounds; GrabCut failures on busy or
    low-contrast scenes trip them."""
    flags: list[str] = []
    H, W = matte.shape
    x, y, w, h = bbox
    if w == 0 or h == 0:
        return ["cutout empty - no subject isolated"]
    coverage = (w * h) / (W * H)
    fill = float((matte[y:y + h, x:x + w] > 127).mean())
    if coverage > 0.80 and fill > 0.80:
        flags.append(f"cutout kept {coverage:.0%} of the frame as a solid slab - "
                     "subject isolation failed (busy or low-contrast background)")
    elif coverage > 0.65 and fill > 0.85:
        flags.append(f"cutout covers {coverage:.0%} of frame at {fill:.0%} fill - "
                     "probable background inclusion; eye-check before use")
    if flag_frame_cut:  # opt-in: default behavior unchanged
        from .boundary import frame_edge_contact
        fe = frame_edge_contact(matte)
        if fe["needs_review"]:
            flags.append("product touches the source frame on " + "/".join(fe["touches"]) +
                         " - hard straight cut edges are the photo's own edge; eye-check")
    return flags


def fidelity_score(report: QAReport) -> int:
    """Map metrics to the 1-5 fidelity rubric (5 = pixel-faithful). Thresholds
    are starting points from the plan; Phase 0 measured data recalibrates them
    (recorded honestly here as provisional)."""
    vals = []
    if report.ssim_product is not None and report.ssim_product == report.ssim_product:
        vals.append(min(5, max(1, int(round((report.ssim_product - 0.70) / 0.06)) + 1)))
    if report.color_drift is not None and report.color_drift == report.color_drift:
        vals.append(5 if report.color_drift < 0.08 else
                    4 if report.color_drift < 0.18 else
                    3 if report.color_drift < 0.30 else
                    2 if report.color_drift < 0.45 else 1)
    if report.edge_retained is not None and report.edge_retained == report.edge_retained:
        vals.append(5 if report.edge_retained > 0.95 else
                    4 if report.edge_retained > 0.85 else
                    3 if report.edge_retained > 0.70 else
                    2 if report.edge_retained > 0.55 else 1)
    score = int(round(float(np.mean(vals)))) if vals else 3
    if report.critical_errors:
        score = min(score, 2)  # any critical flag caps fidelity per the rubric
    return max(1, min(5, score))


def evaluate(source: Image.Image, output_path: Path, route: str, template: str,
             seed: int, src_crop_box: tuple[int, int, int, int] | None = None,
             out_region: tuple[int, int, int, int] | None = None,
             matte: np.ndarray | None = None, notes: str = "") -> QAReport:
    """Compare source crop to the aligned output region inside the matte."""
    report = QAReport(source=str(source), output=str(output_path), route=route,
                      template=template, seed=seed, notes=notes)
    out = np.array(Image.open(output_path).convert("RGB"))
    src_full = _rgb(source)
    if src_crop_box is None or matte is None:
        report.notes += " | no alignment supplied; metrics skipped, eye review only"
        report.critical_errors.append("unscored: no product alignment")
        report.score_fidelity = fidelity_score(report)
        return report
    x, y, w, h = src_crop_box
    src = src_full[y:y + h, x:x + w]
    msk = matte[y:y + h, x:x + w]
    if out_region is not None:
        ox, oy, ow, oh = out_region
        region = out[oy:oy + oh, ox:ox + ow]
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
        out = np.zeros_like(src_full)
        out[:] = 0
        out[y:y + h, x:x + w] = region
        out = out[y:y + h, x:x + w]
    else:
        out = cv2.resize(out, (src_full.shape[1], src_full.shape[0]),
                         interpolation=cv2.INTER_AREA)[y:y + h, x:x + w]
    report.ssim_product = masked_ssim(src, out, msk)
    report.l1_product = masked_l1(src, out, msk)
    report.color_drift = color_drift(src, out, msk)
    report.edge_retained = edge_retention(src, out, msk)
    report.critical_errors = critical_error_checks(src, out, msk)
    report.score_fidelity = fidelity_score(report)
    return report


def report_to_csv_row(r: QAReport) -> dict:
    d = asdict(r)
    d["critical_errors"] = " ; ".join(r.critical_errors)
    return d
