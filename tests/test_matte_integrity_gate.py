"""21 Nov 2026: matte-integrity review gate for paid routes.

Thresholds derive from the measured table in renderello/matte_integrity.py.
Synthetic mattes only; no model weights or network needed.
"""
import numpy as np
import pytest
from renderello.matte_integrity import matte_stats, assess, assess_compact


def _matte(boxes, shape=(400, 400)):
    m = np.zeros(shape, np.uint8)
    for x, y, w, h in boxes:
        m[y:y + h, x:x + w] = 255
    return m


def test_single_compact_product_passes():
    st = matte_stats(_matte([(100, 100, 200, 200)]))
    assert st["components"] == 1 and st["largest_share"] == 1.0
    assert assess(st) is None and assess_compact(st) is None


def test_small_sparse_product_passes():
    # tiny product, one region: low coverage alone never warns
    st = matte_stats(_matte([(180, 180, 40, 40)]))
    assert st["coverage"] < 0.02
    assert assess(st) is None


def test_three_moderate_regions_pass():
    # dominant region must stay >= 40% of the matte (the measured band)
    st = matte_stats(_matte([(10, 10, 80, 80), (150, 10, 50, 50), (300, 300, 55, 55)]))
    assert st["components"] == 3 and st["largest_share"] >= 0.4
    assert assess(st) is None


def test_fragmented_clutter_warns():
    boxes = [(i * 38 % 360, i * 53 % 360, 18, 18) for i in range(9)]
    st = matte_stats(_matte(boxes))
    assert st["components"] > 3
    assert assess(st) is not None and assess_compact(st) is not None


def test_low_largest_share_warns():
    st = matte_stats(_matte([(10, 10, 30, 30), (100, 100, 90, 90), (250, 250, 28, 28)]))
    assert st["components"] == 3 and st["largest_share"] < 0.9
    # largest below 40% only when truly fragmented; here dominant region passes
    assert assess(st) is None
    st2 = matte_stats(_matte([(10, 10, 40, 40), (150, 150, 42, 42), (300, 300, 41, 41)]))
    assert assess(st2) is not None  # 3 near-equal regions, largest < 40%


def test_compact_message_fits_webapp_passthrough():
    boxes = [(i * 38 % 360, i * 53 % 360, 18, 18) for i in range(9)]
    c = assess_compact(matte_stats(_matte(boxes)))
    assert c is not None and len(c) <= 120  # leaves room for prefix+filename


def test_specks_ignored():
    m = _matte([(100, 100, 150, 150)])
    m[0:2, 0:2] = 255  # 4px speck, below 0.2% of the frame
    st = matte_stats(m)
    assert st["components"] == 1
