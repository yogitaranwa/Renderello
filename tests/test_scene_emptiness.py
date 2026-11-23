"""Pre-paste emptiness screen: flags only, never a clearance."""
import numpy as np
import pytest

from renderello.scene_emptiness import screen_scene_emptiness


def _two_tone(h=200, w=200, horizon=120):
    img = np.full((h, w, 3), 127, np.uint8)
    img[horizon:] = 176
    return img


def test_clean_two_tone_scene_not_suspect():
    scene = _two_tone()
    r = screen_scene_emptiness(scene, (60, 100, 80, 90))
    assert r["suspect_object_in_scene"] is False
    assert r["contrast_blob_suspect"] is False
    assert r["straight_edge_suspect"] is False
    assert r["needs_eye_review"] is True      # screen never clears eye review
    assert r["screen_only"] is True


def test_object_blob_flagged():
    scene = _two_tone()
    # A rendered bottle: bright vertical rectangle with dark cap and a label.
    scene[110:185, 80:130] = (235, 230, 222)
    scene[100:112, 92:118] = (40, 40, 42)
    scene[130:160, 85:125] = (200, 60, 60)
    r = screen_scene_emptiness(scene, (70, 95, 80, 95))
    assert r["suspect_object_in_scene"] is True
    assert r["needs_eye_review"] is True


def test_straight_edges_in_placement_flagged():
    scene = _two_tone()
    # Softbox-frame-like parallel lines across the placement rectangle.
    scene[110:113, 60:150] = 20
    scene[125:128, 60:150] = 20
    scene[140:143, 60:150] = 20
    scene[155:158, 60:150] = 20
    r = screen_scene_emptiness(scene, (60, 100, 90, 80))
    assert r["straight_edge_suspect"] is True
    assert r["suspect_object_in_scene"] is True


def test_placement_validated():
    scene = _two_tone()
    with pytest.raises(ValueError):
        screen_scene_emptiness(scene, (150, 100, 80, 90))   # outside width
    with pytest.raises(ValueError):
        screen_scene_emptiness(scene, (60, 100, 0, 90))      # empty rect
    with pytest.raises(ValueError):
        screen_scene_emptiness(np.zeros((10, 10), np.uint8), (0, 0, 5, 5))
