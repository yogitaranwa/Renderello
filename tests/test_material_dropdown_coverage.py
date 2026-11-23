"""Regression: every material value the website dropdown offers must resolve
to a specific overlay, not silently fall back to GENERIC_MATERIAL.

The dropdown (webapp/static/js/app.js) offered glossy_plastic and wood, but
MATERIAL_OVERLAYS only had 'glossy' and no 'wood', so those selections
silently produced the generic prompt.
"""
from renderello.presets import GENERIC_MATERIAL, MATERIAL_OVERLAYS, material_overlay

UI_MATERIALS = ['matte_plastic', 'glossy_plastic', 'glass', 'metal',
                'fabric', 'ceramic', 'wood', 'leather']


def test_every_dropdown_material_has_specific_overlay():
    for m in UI_MATERIALS:
        assert material_overlay(m) != GENERIC_MATERIAL, (
            f"dropdown material {m!r} falls back to the generic overlay")


def test_glossy_plastic_matches_glossy_guidance():
    assert MATERIAL_OVERLAYS['glossy_plastic'] == MATERIAL_OVERLAYS['glossy']


def test_unknown_material_still_falls_back_generic():
    assert material_overlay('unobtanium') == GENERIC_MATERIAL
    assert material_overlay(None) == GENERIC_MATERIAL
