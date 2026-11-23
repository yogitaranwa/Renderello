from renderello.templates import TEMPLATES, all_templates, get_template


def test_at_least_eight_templates():
    assert len(TEMPLATES) >= 8  # problem statement floor


def test_required_categories_present():
    keys = set(TEMPLATES)
    for required in ("clean_studio", "luxury_hero", "editorial_beauty",
                     "lifestyle_kitchen", "natural_outdoor",
                     "minimal_tabletop", "seasonal_holiday", "social_ad"):
        assert required in keys, f"missing required template {required}"


def test_every_template_fully_specified():
    for t in all_templates():
        for field in (t.scene, t.lighting, t.camera, t.mood,
                      t.background_hint, t.shadow):
            assert len(field) > 10, f"{t.key}.{field} too thin"
        assert t.aspect is not None


def test_unknown_template_raises():
    try:
        get_template("does_not_exist")
    except KeyError as e:
        assert "available" in str(e)
    else:
        raise AssertionError("expected KeyError")
