"""Empty-scene prompt builder: long, detailed, never contradictory."""
import pytest

from renderello.empty_scene_prompt import build_empty_scene_prompt


class T:
    scene = "matte ivory stone tabletop against a warm beige seamless wall"
    lighting = "low-contrast diffuse window light from upper left"
    camera = "straight-on at product-label height, 85mm lens"
    mood = "warm restrained editorial finish"


def test_environment_detail_and_exclusions():
    p = build_empty_scene_prompt(T(), aspect_key="1:1", horizon_frac=0.60)
    assert "EMPTY" in p
    for fragment in (T.scene, T.lighting, T.camera, T.mood):
        assert fragment in p            # long detailed environment stays
    for exclusion in ("no product", "no bottle", "no text", "no logo",
                      "no pedestal", "no props", "no packaging"):
        assert exclusion in p
    assert "sixty" in p or "60 percent" in p   # horizon stated approximately
    # No contradiction: no bare product DESCRIPTION precedes the exclusion.
    assert "Place the product" not in p
    assert "Keep the exact product" not in p


def test_environment_only_instruction_included():
    p = build_empty_scene_prompt(T(), instruction="morning light, cooler wall")
    assert "morning light, cooler wall" in p


@pytest.mark.parametrize("bad", ["add the product at center", "bottle on the left",
                                 "show the label clearly", "new packaging"])
def test_product_naming_instruction_refused(bad):
    with pytest.raises(ValueError, match="product"):
        build_empty_scene_prompt(T(), instruction=bad)


def test_horizon_range_validated():
    with pytest.raises(ValueError):
        build_empty_scene_prompt(T(), horizon_frac=0.05)
    with pytest.raises(ValueError):
        build_empty_scene_prompt(T(), horizon_frac=0.95)


def test_minimal_template_still_excludes():
    class Bare:
        scene = lighting = camera = mood = ""
    p = build_empty_scene_prompt(Bare())
    assert "no product" in p and "empty" in p.lower()
