import pytest

from renderello.prompts import (FIDELITY_GUARD, build_harmonize_prompt,
                                build_prompt, validate_instruction)
from renderello.schemas import EditRequest
from renderello.templates import get_template


def _req(instruction: str = "") -> EditRequest:
    from pathlib import Path
    return EditRequest(source_path=Path("x.png"),
                       template=get_template("clean_studio"),
                       custom_instruction=instruction)


def test_guard_leads_prompt():
    p = build_prompt(_req())
    assert p.startswith(FIDELITY_GUARD)
    assert "label text and typography" in p
    assert "Change only the background" in p


def test_instruction_appended_and_capped():
    p = build_prompt(_req("soft morning haze " * 40))
    assert "Additional scene instruction" in p
    assert len(p) < 1800


def test_negative_prompt_covers_critical_classes():
    from renderello.prompts import NEGATIVE_PROMPT
    for cls in ("altered logo", "changed label text", "invented text",
                "deformed packaging", "wrong product"):
        assert cls in NEGATIVE_PROMPT


def test_product_altering_instruction_rejected():
    with pytest.raises(ValueError):
        validate_instruction("please change the label to say NEW")
    with pytest.raises(ValueError):
        validate_instruction("edit the logo color")
    assert validate_instruction("warmer evening light") == "warmer evening light"


def test_harmonize_prompt_forbids_product_motion():
    p = build_harmonize_prompt(_req())
    assert "Do not move, resize, rotate or repaint the product" in p


@pytest.mark.parametrize('key',__import__('renderello.templates',fromlist=['TEMPLATES']).TEMPLATES)
def test_all_original_template_fields_reach_main_edit_prompt(key):
    from pathlib import Path
    t=get_template(key)
    p=build_prompt(EditRequest(Path('source.png'),t))
    for field in ['scene','lighting','camera','mood','background_hint','shadow']:
        assert getattr(t,field) in p
    assert p.startswith(FIDELITY_GUARD)
