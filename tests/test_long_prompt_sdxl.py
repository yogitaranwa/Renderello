import pytest
from renderello.long_prompt_sdxl import encode_long_sdxl


@pytest.mark.parametrize('prompt',[None,'', '   ',42])
def test_invalid_prompt_fails_before_model_access(prompt):
    with pytest.raises(ValueError):encode_long_sdxl(None,prompt)


def test_large_prompt_is_explicitly_rejected_not_truncated():
    with pytest.raises(ValueError,match='no truncation'):encode_long_sdxl(None,'x'*16001)


def test_nontext_negative_prompt_fails_before_model_access():
    with pytest.raises(ValueError):encode_long_sdxl(None,'empty counter',negative_prompt=[])
