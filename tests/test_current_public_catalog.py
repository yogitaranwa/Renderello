"""2026-11-14 owning documentation snapshot, not an account-specific quote."""
import pytest
from renderello.backends.magichour import MODEL_TABLE, MagicHourBackend
from renderello.costs import MODEL_CREDITS_FROM, projected_usd_per_output
from renderello.mhcompare import plan_compare


def test_legacy_nano_banana_two_uses_two_point_one_floor():
    # https://docs.magichour.ai/api-reference/models explicitly maps the alias.
    assert MODEL_TABLE['nano-banana-2']['credits'] == 50
    assert MODEL_CREDITS_FROM['nano-banana-2'] == 50
    assert MODEL_TABLE['nano-banana-2']['credits'] == MODEL_TABLE['nano-banana-2.1']['credits']
    assert MagicHourBackend('nano-banana-2').credits_per_image() == 50


def test_gpt_image_two_is_supported_without_network():
    b = MagicHourBackend('gpt-image-2')
    assert b.live is False and b.credits_per_image() == 50
    assert MODEL_TABLE['gpt-image-2']['max_inputs'] == 9
    assert MODEL_TABLE['gpt-image-2']['tier'] == 'creator'


@pytest.mark.parametrize('model', sorted(MODEL_TABLE))
def test_backend_and_economics_floor_catalogs_agree(model):
    assert MODEL_CREDITS_FROM[model] == MODEL_TABLE[model]['credits']


def test_alias_projection_matches_explicit_replacement_but_is_not_quote():
    alias = plan_compare(['nano-banana-2'])
    new = plan_compare(['nano-banana-2.1'])
    assert alias['kind'] == 'plan_only_no_spend'
    assert alias['total_credits_listed'] == new['total_credits_listed'] == 50
    assert alias['total_projected_usd'] == new['total_projected_usd']
    assert projected_usd_per_output('nano-banana-2', 'creator_annual') == .05
    assert 'lowest-res' in alias['note']
