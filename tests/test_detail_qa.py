import numpy as np
import pytest
from PIL import Image, ImageDraw
from renderello.detail_qa import detail_change_review
from renderello.qa import critical_error_checks, masked_l1


def sample():
    image = Image.new('RGB', (320, 480), (218, 218, 218))
    draw = ImageDraw.Draw(image)
    # Small contrasting packaging detail in a large otherwise plain product.
    for y in range(288, 320, 4):
        draw.line((136, y, 167, y), fill=(20, 20, 20), width=1)
    array = np.array(image)
    return array, np.full(array.shape[:2], 255, np.uint8)


def test_small_changed_detail_not_diluted_by_whole_product_average():
    source, matte = sample()
    output = source.copy()
    output[288:320, 136:168] = 218
    assert masked_l1(source, output, matte) < 1
    review = detail_change_review(source, output, matte)
    assert review['status'] == 'LOCAL_DETAIL_CHANGE_REVIEW_REQUIRED'
    assert review['flagged_patch_count'] > 0
    assert any('local product-detail change' in flag for flag in critical_error_checks(source, output, matte))
    assert review['text_verified'] is False and review['publication_accepted'] is False


def test_identical_detail_has_no_tripwire_and_is_not_a_pass():
    source, matte = sample()
    report = detail_change_review(source, source.copy(), matte)
    assert report['status'] == 'NO_LOCAL_TRIPWIRE_NOT_A_PASS'
    assert not report['flagged_patches']


def test_uniform_brightness_is_not_label_change():
    source, matte = sample()
    output = (source.astype(np.int16) + 15).astype(np.uint8)
    report = detail_change_review(source, output, matte)
    assert not report['flagged_patches']


def test_changed_background_and_fractional_boundary_are_excluded():
    source, matte = sample()
    matte[:] = 0
    matte[40:440, 80:240] = 255
    output = source.copy()
    output[:, :60] = 0
    assert not detail_change_review(source, output, matte)['flagged_patches']
    matte[270:340, 120:180] = 254
    output[288:320, 136:168] = 218
    assert not detail_change_review(source, output, matte)['flagged_patches']


@pytest.mark.parametrize('shape', [(1, 80), (31, 80), (80, 31), (32, 32)])
def test_small_or_plain_images_do_not_invent_evidence(shape):
    source = np.full((*shape, 3), 180, np.uint8)
    report = detail_change_review(source, source, np.full(shape, 255, np.uint8))
    assert report['status'] == 'INSUFFICIENT_DETAIL_EVIDENCE'
    assert report['evaluated_patches'] == 0


def test_rejects_implicit_resizing_and_non_native_matte():
    source, matte = sample()
    with pytest.raises(ValueError): detail_change_review(source, source[:100], matte)
    with pytest.raises(ValueError): detail_change_review(source, source, matte.astype(float))
