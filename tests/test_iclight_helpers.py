"""CPU-only tests for the pure helpers in iclight_gpu (the model path needs a
T4 and was validated separately on Kaggle; it is NOT exercised here)."""
import numpy as np
import pytest

from renderello import iclight_gpu as ic


def test_composite_on_gray_extremes():
    img = np.full((4, 4, 3), 200, np.uint8)
    full = ic.composite_on_gray(img, np.full((4, 4), 255, np.uint8))
    none = ic.composite_on_gray(img, np.zeros((4, 4), np.uint8))
    assert (full == 200).all() and (none == 127).all()


def test_composite_half_alpha():
    img = np.full((2, 2, 3), 227, np.uint8)
    out = ic.composite_on_gray(img, np.full((2, 2), 128, np.uint8))
    assert abs(int(out[0, 0, 0]) - 177) <= 1


@pytest.mark.parametrize("w,h", [(300, 900), (900, 300), (512, 640)])
def test_center_crop_shape(w, h):
    img = np.random.randint(0, 255, (h, w, 3), np.uint8)
    assert ic.resize_and_center_crop(img, 512, 640).shape == (640, 512, 3)


def test_highres_size_multiple_of_64():
    assert ic.highres_size(512, 640, 1.5) == (768, 960)
    w, h = ic.highres_size(500, 600, 1.3)
    assert w % 64 == 0 and h % 64 == 0


def test_no_cuda_raises_cleanly():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("GPU host")
    with pytest.raises(Exception):
        ic.ICLight("/nonexistent/w.safetensors").load()
