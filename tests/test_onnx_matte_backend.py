"""u2netp_onnx backend: registry, auto-order, sha refusal, real tiny-model inference."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from renderello import cutout  # noqa: E402
from renderello import onnx_matte  # noqa: E402


def test_backend_registered_and_auto_order(monkeypatch, tmp_path):
    assert "u2netp_onnx" in cutout._BACKENDS
    # behavior, not source text: with torch absent, auto must still try
    # u2netp_onnx before grabcut (the torch-free free-host path)
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    calls = []
    def spy_u2(img, device="cpu"):
        calls.append("u2netp_onnx")
        return np.full(img.shape[:2], 255, dtype=np.uint8)
    def spy_grab(img):
        calls.append("grabcut")
        return np.zeros(img.shape[:2], dtype=np.uint8)
    monkeypatch.setitem(cutout._BACKENDS, "u2netp_onnx", spy_u2)
    monkeypatch.setitem(cutout._BACKENDS, "grabcut", spy_grab)
    from PIL import Image
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    res = cutout.cutout(img, tmp_path, backend="auto")
    assert calls == ["u2netp_onnx"], calls
    assert res.backend == "u2netp_onnx"


def test_sha_refusal(tmp_path, monkeypatch):
    bad = tmp_path / "u2netp.onnx"
    bad.write_bytes(b"not the real model")
    monkeypatch.setenv("RENDERELLO_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="sha256"):
        onnx_matte._model_path()


def test_real_inference_if_model_cached():
    model = Path("/tmp/u2netp.onnx")
    if not model.is_file():
        pytest.skip("pinned model not downloaded in this environment")
    import shutil
    cache = Path("/tmp/renderello-models")
    cache.mkdir(exist_ok=True)
    shutil.copy(model, cache / "u2netp.onnx")
    img = (np.random.RandomState(0).rand(64, 48, 3) * 255).astype(np.uint8)
    matte = onnx_matte.u2netp_onnx_matte(img)
    assert matte.shape == (64, 48) and matte.dtype == np.uint8
    assert matte.max() > matte.min()
