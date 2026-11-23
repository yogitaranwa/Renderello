"""U2Net-p ONNX learned matting - the free-host learned cutout backend.

Why this exists: full BiRefNet needs torch + a 444 MB fp16 checkpoint (Colab)
and the BiRefNet-tiny ONNX conversion is 224 MB - neither fits a 512 MB free
Render instance. U2Net-p is a 4.6 MB original-publisher saliency model whose
ONNX conversion runs in ~0.3 s on CPU in well under 200 MB extra RAM. Quality
sits between OpenCV grabcut and BiRefNet: smooth learned edges, weaker on
fine/translucent detail than BiRefNet. The backend name is recorded honestly
in manifests and shown in the UI; never presented as BiRefNet.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_URL = ("https://github.com/danielgatis/rembg/releases/download/"
             "v0.0.0/u2netp.onnx")
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
_INPUT = 320
_session = None


def _model_path() -> Path:
    d = Path(os.environ.get("RENDERELLO_MODEL_DIR", "/tmp/renderello-models"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / "u2netp.onnx"
    if not p.is_file():
        tmp = p.with_suffix(".download")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MODEL_SHA256:
            tmp.unlink()
            raise RuntimeError("u2netp.onnx sha256 mismatch; refusing")
        tmp.rename(p)
    if hashlib.sha256(p.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("cached u2netp.onnx sha256 mismatch; refusing")
    return p


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(str(_model_path()), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    return _session


def u2netp_onnx_matte(img_rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 matte for an HxWx3 uint8 RGB array."""
    h, w = img_rgb.shape[:2]
    session = _get_session()
    im = Image.fromarray(img_rgb).resize((_INPUT, _INPUT), Image.Resampling.BILINEAR)
    x = np.asarray(im).astype(np.float32)
    x /= max(float(x.max()), 1.0)
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    raw = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    if not np.isfinite(raw).all():
        raise ValueError("non-finite u2netp output")
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-8:
        raise ValueError("constant saliency; no reliable foreground")
    alpha = (raw - lo) / (hi - lo) * 255.0
    # Saliency outputs a soft plateau, not a binary matte: a 0.98-alpha interior
    # would blend the whole product with the new backdrop (uniform color
    # shift), and faint nonzero background speckle pollutes the bbox. Stretch
    # so the interior saturates to full opacity and the floor to zero, keeping
    # only a narrow soft band at edges.
    alpha = np.clip((alpha - 32.0) * (255.0 / 192.0), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(alpha).resize((w, h), Image.Resampling.BILINEAR))
