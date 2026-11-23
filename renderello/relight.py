"""IC-Light relighting wrapper (GPU, optional). Owning repo:
huggingface.co/lllyasviel/ic-light - code Apache-2.0, weights OpenRAIL-M.

IC-Light's public release ships as a Gradio demo over a diffusers fork rather
than a pip package, so this wrapper isolates the integration behind one
function. Until a GPU host with the IC-Light environment exists, callers get a
deterministic CPU 'grading fallback' (directional luminance + warmth shift)
and the result is labeled fallback=True so QA never mistakes it for the real
relight pass.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def relight_available() -> bool:
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def grading_fallback(img: Image.Image, light_direction: str = "left",
                     warmth: float = 6.0, strength: float = 0.18) -> Image.Image:
    """CPU stand-in: directional luminance gradient + mild color-temperature
    shift. Labeled everywhere as a fallback, never as IC-Light output."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    if light_direction in ("left", "right"):
        g = np.linspace(1, -1, w, dtype=np.float32)[None, :, None]
        if light_direction == "right":
            g = -g
    else:
        g = np.linspace(1, -1, h, dtype=np.float32)[:, None, None]
    arr *= 1.0 + g * strength
    arr[..., 0] = np.clip(arr[..., 0] + warmth, 0, 255)   # warm: +R
    arr[..., 2] = np.clip(arr[..., 2] - warmth * 0.6, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def relight(img: Image.Image, template_lighting: str, prefer_gpu: bool = False
            ) -> tuple[Image.Image, bool]:
    """Returns (image, used_fallback). Real IC-Light integration lands with the
    GPU host; until then every caller receives the labeled fallback."""
    if prefer_gpu and relight_available():
        # Integration point: IC-Light foreground-conditioned relight with the
        # template's lighting description as the light prompt. Kept explicit so
        # the swap is one edit on the GPU host.
        raise NotImplementedError("IC-Light environment not installed on this "
                                  "host; run with prefer_gpu=False (fallback) "
                                  "or set up the ic-light environment first")
    direction = "left" if "left" in template_lighting else (
        "right" if "right" in template_lighting else "top")
    warmth = 8.0 if any(k in template_lighting for k in
                        ("warm", "tungsten", "3800K", "4300K", "4800K")) else 2.0
    return grading_fallback(img, direction, warmth), True
