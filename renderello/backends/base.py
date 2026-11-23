"""Editor backend interface. A backend takes (source image, prompt, negative
prompt, seed, aspect) and returns an edited PIL image. Implementations:
qwen_local (GPU, Apache-2.0), magichour (API, mock by default), stub (tests)."""
from __future__ import annotations

from typing import Protocol

from PIL import Image

from ..schemas import Aspect


class EditorBackend(Protocol):
    name: str

    def edit(self, source: Image.Image, prompt: str, negative_prompt: str,
             seed: int, aspect: Aspect, reference: Image.Image | None = None) -> Image.Image:
        ...


class StubEditor:
    """Deterministic test double: applies a fixed, seed-dependent color grade so
    pipeline/QA behavior is testable without GPU or network. NOT a quality
    signal - it exists only to exercise orchestration code."""
    name = "stub"

    def edit(self, source: Image.Image, prompt: str, negative_prompt: str,
             seed: int, aspect: Aspect, reference: Image.Image | None = None) -> Image.Image:
        import numpy as np
        arr = np.array(source.convert("RGB")).astype(np.float32)
        shift = ((seed % 7) - 3) * 2.0
        arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 255)
        return Image.fromarray(arr.astype(np.uint8)).resize(
            aspect.size, Image.LANCZOS)
