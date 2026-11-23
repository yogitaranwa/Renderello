"""Qwen-Image-Edit-2511 local backend (Apache-2.0, owning repo:
huggingface.co/Qwen/Qwen-Image-Edit-2511). GPU only; imported lazily so CPU
installs never touch torch/diffusers. Inference parameters follow the model
card quick-start (true_cfg_scale 4.0, guidance_scale 1.0, 40 steps)."""
from __future__ import annotations

import threading

from PIL import Image

from ..config import (QWEN_EDIT_MODEL_ID, QWEN_DEFAULT_STEPS, QWEN_GUIDANCE_SCALE,
                      QWEN_TRUE_CFG_SCALE, CACHE_DIR, DeviceConfig)


class QwenEditBackend:
    name = "qwen-image-edit-2511"
    _lock = threading.Lock()

    def __init__(self, device: DeviceConfig | None = None, steps: int = QWEN_DEFAULT_STEPS,
                 low_memory: bool = False):
        self.device = device or DeviceConfig.resolve(prefer_gpu=True)
        if self.device.device != "cuda":
            raise RuntimeError("Qwen-Image-Edit-2511 needs a CUDA GPU here; use "
                               "Route 2 (cutout_compose) on CPU or run on Kaggle/Colab")
        self.steps = steps
        # bf16 weights total ~56.6 GB (HF API file listing, 2026-11-09):
        # they do NOT fit T4-class 15 GB cards in bf16. low_memory=True uses
        # sequential CPU offload (needs accelerate): feasible but SLOW.
        # UNTESTED as of v0.1.5 - first GPU run must validate and report.
        self.low_memory = low_memory
        self._pipe = None

    def _load(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline  # per model card
        with self._lock:
            if self._pipe is None:
                dtype = getattr(torch, self.device.dtype_name)
                self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                    QWEN_EDIT_MODEL_ID, torch_dtype=dtype, cache_dir=str(CACHE_DIR))
                if self.low_memory:
                    self._pipe.enable_sequential_cpu_offload()
                else:
                    self._pipe.to(self.device.device)
                self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def edit(self, source: Image.Image, prompt: str, negative_prompt: str,
             seed: int, aspect, reference: Image.Image | None = None) -> Image.Image:
        import torch
        pipe = self._load()
        images = [source.convert("RGB")]
        if reference is not None:  # 2511 accepts multi-image input (model card)
            images.append(reference.convert("RGB"))
        with torch.inference_mode():
            out = pipe(
                image=images,
                prompt=prompt,
                negative_prompt=negative_prompt or " ",
                generator=torch.manual_seed(seed),
                true_cfg_scale=QWEN_TRUE_CFG_SCALE,
                guidance_scale=QWEN_GUIDANCE_SCALE,
                num_inference_steps=self.steps,
                num_images_per_prompt=1,
            )
        return out.images[0]
