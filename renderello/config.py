"""Runtime configuration: paths, device, model identifiers, seeds.

Model ids are pinned to the exact repos verified in the research plan
(owning-source model cards). Heavy frameworks are imported lazily by the modules
that need them, so the CPU test path never requires torch/diffusers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Owning-source model repos (see README license register).
QWEN_EDIT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"          # Apache-2.0
BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"                 # MIT
RMBG2_MODEL_ID = "briaai/RMBG-2.0"                        # CC BY-NC 4.0 - restricted, see README
IC_LIGHT_REPO = "lllyasviel/ic-light"                     # code Apache-2.0 / weights OpenRAIL-M

MAGICHOUR_API_BASE = os.environ.get("MAGICHOUR_API_BASE", "https://api.magichour.ai")

DEFAULT_OUTPUT_ROOT = Path(os.environ.get("RENDERELLO_OUT", "runs"))
CACHE_DIR = Path(os.environ.get("RENDERELLO_CACHE", ".cache/renderello"))

# Qwen-Image-Edit-2511 inference defaults per the model card quick-start.
QWEN_DEFAULT_STEPS = 40
QWEN_TRUE_CFG_SCALE = 4.0
QWEN_GUIDANCE_SCALE = 1.0

# Working resolutions. Longest side is capped; product is never upscaled past source.
MAX_WORKING_SIDE = 1024
MIN_WORKING_SIDE = 512


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved execution device. GPU work is opt-in; CPU is the default so the
    deterministic route and the whole test suite run anywhere."""
    device: str = "cpu"
    dtype_name: str = "float32"  # bfloat16 on CUDA per Qwen model card

    @staticmethod
    def resolve(prefer_gpu: bool = False) -> "DeviceConfig":
        if prefer_gpu:
            try:
                import torch  # lazy: GPU hosts only
                if torch.cuda.is_available():
                    return DeviceConfig(device="cuda", dtype_name="bfloat16")
            except ImportError:
                pass
        return DeviceConfig()


@dataclass(frozen=True)
class RunConfig:
    """Per-run knobs. Seeds are explicit so benchmark runs are reproducible and
    first-try results stay honestly attributable to a recorded seed."""
    seed: int = 0
    variants: int = 2                 # problem statement: generate 2-4 alternatives
    max_variants: int = 4
    out_root: Path = DEFAULT_OUTPUT_ROOT
    device: DeviceConfig = field(default_factory=DeviceConfig)
    save_intermediates: bool = True   # cutouts, backgrounds, pre-relight - auditability

    def variant_seeds(self) -> list[int]:
        n = max(1, min(self.variants, self.max_variants))
        return [self.seed + i for i in range(n)]
