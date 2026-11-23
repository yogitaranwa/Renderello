"""Generative scene providers - the pluggable backends that create the
environment a verified product cutout is composited into.

Design contract (per the problem statement and owner rules):

- A provider generates the SCENE ONLY. The product is composited afterward
  by compose.composite_with_scene / compose.composite, so product pixels are
  never at the mercy of a generative model on the deterministic routes.
- Every provider declares its cost and its limits honestly:
  procedural   = free, CPU, always available, NOT photoreal-AI quality
  sdxl_lightning = free, needs a GPU (Colab/Kaggle), AI-quality empty scenes
  hf_inference = free-tier hosted API, needs HF token, rate-limited
  magichour    = paid credits, live only with the API key, RENDERELLO_MH_LIVE=1
                 and an owner-authorized credit budget (server-side guards)
- A provider that is not available says so; nothing silently falls back to a
  weaker provider. Callers pick explicitly and the manifest records which
  provider produced each output.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import longprompt
from .compose import procedural_background
from .schemas import TemplateSpec


@dataclass
class ProviderInfo:
    name: str
    label: str
    cost: str
    requires: str
    available: bool
    unavailable_reason: str = ""
    quality_note: str = ""


class SceneProvider:
    name = "base"
    label = "base"
    cost = "unknown"
    quality_note = ""

    def available(self) -> tuple[bool, str]:
        return False, "not implemented"

    def info(self) -> ProviderInfo:
        ok, why = self.available()
        return ProviderInfo(self.name, self.label, self.cost, "",
                            ok, "" if ok else why, self.quality_note)

    def generate_scene(self, size: tuple[int, int], prompt: str, seed: int,
                       template: TemplateSpec,
                       negative: str = "") -> np.ndarray:
        raise NotImplementedError


class ProceduralProvider(SceneProvider):
    """Deterministic procedural backdrop (gradient + texture + bokeh).

    Always available on CPU. Honest labeling: these are graphic backdrops,
    not AI-photoreal scenes. The product itself remains pixel-exact.
    """
    name = "procedural"
    label = "Procedural studio backdrop (CPU, free)"
    cost = "free"
    quality_note = ("Graphic backdrop, not an AI-photoreal scene; product "
                    "pixels stay source-exact.")

    def available(self):
        return True, ""

    def generate_scene(self, size, prompt, seed, template, negative=""):
        return procedural_background(size, template, seed)


class SDXLLightningProvider(SceneProvider):
    """SDXL-Lightning empty-scene generation on a local GPU.

    Free, runs in the Colab notebook (interactive GPU use - the allowed
    Colab usage). 4-step Lightning UNet on SDXL base; long prompts are fed
    through longprompt.encode_long_prompt so no clause is dropped. The scene
    is generated EMPTY (the prompt must not name the product); the verified
    product cutout is composited on top afterward.
    """
    name = "sdxl_lightning"
    label = "SDXL-Lightning AI scene (free, needs GPU)"
    cost = "free (your Colab/Kaggle GPU)"
    quality_note = ("AI-photoreal empty scene; fully opaque product pixels "
                    "composited bit-exact on top (verified scope; edges and "
                    "mask semantics excluded). Lighting/shadow harmony "
                    "reviewed by QA + eye.")

    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self._pipe = None

    def available(self):
        try:
            import torch  # noqa
            import diffusers  # noqa
        except ImportError:
            return False, "torch/diffusers not installed (install on the GPU host)"
        if self.device == "cuda" and not torch.cuda.is_available():
            return False, "no CUDA GPU visible"
        return True, ""

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import (StableDiffusionXLPipeline, EulerDiscreteScheduler,
                               UNet2DConditionModel)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        base_revision = '462165984030d82259a11f4367a4eed129e94a7b'
        lightning_revision = 'c9a24f48e1c025556787b0c58dd67a091ece2e44'
        checkpoint = hf_hub_download(
            repo_id="ByteDance/SDXL-Lightning",
            filename="sdxl_lightning_4step_unet.safetensors",
            revision=lightning_revision)
        unet = UNet2DConditionModel.from_config(base, subfolder="unet",
                                                  revision=base_revision)
        unet = unet.to(self.device, torch.float16)
        weights = load_file(checkpoint, device=self.device)
        unet.load_state_dict(weights)
        del weights
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16",
            revision=base_revision)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to(self.device)
        self._pipe = pipe
        return pipe

    def generate_scene(self, size, prompt, seed, template, negative=""):
        import torch
        pipe = self._load()
        w, h = size
        # SDXL native multiples of 8; keep longest side <= 1024 for speed.
        scale = 1024 / max(w, h)
        gw, gh = max(256, int(w * scale) // 8 * 8), max(256, int(h * scale) // 8 * 8)
        pe, ne, pooled, neg_pooled = longprompt.encode_long_prompt(
            pipe, prompt, self.device, negative)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        kwargs = dict(num_inference_steps=4, guidance_scale=0.0,
                      width=gw, height=gh, generator=gen,
                      prompt_embeds=pe, negative_prompt_embeds=ne)
        if pooled is not None:
            kwargs["pooled_prompt_embeds"] = pooled
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled
        img = pipe(**kwargs).images[0]
        return np.asarray(img.resize((w, h), Image.LANCZOS).convert("RGB"))


class HFInferenceProvider(SceneProvider):
    """Hugging Face hosted inference (free tier, rate-limited).

    Uses the account's token from the HF_TOKEN environment variable - never
    stored in code, Drive or the repo. Scene-only text-to-image; the product
    is composited afterward. Free tier is rate-limited: callers surface
    HTTP 429 honestly instead of retrying in a loop.
    """
    name = "hf_inference"
    label = "Hugging Face hosted model (free tier, rate-limited)"
    cost = "free tier (rate-limited)"
    quality_note = "AI-photoreal empty scene via hosted API; product composited on top."
    DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, model: str | None = None, token_env: str = "HF_TOKEN"):
        self.model = model or os.environ.get("HF_MODEL", self.DEFAULT_MODEL)
        self.token_env = token_env

    def available(self):
        if not os.environ.get(self.token_env):
            return False, f"{self.token_env} not set"
        try:
            import requests  # noqa
        except ImportError:
            return False, "requests not installed"
        return True, ""

    def generate_scene(self, size, prompt, seed, template, negative=""):
        import requests
        token = os.environ[self.token_env]
        url = f"https://router.huggingface.co/hf-inference/models/{self.model}"
        resp = requests.post(
            url, headers={"Authorization": f"Bearer {token}"},
            json={"inputs": prompt, "parameters": {
                "negative_prompt": negative, "seed": seed,
                "width": min(size[0], 1024), "height": min(size[1], 1024),
                "num_inference_steps": 28}},
            timeout=180)
        if resp.status_code == 429:
            raise RuntimeError("HF free-tier rate limit reached (429); "
                               "wait or run the Colab notebook GPU instead")
        if resp.status_code == 503:
            raise RuntimeError("HF model is loading (503); retry shortly")
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.asarray(img.resize(size, Image.LANCZOS))


class MagicHourProvider(SceneProvider):
    """Magic Hour AI - PAID credits, product-preserving scene-compose.

    Live calls stay disabled unless the API key exists AND RENDERELLO_MH_LIVE=1
    is set for the deployment. Every live run also passes the server-side
    aggregate budget guard in studio.py (owner-authorized cap, per-batch
    limit, optional live-balance floor); without a budget the run refuses
    with zero bill.
    """
    name = "magichour"
    label = "Magic Hour AI (paid credits, budget-capped)"
    cost = "paid credits"
    quality_note = ("Background-only generation with a preserved product "
                    "cutout; eye review remains the acceptance gate.")

    def available(self):
        if not os.environ.get("MAGIC_HOUR_API_KEY"):
            return False, "MAGIC_HOUR_API_KEY not set"
        if os.environ.get("RENDERELLO_MH_LIVE") != "1":
            return False, ("not enabled on this deployment "
                           "(RENDERELLO_MH_LIVE unset)")
        return True, ""

    def generate_scene(self, size, prompt, seed, template, negative=""):
        raise RuntimeError(
            "Magic Hour runs through studio.run_studio(provider='magichour') "
            "- the scene-compose route drives scene generation internally")


PROVIDERS: dict[str, type[SceneProvider]] = {
    p.name: p for p in (ProceduralProvider, SDXLLightningProvider,
                        HFInferenceProvider, MagicHourProvider)}


def get_provider(name: str, **kwargs) -> SceneProvider:
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise KeyError(f"unknown scene provider '{name}'. available: "
                       f"{', '.join(sorted(PROVIDERS))}") from None
    return cls(**kwargs) if name == "sdxl_lightning" else (
        cls(**kwargs) if name == "hf_inference" else cls())


def provider_report() -> list[dict]:
    """Availability of every provider, for UIs and manifests."""
    out = []
    for cls in PROVIDERS.values():
        try:
            info = cls().info()
        except Exception as e:
            info = ProviderInfo(cls.name, cls.label, cls.cost, "", False,
                                f"{type(e).__name__}: {e}", "")
        row = info.__dict__
        if cls.name == "magichour":
            try:
                from .backends.magichour import MODEL_TABLE
                row["models"] = {m: t["credits"] for m, t in MODEL_TABLE.items()}
            except Exception:
                pass
        out.append(row)
    return out
