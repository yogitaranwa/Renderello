"""IC-Light GPU relight (foreground-conditioned, text-to-scene).

Owning repo: github.com/lllyasviel/IC-Light (code Apache-2.0, weights
OpenRAIL-M; attribution required in any derivative display). This is the
bg_source=NONE variant of the official gradio_demo.py `process()`: the scene
is generated from the text prompt around a foreground composited on gray 127.

VALIDATED 2026-11-09 on the owner's Kaggle (2x Tesla T4, torch 2.11.0+cu130,
diffusers 0.42.0): 3 relights, 32.1 / 37.9 / 36.5 s each (512x640, 25 steps,
highres x1.5). See COVERAGE.md for the measured limits. IMPORTANT: the output
is a REGENERATION, not a pixel edit - product details drifted in that test
(sneaker LED sole lost, can graphic altered). Never ship these pixels as the
product; use them as scene/lighting only and paste source-exact product pixels
back (the pipeline's compose route does that for the product region).

Requires CUDA + `diffusers safetensors transformers`. Nothing here runs on CPU
hosts; the pure-numpy helpers are unit-tested, the model path is not.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from PIL import Image

SD15_NAME = "stablediffusionapi/realistic-vision-v51"  # official demo default
ICLIGHT_URL = ("https://huggingface.co/lllyasviel/ic-light/resolve/main/"
               "iclight_sd15_fc.safetensors")
DEFAULTS = dict(width=512, height=640, steps=25, cfg=2.0, highres_scale=1.5,
                highres_denoise=0.5, seed=12345,
                a_prompt="best quality",
                n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality")


def composite_on_gray(img_rgb: np.ndarray, matte_u8: np.ndarray) -> np.ndarray:
    """Official run_rmbg composite (sigma=0): 127 + (img-127)*alpha."""
    alpha = matte_u8.astype(np.float32) / 255.0
    return (127 + (img_rgb.astype(np.float32) - 127) * alpha[..., None]
            ).clip(0, 255).astype(np.uint8)


def resize_and_center_crop(image: np.ndarray, tw: int, th: int) -> np.ndarray:
    pil = Image.fromarray(image)
    ow, oh = pil.size
    s = max(tw / ow, th / oh)
    rw, rh = int(round(ow * s)), int(round(oh * s))
    r = pil.resize((rw, rh), Image.LANCZOS)
    l, t = (rw - tw) // 2, (rh - th) // 2
    return np.array(r.crop((l, t, l + tw, t + th)))


def highres_size(w: int, h: int, scale: float) -> tuple[int, int]:
    return (int(round(w * scale / 64.0) * 64), int(round(h * scale / 64.0) * 64))


def lighting_gradient(width, height, direction):
    """Official directional initial image; model output is not guaranteed direction-correct."""
    if direction not in ('none', 'left', 'right', 'top', 'bottom'):
        raise ValueError('unknown initial-light direction')
    if direction == 'none':
        return None
    if type(width) is not int or type(height) is not int or width % 64 or height % 64 or not (256 <= width <= 1024 and 256 <= height <= 1024):
        raise ValueError('initial-light dimensions must be multiples of64 within256..1024')
    if direction in ('left', 'right'):
        gradient = np.linspace(255, 0, width) if direction == 'left' else np.linspace(0, 255, width)
        image = np.tile(gradient, (height, 1))
    else:
        gradient = np.linspace(255, 0, height) if direction == 'top' else np.linspace(0, 255, height)
        image = np.tile(gradient[:, None], (1, width))
    return np.stack((image,) * 3, axis=-1).astype(np.uint8)


class ICLight:
    """Lazy-loaded IC-Light (SD1.5 fc). Call .relight(); call .close() after."""

    def __init__(self, weights_path: str = "models/iclight_sd15_fc.safetensors"):
        self.weights_path = weights_path
        self._loaded = False

    def _ensure_weights(self) -> None:
        if os.path.exists(self.weights_path):
            return
        from torch.hub import download_url_to_file
        os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
        download_url_to_file(ICLIGHT_URL, dst=self.weights_path)

    def load(self) -> None:
        import torch
        import safetensors.torch as sf
        from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                               StableDiffusionImg2ImgPipeline,
                               StableDiffusionPipeline, UNet2DConditionModel)
        from diffusers.models.attention_processor import AttnProcessor2_0
        from transformers import CLIPTextModel, CLIPTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("IC-Light needs CUDA; no GPU available")
        self._ensure_weights()
        self.torch = torch
        self.tokenizer = CLIPTokenizer.from_pretrained(SD15_NAME, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(SD15_NAME, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(SD15_NAME, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(SD15_NAME, subfolder="unet")
        with torch.no_grad():  # 8-channel conv_in hack from the official demo
            nc = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size,
                                 unet.conv_in.stride, unet.conv_in.padding)
            nc.weight.zero_()
            nc.weight[:, :4].copy_(unet.conv_in.weight)
            nc.bias = unet.conv_in.bias
            unet.conv_in = nc
        orig_forward = unet.forward

        def hooked(sample, timestep, encoder_hidden_states, **kw):
            cc = kw["cross_attention_kwargs"]["concat_conds"].to(sample)
            cc = torch.cat([cc] * (sample.shape[0] // cc.shape[0]), dim=0)
            kw["cross_attention_kwargs"] = {}
            return orig_forward(torch.cat([sample, cc], dim=1), timestep,
                                encoder_hidden_states, **kw)
        unet.forward = hooked
        off = sf.load_file(self.weights_path)
        base = unet.state_dict()
        unet.load_state_dict({k: base[k] + off[k] for k in base}, strict=True)
        del off, base
        dev = torch.device("cuda")
        self.device = dev
        self.te = te.to(device=dev, dtype=torch.float16)
        self.vae = vae.to(device=dev, dtype=torch.bfloat16)
        self.unet = unet.to(device=dev, dtype=torch.float16)
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())
        sched = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1)
        kw = dict(vae=self.vae, text_encoder=self.te, tokenizer=self.tokenizer,
                  unet=self.unet, scheduler=sched, safety_checker=None,
                  requires_safety_checker=False, feature_extractor=None,
                  image_encoder=None)
        self.t2i = StableDiffusionPipeline(**kw)
        self.i2i = StableDiffusionImg2ImgPipeline(**kw)
        self._loaded = True

    def _enc(self, txt):
        torch = self.torch
        ml = self.tokenizer.model_max_length
        cl = ml - 2
        s, e = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        toks = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[s] + toks[i:i + cl] + [e] for i in range(0, len(toks), cl)]
        chunks = [(c[:ml] if len(c) >= ml else c + [e] * (ml - len(c))) for c in chunks]
        ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        return self.te(ids).last_hidden_state

    def _pair(self, pos, neg):
        torch = self.torch
        c, uc = self._enc(pos), self._enc(neg)
        n = max(len(c), len(uc))
        c = torch.cat([c] * math.ceil(n / len(c)), dim=0)[:n]
        uc = torch.cat([uc] * math.ceil(n / len(uc)), dim=0)[:n]
        return (torch.cat([p[None] for p in c], dim=1),
                torch.cat([p[None] for p in uc], dim=1))

    def _np2pt(self, imgs):
        h = self.torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        return h.movedim(-1, 1)

    def _pt2np(self, imgs):
        return [(x.movedim(0, -1) * 127.5 + 127.5).detach().float().cpu().numpy()
                .clip(0, 255).astype(np.uint8) for x in imgs]

    def relight(self, fg_on_gray: np.ndarray, prompt: str, **over) -> tuple[np.ndarray, float]:
        """fg_on_gray: HxWx3 uint8 (see composite_on_gray). Returns (image, seconds)."""
        p = {**DEFAULTS, **over}
        initial = lighting_gradient(p['width'], p['height'], p.get('initial_light', 'none'))
        lowres_denoise = p.get('lowres_denoise', .9)
        if isinstance(lowres_denoise, bool) or not np.isfinite(lowres_denoise) or not .1 <= lowres_denoise <= 1:
            raise ValueError('lowres_denoise must be finite0.1..1')
        if not self._loaded:
            self.load()
        torch = self.torch
        t0 = time.time()
        with torch.inference_mode():
            rng = torch.Generator(device=self.device).manual_seed(int(p["seed"]))
            w, h = p["width"], p["height"]
            fg = resize_and_center_crop(fg_on_gray, w, h)
            cc = self._np2pt([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
            cc = self.vae.encode(cc).latent_dist.mode() * self.vae.config.scaling_factor
            conds, unconds = self._pair(prompt + ", " + p["a_prompt"], p["n_prompt"])
            if initial is None:
                lat = self.t2i(prompt_embeds=conds, negative_prompt_embeds=unconds,
                               width=w, height=h, num_inference_steps=p["steps"],
                               num_images_per_prompt=1, generator=rng, output_type="latent",
                               guidance_scale=p["cfg"],
                               cross_attention_kwargs={"concat_conds": cc}
                               ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
            else:
                # Official IC-Light directional initial-latent path, not post-hoc grading.
                bg = self._np2pt([initial]).to(device=self.vae.device, dtype=self.vae.dtype)
                bg_latent = self.vae.encode(bg).latent_dist.mode() * self.vae.config.scaling_factor
                lat = self.i2i(image=bg_latent, strength=lowres_denoise,
                    prompt_embeds=conds, negative_prompt_embeds=unconds,
                    width=w, height=h, num_inference_steps=int(round(p['steps'] / lowres_denoise)),
                    num_images_per_prompt=1, generator=rng, output_type='latent',
                    guidance_scale=p['cfg'], cross_attention_kwargs={'concat_conds': cc}
                    ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
            px = self._pt2np(self.vae.decode(lat).sample)
            hw, hh = highres_size(w, h, p["highres_scale"])
            px = [np.array(Image.fromarray(x).resize((hw, hh), Image.LANCZOS)) for x in px]
            px = self._np2pt(px).to(device=self.vae.device, dtype=self.vae.dtype)
            lat = self.vae.encode(px).latent_dist.mode() * self.vae.config.scaling_factor
            lat = lat.to(device=self.unet.device, dtype=self.unet.dtype)
            h2, w2 = lat.shape[2] * 8, lat.shape[3] * 8
            fg = resize_and_center_crop(fg_on_gray, w2, h2)
            cc = self._np2pt([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
            cc = self.vae.encode(cc).latent_dist.mode() * self.vae.config.scaling_factor
            lat = self.i2i(image=lat, strength=p["highres_denoise"], prompt_embeds=conds,
                           negative_prompt_embeds=unconds, width=w2, height=h2,
                           num_inference_steps=int(round(p["steps"] / p["highres_denoise"])),
                           num_images_per_prompt=1, generator=rng, output_type="latent",
                           guidance_scale=p["cfg"],
                           cross_attention_kwargs={"concat_conds": cc}
                           ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
            out = self._pt2np(self.vae.decode(lat).sample)[0]
        return out, round(time.time() - t0, 2)

    def close(self) -> None:
        if self._loaded:
            for a in ("t2i", "i2i", "unet", "vae", "te"):
                delattr(self, a)
            self.torch.cuda.empty_cache()
            self._loaded = False


class SupportPlaneCallback:
    """Experimental plane-only conditioning; compositor removes all product RGB.

    No guarantee of a matching support plane. Must be tested on actual GPU output.
    """
    support_plane = True

    def __init__(self, model, seed=0):
        self.model,self.seed=model,int(seed)

    def __call__(self, canvas, prompt):
        if canvas.ndim!=3 or canvas.shape[2]!=3 or canvas.dtype!=np.uint8:
            raise ValueError('plane canvas must be HxWx3 uint8')
        return self.model.relight(canvas,prompt,seed=self.seed,cfg=4.0,
            n_prompt='bottle, jar, can, box, glass, cup, watch, product, packaging, '
                     'logo, text, person, hands, building, doorway, stairs, pillars, '
                     'floating platform, vertical tabletop, clutter, low quality, distortion')[0]


class FlatlaySurfaceCallback:
    """Opt-in matched-view surface with an explicit designer-selected material."""
    flatlay_surface=True
    MATERIALS={
      'linen':'very fine woven ivory linen fabric, microscopic weave, soft folds only at the far corners',
      'slate':'smooth dark charcoal slate sheet, subtle fine stone grain, no cracks, no large rocks',
      'paper':'smooth matte pale sage green photographic paper, nearly invisible fine grain',
    }
    def __init__(self,model,seed=0,material='linen'):
        if material not in self.MATERIALS:raise ValueError('unknown flatlay material')
        self.model,self.seed,self.material=model,int(seed),material
    def __call__(self,canvas,prompt):
        if canvas.dtype!=np.uint8 or canvas.ndim!=3 or canvas.shape[2]!=3:
            raise ValueError('flatlay canvas must be RGB uint8')
        prompt=('professional commercial product flatlay backdrop, '+self.MATERIALS[self.material]+
                ', true directly overhead camera, continuous flat surface filling every pixel, '
                'empty center for later product placement, subtle soft diagonal window lighting, '
                'gentle tonal gradient, low contrast, high resolution photograph, no walls, no horizon')
        return self.model.relight(canvas,prompt,seed=self.seed,cfg=6.0,
          n_prompt='product, packaging, bottle, watch, glass, cup, text, logo, hands, '
                   'person, wall, horizon, kitchen cabinets, vertical view, side view, '
                   'pedestal, stairs, cracks, coarse texture, stucco, gravel, pebbles, '
                   'leather, embossed pattern, busy texture, cartoon, low quality, distortion')[0]


class CameraMatchedCallback:
    """Reviewer-declared front/elevated empty sweep, without product conditioning.

    Same-angle prompting is a hypothesis to test, never a pose-estimation claim.
    """
    camera_matched = True
    CAMERAS = {
        'front': 'eye-level camera parallel to the vertical backdrop, level camera, low horizon',
        'elevated': 'camera looking slightly down at the floor at 20 degrees, three-quarter product photography angle',
    }
    STYLES = {
        'clean_studio': ('seamless matte ivory paper sweep', 'broad softbox illumination from upper left, subtle natural brightness gradient'),
        'luxury_hero': ('seamless smooth deep charcoal sweep, softly lit floor, restrained dark gray tonal gradient', 'large softbox from upper left, gentle rim light from upper right, readable softly lit center'),
        'editorial_beauty': ('seamless smooth pale blush pink sweep, muted warm pink floor, clean beauty editorial studio', 'large diffused beauty softbox from upper left, bright soft daylight')
    }
    def __init__(self,model,seed=0,pose='front',style='clean_studio'):
        if pose not in self.CAMERAS:raise ValueError('matched sweep supports front or elevated sources only')
        if style not in self.STYLES:raise ValueError('unsupported camera studio style')
        self.model,self.seed,self.camera_pose,self.style=model,int(seed),pose,style
    def __call__(self,canvas,prompt):
        if canvas.dtype!=np.uint8 or canvas.ndim!=3 or canvas.shape[2]!=3:
            raise ValueError('matched scene canvas must be RGB uint8')
        surface,lighting=self.STYLES[self.style]
        prompt=('empty professional photographic studio cyclorama, '+surface+', '
                'continuous flat level floor occupying lower two thirds of photograph, '
                'smooth curved infinity backdrop without a visible wall joint, '+self.CAMERAS[self.camera_pose]+
                ', '+lighting+', '
                'empty studio, no objects, commercially photographed clean backdrop')
        return self.model.relight(canvas,prompt,seed=self.seed,cfg=6.,
            n_prompt='product, bottle, shoe, bag, packaging, text, logo, person, hands, '
                     'table edge, furniture, cabinet, shelf, window, rocks, raised platform, '
                     'pedestal, coarse fabric, busy texture, overhead camera, topdown, '
                     'stairs, corners, dramatic shadows, low quality')[0]


class DarkSweepConditioningCallback(CameraMatchedCallback):
    """Experimental conditioning-only ablation. Same luxury prompt and CFG.

    A dark smooth sweep replaces fixed bright-gray conditioning. No product RGB,
    geometry, or generated-room retouch. Never a physical-lighting/quality claim.
    """
    @staticmethod
    def conditioning_canvas(shape):
        h,w=shape[:2]
        yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
        x=(xx+.5)/w;y=(yy+.5)/h
        spot=np.exp(-((x-.44)**2/.16+(y-.62)**2/.26))
        floor=1/(1+np.exp(-(y-.68)*14))
        lum=18+25*spot+12*floor
        return np.repeat(lum[...,None],3,axis=2).clip(0,255).astype(np.uint8)
    def __init__(self,model,seed=0,pose='front',style='luxury_hero'):
        if style!='luxury_hero':raise ValueError('dark sweep ablation is luxury only')
        super().__init__(model,seed,pose,style)
    def __call__(self,canvas,prompt):
        return super().__call__(self.conditioning_canvas(canvas.shape),prompt)
