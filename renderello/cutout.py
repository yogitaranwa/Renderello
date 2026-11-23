"""Subject cutout. Backend order: BiRefNet (MIT) -> RMBG-2.0 (restricted license,
only if explicitly enabled) -> OpenCV GrabCut (CPU fallback, no download).

Model backends lazy-load and run on the resolved device; the GrabCut fallback
keeps the whole pipeline runnable and testable on CPU with zero downloads. Matte
quality differs by backend and is recorded on the CutoutResult for QA honesty.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import BIREFNET_MODEL_ID, RMBG2_MODEL_ID, CACHE_DIR
from .schemas import CutoutResult


def _bbox_from_matte(matte: np.ndarray) -> tuple[int, int, int, int]:
    # Bounding geometry must contain every saved nonzero alpha pixel. A display
    # threshold here silently discards soft edges later when compose crops.
    ys, xs = np.where(matte > 0)
    if len(xs) == 0:
        raise ValueError("empty cutout matte - no product found in source image")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)


def _save_cutout(img_rgb: np.ndarray, matte: np.ndarray, out_dir: Path,
                 stem: str, backend: str) -> CutoutResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.dstack([img_rgb, matte])
    rgba_path = out_dir / f"{stem}_cutout.png"
    matte_path = out_dir / f"{stem}_matte.png"
    Image.fromarray(rgba, "RGBA").save(rgba_path)
    Image.fromarray(matte, "L").save(matte_path)
    return CutoutResult(
        rgba_path=rgba_path, matte_path=matte_path,
        bbox=_bbox_from_matte(matte), backend=backend,
        coverage=float((matte > 127).mean()),
    )


def grabcut_matte(img_rgb: np.ndarray, iters: int = 8, *, max_pixels: int = 4_000_000) -> np.ndarray:
    """CPU fallback. Two seedings: a centered border-rect for normal framing,
    then a border-color mask for products touching the frame edge (the rect
    seed alone starves GrabCut of foreground there). Raises ValueError when no
    product can be isolated - loud failure beats a silently blank matte."""
    h, w = img_rgb.shape[:2]
    # OpenCV's graph allocation can kill the process before Python can record
    # a failure. Keep a declared CPU working budget, never silently resize the
    # photograph or upscale a proxy matte. Supplied native alpha bypasses this
    # fallback, while explicit callers can choose a different allocation budget.
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("positive integer GrabCut pixel budget required")
    if h * w > max_pixels:
        raise ValueError(f"CPU GrabCut allocation budget exceeded: {w}x{h} > {max_pixels} pixels; "
                         "supply a reviewed native-size alpha or choose an explicit model backend; "
                         "source was not resized")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    def _rect_pass() -> np.ndarray:
        mask = np.zeros((h, w), np.uint8)
        rect = (max(1, w // 50), max(1, h // 50), w - 2 * max(1, w // 50),
                h - 2 * max(1, h // 50))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8)

    def _border_pass() -> np.ndarray:
        border = np.concatenate([
            img_rgb[:5].reshape(-1, 3), img_rgb[-5:].reshape(-1, 3),
            img_rgb[:, :5].reshape(-1, 3), img_rgb[:, -5:].reshape(-1, 3)])
        med = np.median(border, axis=0)
        dist = np.linalg.norm(img_rgb.astype(np.float32) - med, axis=2)
        mask = np.where(dist > 25, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8)

    matte = _rect_pass()
    if (matte > 127).mean() < 0.01:
        matte = _border_pass()
    if (matte > 127).mean() < 0.005:
        raise ValueError("empty cutout matte - no product found in source image")
    # Clean speckle noise and fill pinholes.
    k = np.ones((5, 5), np.uint8)
    matte = cv2.morphologyEx(matte, cv2.MORPH_OPEN, k)
    matte = cv2.morphologyEx(matte, cv2.MORPH_CLOSE, k, iterations=2)
    # Feather edges 1px for compositing.
    return cv2.GaussianBlur(matte, (3, 3), 0)


def _u2netp_entry(img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    from renderello.onnx_matte import u2netp_onnx_matte
    return u2netp_onnx_matte(img_rgb)


def _load_birefnet(device: str):
    from transformers import AutoModelForImageSegmentation  # lazy, GPU host
    kwargs: dict = {}
    if not device.startswith("cuda"):
        # Published BiRefNet weights are fp16; on CPU the default load keeps
        # fp16 biases while inputs are float32 and inference dies with
        # "Input type (float) and bias type (c10::Half) should be the same".
        import torch
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_MODEL_ID, trust_remote_code=True, cache_dir=str(CACHE_DIR),
        **kwargs)
    model.to(device).eval()
    if device.startswith("cuda"):
        # fp16 halves VRAM and is ~2x faster; validated on Kaggle T4 x2
        # (2026-11-09): 0.36 s/img steady-state, matte quality unchanged
        # in the 5-photo comparison test.
        model.half()
    return model


def birefnet_matte(img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """BiRefNet inference (MIT, github.com/ZhengPeng7/BiRefNet). Input normalized
    per the repo's inference recipe (1024x1024, ImageNet stats)."""
    import torch
    from torchvision import transforms
    model = _load_birefnet(device)
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img_rgb).unsqueeze(0).to(device)
    if device.startswith("cuda"):
        x = x.half()  # match model.half() above
    with torch.inference_mode():
        pred = model(x)[-1].sigmoid().float().cpu()
    matte = (pred[0].squeeze().numpy() * 255).astype(np.uint8)
    return cv2.resize(matte, (img_rgb.shape[1], img_rgb.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def rmbg2_matte(img_rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    """BRIA RMBG-2.0. License CC BY-NC 4.0 (non-commercial) - enabled only via
    pipeline flag; restriction recorded in README/plan, owner decision pending."""
    import torch
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    model = AutoModelForImageSegmentation.from_pretrained(
        RMBG2_MODEL_ID, trust_remote_code=True, cache_dir=str(CACHE_DIR))
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    ])
    with torch.inference_mode():
        pred = model(tf(img_rgb).unsqueeze(0).to(device))[-1].sigmoid().cpu()
    matte = (pred[0].squeeze().numpy() * 255).astype(np.uint8)
    return cv2.resize(matte, (img_rgb.shape[1], img_rgb.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


_BACKENDS = {"grabcut": grabcut_matte, "birefnet": birefnet_matte,
             "rmbg2": rmbg2_matte, "u2netp_onnx": _u2netp_entry}


def cutout(source: Image.Image, out_dir: Path, stem: str = "product",
           backend: str = "auto", device: str = "cpu",
           allow_restricted: bool = False) -> CutoutResult:
    """Produce an RGBA cutout + matte. backend='auto' picks the best available: birefnet when torch is
    installed, else u2netp_onnx when onnxruntime is installed, else grabcut. Restricted-license backends must
    be requested explicitly (allow_restricted)."""
    img_rgb = np.array(source.convert("RGB"))
    order: list[str]
    if backend == "auto":
        order = []
        try:
            import torch  # noqa: F401
            order.append("birefnet")
        except ImportError:
            pass
        try:
            import onnxruntime  # noqa: F401
            order.append("u2netp_onnx")
        except ImportError:
            pass
        order.append("grabcut")
    else:
        order = [backend]
    if any(b == "rmbg2" for b in order) and not allow_restricted:
        raise PermissionError("RMBG-2.0 is CC BY-NC 4.0; pass allow_restricted=True "
                              "only after the owner resolves the license question")
    last_err: Exception | None = None
    for name in order:
        try:
            matte = _BACKENDS[name](img_rgb, device) if name != "grabcut" \
                else _BACKENDS[name](img_rgb)
            return _save_cutout(img_rgb, matte, out_dir, stem, name)
        except Exception as e:  # fall through to next backend
            last_err = e
            if backend != "auto":
                raise
            remaining = order[order.index(name) + 1:]
            if remaining:
                warnings.warn(
                    f"cutout backend '{name}' failed "
                    f"({type(e).__name__}: {e}); falling back to "
                    f"'{remaining[0]}' - matte quality is reduced (crude "
                    f"edges, not publish-quality). Fix the error above to "
                    f"use the learned backend.",
                    RuntimeWarning, stacklevel=2)
    raise RuntimeError(f"all cutout backends failed; last error: {last_err}")


def supplied_mask_cutout(source: Image.Image, mask_path: Path, out_dir: Path,
                         stem: str = "product") -> CutoutResult:
    """Use explicit alpha with no model/fallback. Never resize a misaligned mask.
    White=foreground, black=background; grayscale values are soft opacity.
    Mask correctness still requires eye review; source RGB is not edited.
    """
    with Image.open(mask_path) as image:
        if image.mode != "L":
            raise ValueError("supplied mask must be an 8-bit grayscale L-mode image")
        if image.size != source.size:
            raise ValueError("supplied mask must exactly match source dimensions; no auto-resize")
        matte = np.array(image)
    if not np.any(matte > 127):
        raise ValueError("supplied mask has no foreground")
    if not np.any(matte < 128):
        raise ValueError("supplied mask has no background")
    return _save_cutout(np.array(source.convert("RGB")), matte, out_dir, stem, "supplied_mask")
