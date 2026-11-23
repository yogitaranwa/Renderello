"""Deterministic compositing (Route 2 core, CPU-fast, no models):

  procedural background from the template's background_hint
  -> product placement (scale to template framing, canvas from aspect)
  -> synthetic contact shadow matched to the template's shadow spec
  -> alpha blend; product pixels are pasted, never resampled twice, so the
     untransformed interior stays bit-identical to the source cutout.

The generative routes never pass through here; this module is the fidelity-safe
path the benchmark uses to separate 'model quality' from 'pipeline correctness'.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image

from .schemas import Aspect, CutoutResult, TemplateSpec


def _rng_for(seed: int, *parts: str) -> np.random.Generator:
    h = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _parse_hint_color(hint: str) -> tuple[int, int, int]:
    """Map the template's background_hint language to a base color. Deliberately
    small vocabulary - the hint text is ours, so keep the mapping explicit."""
    table = [
        ("white", (250, 250, 250)), ("light gray", (225, 225, 226)),
        ("gray", (200, 200, 202)), ("beige", (232, 222, 208)),
        ("pink", (240, 226, 224)), ("cream", (243, 234, 220)),
        ("green", (168, 190, 158)), ("amber", (176, 124, 62)),
        ("gold", (190, 150, 80)), ("black", (22, 22, 24)),
        ("near-black", (30, 30, 33)), ("dark", (44, 44, 48)),
        ("saturated", (210, 60, 70)),
    ]
    low = hint.lower()
    for key, rgb in table:
        if key in low:
            return rgb
    return (235, 235, 236)


def placement_params(seed: int, template_key: str) -> tuple[float, float, float]:
    """Per-variant composition knobs (fill, x-jitter, baseline). This is what
    makes 2-4 alternatives genuinely distinct compositions instead of re-noised
    copies: only position/scale/background vary, with no generative repaint.
    Downscaling can change product RGB; native exactness needs verification. Centralized so compositing and QA alignment can never drift
    apart silently."""
    rng = _rng_for(seed, template_key, "compose")
    return (float(rng.uniform(0.60, 0.78)),    # fill fraction of canvas height
            float(rng.uniform(-0.09, 0.09)),   # x jitter, fraction of canvas width
            float(rng.uniform(0.84, 0.93)))    # baseline, fraction of canvas height


def procedural_background(size: tuple[int, int], template: TemplateSpec,
                          seed: int) -> np.ndarray:
    """Gradient background (seed-varying angle) with a soft radial halo behind
    the product anchor and optional diagonal light band ('social_ad' style).
    Deterministic per (seed, template)."""
    w, h = size
    base = np.array(_parse_hint_color(template.background_hint), np.float32)
    rng = _rng_for(seed, template.key, "halo")
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    angle = float(rng.uniform(0.0, np.pi))
    ca, sa = abs(np.cos(angle)), abs(np.sin(angle))
    proj = (X * np.cos(angle) + Y * np.sin(angle)) / (w * ca + h * sa + 1e-6)
    direction = -1.0 if "dark" in template.background_hint or "black" in template.background_hint else 1.0
    grad = base[None, None, :] + direction * (proj[..., None] - 0.5) * 36.0
    img = grad.astype(np.float32).copy()

    # Radial halo near the product anchor; center jitters per seed.
    cx = w * (0.5 + float(rng.uniform(-0.06, 0.06)))
    cy = h * (0.42 + float(rng.uniform(-0.05, 0.05)))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / (0.6 * max(w, h))
    halo_strength = 14.0 if "glow" in template.background_hint or "halo" in template.background_hint else 7.0
    img += (1.0 - np.clip(r, 0, 1))[..., None] * halo_strength

    if "diagonal" in template.background_hint:
        band = np.clip(1.0 - np.abs((X / w) - (Y / h) - 0.15) * 3.0, 0, 1)
        img += band[..., None] * 26.0
    # Fine grain keeps gradients from banding after JPEG export.
    img += rng.normal(0, 0.9, (h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def fit_product(cutout_rgba: np.ndarray, bbox: tuple[int, int, int, int],
                canvas: tuple[int, int], fill: float = 0.7, jx: float = 0.0,
                baseline: float = 0.88) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the matte bbox and scale so the product height is `fill` of the
    canvas height (template camera language says ~70%). Never upscales beyond
    source resolution. jx shifts placement horizontally (fraction of canvas
    width); baseline sets where the product's foot sits (fraction of canvas
    height). Returns placed-crop RGBA and its (x, y) offset."""
    x, y, w, h = bbox
    crop = cutout_rgba[y:y + h, x:x + w]
    cw, ch = canvas
    scale = min((ch * fill) / h, (cw * 0.9) / w, 1.0)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) != (w, h):
        # fit_product never upscales. Average premultiplied colour so RGB
        # hidden by zero alpha cannot leak into visible antialiased edges.
        # Keep opaque resizes on the original byte-for-byte OpenCV path.
        if np.all(crop[..., 3] == 255):
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            alpha = crop[..., 3].astype(np.float64) / 255.0
            premultiplied = crop[..., :3].astype(np.float64) * alpha[..., None]
            resized_alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
            resized_rgb = cv2.resize(premultiplied, (nw, nh), interpolation=cv2.INTER_AREA)
            straight = np.zeros_like(resized_rgb)
            np.divide(resized_rgb, resized_alpha[..., None], out=straight,
                      where=resized_alpha[..., None] > 0)
            legacy = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb_bytes = np.clip(np.rint(straight), 0, 255).astype(np.uint8)
            # Where every contributor is opaque, preserve the existing byte
            # resampler exactly, including its integer rounding convention.
            covered = resized_alpha >= 1.0 - 1e-10
            rgb_bytes[covered] = legacy[..., :3][covered]
            crop = np.dstack((rgb_bytes, legacy[..., 3]))
    px = max(0, min((cw - nw) // 2 + int(jx * cw), cw - nw))
    py = max(0, min(int(ch * baseline) - nh, ch - nh))
    return crop, (px, py)


def snap_to_canvas_edges(alpha_full: np.ndarray, placement: tuple[int, int],
                         prod_size: tuple[int, int], canvas: tuple[int, int]) -> tuple[int, int]:
    """If the source matte touches the photo frame on a side, that side of the
    product is a straight cut (the photo's own edge). Float it mid-canvas and it
    reads as a rectangle (v0.4.9 watch1). Snap each such side to the matching
    canvas border so the cut reads as an intentional crop. Pixels are untouched."""
    from .boundary import frame_edge_contact
    px, py = placement; pw, ph = prod_size; cw, ch = canvas
    t = frame_edge_contact(alpha_full)["touches"]
    if "left" in t and "right" not in t:
        px = 0
    elif "right" in t and "left" not in t:
        px = cw - pw
    if "top" in t and "bottom" not in t:
        py = 0
    elif "bottom" in t and "top" not in t:
        py = ch - ph
    return px, py


def cover_fit_if_cut_both_sides(rgba_full: np.ndarray, canvas: tuple[int, int]):
    """When the matte touches BOTH opposite frame edges on an axis (watch1: strap
    runs off two corners), no placement can hide the straight cuts. Scale the whole
    source frame to cover the canvas (center crop) so the cuts fall off-canvas.
    Returns a canvas-sized RGBA or None when this case does not apply. May upscale
    the source (reported by the caller via the crop size, never silent in diag)."""
    from .boundary import frame_edge_contact
    t = frame_edge_contact(rgba_full[..., 3])["touches"]
    if not (("left" in t and "right" in t) or ("top" in t and "bottom" in t)):
        return None
    cw, ch = canvas
    H, W = rgba_full.shape[:2]
    sc = max(cw / W, ch / H)
    nw, nh = max(cw, round(W * sc)), max(ch, round(H * sc))
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4
    # This branch used to filter straight RGBA, allowing hidden background RGB
    # to contaminate visible cover-fit edges, especially with Lanczos upscaling.
    from .alpha_resample import resize_rgba
    big = resize_rgba(rgba_full, (nw, nh), interpolation=interp)
    x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    return big[y0:y0 + ch, x0:x0 + cw].copy()


def anchored_contact_shadow(canvas, placement, matte):
    """Opt-in tight support-plane shadow per connected product component.

    Uses each component's actual lowest alpha support, never the whole image or
    a scene-inferred surface. This is a 2D contact cue, not a physical floor or
    pose detector. Eye review remains necessary, especially for hanging objects.
    """
    cw,ch=canvas;px,py=placement
    if matte.ndim!=2:raise ValueError('matte must be2D')
    h,w=matte.shape
    if not (0<=px<=cw-w and 0<=py<=ch-h):raise ValueError('matte outside canvas')
    layer=np.zeros((ch,cw),np.float32)
    n,labels,stats,_=cv2.connectedComponentsWithStats((matte>127).astype(np.uint8),8)
    for label in range(1,n):
        if stats[label,cv2.CC_STAT_AREA]<max(3,matte.size//10000):continue
        ys,xs=np.where(labels==label);bottom=int(ys.max())
        # Only the lowest narrow band defines contact, not the full lower sixth.
        support=xs[ys>=bottom-max(1,int(stats[label,cv2.CC_STAT_HEIGHT]*.015))]
        left,right=int(support.min()),int(support.max())
        center=px+(left+right)/2;radius=max(3,(right-left+1)/2)
        yy,xx=np.ogrid[:ch,:cw]
        sigma_x=max(3,radius*.65);sigma_y=max(1.5,min(4,h*.009))
        contact=np.exp(-.5*(((xx-center)/sigma_x)**2+((yy-(py+bottom+1))/sigma_y)**2))
        layer=np.maximum(layer,contact*.50)
    return (np.clip(layer,0,1)*255).astype(np.uint8)


def base_footprint_shadow(canvas, placement, matte):
    """Opt-in soft local support cue using per-column lowest opaque alpha.

    This is hand-selected 2D finishing for known upright floor-contact sources,
    not inferred scene depth, light direction or physical cast-shadow recovery.
    """
    cw,ch=canvas;px,py=placement
    if matte.ndim!=2:raise ValueError('matte must be2D')
    h,w=matte.shape
    if not (0<=px<=cw-w and 0<=py<=ch-h):raise ValueError('matte outside canvas')
    visible=matte>127;layer=np.zeros((ch,cw),np.float32)
    yy,xx=np.where(visible)
    if not len(xx):return layer.astype(np.uint8)
    floor=int(yy.max());band=max(2,round(h*.035))
    # Actual bottom contours support indented feet without turning a whole
    # lower-product rectangle into a detached block shadow.
    for x in range(w):
        ys=np.flatnonzero(visible[:,x])
        if len(ys) and ys[-1]>=floor-band:
            y=py+int(ys[-1])+1
            if y<ch:layer[y,px+x]=1.
    sigma_x=max(1.,min(5.,w*.015));sigma_y=max(1.,min(5.,h*.006))
    layer=cv2.GaussianBlur(layer,(0,0),sigmaX=sigma_x,sigmaY=sigma_y)
    peak=float(layer.max())
    if peak>0:layer*=.25/peak
    return np.clip(layer*255,0,64).astype(np.uint8)


def contact_shadow(canvas: tuple[int, int], placement: tuple[int, int],
                   prod_size: tuple[int, int], matte: np.ndarray,
                   style: str) -> np.ndarray:
    """Shadow layer (grayscale alpha). Style words from the template shadow spec:
    'soft/faint/diffused/long' -> large blur; 'crisp/hard/graphic' -> tight blur
    + offset; 'reflection' handled separately by the caller."""
    if style == "base footprint soft":
        return base_footprint_shadow(canvas,placement,matte)
    if "anchored" in style:
        return anchored_contact_shadow(canvas,placement,matte)
    cw, ch = canvas
    px, py = placement
    pw, ph = prod_size
    layer = np.zeros((ch, cw), np.float32)
    # Anchor on the actual matte extent; supports transparent padding supplied
    # by callers. fit_product itself does NOT add padding.
    rows = np.nonzero((matte > 16).sum(axis=1))[0] if len(matte) else []
    if len(rows):
        y1 = int(rows[-1])
        foot_h = min(y1 + 1, max(2, (y1 - int(rows[0]) + 1) // 6))
        foot = matte[y1 - foot_h + 1:y1 + 1, :]
        fh, fw = foot.shape
        region = layer[py + y1 - fh + 1:py + y1 + 1, px:px + pw]
        if region.shape == foot.shape:
            region[:] = np.maximum(region, foot.astype(np.float32) / 255.0)
    hard = any(k in style for k in ("crisp", "hard", "graphic"))
    blur = 7 if hard else 31
    opacity, (dx, dy) = (0.55, (18, 10)) if hard else (0.30, (4, 8))
    if "long" in style:
        dx, dy = 40, 6
    layer = cv2.GaussianBlur(layer, (blur | 1, blur | 1), 0)
    # Translate without wraparound (np.roll made shadows reappear at the
    # opposite canvas edge for frame-touching products).
    layer = cv2.warpAffine(layer, np.float32([[1, 0, dx], [0, 1, dy]]),
                           (cw, ch), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (np.clip(layer, 0, 1) * opacity * 255).astype(np.uint8)


def cover_to_canvas(img_rgb: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Scale an RGB image to cover the canvas and center-crop the overflow.
    Used to fit an owner-supplied reference/backdrop photo to the working aspect."""
    cw, ch = canvas
    H, W = img_rgb.shape[:2]
    sc = max(cw / W, ch / H)
    nw, nh = max(cw, round(W * sc)), max(ch, round(H * sc))
    interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4
    big = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
    x0, y0 = (nw - cw) // 2, (nh - ch) // 2
    return big[y0:y0 + ch, x0:x0 + cw].copy()


def composite(cutout: CutoutResult, template: TemplateSpec, aspect: Aspect,
              seed: int, reflection: bool = False,
              bg_transform=None, reference: np.ndarray | None = None) -> Image.Image:
    """Full Route 2 composite. Returns RGB canvas image. bg_transform, when
    given, grades the BACKGROUND ONLY (e.g. relight fallback) before the
    product is pasted. Native product RGB is used before optional downscaling;
    output exactness is not guaranteed by this routine alone. When
    reference (HxWx3 uint8) is given, it replaces the procedural background:
    the reference photo is cover-fitted to the canvas (center crop)."""
    cw, ch = aspect.size
    rgba = np.array(Image.open(cutout.rgba_path).convert("RGBA"))
    matte = np.array(Image.open(cutout.matte_path).convert("L"))
    fill, jx, baseline = placement_params(seed, template.key)
    crop, (px, py) = fit_product(rgba, cutout.bbox, (cw, ch),
                                 fill=fill, jx=jx, baseline=baseline)
    ph, pw = crop.shape[:2]
    scaled_matte = cv2.resize(matte[cutout.bbox[1]:cutout.bbox[1] + cutout.bbox[3],
                                    cutout.bbox[0]:cutout.bbox[0] + cutout.bbox[2]],
                              (pw, ph), interpolation=cv2.INTER_AREA)

    if reference is not None:
        bg = cover_to_canvas(reference, (cw, ch)).astype(np.float32)
    else:
        bg = procedural_background((cw, ch), template, seed).astype(np.float32)
    # Shadow under the product, matched to the template's shadow language.
    shadow = contact_shadow((cw, ch), (px, py), (pw, ph), scaled_matte,
                            template.shadow).astype(np.float32) / 255.0
    bg *= (1.0 - shadow[..., None] * 0.85)

    # Optional mirror reflection (reflective_surface template).
    if reflection:
        refl = cv2.flip(crop, 0)
        ra = (refl[..., 3:4].astype(np.float32) / 255.0) * 0.35
        fade = np.linspace(0.5, 0.0, ph, dtype=np.float32)[:, None, None]
        ra *= fade
        ry = py + ph
        # A reflection may extend beyond the canvas. Draw its visible prefix
        # rather than dropping the whole reflection when the floor is short.
        # Keep the full-product fade so clipping does not change its falloff.
        visible_h = min(ph, ch - ry)
        if visible_h > 0:
            region = bg[ry:ry + visible_h, px:px + pw]
            visible_alpha = ra[:visible_h]
            region[:] = (region * (1 - visible_alpha)
                         + refl[:visible_h, :, :3].astype(np.float32) * visible_alpha)

    if bg_transform is not None:
        bg_img = bg_transform(Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)))
        bg = np.array(bg_img.convert("RGB")).astype(np.float32)

    # Final paste: straight alpha over, product interior untouched.
    a = (crop[..., 3:4].astype(np.float32) / 255.0)
    region = bg[py:py + ph, px:px + pw]
    region[:] = region * (1 - a) + crop[..., :3].astype(np.float32) * a
    # No global sharpening. fit_product may already downscale for framing;
    # untransformed opaque pixels can be verified separately. Background grain
    # is applied before the final paste.
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")



SCENE_NATIVE_ASPECT = 512 / 640  # IC-Light working canvas (w/h); relight() center-crops to this


def aspect_safe_scene_call(scene_fn, gray: np.ndarray, prompt: str, native: float = SCENE_NATIVE_ASPECT, pad_value: int = 127):
    """Call scene_fn on a canvas whose aspect matches the model's native aspect.

    ICLight.relight() center-crops its input to 512x640. On a non-4:5 canvas
    (landscape/square) that crop shifts and rescales the product, so the
    generated product no longer sits under the pasted one (ghost product
    behind the paste). Fix: pad the gray-127 canvas to the native aspect
    (product untouched, still at the same pixel scale relative to the pad),
    run the model, then crop the original region back out. Canvases already at
    the native aspect are passed through unchanged.
    """
    h, w = gray.shape[:2]
    if abs(w / h - native) < 0.01:
        return np.asarray(scene_fn(gray, prompt))
    if w / h > native:      # too wide: pad top/bottom
        H, W = int(round(w / native)), w
    else:                   # too tall: pad left/right
        H, W = h, int(round(h * native))
    y0, x0 = (H - h) // 2, (W - w) // 2
    padded = np.full((H, W, 3), pad_value, np.uint8)
    padded[y0:y0 + h, x0:x0 + w] = gray
    out = np.asarray(scene_fn(padded, prompt))
    if out.shape[:2] != (H, W):
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LANCZOS4)
    return out[y0:y0 + h, x0:x0 + w]


def support_plane_canvas(shape, support_row):
    """Experimental empty horizontal plane conditioning. No source RGB or shape.

    A single monochrome horizon is18% of height behind the desired foot row.
    This is a layout hypothesis, never evidence of a generated usable surface.
    """
    h,w=shape[:2]
    if not 0<=support_row<h:raise ValueError('support row outside canvas')
    horizon=max(0,min(h-1,int(round(support_row-.18*h))))
    image=np.full((h,w,3),127,np.uint8)
    image[horizon:]=176
    return image,horizon


def composite_with_scene(cutout: CutoutResult, template: TemplateSpec, aspect: Aspect,
                         seed: int, scene_fn, prompt: str | None = None,
                         shadow_style: str | None = None,
                         scene_grade: float = 0.0,
                         snap_frame_cut: bool = False,
                         glass_backdrop=None,
                         scene_auto_lift: bool = False, cache_dir=None,
                         scene_is_aligned: bool = False,
                         scene_accept=None, scene_max_attempts: int = 1,
                         conditioning: str = "product_gray"):
    """Route 2b: scene generated by a relight/scene model (IC-Light) around the
    placed product, then SOURCE-EXACT product pixels pasted back on top.

    scene_fn(canvas_rgb_gray127: HxWx3 uint8, prompt: str) -> HxWx3 uint8 of the
    same size as the canvas (use iclight_gpu.ICLight.relight + a resize).
    Returns (image, diagnostics). The generated product silhouette/shadow can
    survive around the pasted product, so diagnostics reports a ring statistic
    for EYE REVIEW; it is a screen, not proof the result is clean.

    conditioning selects what the model actually receives:
    - "product_gray" (default, legacy): the gray-127 canvas carrying the
      product's RGB-modulated silhouette. The model can see the product shape
      and may render a second one around it (the 20 Nov 2026 duplicate fail).
    - "empty_guide": a product-free two-tone floor/wall guide from
      support_plane_canvas, the horizon pinned at the product's foot row.
      The model never receives any product pixel or silhouette; the caller
      MUST pass an environment-only prompt (renderello.empty_scene_prompt).
      This restores the measured-clean 13 Nov 2026 conditioning.
    """
    if conditioning not in ("product_gray", "empty_guide"):
        raise ValueError(f"unknown conditioning {conditioning!r}")
    cw, ch = aspect.size
    rgba = np.array(Image.open(cutout.rgba_path).convert("RGBA"))
    fill, jx, baseline = placement_params(seed, template.key)
    crop, (px, py) = fit_product(rgba, cutout.bbox, (cw, ch),
                                 fill=fill, jx=jx, baseline=baseline)
    reviewed_overhead = (getattr(prompt, 'staging', None) == 'overhead')
    if reviewed_overhead and (template.key != 'flat_lay' or
                              getattr(prompt, 'reviewed_source_view', None) != 'overhead'):
        raise ValueError('overhead staging requires a reviewed overhead source and flat_lay template')
    if reviewed_overhead:
        import hashlib
        if getattr(prompt, 'reviewed_alpha_sha256', None) != hashlib.sha256(rgba[..., 3].tobytes()).hexdigest():
            raise ValueError('overhead alpha differs from pixel-reviewed complete product matte; callback not called')
    if reviewed_overhead or getattr(scene_fn,'flatlay_surface',False):
        # Flatlay has no floor baseline. Center the actual alpha bounds, not
        # the original photo frame; keep the existing no-upscale policy.
        crop,(px,py)=fit_product(rgba,cutout.bbox,(cw,ch),fill=.76,jx=0.,baseline=.88)
        px,py=flatlay_center(crop[...,3],(cw,ch))
    ph, pw = crop.shape[:2]
    cover_used = False
    if snap_frame_cut:  # opt-in; default placement unchanged
        cover = cover_fit_if_cut_both_sides(rgba, (cw, ch))
        if cover is not None:
            cover_used = True   # product runs off opposite frame edges: keep source framing
            crop, (px, py) = cover, (0, 0)
            ph, pw = crop.shape[:2]
        else:
            px, py = snap_to_canvas_edges(rgba[..., 3], (px, py), (pw, ph), (cw, ch))
    a_full = np.zeros((ch, cw), np.float32)
    a_full[py:py + ph, px:px + pw] = crop[..., 3].astype(np.float32) / 255.0
    gray = np.full((ch, cw, 3), 127, np.float32)
    gray[py:py + ph, px:px + pw] = (127 + (crop[..., :3].astype(np.float32) - 127)
                                   * a_full[py:py + ph, px:px + pw, None])
    prompt = prompt or f"{template.scene}, {template.lighting}, product photography"
    canvas = gray.clip(0, 255).astype(np.uint8)
    plane_horizon=None
    if conditioning == "empty_guide":
        # Product-free model input: only the two-tone floor/wall guide, horizon
        # pinned at the product's lowest opaque row so the model renders the
        # support surface exactly where the product will stand. The prompt
        # MUST come from empty_scene_prompt.build_empty_scene_prompt: a
        # hand-written prompt can carry a product description with a
        # contradictory "no product" tail, which is exactly the measured
        # 20 Nov 2026 duplicate-product failure shape.
        from .empty_scene_prompt import EmptyScenePrompt
        if not isinstance(prompt, EmptyScenePrompt):
            raise ValueError(
                "empty_guide conditioning requires an EmptyScenePrompt from "
                "renderello.empty_scene_prompt.build_empty_scene_prompt")
        support_rows = np.where((crop[..., 3] > 127).any(axis=1))[0]
        if not len(support_rows):
            raise ValueError('empty matte for support plane')
        canvas, plane_horizon = support_plane_canvas(
            gray.shape, py + int(support_rows[-1]))
    elif getattr(scene_fn,'support_plane',False):
        support_rows=np.where((crop[...,3]>127).any(axis=1))[0]
        if not len(support_rows):raise ValueError('empty matte for support plane')
        canvas,plane_horizon=support_plane_canvas(gray.shape,py+int(support_rows[-1]))
        prompt=(f"{template.scene}, {template.lighting}, empty bare level tabletop spanning "
                "the entire foreground, continuous flat horizontal support surface, "
                "table surface clearly visible below a low distant horizon, studio set photography")

    flatlay_used = reviewed_overhead or bool(getattr(scene_fn,'flatlay_surface',False))
    if flatlay_used:
        canvas=np.full_like(canvas,176)
        plane_horizon = None
        if not reviewed_overhead:
            prompt=(f"{template.scene}, {template.lighting}, {template.camera}, "
                "true overhead flat lay photograph, entire image is one continuous "
                "horizontal textured surface seen directly from above, empty center, "
                "no wall, no horizon, no product, no raised platform")
        shadow_style='surface contact'

    if getattr(scene_fn,'camera_matched',False):
        # Empty scene only. Never feed source silhouette/RGB into generation.
        canvas=np.full_like(canvas,176)

    # Direct backdrop is already fitted to the requested aspect. Only model
    # callbacks need IC-Light native-aspect padding/crop correction.
    pad_value = 176 if (flatlay_used or getattr(scene_fn, "camera_matched", False)) else 127

    def _gen_scene():
        s = (np.asarray(scene_fn(canvas, prompt)) if scene_is_aligned else
             aspect_safe_scene_call(scene_fn, canvas, prompt, pad_value=pad_value))
        if s.shape != (ch, cw, 3):
            s = cv2.resize(s, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        return s

    scene = _gen_scene()
    scene_attempts = 1
    scene_acceptance = None
    emptiness = None

    def _screens():
        """Combined acceptance dict: caller scene_accept plus the pre-paste
        emptiness screen (empty_guide conditioning only). Either suspect
        verdict counts as an intrusion so the retry budget regenerates."""
        nonlocal emptiness
        combined = {}
        if scene_accept is not None:
            combined.update(scene_accept(scene))
        if conditioning == "empty_guide":
            from .scene_emptiness import screen_scene_emptiness
            emptiness = screen_scene_emptiness(scene, (px, py, pw, ph))
            combined["emptiness"] = emptiness
            if emptiness["suspect_object_in_scene"]:
                combined["intrusion"] = True
        return combined

    if scene_accept is not None or conditioning == "empty_guide":
        # Heuristic rejection of generated scenes with equipment/frame
        # intrusions (see scene_acceptance.py) or a suspected object where
        # the product will be pasted (see scene_emptiness.py). Regeneration
        # is capped by the CALLER via scene_max_attempts; each attempt may
        # be a paid generation.
        scene_acceptance = _screens()
        while scene_acceptance.get("intrusion") and scene_attempts < max(1, scene_max_attempts):
            scene = _gen_scene()
            scene_attempts += 1
            scene_acceptance = _screens()
    if cache_dir is not None:
        # Lossless layers at the exact aligned output coordinates. No need to
        # run GPU again to try shadows/scene grades or inspect the true alpha.
        from pathlib import Path
        import json
        cd = Path(cache_dir); cd.mkdir(parents=True, exist_ok=True)
        Image.fromarray(scene.astype(np.uint8)).save(cd / "scene.png")
        Image.fromarray(crop).save(cd / "product_rgba.png")
        (cd / "layers.json").write_text(json.dumps({
            "placement": [px, py], "cover_used": cover_used,
            "glass_backdrop": None if glass_backdrop is None else np.asarray(glass_backdrop).tolist(),
            "shadow_style": shadow_style, "scene_grade": scene_grade,
            "scene_auto_lift": scene_auto_lift}, indent=2))
    grounded = False
    if shadow_style and not cover_used and glass_backdrop is None and not reviewed_overhead:
        # Generated scenes carry no real floor contact, so an ungrounded product
        # reads as floating. Darken the SCENE ONLY with a per-component contact
        # ellipse at the product's actual lowest alpha; the product is pasted
        # afterwards by finish_scene and stays bit-exact.
        anchor = anchored_contact_shadow((cw, ch), (px, py), crop[..., 3])
        scene = scene.astype(np.float32) * (1.0 - anchor.astype(np.float32)[..., None] / 255.0)
        grounded = True
    out, scene_lift = finish_scene(scene, crop, (px,py), shadow_style=shadow_style,
                                  scene_grade=scene_grade, glass_backdrop=glass_backdrop,
                                  scene_auto_lift=scene_auto_lift, cover_used=cover_used)
    # Ring diagnostic: generated pixels just outside the pasted product vs the
    # scene far away. Large difference => likely leftover generated silhouette
    # or shadow hugging the product (needs eye review).
    m = (a_full > 0.5).astype(np.uint8)
    k = max(3, int(round(0.01 * max(cw, ch))))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = cv2.dilate(m, kern) - m
    far = 1 - cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * k + 1, 6 * k + 1)))
    diag = {"ring_px": int(ring.sum()), "far_px": int(far.sum())}
    if flatlay_used:
        diag["scene_mode"] = "flatlay_surface_experimental"
        if reviewed_overhead:
            diag["source_view_review"] = "overhead explicitly declared by evaluator"
            diag["foreground_view_change"] = "none"
    if getattr(scene_fn,'camera_matched',False):
        diag['scene_mode']='camera_matched_experimental'
        diag['declared_camera_pose']=scene_fn.camera_pose
    if plane_horizon is not None:
        diag["support_plane_horizon"] = plane_horizon
        diag["scene_mode"] = ("empty_guide" if conditioning == "empty_guide"
                              else "support_plane_experimental")
    diag["conditioning"] = conditioning
    if glass_backdrop is not None:
        diag["glass_mode"] = True
    if ring.sum() and far.sum():
        rl = scene[ring.astype(bool)].mean()
        fl = scene[far.astype(bool)].mean()
        diag["ring_vs_far_luma_diff"] = round(float(abs(rl - fl)), 2)
        diag["needs_eye_review"] = diag["ring_vs_far_luma_diff"] > 25.0
    else:
        diag["needs_eye_review"] = True
    # A quiet ring only clears that heuristic. It cannot certify camera,
    # lighting/contact realism, mask accuracy or publication usefulness.
    diag["ring_alert"] = diag["needs_eye_review"]
    diag["needs_eye_review"] = True
    diag["publication_status"] = "NOT_EVALUATED"
    diag["review_required"] = ["camera compatibility", "contact and light", "mask edges", "composition"]
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    diag["placement"] = (int(px), int(py), int(pw), int(ph))
    diag["scene_lift"] = scene_lift
    diag["shadow_style"] = shadow_style
    diag["grounding_shadow"] = "anchored_contact" if grounded else None
    if scene_acceptance is not None:
        diag["scene_acceptance"] = scene_acceptance
        diag["scene_attempts"] = scene_attempts
        if emptiness is not None:
            diag["emptiness_screen"] = emptiness
    diag["cover_fit"] = bool(cover_used)
    if glass_backdrop is None:
        inner = crop[..., 3] == 255
        diag["paste_bit_exact"] = bool(np.array_equal(out_u8[py:py + ph, px:px + pw][inner], crop[..., :3][inner]))
    return Image.fromarray(out_u8, "RGB"), diag


def finish_scene(scene, crop, placement, *, shadow_style=None, scene_grade=0.0,
                 glass_backdrop=None, scene_auto_lift=False, cover_used=False):
    """CPU-only finishing shared by live compose and exact cached replay."""
    ch, cw = scene.shape[:2]
    ph, pw = crop.shape[:2]
    px, py = placement
    a_full = np.zeros((ch, cw), np.float32)
    a_full[py:py+ph, px:px+pw] = crop[...,3].astype(np.float32) / 255.0
    out = scene.astype(np.float32)
    a = a_full[..., None]
    scene_lift = 1.0
    if scene_auto_lift:
        # Dark-scene guard: generated scenes that come back near-black (e.g. the
        # barber slate scene) make dark products unreadable. Lift the SCENE ONLY
        # toward a readable mean luma; product pixels are pasted afterwards and
        # stay bit-exact. Recorded in diagnostics.
        luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])
        mean_l = float(luma.mean())
        if mean_l < 70.0:
            target = 100.0
            gamma = np.log(max(target, 1.0) / 255.0) / np.log(max(mean_l, 1.0) / 255.0)
            gamma = float(np.clip(gamma, 1.0 / 1.6, 1.0))
            out = 255.0 * (out / 255.0) ** gamma
            scene_lift = round(1.0 / gamma, 3)
    if scene_grade > 0:
        # Scene-only chroma grade toward the product's colour balance. Never
        # touches product pixels (pasted afterwards). Gain is clipped.
        s_ = float(min(scene_grade, 0.5))
        pm = crop[..., :3][crop[..., 3] > 200].reshape(-1, 3).astype(np.float32)
        if len(pm):
            pm = pm.mean(0); sm = out.reshape(-1, 3).mean(0) + 1e-3
            gain = (pm / (pm.mean() + 1e-3)) / (sm / sm.mean())
            out = out * np.clip(gain ** s_, 0.8, 1.25)
    if shadow_style and not cover_used:
        # Contact shadow darkens the generated scene only; product pixels are
        # pasted afterwards so the interior stays bit-exact. Skipped for
        # cover-fit placements (full-bleed source framing has no floor contact).
        if shadow_style=='surface contact':
            sh=surface_contact_shadow((cw,ch),(px,py),crop[...,3])
        else:
            sh = contact_shadow((cw, ch), (px, py), (pw, ph), crop[..., 3], shadow_style)
        out = out * (1.0 - sh.astype(np.float32)[..., None] / 255.0)
    if glass_backdrop is not None:
        # Clear-glass mode: modulate the NEW scene by measured transmission instead of
        # pasting source pixels (so the old backdrop cannot show through). NOT bit-exact.
        full = gray_src_paste(crop, px, py, ch, cw)
        T = np.clip(full / np.maximum(np.asarray(glass_backdrop, np.float32)[None, None, :], 1.0), 0.0, 1.0)
        out = out * (1 - a) + out * T * a
    else:
        out = out * (1 - a) + gray_src_paste(crop, px, py, ch, cw) * a
    return np.clip(out, 0, 255).astype(np.uint8), scene_lift


def gray_src_paste(crop: np.ndarray, px: int, py: int, ch: int, cw: int) -> np.ndarray:
    """Full-canvas float RGB holding the source-exact product crop at (px,py)."""
    full = np.zeros((ch, cw, 3), np.float32)
    full[py:py + crop.shape[0], px:px + crop.shape[1]] = crop[..., :3]
    return full


def estimate_clean_backdrop(rgb: np.ndarray, border_frac: float = 0.06, max_std: float = 28.0):
    """Median colour of the photo border when it is a near-uniform backdrop.
    Returns (B(3,), ok). ok=False for cluttered backdrops (glass2-style scenes),
    where transmission compositing cannot be used."""
    h, w = rgb.shape[:2]
    t = max(2, int(min(h, w) * border_frac))
    b = np.concatenate([rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3),
                        rgb[:, :t].reshape(-1, 3), rgb[:, -t:].reshape(-1, 3)]).astype(np.float32)
    return np.median(b, axis=0), bool(b.std(axis=0).max() <= max_std)


def silhouette_from_backdrop(rgb: np.ndarray, backdrop: np.ndarray, thresh: float = 14.0) -> np.ndarray:
    """Soft silhouette of an object on a CLEAN backdrop: difference to the backdrop,
    holes filled by flooding from the border (clear glass interiors are near-backdrop)."""
    d = np.abs(rgb.astype(np.float32) - backdrop[None, None, :]).max(axis=2)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = m.shape
    ff = m.copy(); pad = np.zeros((h + 2, w + 2), np.uint8)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if ff[seed[1], seed[0]] == 0:
            cv2.floodFill(ff, pad, seed, 2)
    filled = (ff != 2).astype(np.uint8)
    k = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.2)
    return np.clip(k, 0, 1)


def transmission_composite(src_rgb: np.ndarray, scene_rgb: np.ndarray, matte: np.ndarray,
                           backdrop: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Clear-glass mode (opt-in, clean-backdrop sources only). Inside the matte the
    NEW scene is multiplied by the glass's measured transmission T = I / backdrop
    (clipped 0..1), so the original background does not show through and the glass
    keeps its darkening, tint and edge lines. Outside the matte: scene unchanged.
    Specular highlights equal to the backdrop (T=1) are NOT recovered (known limit).
    Not bit-exact by design: glass pixels are modulated, not pasted."""
    T = np.clip(src_rgb.astype(np.float32) / np.maximum(backdrop[None, None, :], 1.0), 0.0, 1.0)
    T = 1.0 - strength * (1.0 - T)
    M = matte.astype(np.float32)[..., None]
    sc = scene_rgb.astype(np.float32)
    out = sc * (1 - M) + sc * T * M
    return np.clip(out, 0, 255).astype(np.uint8)


def replay_scene(cache_dir, **overrides):
    """CPU-only exact-layer replay. Overrides are finishing options, no model calls.
    Returns an image and opaque RGB invariant, not a visual-quality verdict."""
    from pathlib import Path
    import json
    cd = Path(cache_dir)
    options = json.loads((cd / "layers.json").read_text())
    placement = options.pop("placement")
    options.pop("scene_mode", None)  # recorded callback provenance, not a finish argument
    allowed = {"shadow_style", "scene_grade", "scene_auto_lift"}
    if set(overrides) - allowed:
        raise ValueError("Only shadow_style, scene_grade, scene_auto_lift may be overridden")
    options.update(overrides)
    scene = np.array(Image.open(cd / "scene.png").convert("RGB"))
    crop = np.array(Image.open(cd / "product_rgba.png").convert("RGBA"))
    out, lift = finish_scene(scene, crop, placement, **options)
    px, py = placement; ph, pw = crop.shape[:2]
    exact = None if options.get("glass_backdrop") is not None else bool(np.array_equal(
        out[py:py+ph,px:px+pw][crop[...,3] == 255], crop[...,:3][crop[...,3] == 255]))
    return Image.fromarray(out), {"paste_bit_exact": exact, "scene_lift": lift}


def surface_contact_shadow(canvas, placement, matte):
    """Small full-silhouette shadow for explicitly topdown scenes only."""
    w,h=canvas;x,y=placement;mh,mw=matte.shape
    if x<0 or y<0 or x+mw>w or y+mh>h:raise ValueError('matte outside canvas')
    a=np.zeros((h,w),np.float32);a[y:y+mh,x:x+mw]=matte.astype(np.float32)/255
    sigma=max(1.,min(w,h)*.003)
    M=np.float32([[1,0,sigma],[0,1,2*sigma]])
    shifted=cv2.warpAffine(a,M,(w,h),borderMode=cv2.BORDER_CONSTANT)
    return (cv2.GaussianBlur(shifted,(0,0),sigma)*65).clip(0,65).astype(np.uint8)


def flatlay_center(matte, canvas):
    w,h=canvas;mh,mw=matte.shape
    yy,xx=np.where(matte>127)
    if not len(xx):raise ValueError('empty matte')
    if mw>w or mh>h:raise ValueError('crop larger than canvas')
    x=round(w/2-(int(xx.min())+int(xx.max()))/2)
    y=round(h/2-(int(yy.min())+int(yy.max()))/2)
    return max(0,min(w-mw,x)),max(0,min(h-mh,y))


def audit_composite(cutout, source, output, aspect, seed, template_key):
    """Audit decoded saved single-product output; resizing is not native exactness.

    Uses the same fit_product as the compositor. No alpha semantics, edge or
    lighting acceptance is inferred. Multi-product layouts need their own audit.
    """
    from .native_pixel_audit import audit_native_pixels
    rgba = np.array(Image.open(cutout.rgba_path).convert("RGBA"))
    alpha = np.array(Image.open(cutout.matte_path).convert("L"))
    fill, jx, baseline = placement_params(seed, template_key)
    crop, (px, py) = fit_product(rgba, cutout.bbox, aspect.size, fill, jx, baseline)
    result = audit_native_pixels(source, alpha, output, list(cutout.bbox),
                                 [px, py, crop.shape[1], crop.shape[0]])
    result["placement_basis"] = "COMPOSITOR_FIT_PRODUCT_COORDINATES"
    return result
