"""Opt-in native ViTMatte research, never a production quality acceptance.

No source resizing, repainting or silent mode conversion. Hashes bind supplied
bytes, not semantic product identity. A binary support mask and derived trimap
are caller evidence, not independently annotated boundary truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

MODEL = 'hustvl/vitmatte-small-composition-1k'
REVISION = '6a58ad7646403c1df626fbd746900aec7361ea1d'
WEIGHTS_SHA256 = 'bda9289db1bb6762d978b42d1c62ae3f34daf7497171a347a1d09657efd788cb'
MAX_PIXELS = 1_000_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bound(source, mask, source_sha256, mask_sha256):
    for path, expected in ((source, source_sha256), (mask, mask_sha256)):
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('explicit lowercase SHA-256 required')
        if sha256(path) != expected:
            raise ValueError('input file hash mismatch: ' + str(path))
    with Image.open(source) as s, Image.open(mask) as m:
        if s.mode != 'RGB' or m.mode != 'L':
            raise ValueError('native RGB source and L mask required; no mode conversion')
        if s.size != m.size:
            raise ValueError('native source/mask dimensions differ; no resizing')
        if min(s.size) < 2 or s.width * s.height > MAX_PIXELS:
            raise ValueError('source dimensions outside bounded native research limit (2px minimum, 1000000 pixels maximum)')
        rgb, alpha = np.array(s), np.array(m)
    if not np.isin(alpha, [0, 255]).all():
        raise ValueError('explicit binary 0/255 support mask required; no hidden threshold')
    if not (alpha == 0).any() or not (alpha == 255).any():
        raise ValueError('both source foreground and background support required')
    return rgb, alpha


def derive_trimap(mask, radius):
    from scipy.ndimage import binary_erosion, binary_dilation
    if type(radius) is not int or not 1 <= radius <= 16:
        raise ValueError('integer trimap radius 1..16 required')
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
        raise ValueError('two-dimensional binary uint8 mask required')
    support = mask == 255
    fg = binary_erosion(support, iterations=radius)
    bg = ~binary_dilation(support, iterations=radius)
    if not fg.any() or not bg.any():
        raise ValueError('trimap has no known foreground or background')
    tri = np.full(mask.shape, 128, np.uint8)
    tri[fg], tri[bg] = 255, 0
    return tri


def checked_alpha(raw, trimap):
    if not isinstance(trimap, np.ndarray) or trimap.ndim != 2 or trimap.dtype != np.uint8 or not np.isin(trimap, [0, 128, 255]).all():
        raise ValueError('explicit 0/128/255 uint8 trimap required')
    if not isinstance(raw, np.ndarray) or raw.dtype.kind != 'f' or raw.shape != trimap.shape:
        raise ValueError('source-native floating alpha required')
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError('finite alpha in [0,1] required; no clipping invalid predictions')
    alpha = raw.copy()
    alpha[trimap == 255], alpha[trimap == 0] = 1, 0
    return alpha, np.round(alpha * 255).astype(np.uint8)


def composite(rgb, alpha8, background):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('native uint8 RGB required')
    if not isinstance(alpha8, np.ndarray) or alpha8.dtype != np.uint8 or alpha8.shape != rgb.shape[:2]:
        raise ValueError('matching uint8 alpha required')
    if type(background) is not int or not 0 <= background <= 255:
        raise ValueError('integer diagnostic background 0..255 required')
    a = alpha8.astype(np.float64)[:, :, None] / 255
    return np.round(rgb * a + background * (1 - a)).astype(np.uint8)


class CpuPredictor:
    """Pinned weights, strict load; dependencies are optional outside this route."""
    def __init__(self, threads=2, *, device='cpu'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('explicit cpu or cuda device required')
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('CPU threads 1..4 required')
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        self.device = device
        from huggingface_hub import hf_hub_download
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        import transformers
        torch.set_num_threads(threads)
        self.torch = torch
        t = time.monotonic()
        weights = hf_hub_download(MODEL, 'model.safetensors', revision=REVISION)
        if sha256(weights) != WEIGHTS_SHA256:
            raise ValueError('pinned model weights hash mismatch')
        self.processor = VitMatteImageProcessor.from_pretrained(MODEL, revision=REVISION)
        self.model, info = VitMatteForImageMatting.from_pretrained(
            MODEL, revision=REVISION, use_safetensors=True, output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('strict checkpoint load failed: ' + repr(info))
        self.model.eval().to(device)
        self.info = dict(model=MODEL, revision=REVISION, weights_sha256=WEIGHTS_SHA256,
                         load_seconds=time.monotonic() - t, loading_info=info,
                         torch_version=torch.__version__, transformers_version=transformers.__version__,
                         device=device, threads=threads,
                         gpu_name=torch.cuda.get_device_name(0) if device == 'cuda' else None)

    def __call__(self, rgb, trimap):
        h, w = trimap.shape
        padded = ((h + 31) // 32 * 32, (w + 31) // 32 * 32)
        inp = self.processor(images=Image.fromarray(rgb), trimaps=Image.fromarray(trimap), return_tensors='pt')
        if tuple(inp['pixel_values'].shape) != (1, 4, *padded):
            raise ValueError('processor dimensions changed beyond expected padding')
        inp = {key: value.to(self.device) for key, value in inp.items()}
        if self.device == 'cuda': self.torch.cuda.synchronize()
        t = time.monotonic()
        with self.torch.inference_mode():
            pred = self.model(**inp).alphas
        if self.device == 'cuda': self.torch.cuda.synchronize()
        if tuple(pred.shape) != (1, 1, *padded):
            raise ValueError('unexpected model output shape')
        return pred[0, 0, :h, :w].detach().cpu().numpy().copy(), dict(
            inference_seconds=time.monotonic() - t, padded_shape=list(padded))


def run(source, mask, output_dir, *, source_sha256, mask_sha256, radius=3, threads=2, predictor=None):
    dest = Path(output_dir)
    if dest.exists():
        raise ValueError('output directory exists; prior evidence cannot be replaced')
    rgb, prior = read_bound(source, mask, source_sha256, mask_sha256)
    tri = derive_trimap(prior, radius)
    worker = predictor if predictor is not None else CpuPredictor(threads)
    raw, timing = worker(rgb, tri)
    alpha, a8 = checked_alpha(raw, tri)
    metrics = dict(unknown_pixels=int((tri == 128).sum()), fractional_alpha_pixels=int(((a8 > 0) & (a8 < 255)).sum()),
                   changed_coverage=int((prior != a8).sum()), support_added=int(((prior == 0) & (a8 > 0)).sum()),
                   support_removed=int(((prior > 0) & (a8 == 0)).sum()), source_rgb_repainted=0,
                   source_rgb_resampled=False, boundary_accuracy=None, publication_accepted=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='native-matting-', dir=dest.parent))
    try:
        Image.fromarray(rgb).save(temp / 'source-native.png')
        Image.fromarray(prior).save(temp / 'prior-support-mask.png')
        Image.fromarray(tri).save(temp / 'derived-trimap.png')
        Image.fromarray(a8).save(temp / 'alpha.png')
        Image.fromarray(np.dstack([rgb, a8])).save(temp / 'component-rgba.png')
        np.save(temp / 'raw-alpha.npy', raw, allow_pickle=False)
        np.save(temp / 'enforced-alpha.npy', alpha, allow_pickle=False)
        for name, background in (('dark', 35), ('light', 242)):
            Image.fromarray(composite(rgb, a8, background)).save(temp / (name + '.png'))
        outputs = {p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in temp.iterdir()}
        manifest = dict(inputs=dict(source_sha256=source_sha256, mask_sha256=mask_sha256, native_shape=list(rgb.shape)),
                        radius=radius, predictor=getattr(worker, 'info', {'unverified_injected_predictor': True}),
                        timing=timing, metrics=metrics, outputs=outputs,
                        status='EXPERIMENTAL_NOT_PHOTO_ACCEPTED', trimap_is_derived_not_independent_truth=True,
                        composite_alpha='exported_quantized_uint8', production_default_changed=False)
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        dest.mkdir()  # exclusive reservation, no overwriting a competing writer
        try:
            for p in temp.iterdir():
                p.rename(dest / p.name)
        except BaseException:
            shutil.rmtree(dest)
            raise
        return manifest
    finally:
        shutil.rmtree(temp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'mask', 'source-sha256', 'mask-sha256', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--radius', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    try:
        result = run(a.source, a.mask, a.output_dir, source_sha256=a.source_sha256,
                     mask_sha256=a.mask_sha256, radius=a.radius, threads=a.threads)
    except (ValueError, OSError) as e:
        p.exit(2, 'REFUSED: ' + str(e) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
