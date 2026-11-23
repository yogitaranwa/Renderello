"""Explicit withoutBG learned-matting-only CPU diagnostic. Never auto fallback.

The normal router's chosen category is recorded, not obeyed. This route is for
reviewing a smaller branch where the normal routed test could not complete.
Predicted alpha is model-resized; source RGB and native export size are fixed.
No model downloads, API calls, default route changes or quality acceptance.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time
import numpy as np
from PIL import Image

MAX_PIXELS = 1_000_000


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _bound(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('explicit lowercase SHA256 required')
    if file_hash(path) != expected:
        raise ValueError('file binding mismatch')


def quantize_native_alpha(raw, height, width, resize):
    if not isinstance(raw, np.ndarray) or raw.dtype.kind != 'f' or raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError('floating single-channel coarse alpha required')
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError('coarse alpha must be finite in [0,1], no invalid clipping')
    if any(type(x) is not int or x < 2 for x in (height, width)) or height * width > MAX_PIXELS:
        raise ValueError('native 2px minimum and 1MP maximum required')
    a = resize(raw, height, width)
    if not isinstance(a, np.ndarray) or a.shape != (1, height, width) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('native alpha resize returned invalid result')
    return np.floor(a[0] * 255 + .5).astype(np.uint8)


def local_pipeline(sidecar, model_dir):
    import onnxruntime as ort
    from withoutbg.routed import RoutedPipeline
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    root = Path(model_dir).resolve()
    def resolve(name):
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError('existing local model inside supplied bundle required')
        return candidate
    return RoutedPipeline(sidecar, resolve, providers=['CPUExecutionProvider'],
        session_factory=lambda p, providers: ort.InferenceSession(p, sess_options=options, providers=providers))


def run(source, model_dir, output_dir, *, source_sha256, sidecar_sha256,
        reviewed_diagnostic=False, review_note='', pipeline_factory=None, resize=None):
    if reviewed_diagnostic is not True or not isinstance(review_note, str) or not review_note.strip() or len(review_note) > 2000:
        raise ValueError('explicit diagnostic review and bounded note required')
    source, model_dir, dest = Path(source), Path(model_dir), Path(output_dir)
    if dest.exists():
        raise ValueError('output exists; preserve prior evidence')
    sidecar_path = model_dir / 'withoutbg-open-weights.onnx.json'
    _bound(source, source_sha256)
    _bound(sidecar_path, sidecar_sha256)
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get('schema_version') != 3 or sidecar.get('pipeline') != 'routed':
        raise ValueError('explicit schema3 routed bundle required')
    with Image.open(source) as im:
        if im.mode != 'RGB' or min(im.size) < 2 or im.width * im.height > MAX_PIXELS:
            raise ValueError('native RGB 2px..1MP source required; no silent conversion')
        rgb = np.array(im)
        icc = im.info.get('icc_profile')
    if resize is None:
        from withoutbg.routed import resize_bilinear
        resize = resize_bilinear
    worker = (pipeline_factory or local_pipeline)(sidecar, model_dir)
    start = time.monotonic()
    normalized = rgb.astype(np.float32) / 255
    category, feeds, features = worker._route(normalized)
    if category not in worker.categories:
        raise ValueError('router category outside declared bundle')
    # Deliberate review route. Never replace a failed default result silently.
    coarse = worker._coarse(normalized, feeds, features)
    alpha = quantize_native_alpha(coarse, *rgb.shape[:2], resize)
    providers = {name: session.get_providers() for name, session in worker.sessions.items()}
    if set(providers) != {'router', 'coarse'} or any(p != ['CPUExecutionProvider'] for p in providers.values()):
        raise ValueError('only requested CPU router/coarse sessions permitted')
    result = dict(schema='withoutbg-matting-only-diagnostic.v1', source_sha256=source_sha256,
        sidecar_sha256=sidecar_sha256, model_version=sidecar.get('model_version'),
        router_category=category, normal_route='birefnet' if category in worker.birefnet_categories else 'matting',
        executed_route='FORCED_LEARNED_MATTING_DIAGNOSTIC', review_note=review_note,
        native_size=[rgb.shape[1], rgb.shape[0]], providers=providers,
        inference_seconds=time.monotonic()-start, source_rgb_repainted=0,
        source_rgb_resampled=False, model_internally_resizes=True,
        default_route_changed=False, automatic_fallback=False, boundary_ground_truth=False,
        physical_relighting=False, photographic_acceptance=False, paid_calls=0)
    # Structural flags must travel with every result. They never choose which
    # component is the product or authorize removal of small parts.
    if min(alpha.shape) >= 3:
        from renderello.candidate_diagnostics import mask_diagnostics
        result['mask_review'] = mask_diagnostics(rgb, alpha)
    else:
        result['mask_review'] = dict(flags=['TOO_SMALL_FOR_STRUCTURAL_REVIEW'],
            semantic_completeness='NOT_EVALUATED', automatic_acceptance=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='matting-diagnostic-', dir=dest.parent))
    try:
        kw = {'icc_profile': icc} if icc else {}
        Image.fromarray(alpha).save(temp/'alpha.png')
        Image.fromarray(np.dstack([rgb, alpha])).save(temp/'component-rgba.png', **kw)
        saved = np.array(Image.open(temp/'component-rgba.png'))
        if not np.array_equal(saved[:, :, :3], rgb) or not np.array_equal(saved[:, :, 3], alpha):
            raise RuntimeError('native export invariant failed')
        result['rgba_sha256'] = file_hash(temp/'component-rgba.png')
        (temp/'result.json').write_text(json.dumps(result, indent=2))
        if dest.exists():
            raise ValueError('output appeared during run; preserve it')
        temp.rename(dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return result


def main(argv=None):
    """Explicit opt-in command, separate from production segmentation defaults."""
    import argparse
    parser = argparse.ArgumentParser(description='Reviewed CPU learned-matting-only diagnostic. Not a production fallback or quality acceptance.')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--sidecar-sha256', required=True)
    parser.add_argument('--review-note', required=True)
    parser.add_argument('--reviewed-diagnostic', action='store_true', required=True)
    args = parser.parse_args(argv)
    result = run(args.source, args.model_dir, args.output_dir,
        source_sha256=args.source_sha256, sidecar_sha256=args.sidecar_sha256,
        review_note=args.review_note, reviewed_diagnostic=args.reviewed_diagnostic)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
