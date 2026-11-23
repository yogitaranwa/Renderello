"""Native, explicit opt-in reviewed source-floor illustration. No generation/API.

Light flat targets only. Annotations and uncalibrated source-floor intensity are
not physical geometry, relighting or photographic acceptance. Defaults untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tempfile
from io import BytesIO

MAX_INPUT_BYTES = 16_000_000


def _snapshot(path):
    with Path(path).open('rb') as f:
        data = f.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError('input exceeds16MB bounded snapshot')
    return data
import numpy as np
from PIL import Image
from .candidate_diagnostics import digest
from .reviewed_source_floor import tint_reviewed_source_floor


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run_files(source, alpha, selection, output, *, source_sha256,
              alpha_sha256, selection_sha256, background_rgb,
              reviewed=False, review_note=''):
    paths = [Path(source), Path(alpha), Path(selection)]
    output = Path(output)
    if output.exists():
        raise ValueError('preserve previous output')
    snapshots = []
    for path, expected in zip(paths, [source_sha256, alpha_sha256, selection_sha256]):
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('explicit lowercase file SHA256 required')
        data = _snapshot(path)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source/alpha/selection file binding mismatch')
        snapshots.append(data)
    if not isinstance(background_rgb, (list, tuple)) or len(background_rgb) != 3 or any(type(v) is not int or not 160 <= v <= 255 for v in background_rgb):
        raise ValueError('three integer light-background channels160..255 required')
    arrays = []
    icc = None
    for index, data in enumerate(snapshots):
        with Image.open(BytesIO(data)) as im:
            if im.mode != ('RGB' if index == 0 else 'L'):
                raise ValueError('native RGB source and L alpha/selection, no conversion')
            if im.width * im.height > 1_000_000 or min(im.size) < 2:
                raise ValueError('native2px..1MP required, no resize')
            if index == 0:
                icc = im.info.get('icc_profile')
            arrays.append(np.array(im))
    rgb, a, selected = arrays
    if a.shape != rgb.shape[:2] or selected.shape != a.shape:
        raise ValueError('native dimensions must match')
    bg = np.full_like(rgb, background_rgb)
    out, record = tint_reviewed_source_floor(rgb, a, bg, selected,
        source_sha256=digest(rgb), alpha_sha256=digest(a),
        background_sha256=digest(bg), selection_sha256=digest(selected),
        reviewed=reviewed, source_floor='neutral_white_opaque', review_note=review_note)
    base = np.rint(rgb * (a[:, :, None] / 255) + bg * (1 - a[:, :, None] / 255)).astype(np.uint8)
    record.update(file_bindings=dict(zip(['source', 'alpha', 'selection'],
        [source_sha256, alpha_sha256, selection_sha256])),
        background_rgb=list(background_rgb), source_size=[rgb.shape[1], rgb.shape[0]],
        original_alpha_unchanged=True, paid_calls=0, local_review_only=True,
        input_read_semantics='bounded_immutable_byte_snapshots_hashed_before_decode',
        max_input_file_bytes=MAX_INPUT_BYTES)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='reviewed-source-floor-', dir=output.parent))
    try:
        kw = {'icc_profile': icc} if icc else {}
        for name, value in [('no-floor.png', base), ('reviewed-floor.png', out)]:
            Image.fromarray(value).save(temp / name, **kw)
            saved = Image.open(temp / name)
            if not np.array_equal(np.array(saved), value) or saved.info.get('icc_profile') != icc:
                raise RuntimeError('export RGB/profile verification failed')
        (temp / 'review.json').write_text(json.dumps(record, indent=2))
        if output.exists():
            raise ValueError('output appeared during run; preserve it')
        temp.rename(output)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'alpha', 'selection', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    for name in ['source-sha256', 'alpha-sha256', 'selection-sha256', 'review-note']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--background-rgb', type=int, nargs=3, required=True)
    p.add_argument('--reviewed-neutral-white-floor', action='store_true', required=True)
    a = p.parse_args(argv)
    result = run_files(a.source, a.alpha, a.selection, a.output,
        source_sha256=a.source_sha256, alpha_sha256=a.alpha_sha256,
        selection_sha256=a.selection_sha256, background_rgb=a.background_rgb,
        reviewed=a.reviewed_neutral_white_floor, review_note=a.review_note)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
