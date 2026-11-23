"""Replay a saved native-matting result before it becomes a studio candidate.

This proves internal consistency and source pixel pairing, not model authorship,
semantic segmentation truth, optical correctness or publication acceptance.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from .candidate_diagnostics import native_rgb, native_alpha, digest
from .checked_research.native_matting import derive_trimap, checked_alpha, composite
from .matte_change_review import audit_trimap, review_change

EXPECTED = frozenset(('source-native.png', 'prior-support-mask.png', 'derived-trimap.png',
    'alpha.png', 'component-rgba.png', 'raw-alpha.npy', 'enforced-alpha.npy', 'dark.png', 'light.png'))


def load_verified_result(directory, source_rgb, initial_alpha):
    native_rgb(source_rgb)
    native_alpha(initial_alpha, source_rgb.shape[:2])
    if source_rgb.shape[0] * source_rgb.shape[1] > 1_000_000:
        raise ValueError('Native result exceeds one-megapixel replay budget')
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Regular result directory required')

    def bounded(name, limit=20_000_000):
        p = root / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > limit:
            raise ValueError('Missing, unsafe or oversized result file: ' + name)
        data = p.read_bytes()
        if len(data) > limit:
            raise ValueError('Result file grew beyond read budget')
        return data

    manifest_bytes = bounded('manifest.json', 65_536)
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or set(manifest.get('outputs', {})) != EXPECTED:
        raise ValueError('Exactly the nine native result outputs are required')
    if manifest.get('status') != 'EXPERIMENTAL_NOT_PHOTO_ACCEPTED':
        raise ValueError('Unexpected acceptance/status claim in research result')
    if manifest.get('inputs', {}).get('native_shape') != list(source_rgb.shape):
        raise ValueError('Manifest native shape differs from source')
    buffers = {}
    for name in sorted(EXPECTED):
        data = bounded(name)
        entry = manifest['outputs'][name]
        if not isinstance(entry, dict) or entry.get('bytes') != len(data) or entry.get('sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError('Native result hash or size mismatch: ' + name)
        buffers[name] = data
    # Decode only the verified bytes, so later on-disk changes cannot switch input.
    from io import BytesIO
    def image(name, mode):
        with Image.open(BytesIO(buffers[name])) as im:
            if im.mode != mode or im.size != (source_rgb.shape[1], source_rgb.shape[0]) or getattr(im, 'n_frames', 1) != 1:
                raise ValueError('Native result image mode/shape changed: ' + name)
            if im.getexif().get(274, 1) != 1 or im.info.get('icc_profile'):
                raise ValueError('Result orientation/profile transform is not permitted')
            return np.array(im)
    src = image('source-native.png', 'RGB')
    rgba = image('component-rgba.png', 'RGBA')
    alpha = image('alpha.png', 'L')
    binary = image('prior-support-mask.png', 'L')
    trimap = image('derived-trimap.png', 'L')
    if not np.array_equal(src, source_rgb) or not np.array_equal(rgba[:, :, :3], source_rgb):
        raise ValueError('Result belongs to different or repainted source RGB')
    if not np.array_equal(rgba[:, :, 3], alpha):
        raise ValueError('RGBA alpha differs from standalone alpha')
    if not np.array_equal(binary, np.where(initial_alpha >= 128, 255, 0).astype(np.uint8)):
        raise ValueError('Result support mask differs from supplied initial alpha threshold')
    if not np.array_equal(derive_trimap(binary, manifest.get('radius')), trimap):
        raise ValueError('Derived trimap replay differs')
    raw = np.load(BytesIO(buffers['raw-alpha.npy']), allow_pickle=False, max_header_size=1024)
    enforced = np.load(BytesIO(buffers['enforced-alpha.npy']), allow_pickle=False, max_header_size=1024)
    expected_enforced, expected_alpha = checked_alpha(raw, trimap)
    if not np.array_equal(expected_enforced, enforced) or not np.array_equal(expected_alpha, alpha):
        raise ValueError('Alpha enforcement replay differs')
    for name, background in [('dark.png', 35), ('light.png', 242)]:
        if not np.array_equal(composite(source_rgb, alpha, background), image(name, 'RGB')):
            raise ValueError('Composite replay differs: ' + name)
    seeds, _ = audit_trimap(initial_alpha, trimap)
    changes, _ = review_change(initial_alpha, alpha)
    return alpha.copy(), dict(
        schema='verified-native-result-import.v1',
        source_pixels_sha256=digest(source_rgb), initial_alpha_sha256=digest(initial_alpha),
        imported_alpha_sha256=digest(alpha), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        files_verified=len(EXPECTED), source_rgb_exact=True, replay_exact=True,
        model_execution_provenance='MANIFEST_CLAIM_NOT_AUTHENTICATED_BY_REPLAY',
        no_new_inference=True, seeds=seeds, changes=changes,
        automatic_acceptance=False, physical_relighting=False, glass_optics_verified=False)
