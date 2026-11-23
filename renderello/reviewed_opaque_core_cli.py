"""File-bound reviewed near-opaque interior repair. Never automatic alpha saturation."""
from pathlib import Path
import argparse
import json
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from .image_intake import _read_bounded
from .candidate_diagnostics import digest
from .reviewed_edge_cli import read_image, sha, _json_pairs
from .reviewed_opaque_core import restore_reviewed_opaque_core

NAMES = ('source', 'alpha', 'region')


def load_spec(path, expected):
    if Path(path).is_symlink():
        raise ValueError('regular spec file required')
    raw = _read_bounded(Path(path), 65_536)
    if sha(raw) != expected:
        raise ValueError('spec file hash mismatch')
    def nonfinite(value):
        raise ValueError('nonfinite JSON')
    s = json.loads(raw, object_pairs_hook=_json_pairs, parse_constant=nonfinite)
    fields = {'schema', 'files', 'reviewed', 'material', 'review_note', 'controls'}
    if not isinstance(s, dict) or set(s) != fields or s['schema'] != 'reviewed-opaque-core-files.v1':
        raise ValueError('exact opaque-core spec required')
    if not isinstance(s['files'], dict) or set(s['files']) != set(NAMES):
        raise ValueError('all three file bindings required')
    for h in s['files'].values():
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('lowercase file SHA-256 required')
    if not isinstance(s['controls'], dict) or set(s['controls']) != {'max_deficit', 'clearance', 'feather'}:
        raise ValueError('exact explicit opaque-core controls required')
    return s


def execute(*, source, alpha, region, spec, spec_sha256, output_dir):
    dest = Path(output_dir)
    if dest.exists() or dest.is_symlink():
        raise ValueError('output exists; preserve earlier evidence')
    s = load_spec(spec, spec_sha256)
    arrays, records = [], {}
    for n, path in zip(NAMES, (source, alpha, region)):
        if Path(path).is_symlink():
            raise ValueError('regular image file required')
        a, record = read_image(path, s['files'][n], n)
        arrays.append(a); records[n] = record
    rgb, a, region = arrays
    out_rgb, out_alpha, metrics = restore_reviewed_opaque_core(rgb, a, region,
        source_sha256=digest(rgb), alpha_sha256=digest(a), region_sha256=digest(region), reviewed=s['reviewed'], material=s['material'],
        review_note=s['review_note'], **s['controls'])
    if not np.array_equal(out_rgb, rgb) or not np.array_equal(out_alpha[~region], a[~region]):
        raise RuntimeError('source or outside-region invariant failed')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.opaque-core-stage-', dir=dest.parent))
    try:
        Image.fromarray(out_rgb).save(temp/'source-rgb.png')
        Image.fromarray(out_alpha).save(temp/'alpha.png')
        Image.fromarray(np.dstack([out_rgb, out_alpha])).save(temp/'component-rgba.png')
        for label, value in [('dark', 35), ('light', 242)]:
            aa = out_alpha.astype(np.uint32)[:, :, None]
            comp = ((out_rgb.astype(np.uint32)*aa + value*(255-aa)+127)//255).astype(np.uint8)
            Image.fromarray(comp).save(temp/(label+'.png'))
        manifest = dict(schema='reviewed-opaque-core-file-result.v1', spec_sha256=spec_sha256,
            inputs=records, controls=s['controls'], metrics=metrics,
            outputs={p.name: sha(p.read_bytes()) for p in temp.iterdir()},
            review_is_caller_declaration=True, source_rgb_exact=True, accepted=False,
            source_resampled=False, default_pipeline_changed=False)
        (temp/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        dest.mkdir()
        try:
            for p in temp.iterdir(): os.rename(p, dest/p.name)
        except BaseException:
            shutil.rmtree(dest)
            raise
        return manifest
    finally:
        shutil.rmtree(temp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for n in NAMES + ('spec', 'spec-sha256', 'output-dir'): p.add_argument('--'+n, required=True)
    a = p.parse_args(argv)
    try: result = execute(**vars(a))
    except (ValueError, OSError) as exc: p.exit(2, 'REFUSED: '+str(exc)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()
