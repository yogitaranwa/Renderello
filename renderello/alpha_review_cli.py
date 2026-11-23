"""Source-bound command-line export of explicit review-only alpha annotations.

Run: python -m renderello.alpha_review_cli --source native.png --alpha alpha.png
     --request annotation.json --output new-review-directory

The request binds decoded RGB/alpha hashes and native dimensions. It cannot
approve production use. Reviewers must inspect both source and final composites.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .explicit_alpha_patch import export_explicit_alpha_review


def export_bound_review(source_path, alpha_path, request_path, output):
    """Refuse a mismatched source/baseline before creating any output files."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('preserve previous output')
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    required = {'schema_version', 'native_size', 'source_rgb_sha256',
                'baseline_alpha_sha256', 'protected_regions', 'patches'}
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError('exact source-bound request fields required')
    if type(request['schema_version']) is not int or request['schema_version'] != 1:
        raise ValueError('unsupported request schema')
    size = request['native_size']
    if (not isinstance(size, list) or len(size) != 2
            or any(type(n) is not int or n <= 0 for n in size)
            or size[0] * size[1] > 4_000_000):
        raise ValueError('invalid native dimensions')
    for field in ['source_rgb_sha256', 'baseline_alpha_sha256']:
        value = request[field]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('lowercase SHA-256 binding required')
    with Image.open(source_path) as image:
        if image.mode != 'RGB' or image.size != tuple(size):
            raise ValueError('native RGB mode and dimensions must match request')
        source = image.copy()
    with Image.open(alpha_path) as image:
        if image.mode != 'L' or image.size != tuple(size):
            raise ValueError('native L alpha and dimensions must match request')
        alpha = image.copy()
    if hashlib.sha256(source.tobytes()).hexdigest() != request['source_rgb_sha256']:
        raise ValueError('source RGB hash mismatch; inspect and re-annotate the intended source')
    if hashlib.sha256(alpha.tobytes()).hexdigest() != request['baseline_alpha_sha256']:
        raise ValueError('baseline alpha hash mismatch; inspect the new baseline before replay')
    return export_explicit_alpha_review(source, alpha, request['protected_regions'],
                                        request['patches'], output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'alpha', 'request', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = export_bound_review(args.source, args.alpha, args.request, args.output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f'Review export refused: {exc}\n')
    print(json.dumps({'output': str(Path(args.output).resolve()),
                      'alpha_changed_pixels': manifest['alpha_changed_pixels'],
                      'kind': manifest['kind'], 'quality_accepted': False,
                      'production_review_created': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
