"""Source-bound paired lighting diagnostic. Does not generate or accept photographs.

Hashes bind file bytes; alignment remains an explicit caller declaration, not
independent proof. Native-sized gain arrays must originate from the same source.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .evidence_checks import source_bound_alpha
from .lighting_diagnostics import paired_horizontal_gain
from .research_cli import file_record


def execute(args):
    dest = Path(args.output_dir)
    if dest.exists():
        raise ValueError('output directory already exists; preserve prior evidence')
    binding = source_bound_alpha(args.source, args.alpha, args.source_sha256,
                                 expected_alpha_sha256=args.alpha_sha256)
    records = {}
    gains = []
    for name in ('left_gain', 'right_gain'):
        path = getattr(args, name)
        records[name] = file_record(path, getattr(args, name + '_sha256'))
        if Path(path).stat().st_size > 64 * 1024 * 1024:
            raise ValueError('gain file exceeds 64 MiB research limit')
        gain = np.load(path, allow_pickle=False)
        if not isinstance(gain, np.ndarray):
            gain.close()
            raise ValueError('single native NumPy gain array required, not an archive')
        gains.append(gain)
    with Image.open(args.alpha) as im:
        opaque = np.array(im) == 255
    metrics = paired_horizontal_gain(*gains, opaque,
                                     aligned_pair_reviewed=args.aligned_pair_reviewed)
    manifest = dict(command='lighting-pair', inputs=dict(binding=binding, **records),
                    metrics=metrics, publication_accepted=False, quality_accuracy=None,
                    alignment_review_is_caller_declaration=True,
                    source_identity_independently_verified=False)
    # Validate before creating output; reserve exclusively and never overwrite evidence.
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='renderello-lighting-pair-', dir=dest.parent))
    try:
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        dest.mkdir()
        try:
            (temp / 'manifest.json').rename(dest / 'manifest.json')
        except BaseException:
            shutil.rmtree(dest)
            raise
    finally:
        shutil.rmtree(temp)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'alpha', 'left-gain', 'right-gain'):
        p.add_argument('--' + name, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--aligned-pair-reviewed', action='store_true')
    p.add_argument('--output-dir', required=True)
    return p


if __name__ == '__main__':
    p = parser()
    args = p.parse_args()
    try:
        result = execute(args)
    except (ValueError, OSError) as error:
        p.exit(2, 'REFUSED: ' + str(error) + '\n')
    print(json.dumps(result, indent=2))
