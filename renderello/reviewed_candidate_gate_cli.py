"""File-bound exact intervention audit. No candidate image export or promotion."""
from pathlib import Path
from io import BytesIO
import argparse, hashlib, json, os, shutil, tempfile
import numpy as np
from PIL import Image
from .image_intake import _read_bounded
from .candidate_diagnostics import digest
from .reviewed_edge_cli import _json_pairs
from .reviewed_candidate_gate import review_candidate
from .matte_change_review import ChangeBudget

NAMES = ('source', 'before', 'candidate', 'allowed', 'protected')
sha = lambda data: hashlib.sha256(data).hexdigest()


def load_policy(path, expected):
    raw = _read_bounded(Path(path), 65536)
    if sha(raw) != expected: raise ValueError('policy file hash mismatch')
    try:
        p = json.loads(raw, object_pairs_hook=_json_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError('invalid policy JSON') from e
    fields = {'schema', 'files', 'reviewed', 'material', 'framing', 'review_note', 'budget'}
    if not isinstance(p, dict) or set(p) != fields or p['schema'] != 'reviewed-candidate-files.v1':
        raise ValueError('exact candidate policy schema required')
    if not isinstance(p['files'], dict) or set(p['files']) != set(NAMES):
        raise ValueError('exact five file bindings required')
    for h in p['files'].values():
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('lowercase file SHA256 required')
    keys = set(ChangeBudget.__dataclass_fields__)
    if not isinstance(p['budget'], dict) or set(p['budget']) != keys:
        raise ValueError('every topology change control must be explicit')
    try: budget = ChangeBudget(**p['budget'])
    except (TypeError, ValueError) as e: raise ValueError('invalid topology budget') from e
    return p, budget


def read_bound(path, expected, name):
    raw = _read_bounded(Path(path), 8_000_000)
    if sha(raw) != expected: raise ValueError(name + ' file hash mismatch')
    if not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw) < 29 or raw[24] != 8:
        raise ValueError(name + ' requires8bit PNG')
    with Image.open(BytesIO(raw)) as im:
        mode = 'RGB' if name == 'source' else ('RGBA' if name == 'candidate' else 'L')
        if im.format != 'PNG' or im.mode != mode: raise ValueError(name + ' exact ' + mode + ' required')
        if getattr(im, 'n_frames', 1) != 1 or im.getexif() or any(k in im.info for k in ('icc_profile','gamma','srgb','transparency')):
            raise ValueError(name + ' metadata requires separate intake')
        if min(im.size) < 3 or im.width * im.height > 1_000_000:
            raise ValueError(name + ' native3px..1MP limit; no resize')
        arr = np.array(im)
    if name in ('allowed','protected'):
        if not np.isin(arr,[0,255]).all(): raise ValueError(name + ' requires binary0/255; no threshold')
        arr = arr == 255
    return arr, dict(file_sha256=sha(raw), pixel_sha256=digest(arr), bytes=len(raw), shape=list(arr.shape))


def execute(*, source, before, candidate, allowed, protected, policy, policy_sha256, output_dir):
    dest = Path(output_dir)
    if dest.exists() or dest.is_symlink(): raise ValueError('output exists; preserve earlier evidence')
    p, budget = load_policy(policy, policy_sha256)
    arrays = {}; records = {}
    for n, path in zip(NAMES,(source,before,candidate,allowed,protected)):
        arrays[n], records[n] = read_bound(path,p['files'][n],n)
    copies = {n:a.copy()for n,a in arrays.items()}
    report, delta = review_candidate(arrays['source'], arrays['before'], arrays['candidate'],
        arrays['allowed'], arrays['protected'], bindings={n:digest(a)for n,a in arrays.items()},
        reviewed=p['reviewed'], material=p['material'], framing=p['framing'],
        review_note=p['review_note'], budget=budget)
    if any(not np.array_equal(a,copies[n])for n,a in arrays.items()): raise RuntimeError('input mutation')
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.candidate-audit-stage-',dir=dest.parent))
    try:
        Image.fromarray(delta).save(temp/'alpha-change-visualization.png')
        report.update(policy_file_sha256=policy_sha256, files=records,
            candidate_image_exported=False, audit_only=True,
            decision='INTERVENTION_CONSTRAINTS_ONLY' if report['intervention_constraints_satisfied'] else 'REFUSED_INTERVENTION_SCOPE',
            visualization_semantics='Red alpha increase, blue decrease. Magnitude is not error or quality.')
        report['output_sha256']={'alpha-change-visualization.png':sha((temp/'alpha-change-visualization.png').read_bytes())}
        (temp/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        dest.mkdir()
        try:
            for q in temp.iterdir(): os.rename(q,dest/q.name)
        except BaseException:
            shutil.rmtree(dest);raise
        return report
    finally: shutil.rmtree(temp)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in (*NAMES,'policy','policy-sha256','output-dir'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    try: report=execute(**vars(a))
    except (ValueError,OSError)as e:p.exit(2,'REFUSED INPUT: '+str(e)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    # A valid audit may refuse the candidate. The report is evidence, not approval.
    if not report['intervention_constraints_satisfied']:p.exit(3)

if __name__=='__main__':main()
