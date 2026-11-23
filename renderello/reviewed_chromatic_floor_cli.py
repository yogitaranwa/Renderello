"""Offline native color-bearing floor illustration. Explicit reviewed inputs only.

Does not identify a floor, infer a shadow, relight a product or authorize publishing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from .cached_native_scene_cli import snapshot
from .candidate_diagnostics import digest
from .reviewed_chromatic_floor import tint_reviewed_chromatic_floor


def run(source, alpha, selection, out, *, source_sha256, alpha_sha256,
        selection_sha256, background_rgb, reviewed=False, review_note=''):
    out=Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('preserve prior result; choose new output directory')
    if not isinstance(background_rgb,(tuple,list)) or len(background_rgb)!=3 or any(type(v)is not int or not 160<=v<=255 for v in background_rgb):
        raise ValueError('three integer light background channels160..255 required')
    inputs={}
    for name,path,mode,expected in [('source',source,'RGB',source_sha256),
            ('alpha',alpha,'L',alpha_sha256),('selection',selection,'L',selection_sha256)]:
        if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('explicit lowercase input file SHA256 required')
        data,array=snapshot(path,mode,1_000_000)
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError('input file binding mismatch')
        inputs[name]=(data,array)
    rgb,a,s=(inputs[n][1] for n in ['source','alpha','selection'])
    bg=np.full_like(rgb,background_rgb)
    output,review=tint_reviewed_chromatic_floor(rgb,a,bg,s,
        source_sha256=digest(rgb),alpha_sha256=digest(a),
        background_sha256=digest(bg),selection_sha256=digest(s),reviewed=reviewed,
        source_floor='white_opaque_with_color_reflection',review_note=review_note)
    baseline=np.rint(rgb*(a[:,:,None]/255)+bg*(1-a[:,:,None]/255)).astype(np.uint8)
    # Reserve only after validation. A failed write retains the inputs and an
    # INCOMPLETE marker, never a misleading successful manifest.
    out.mkdir(parents=True)
    (out/'INCOMPLETE').touch()
    for name,(data,_) in inputs.items():(out/(name+'.png')).write_bytes(data)
    for name,array in [('no-floor.png',baseline),('reviewed-floor.png',output)]:
        Image.fromarray(array).save(out/name)
        with Image.open(out/name)as im:
            if im.mode!='RGB' or not np.array_equal(np.array(im),array):
                raise RuntimeError('saved RGB export mismatch')
    record=dict(schema='offline-reviewed-chromatic-floor-result.v1',
        status='EXECUTED_REVIEW_REQUIRED',provider_calls=0,credits_spent=0,
        input_file_sha256={n:hashlib.sha256(data).hexdigest() for n,(data,_) in inputs.items()},
        output_file_sha256=hashlib.sha256((out/'reviewed-floor.png').read_bytes()).hexdigest(),
        background_rgb=list(background_rgb),review=review)
    (out/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False))
    (out/'INCOMPLETE').unlink()
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','alpha','selection','out']:p.add_argument('--'+n,type=Path,required=True)
    for n in ['source-sha256','alpha-sha256','selection-sha256','review-note']:p.add_argument('--'+n,required=True)
    p.add_argument('--background-rgb',type=int,nargs=3,required=True)
    p.add_argument('--reviewed',action='store_true',required=True)
    args=p.parse_args(argv)
    print(json.dumps(run(**vars(args)),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
