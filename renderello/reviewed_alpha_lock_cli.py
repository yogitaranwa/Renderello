"""Executable reviewed alpha-lock route with guarded CF and hash-bound inputs.

Research only. Requires explicit reviewer declarations. Never accepts quality.
Region file is binary L PNG; partial-valued labels are refused, not thresholded.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image
from .reviewed_alpha_lock import prepare_alpha_lock,finish_alpha_lock
from .matting import guarded_file
from .candidate_diagnostics import digest
from .reviewed_parts import audit_parts


def execute(source,alpha,region,out,*,source_file_sha256,alpha_file_sha256,
        region_file_sha256,review_note,reviewed=False,material=None,radius=8):
    source,alpha,region,out=map(Path,[source,alpha,region,out])
    if out.exists():raise ValueError('output exists; preserve prior attempt')
    for p,h in [(source,source_file_sha256),(alpha,alpha_file_sha256),(region,region_file_sha256)]:
        if not p.is_file() or p.stat().st_size>32*1024*1024:raise ValueError('input file missing or exceeds32MiB')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('input file hash mismatch: '+p.name)
    with Image.open(source) as im:
        if im.mode!='RGB':raise ValueError('normalized RGB source required; no implicit orientation/color conversion')
        rgb=np.array(im)
    arrays=[]
    for p in [alpha,region]:
        with Image.open(p) as im:
            if im.mode!='L':raise ValueError('alpha and region must be native8bitL')
            arrays.append(np.array(im))
    a,f=arrays
    if not set(np.unique(f)).issubset({0,255}):raise ValueError('binary0/255region file required; no threshold guessing')
    frozen=f==255
    review=dict(source_sha256=digest(rgb),alpha_sha256=digest(a),frozen_sha256=digest(frozen),
        review_note=review_note,reviewed=reviewed,material=material,radius=radius)
    tri,prep=prepare_alpha_lock(rgb,a,frozen,**review)
    # Exclusive reserve only after all input/region declarations validate.
    out.mkdir(parents=True,exist_ok=False)
    (out/'attempt.json').write_text(json.dumps(dict(status='STARTED_NOT_ACCEPTED',review=review,
        source_file_sha256=source_file_sha256,alpha_file_sha256=alpha_file_sha256,region_file_sha256=region_file_sha256),indent=2))
    try:
        Image.fromarray(tri).save(out/'reviewed-trimap.png')
        worker=guarded_file(source,alpha,out/'worker',trimap_path=out/'reviewed-trimap.png',method='cf',radius=radius,edge_colors=False)
        fg=np.array(Image.open(out/'worker/foreground.png'));solved=np.array(Image.open(out/'worker/refined-alpha.png'))
        fg,matte,tri,meta=finish_alpha_lock(rgb,a,frozen,fg,solved,tri,**review)
        part_audit=audit_parts(rgb,a,fg,matte,[dict(name='reviewed-alpha-region',mask=frozen,mask_sha256=digest(frozen))],
            source_sha256=digest(rgb),prior_alpha_sha256=digest(a),reviewed=reviewed,review_note=review_note)
        if not part_audit['all_named_parts_preserved']:raise RuntimeError('reviewed region preservation failed')
        (out/'part-preservation.json').write_text(json.dumps(part_audit,indent=2))
        meta['part_preservation']=part_audit
        Image.fromarray(matte).save(out/'locked-alpha.png');Image.fromarray(np.dstack([fg,matte])).save(out/'locked-cutout.png')
        for name,c in [('light',(250,250,250)),('dark',(20,20,20))]:
            im=Image.new('RGB',(rgb.shape[1],rgb.shape[0]),c);im.paste(Image.fromarray(fg),mask=Image.fromarray(matte));im.save(out/(name+'.png'))
        meta.update(worker=worker,status='EXECUTED_NOT_ACCEPTED',input_files=dict(source=str(source.resolve()),alpha=str(alpha.resolve()),region=str(region.resolve())),
            input_file_sha256=dict(source=source_file_sha256,alpha=alpha_file_sha256,region=region_file_sha256),
            output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.png')})
        (out/'manifest.json').write_text(json.dumps(meta,indent=2));return meta
    except Exception as exc:
        (out/'refused.json').write_text(json.dumps(dict(status='REFUSED_NOT_ACCEPTED',error=str(exc),accepted=False),indent=2))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','alpha','region']:
        p.add_argument('--'+n,required=True,type=Path);p.add_argument('--'+n+'-file-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--review-note',required=True)
    p.add_argument('--reviewed',action='store_true');p.add_argument('--material',choices=['opaque'],required=True);p.add_argument('--radius',type=int,default=8)
    args=vars(p.parse_args())
    try:print(json.dumps(execute(**args),indent=2))
    except (ValueError,RuntimeError,OSError) as e:p.exit(2,'REFUSED: '+str(e)+'\n')

if __name__=='__main__':main()
