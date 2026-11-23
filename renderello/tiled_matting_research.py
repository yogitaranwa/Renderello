"""Opt-in guarded native-mask research, not a production compositor or quality gate.

Original RGB is copied without resizing. Alpha is estimated separately on bounded
native tiles. Seam agreement is a numerical consistency limit, not semantic truth,
photographic acceptance, or equivalence to a global matting solution.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import shutil
import tempfile

import numpy as np
from PIL import Image
from .matting import guarded_file

MAX_SOURCE_PIXELS = 4_000_000
MAX_TILE_PIXELS = 1_000_000
MAX_FILE_BYTES = 32_000_000


def tile_starts(length, tile, overlap):
    if any(type(x) is not int for x in (length, tile, overlap)) or length < 1 or tile < 3 or not 1 <= overlap < tile:
        raise ValueError('integer length, tile >=3 and overlap1..tile-1 required')
    if length <= tile:
        return [0]
    return sorted(set([*range(0, length-tile, tile-overlap), length-tile]))


def _read_image(path, mode):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('bounded regular image file required')
    raw = p.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError('image bytes exceed bound')
    with Image.open(BytesIO(raw)) as im:
        if im.mode != mode or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('single-frame native '+mode+' image required; no conversion')
        if im.getexif().get(274, 1) != 1 or im.info.get('icc_profile') or 'transparency' in im.info:
            raise ValueError('normalize orientation and color metadata explicitly first')
        if im.width*im.height > MAX_SOURCE_PIXELS:
            raise ValueError('research source exceeds four million pixels; no resize')
        im.load()
        return np.array(im), hashlib.sha256(raw).hexdigest()


def _weight(height, width, overlap):
    def ramp(n):
        return np.minimum(1., np.minimum((np.arange(n)+1)/overlap, (n-np.arange(n))/overlap))
    return ramp(height)[:, None]*ramp(width)[None, :]


def stitch_tiles(shape, tiles, overlap, seam_limit):
    """Compare EVERY intersecting tile before deciding whether alpha may be saved."""
    if type(seam_limit) is not int or not 0 <= seam_limit <= 255:
        raise ValueError('integer alpha seam limit0to255 required')
    if not isinstance(shape,(tuple,list)) or len(shape)!=2 or any(type(x) is not int or x<1 for x in shape) or shape[0]*shape[1]>MAX_SOURCE_PIXELS:
        raise ValueError('bounded native shape required')
    if type(overlap) is not int or not 1<=overlap<=1000 or not 1<=len(tiles)<=64:
        raise ValueError('bounded overlap and1to64 tiles required')
    h, w = shape
    sums = np.zeros(shape, np.float64)
    weights = np.zeros(shape, np.float64)
    seams = []
    for i, (x, y, a) in enumerate(tiles):
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.ndim != 2 or a.size > MAX_TILE_PIXELS:
            raise ValueError('bounded uint8 tile required')
        if type(x) is not int or type(y) is not int:raise ValueError('integer tile origin required')
        th, tw = a.shape
        if min(th, tw) < 1 or x < 0 or y < 0 or x+tw > w or y+th > h:
            raise ValueError('tile outside native source')
        for j, (px, py, pa) in enumerate(tiles[:i]):
            left, top = max(x, px), max(y, py)
            right, bottom = min(x+tw, px+pa.shape[1]), min(y+th, py+pa.shape[0])
            if left >= right or top >= bottom:
                continue
            diff = np.abs(a[top-y:bottom-y, left-x:right-x].astype(np.int16)-pa[top-py:bottom-py, left-px:right-px].astype(np.int16))
            seams.append(dict(tiles=[j,i], box=[left,top,right,bottom], max_difference=int(diff.max()), differing_pixels=int(np.count_nonzero(diff)), over_limit=int(np.count_nonzero(diff>seam_limit)), mean_difference=float(diff.mean())))
        weight = _weight(th, tw, overlap)
        sums[y:y+th,x:x+tw] += a*weight
        weights[y:y+th,x:x+tw] += weight
    if not np.all(weights > 0):
        raise ValueError('native source has uncovered pixels')
    accepted = not any(s['over_limit'] for s in seams)
    return (np.rint(sums/weights).astype(np.uint8) if accepted else None), seams


def run(source, initial_alpha, out, *, tile_size=850, overlap=100, seam_limit=16):
    """Retain failed tile evidence but NEVER save a stitched mask on seam refusal."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('preserve older research output')
    if type(tile_size) is not int or not 3 <= tile_size <= 1000 or tile_size*tile_size > MAX_TILE_PIXELS:
        raise ValueError('tile size3to1000 required; million-pixel guard unchanged')
    if type(overlap) is not int or not 1 <= overlap < tile_size:
        raise ValueError('overlap1..tile-1 required')
    if type(seam_limit) is not int or not 0 <= seam_limit <= 255:
        raise ValueError('integer alpha seam limit0to255 required')
    rgb, source_hash = _read_image(source, 'RGB')
    alpha, alpha_hash = _read_image(initial_alpha, 'L')
    if rgb.shape[:2] != alpha.shape or alpha.size < 9:
        raise ValueError('native source and alpha must align')
    h, w = alpha.shape
    xs, ys = tile_starts(w,tile_size,overlap), tile_starts(h,tile_size,overlap)
    if len(xs)*len(ys) > 64:
        raise ValueError('research capped at64 tiles')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+out.name+'-',dir=out.parent))
    report = dict(status='INCOMPLETE', source_sha256=source_hash, initial_alpha_sha256=alpha_hash, source_shape=list(rgb.shape), source_rgb_resampled=False, tile_size=tile_size, overlap=overlap, seam_limit=seam_limit, quality='NOT_EVALUATED', semantics='NOT_VERIFIED', global_solver_equivalence=False, defaults_changed=False, tiles=[])
    try:
        Image.fromarray(rgb).save(stage/'source.png')
        Image.fromarray(alpha).save(stage/'initial-alpha.png')
        tiles=[]
        for y in ys:
            for x in xs:
                i=len(tiles); a=alpha[y:y+tile_size,x:x+tile_size]; c=rgb[y:y+tile_size,x:x+tile_size]
                sp,ap=stage/f'tile{i}-source.png',stage/f'tile{i}-alpha.png'
                Image.fromarray(c).save(sp);Image.fromarray(a).save(ap)
                guarded_file(sp,ap,stage/f'tile{i}-cf',method='cf',radius=3,edge_colors=False)
                with Image.open(stage/f'tile{i}-cf/source.png') as im:
                    if not np.array_equal(np.array(im),c):raise ValueError('tile source RGB changed')
                with Image.open(stage/f'tile{i}-cf/refined-alpha.png') as im:
                    if im.mode!='L' or im.size!=(a.shape[1],a.shape[0]):raise ValueError('tile alpha shape/mode changed')
                    refined=np.array(im)
                with Image.open(stage/f'tile{i}-cf/trimap.png') as im:tri=np.array(im)
                if not np.all(refined[tri==0]==0) or not np.all(refined[tri==255]==255):raise ValueError('protected trimap core changed')
                tiles.append((x,y,refined));report['tiles'].append(dict(index=i,origin=[x,y],shape=list(a.shape),pixels=a.size,source_rgb_exact=True,trimap_cores_exact=True))
        stitched,seams=stitch_tiles(alpha.shape,tiles,overlap,seam_limit);report['seams']=seams
        report['status']='SEAM_LIMIT_PASSED_NOT_QUALITY_ACCEPTED' if stitched is not None else 'REJECTED_SEAM_LIMIT'
        if stitched is not None:
            Image.fromarray(stitched).save(stage/'stitched-alpha.png')
            report['stitched_alpha_sha256']=hashlib.sha256((stage/'stitched-alpha.png').read_bytes()).hexdigest()
        (stage/'research-manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        # mkdir, not replace: never overwrite output created during processing.
        out.mkdir()
        for p in stage.iterdir():p.rename(out/p.name)
        return report
    finally:
        shutil.rmtree(stage,ignore_errors=True)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','alpha','out'):p.add_argument('--'+flag,required=True,type=Path)
    p.add_argument('--tile-size',type=int,default=850);p.add_argument('--overlap',type=int,default=100);p.add_argument('--seam-limit',type=int,default=16)
    a=p.parse_args(argv);r=run(a.source,a.alpha,a.out,tile_size=a.tile_size,overlap=a.overlap,seam_limit=a.seam_limit);print(json.dumps(r,indent=2));return 0 if r['status'].startswith('SEAM_LIMIT_PASSED') else 2

if __name__=='__main__':raise SystemExit(main())
