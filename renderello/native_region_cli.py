"""File entry point for opt-in reviewed native-region matting. No network."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
from .matting import refine_native_region
from .image_intake import inspect_image,IntakePolicy
from .candidate_diagnostics import digest


def _sha(p):
    from .image_intake import _read_bounded
    p=Path(p)
    if p.is_symlink():raise ValueError('regular non-symlink input required')
    return hashlib.sha256(_read_bounded(p,20_000_000)).hexdigest()


def run(source,alpha,spec,out,*,trimap=None):
    source,alpha,spec,out=map(Path,(source,alpha,spec,out))
    if out.exists():raise FileExistsError('preserve previous output directory')
    if spec.is_symlink() or not spec.is_file() or spec.stat().st_size>20000:raise ValueError('bounded regular JSON spec required')
    request=json.loads(spec.read_text())
    required={'source_file_sha256','alpha_file_sha256','roi','reviewed','review_note','material','method','edge_colors','feather'}
    if not isinstance(request,dict) or set(request)!=required|({'trimap_file_sha256'} if trimap else set()):
        raise ValueError('exact explicit native-region spec fields required')
    before={source:_sha(source),alpha:_sha(alpha),spec:_sha(spec)}
    if before[source]!=request['source_file_sha256'] or before[alpha]!=request['alpha_file_sha256']:
        raise ValueError('encoded source/alpha hash mismatch')
    policy=IntakePolicy(max_bytes=20_000_000,max_pixels=8_000_000,allow_colour_conversion=False)
    src=inspect_image(source,policy);mask=inspect_image(alpha,policy);tri=None
    try:
        # Reject transformations because ROI/masks name the original coordinate grid.
        if src.record.get('transformations') or set(mask.record.get('transformations',[])) - {'lossless_8bit_channel_expansion'}:
            raise ValueError('normalize source and masks explicitly before source-space region review')
        with Image.open(source) as im:
            if im.mode!='RGB' or im.getexif().get(274,1)!=1:raise ValueError('orientation-normalized RGB source required')
        with Image.open(alpha) as im:
            if im.mode!='L' or im.getexif().get(274,1)!=1:raise ValueError('native unrotated grayscale alpha required')
            aa=np.array(im)
        rgb=np.array(src.image)
        if trimap:
            trimap=Path(trimap);before[trimap]=_sha(trimap)
            if before[trimap]!=request['trimap_file_sha256']:raise ValueError('trimap hash mismatch')
            with inspect_image(trimap,policy).image as checked:
                if checked.size!=(request['roi'][2]-request['roi'][0],request['roi'][3]-request['roi'][1]):raise ValueError('ROI-sized trimap required')
            with Image.open(trimap) as im:
                if im.mode!='L' or im.getexif().get(274,1)!=1 or im.info.get('icc_profile'):raise ValueError('plain unrotated grayscale trimap required')
                tri=np.array(im)
        fg,a,meta=refine_native_region(rgb,aa,request['roi'],source_sha256=digest(rgb),alpha_sha256=digest(aa),
            reviewed=request['reviewed'],review_note=request['review_note'],material=request['material'],method=request['method'],
            edge_colors=request['edge_colors'],feather=request['feather'],trimap=tri)
        if any(_sha(p)!=h for p,h in before.items()):raise ValueError('input changed during matting')
        out.mkdir(parents=True)
        # Save native source RGB as a plain PNG for the source-exact UI mask path.
        Image.fromarray(rgb).save(out/'normalized-source.png');Image.fromarray(a).save(out/'alpha.png')
        Image.fromarray(np.dstack([fg,a])).save(out/'candidate.png')
        (out/'original-source.bin').write_bytes(source.read_bytes());(out/'request.json').write_bytes(spec.read_bytes())
        x0,y0,x1,y1=request['roi'];patch=fg[y0:y1,x0:x1];z=a[y0:y1,x0:x1,None].astype(float)/255
        for label,color in [('white',(255,255,255)),('dark',(30,30,30)),('magenta',(190,0,150))]:
            Image.fromarray(np.rint(patch*z+np.array(color)*(1-z)).astype(np.uint8)).save(out/(label+'-roi.png'))
        meta.update(input_file_sha256={str(p):h for p,h in before.items()},
            dashboard_handoff='normalized-source.png plus alpha.png in existing explicit mask input; review source pose/material/framing separately',
            estimated_color_handoff='candidate.png is research-only when edge colours enabled; explicit alpha upload keeps source RGB, not estimated foreground',
            external_rgba_import='existing UI RGBA importer is capped at1MP; use explicit mask input for larger sources',
            runtime_guard='size/unknown-band checks only, no file-worker wall-clock/RSS enforcement',live_calls=0)
        meta['outputs_sha256']={p.name:_sha(p) for p in out.iterdir() if p.is_file()}
        (out/'report.json').write_text(json.dumps(meta,indent=2));return meta
    finally:src.close();mask.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','alpha','spec','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--trimap',type=Path)
    a=p.parse_args(argv)
    try:r=run(a.source,a.alpha,a.spec,a.out,trimap=a.trimap)
    except (ValueError,TypeError,KeyError,OSError) as exc:p.exit(2,'REFUSED: '+str(exc)+'\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
