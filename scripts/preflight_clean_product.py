"""Zero-spend diagnostic using Renderello's actual u2netp cutout code.
Does not accept a subject, authorize a payment or declare a quality pass.
Run with PYTHONPATH pointing to the deployed Renderello source checkout.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image
from renderello.cutout import cutout
from renderello.matte_integrity import matte_stats, assess_compact

def inspect_arrays(source, rgba, alpha):
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('native RGB uint8 source required')
    if rgba.shape != (*source.shape[:2],4) or alpha.shape != source.shape[:2]:
        raise ValueError('native dimensions must match; no resizing')
    if alpha.dtype != np.uint8 or rgba.dtype != np.uint8:
        raise ValueError('uint8 outputs required')
    if not np.array_equal(rgba[:,:,:3],source):
        raise ValueError('source RGB changed')
    if not np.array_equal(rgba[:,:,3],alpha):
        raise ValueError('RGBA alpha differs from saved matte')
    if not np.any(alpha>=128) or not np.any(alpha<128):
        raise ValueError('empty or full matte')
    stats=matte_stats(alpha)
    return dict(native_dimensions_match=True, source_rgb_changed=0, stats=stats,
                integrity_warning=assess_compact(stats),
                visual_review='PENDING_EXACT_SOURCE_AND_NATIVE_MASK',
                semantic_completeness='NOT_PROVEN_BY_COMPONENTS',
                edges_and_lighting='NOT_EVALUATED', quality_pass=False,
                paid_calls=0, paid_execution_implemented=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('preserve prior output; choose new directory')
    raw=a.source.read_bytes()
    with Image.open(a.source) as im:
        if im.mode!='RGB' or getattr(im,'n_frames',1)!=1:
            raise ValueError('explicit native RGB single-frame source required')
        if im.getexif().get(274,1)!=1 or im.info.get('icc_profile'):
            raise ValueError('explicit orientation/color normalization required')
        if im.width*im.height>4_000_000:raise ValueError('4MP diagnostic budget exceeded')
        source=np.array(im);co=cutout(im,a.out,'product',backend='u2netp_onnx')
    rgba=np.array(Image.open(co.rgba_path));alpha=np.array(Image.open(co.matte_path))
    report=inspect_arrays(source,rgba,alpha)
    report.update(source_sha256=hashlib.sha256(raw).hexdigest(),source_size=[source.shape[1],source.shape[0]],backend=co.backend)
    for name,col in [('white',(255,255,255)),('magenta',(190,0,150))]:
        # Alpha-composited diagnostic, not Magic Hour output.
        f=alpha.astype(np.float32)[:,:,None]/255
        preview=np.rint(source*f+np.array(col)*(1-f)).astype(np.uint8)
        Image.fromarray(preview).save(a.out/(name+'-matte-review.png'))
    Image.fromarray(source).save(a.out/'Source.png')
    report['outputs_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir() if f.is_file()}
    (a.out/'preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
