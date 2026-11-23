"""Explicit whole-photograph copyspace exports, not product staging.

Run python -m renderello.context_reframe_cli --help. Does not call models,
resample products, infer a review, relight, or create photoshoot templates.
"""
from pathlib import Path
from io import BytesIO
import argparse,hashlib,json
import numpy as np
from PIL import Image
from .candidate_diagnostics import digest,native_rgb,native_alpha
from .source_context_extension import extend_source_context

ASPECTS={'square':(1024,1024),'portrait':(896,1120),'landscape':(1056,704)}


def _bytes(path,limit):
    with Path(path).open('rb')as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('bounded input bytes required')
    return data


def _image(data,mode):
    with Image.open(BytesIO(data))as im:
        if im.mode!=mode or getattr(im,'n_frames',1)!=1:raise ValueError('single-frame '+mode+' required; no silent colour conversion')
        if im.getexif().get(274,1)!=1:raise ValueError('normalize orientation before review')
        if im.width*im.height>1_500_000:raise ValueError('native1.5MP budget; no downscale')
        return np.array(im)


def run(source,alpha,review,out,*,aspects=('square','portrait','landscape')):
    out=Path(out)
    if out.exists():raise FileExistsError('preserve earlier exports; choose a new directory')
    if not aspects or len(set(aspects))!=len(aspects)or any(x not in ASPECTS for x in aspects):raise ValueError('unique supported aspect names required')
    blobs={n:_bytes(p,32_000_000 if n!='review'else 64_000)for n,p in [('source',source),('alpha',alpha),('review',review)]}
    rgb=_image(blobs['source'],'RGB');a=_image(blobs['alpha'],'L');native_rgb(rgb);native_alpha(a,rgb.shape[:2])
    spec=json.loads(blobs['review'])
    keys={'schema','source_pixels_sha256','alpha_pixels_sha256','material','reviewed','note'}
    if not isinstance(spec,dict)or set(spec)!=keys or spec['schema']!='renderello-whole-photo-reframe-review-v1':raise ValueError('exact supported review schema required')
    if spec['reviewed']is not True or spec['material']!='opaque':raise ValueError('explicit opaque review required; no inferred material or glass recovery')
    if not isinstance(spec['note'],str)or not spec['note'].strip()or len(spec['note'])>2000:raise ValueError('bounded nonempty review note required')
    if spec['source_pixels_sha256']!=digest(rgb)or spec['alpha_pixels_sha256']!=digest(a):raise ValueError('review belongs to different source/alpha pixels')
    out.mkdir(parents=True);(out/'INCOMPLETE').touch()
    Image.fromarray(rgb).save(out/'source.png');Image.fromarray(a).save(out/'alpha.png');(out/'review.json').write_bytes(blobs['review'])
    rows=[];h,w=a.shape
    for name in aspects:
        canvas=ASPECTS[name];offset=((canvas[0]-w)//2,(canvas[1]-h)//2)
        try:
            result,report=extend_source_context(rgb,a,material='opaque',reviewed_source_sha256=digest(rgb),reviewed_alpha_sha256=digest(a),canvas=canvas,offset=offset,smooth_outer=True)
            dx,dy=offset
            if not np.array_equal(result[dy:dy+h,dx:dx+w],rgb):raise RuntimeError('whole-source exactness failed')
            path=out/(name+'.png');Image.fromarray(result).save(path)
            rows.append(dict(aspect=name,status='EXECUTED_REVIEW_REQUIRED',file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),report=report))
        except ValueError as exc:rows.append(dict(aspect=name,status='REFUSED',reason=str(exc),canvas=list(canvas)))
    manifest=dict(schema='renderello-whole-photo-reframe-result-v1',purpose='ADDED_COPYSPACE_ONLY_NOT_PHOTOSHOOT_TEMPLATE',
        input_file_sha256={n:hashlib.sha256(b).hexdigest()for n,b in blobs.items()},input_pixels_sha256={'source':digest(rgb),'alpha':digest(a)},
        input_semantics='bounded_snapshots_no_resampling_or_colour_conversion',review_is_caller_declaration_not_owner_quality_approval=True,
        planned_exports=len(rows),rendered_exports=sum(r['status']=='EXECUTED_REVIEW_REQUIRED'for r in rows),
        refused_exports=sum(r['status']=='REFUSED'for r in rows),distinct_sources=1,new_template_accepts=0,photo_quality='NOT_EVALUATED',rows=rows,
        limitations=['Entire old photograph is retained, including its camera, light, background and contact.',
        'Estimated alpha is not ground truth; source-border review and native visual inspection remain required.',
        'No new scene, relighting, pose, transparent optics or polished-template acceptance.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'INCOMPLETE').unlink();return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ['source','alpha','review','out']:parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--aspects',nargs='+',choices=tuple(ASPECTS),default=list(ASPECTS));args=parser.parse_args(argv)
    result=run(args.source,args.alpha,args.review,args.out,aspects=tuple(args.aspects));print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
