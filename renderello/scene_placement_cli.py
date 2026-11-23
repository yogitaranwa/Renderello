"""Explicit actual-scene native placement command. No model or fallback.

Caller declarations are not owner authority, calibrated geometry or photo approval.
"""
from pathlib import Path
from io import BytesIO
import argparse, hashlib, json
import numpy as np
from PIL import Image
from .reviewed_scene_placement import ScenePlacementReview, plan_reviewed_placement, compose_reviewed_placement
from .contact_spans import ContactPolicy, ContactSpan

NAMES=('source','alpha','scene','surface','exclusion')


def _snapshot(path,limit):
    with Path(path).open('rb') as stream:
        value=stream.read(limit+1)
    if len(value)>limit:raise ValueError('bounded input snapshot required')
    return value


def _decode(data,mode,budget):
    with Image.open(BytesIO(data)) as im:
        if im.format!='PNG' or im.mode!=mode or getattr(im,'n_frames',1)!=1:
            raise ValueError('single-frame native '+mode+' PNG required; no conversion')
        if im.getexif().get(274,1)!=1:raise ValueError('normalize orientation before reviewing')
        if im.width*im.height>budget:raise ValueError('native allocation budget exceeded; no resize')
        im.load()
        return np.array(im)


def run(source,alpha,scene,surface,exclusion,review,out):
    out=Path(out)
    if out.exists():raise FileExistsError('preserve earlier output; choose a new directory')
    blobs={n:_snapshot(p,32_000_000) for n,p in zip(NAMES,(source,alpha,scene,surface,exclusion))}
    blobs['review']=_snapshot(review,64_000)
    spec=json.loads(blobs['review'])
    if not isinstance(spec,dict) or set(spec)!={'schema','files','review','contact_policy','spans','translation'} or spec['schema']!='renderello-actual-scene-placement-review.v1':
        raise ValueError('exact scene placement command review schema required')
    hashes={n:hashlib.sha256(blobs[n]).hexdigest() for n in NAMES}
    if spec['files']!=hashes:raise ValueError('encoded input snapshot changed')
    rgb=_decode(blobs['source'],'RGB',1_000_000);a=_decode(blobs['alpha'],'L',1_000_000)
    bg=_decode(blobs['scene'],'RGB',2_000_000)
    plane=_decode(blobs['surface'],'L',2_000_000);ex=_decode(blobs['exclusion'],'L',2_000_000)
    if any(not np.isin(m,[0,255]).all() for m in (plane,ex)):raise ValueError('binary0/255 scene masks required')
    if not isinstance(spec['spans'],list) or len(spec['spans'])>64:raise ValueError('bounded native span list required')
    r=ScenePlacementReview(**spec['review']);p=ContactPolicy(**spec['contact_policy']);spans=[ContactSpan(**s) for s in spec['spans']]
    plane=plane>0;ex=ex>0
    plan=plan_reviewed_placement(rgb,a,bg,plane,ex,r,p,spans,translation=spec['translation'])
    # All parsing, binding and geometry validation precedes creating the result.
    # Unexpected errors after this point leave the marker, not a success manifest.
    out.mkdir(parents=True);(out/'INCOMPLETE').touch()
    for n in NAMES:(out/(n+'.png')).write_bytes(blobs[n])
    (out/'review.json').write_bytes(blobs['review'])
    manifest=dict(schema='renderello-actual-scene-placement-result.v1',status='REFUSED',
        input_file_sha256={n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()},
        report=plan,photo_accepted=False,polished_templates=0,models_called=0,
        review_is_caller_declaration_not_owner_or_quality_approval=True,outputs=[])
    if not plan['blockers']:
        result,shadow,report=compose_reviewed_placement(rgb,a,bg,plane,ex,r,p,spans,translation=spec['translation'])
        Image.fromarray(result).save(out/'designed-preview.png')
        # PNG is a rounded visualization only; retain actual float attenuation too.
        Image.fromarray(np.rint(shadow*255).astype(np.uint8)).save(out/'shadow-visualization.png')
        np.save(out/'shadow.npy',shadow,allow_pickle=False)
        manifest.update(status='EXECUTED_REVIEW_REQUIRED',report=report,outputs=[dict(path=n,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in ('designed-preview.png','shadow-visualization.png','shadow.npy')])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'INCOMPLETE').unlink()
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in NAMES+('review','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    result=run(**vars(args));print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
