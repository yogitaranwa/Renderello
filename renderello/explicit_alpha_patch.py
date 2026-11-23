"""Opt-in native alpha annotations. No semantic inference or quality approval.

A source review note records the annotator's rationale; its presence does not
prove correctness. Restoring opacity reuses photographed RGB, including any old
scene contamination. Callers must inspect the resulting composites.
"""
import hashlib
import numpy as np
from PIL import Image, ImageDraw


def _orientation(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersects(a,b,c,d):
    def on(a,b,c):
        return _orientation(a,b,c)==0 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
    x,y,z,t=_orientation(a,b,c),_orientation(a,b,d),_orientation(c,d,a),_orientation(c,d,b)
    return ((x>0)!=(y>0) and (z>0)!=(t>0)) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)


def _polygon(points,width,height):
    if not isinstance(points,(list,tuple)) or len(points)<3:
        raise ValueError('polygon needs at least three vertices')
    for p in points:
        if not isinstance(p,(list,tuple)) or len(p)!=2 or any(type(x)!=int for x in p):
            raise ValueError('integer source coordinates required')
        if not (0<=p[0]<width and 0<=p[1]<height):
            raise ValueError('polygon outside source')
    if len(set(map(tuple,points)))!=len(points):raise ValueError('repeated vertex')
    edges=list(zip(points,points[1:]+points[:1]))
    if sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)==0:raise ValueError('zero area polygon')
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j<=i or j==i+1 or (i==0 and j==len(edges)-1):continue
            if _intersects(a,b,c,d):raise ValueError('self-intersecting polygon')
    im=Image.new('L',(width,height));ImageDraw.Draw(im).polygon(points,fill=255)
    return np.asarray(im)>0


def apply_explicit_alpha_patches(alpha, protected_regions, patches):
    """Return a new L alpha and exact delta ledger. Never modifies input/defaults.

    Patches: {operation: 'exclude'|'restore_opaque', polygon: [[x,y],...],
              source_review_note: nonempty str}. Overlap and protected-region
    intersection are refused. Native pixel coordinates include polygon borders.
    """
    if not isinstance(alpha,Image.Image) or alpha.mode!='L':raise ValueError('native L alpha required')
    a=np.asarray(alpha);height,width=a.shape
    if width*height>4_000_000:raise ValueError('source too large')
    if not isinstance(protected_regions,dict) or not protected_regions:raise ValueError('named protected regions required')
    protected=np.zeros(a.shape,bool)
    for name,box in protected_regions.items():
        if not isinstance(name,str) or not name.strip() or not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(x)!=int for x in box):raise ValueError('invalid protected region')
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('invalid protected region bounds')
        if np.any(a[y1:y2,x1:x2]!=255):raise ValueError('protected region must be opaque')
        protected[y1:y2,x1:x2]=True
    if not isinstance(patches,(list,tuple)):raise ValueError('explicit patch list required')
    b=a.copy();used=np.zeros(a.shape,bool);ledger=[]
    for patch in patches:
        if not isinstance(patch,dict) or set(patch)!={'operation','polygon','source_review_note'}:raise ValueError('explicit patch fields required')
        op=patch['operation'];note=patch['source_review_note']
        if op not in ('exclude','restore_opaque'):raise ValueError('invalid patch operation')
        if not isinstance(note,str) or not note.strip():raise ValueError('source review rationale required')
        mask=_polygon(patch['polygon'],width,height)
        if np.any(mask&protected):raise ValueError('patch crosses protected region')
        if np.any(mask&used):raise ValueError('overlapping patches require one unambiguous annotation')
        used|=mask;b[mask]=0 if op=='exclude' else 255
        ledger.append({**patch,'selected_pixels':int(mask.sum())})
    manifest={'kind':'EXPLICIT_ALPHA_ANNOTATION_NOT_PRODUCTION_APPROVAL','patches':ledger,'alpha_changed_pixels':int((b!=a).sum()),'support_removed':int(((a>0)&(b==0)).sum()),'support_added':int(((a==0)&(b>0)).sum()),'protected_alpha_changed':int(((b!=a)&protected).sum()),'source_resampled':False,'quality_accepted':False,'independent_ground_truth':False,'input_alpha_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'output_alpha_sha256':hashlib.sha256(b.tobytes()).hexdigest()}
    return Image.fromarray(b),manifest


def export_explicit_alpha_review(source, alpha, protected_regions, patches, output):
    """Write a new review-only bundle with native light/dark audit previews.

    No acceptance marker or production-review bundle is created. Existing output
    directories are preserved, and validation finishes before files are written.
    """
    from pathlib import Path
    import json, tempfile
    if not isinstance(source,Image.Image) or source.mode!='RGB' or source.size!=alpha.size:
        raise ValueError('native RGB source must match alpha dimensions')
    output=Path(output)
    if output.exists():raise ValueError('preserve previous output')
    candidate,manifest=apply_explicit_alpha_patches(alpha,protected_regions,patches)
    rgb=np.asarray(source);a=np.asarray(candidate)
    previews={}
    for name,color in [('light',(245,245,240)),('dark',(25,25,30))]:
        im=Image.new('RGB',source.size,color);im.paste(source,mask=candidate)
        seen=np.asarray(im);opaque=a==255
        if not np.array_equal(seen[opaque],rgb[opaque]):raise RuntimeError('opaque RGB altered')
        previews[name]=im
    manifest.update(source_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),retained_opaque_rgb_changed=0,source_rgb_repainted=False,protected_regions=protected_regions,production_review_created=False,caption='CPU source-locked alpha-annotation preview, not Magic Hour. Requires visual review; not publication approval.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.alpha-review-') as tmp:
        tmp=Path(tmp);source.save(tmp/'source-native.png');alpha.save(tmp/'baseline-alpha.png');candidate.save(tmp/'candidate-alpha.png')
        for name,im in previews.items():
            im.save(tmp/f'{name}-native.png')
            plate=Image.new('RGB',(max(source.width,800),source.height+72),'white');plate.paste(im,(0,0));draw=ImageDraw.Draw(plate)
            draw.text((8,source.height+8),f'{name}: CPU source-locked alpha-annotation preview, not Magic Hour.',fill='black')
            draw.text((8,source.height+30),'Manual source-coordinate candidate; requires visual review, not publication approval.',fill='black')
            draw.text((8,source.height+50),'Native pixels. No RGB repainting/resampling. Protected alpha changes: 0.',fill='black')
            plate.save(tmp/f'{name}-captioned.png')
        (tmp/'manifest.json').write_text(json.dumps(manifest,indent=2))
        # Final directory appears only after every file has been written.
        if output.exists():raise ValueError('preserve previous output')
        tmp.rename(output)
    return manifest
