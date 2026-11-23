"""Opt-in plain-backdrop CPU silhouette control. No optics or scene reconstruction."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image

def plain_backdrop_alpha(image,*,material,threshold=24,background_points=(),protected_regions=None):
    if material!='opaque':raise ValueError('This route requires an explicitly opaque product; no glass optics recovery')
    if threshold not in (8,16,24):raise ValueError('Choose a tested threshold: 8,16,24')
    a=np.array(image.convert('RGB'));h,w=a.shape[:2]
    if h*w>16000000:raise ValueError('Source too large for bounded CPU route')
    band=max(1,min(h,w)//100);border=np.concatenate([a[:band].reshape(-1,3),a[-band:].reshape(-1,3),a[:,:band].reshape(-1,3),a[:,-band:].reshape(-1,3)]);med=np.median(border,axis=0);spread=float(np.percentile(np.linalg.norm(border-med,axis=1),90))
    if spread>=28:raise ValueError('Nonuniform source border: refuse this plain-backdrop route')
    candidate=(np.linalg.norm(a.astype(np.float32)-med,axis=2)<=threshold).astype('uint8')
    _,lab=cv2.connectedComponents(candidate,connectivity=8)
    ids=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]));ids=set(int(v) for v in ids if v!=0)
    # Explicit image-coordinate background seeds can remove enclosed source
    # backdrop, such as a handle opening. Never infer these holes from colour.
    if not isinstance(background_points,(list,tuple)) or len(background_points)>64:
        raise ValueError('At most 64 explicit background points required')
    validated=[]
    for point in background_points:
        if not isinstance(point,(list,tuple)) or len(point)!=2 or any(type(v) is not int for v in point):
            raise ValueError('Background points must be integer source coordinates')
        x,y=point
        if not (0<=x<w and 0<=y<h):raise ValueError('Background point outside source')
        if not candidate[y,x]:raise ValueError('Background point is not in the estimated backdrop colour')
        ids.add(int(lab[y,x]));validated.append([x,y])
    alpha=(~np.isin(lab,list(ids))).astype('uint8')*255
    protected_regions={} if protected_regions is None else protected_regions
    if not isinstance(protected_regions,dict):raise ValueError('Protected regions must be a named mapping')
    for name,box in protected_regions.items():
        if not isinstance(name,str) or not name or not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(v) is not int for v in box):
            raise ValueError('Named protected boxes require integer coordinates')
        x1,y1,x2,y2=box
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Protected box outside source')
        if np.any(alpha[y1:y2,x1:x2]!=255):raise ValueError('Removal crosses protected region; review mask and seeds')
    coverage=float(np.mean(alpha>0))
    if not .005<coverage<.95:raise ValueError('Implausible coverage; manual mask review required')
    return Image.fromarray(alpha),{'background_points':validated,'protected_regions':protected_regions,'seed_origin':'EXPLICIT_CALLER_COORDINATES_NOT_AUTOMATIC_REVIEW','enclosed_background_removed_only_when_seeded':True,'border90_rgb_distance':spread,'threshold':threshold,'coverage':coverage,'limitations':['Binary silhouette can retain pale fringes and old shadows.','Near-background-color product parts touching the outside can be lost.','No protected label regions or independent silhouette ground truth supplied.','Not suitable for transparent optics or complex scene separation.']}

def preview_plain_backdrop(source,root,*,material,template='minimal_tabletop',instruction='',alternatives=2,background_points=(),protected_regions=None):
    from .colab_demo import validate_choices
    from .compose import procedural_background
    import shutil
    spec=validate_choices(template,instruction,alternatives);source=Path(source);root=Path(root)
    if root.exists():raise ValueError('Preserve earlier output')
    im=Image.open(source).convert('RGB');alpha,audit=plain_backdrop_alpha(im,material=material,background_points=background_points,protected_regions=protected_regions);root.mkdir();shutil.copyfile(source,root/('source-original'+source.suffix));alpha.save(root/'experimental-alpha.png');outputs=[];a=np.asarray(alpha);ys,xs=np.nonzero(a);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;canvas=(max(1024,im.width),max(1024,im.height));dx=int(canvas[0]//2-cx);dy=int(canvas[1]//2-cy)
    if xs.min()+dx<0 or ys.min()+dy<0 or xs.max()+dx>=canvas[0] or ys.max()+dy>=canvas[1]:raise ValueError('Source placement clipped')
    src=np.asarray(im)
    for i in range(alternatives):
        bg=Image.fromarray(procedural_background(canvas,spec,i)).convert('RGB');bg.paste(im,(dx,dy),alpha);path=root/('alternative-'+str(i+1)+'.png');bg.save(path);saved=np.asarray(Image.open(path));assert np.array_equal(src[ys,xs],saved[ys+dy,xs+dx]);outputs.append(str(path))
    manifest={'kind':'EXPERIMENTAL_PLAIN_BACKDROP_OPAQUE_PREVIEW_NOT_ACCEPTED','mask':audit,'opaque_rgb_changed':0,'kept_source_pixels':len(xs),'integer_translation':[dx,dy],'source_resampled':False,'outputs':outputs,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'custom_instruction_recorded_not_generatively_executed':instruction,'paid_credits':0};(root/'manifest.json').write_text(json.dumps(manifest,indent=2));return {'kind':manifest['kind'],'manifest':str(root/'manifest.json'),'outputs':outputs,'root':str(root)}
