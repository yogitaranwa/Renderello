"""Native-pixel scene control, not generative relighting or automatic mask approval.

A reviewed mask and explicitly reviewed opaque regions are required. The source
is translated by whole pixels only. No resampling, warp or RGB repaint occurs.
Transparent packaging retains old-scene reflections/transmission and is not
claimed optically correct. Output is a preview until separately reviewed.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageFilter


def compose_native(source, alpha_path, out, *, protected_regions, canvas=(1024,1024), offset=None):
    source,alpha_path,out=map(Path,(source,alpha_path,out))
    if out.exists():raise ValueError('output exists; preserve earlier versions')
    im=Image.open(source).convert('RGB');alpha=Image.open(alpha_path).convert('L')
    if alpha.size!=im.size:raise ValueError('mask must match native source size')
    if im.width*im.height>4_000_000:raise ValueError('source too large for bounded preview')
    if not protected_regions:raise ValueError('reviewed opaque regions required')
    if len(canvas)!=2 or any(type(x)!=int or x<1 for x in canvas) or canvas[0]*canvas[1]>4_000_000:raise ValueError('invalid or oversized canvas')
    bbox=alpha.getbbox()
    if bbox is None:raise ValueError('empty mask')
    if offset is None:offset=((canvas[0]-(bbox[2]-bbox[0]))//2-bbox[0],(canvas[1]-(bbox[3]-bbox[1]))//2-bbox[1])
    if len(offset)!=2 or any(type(x)!=int for x in offset):raise ValueError('integer translation only')
    dx,dy=offset
    if bbox[0]+dx<0 or bbox[1]+dy<0 or bbox[2]+dx>canvas[0] or bbox[3]+dy>canvas[1]:raise ValueError('mask would be clipped')
    aa=np.array(alpha);rgb=np.array(im);checks=[]
    for name,box in protected_regions.items():
        if len(box)!=4 or any(type(x)!=int for x in box):raise ValueError('integer boxes required')
        x1,y1,x2,y2=box
        if not(0<=x1<x2<=im.width and 0<=y1<y2<=im.height):raise ValueError('box outside source')
        if np.any(aa[y1:y2,x1:x2]!=255):raise ValueError('protected region must already be opaque in reviewed mask')
        checks.append({'name':name,'source_box':box,'output_box':[x1+dx,y1+dy,x2+dx,y2+dy]})
    # Flat neutral scene, low contrast falloff. Product RGB never relit.
    yy,xx=np.mgrid[:canvas[1],:canvas[0]];grad=np.clip((yy/canvas[1])*8+((xx-canvas[0]*.4)/canvas[0])**2*8,0,15)
    bg=np.stack([242-grad,239-grad,234-grad],axis=-1).astype('uint8');base=Image.fromarray(bg)
    # Explicitly synthetic contact ellipse, no claim of geometry correctness.
    shadow=np.zeros((canvas[1],canvas[0]),dtype='uint8');cx=(bbox[0]+bbox[2])/2+dx;cy=bbox[3]+dy-2
    ex=max(20,(bbox[2]-bbox[0])*.38);ey=12
    shadow=((np.exp(-((xx-cx)/ex)**2-((yy-cy)/ey)**2))*45).astype('uint8')
    shade=Image.fromarray(shadow).filter(ImageFilter.GaussianBlur(6));base.paste(Image.new('RGB',canvas,(85,80,75)),(0,0),shade)
    base.paste(im,(dx,dy),alpha)
    seen=np.array(base)
    for c in checks:
        x1,y1,x2,y2=c['source_box'];u1,v1,u2,v2=c['output_box'];diff=np.abs(seen[v1:v2,u1:u2].astype('int16')-rgb[y1:y2,x1:x2].astype('int16'))
        c.update(changed_pixels=int(np.any(diff,axis=2).sum()),max_channel_error=int(diff.max()),pixels=(x2-x1)*(y2-y1))
        if c['changed_pixels']:raise RuntimeError('protected source region changed in actual output')
    ys,xs=np.nonzero(aa==255)
    opaque_error=np.abs(seen[ys+dy,xs+dx].astype('int16')-rgb[ys,xs].astype('int16'))
    all_opaque={'pixels':len(xs),'changed_pixels':int(np.any(opaque_error,axis=1).sum()),'max_channel_error':int(opaque_error.max())}
    if all_opaque['changed_pixels']:raise RuntimeError('opaque source pixels changed')
    out.mkdir(parents=True);base.save(out/'output.png');alpha.save(out/'reviewed-alpha.png')
    rgba=im.copy();rgba.putalpha(alpha);rgba.save(out/'source-native-cutout.png')
    manifest={'kind':'SOURCE_LOCKED_NATIVE_PREVIEW_NOT_QUALITY_APPROVAL','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_size':list(im.size),'canvas':list(canvas),'integer_translation':list(offset),'source_resampled':False,'rgb_relit':False,'protected_regions':checks,'all_opaque_audit':all_opaque,'mask_source':'caller-supplied reviewed mask, not automatic approval','limitations':['Source blur and viewpoint remain.','Transparent material retains old-scene transmission/reflections.','Contact ellipse is procedural, not physically estimated.','Mask edges need visual inspection.'],'output_sha256':hashlib.sha256((out/'output.png').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
