"""Opt-in two-layer floor-shadow replay. No inferred light or product editing.

A designer declares a flat-floor setting and a shadow direction. Alpha-only
projection cannot recover3D geometry, so outputs remain designed candidates.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageFilter
from .intent_mask import sha

FLOOR_STYLES = ('porcelain_sweep','sage_arch','noir_halo','amber_paper','linen_daylight','cobalt_aperture')


def shadow_layers(alpha, size, placement, *, direction=0.45, depth=0.12,
                  cast_opacity=0.08, contact_opacity=0.42):
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.ndim!=2:
        raise ValueError('alpha must be a uint8 plane')
    if len(size)!=2 or any(type(n)!=int or not 64<=n<=4096 for n in size):
        raise ValueError('canvas dimensions must be64-4096')
    if len(placement)!=2 or any(type(n)!=int for n in placement):
        raise ValueError('placement must be two integers')
    vals=(direction,depth,cast_opacity,contact_opacity)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in vals):
        raise ValueError('shadow parameters must be finite numbers')
    if not -1<=direction<=1 or not 0.02<=depth<=0.3 or not 0<=cast_opacity<=0.4 or not 0<=contact_opacity<=0.6:
        raise ValueError('shadow parameters outside reviewed bounds')
    w,h=size;left,top=placement;ph,pw=alpha.shape
    if not (0<=left and 0<=top and left+pw<=w and top+ph<=h):
        raise ValueError('product outside canvas')
    ys,xs=np.where(alpha>127)
    if not len(ys):raise ValueError('support pixels above127 required')
    foot=int(ys.max());floor=top+foot
    if floor+3>=h:raise ValueError('insufficient floor room')
    # A small band of actual base columns, not an ellipse across empty foot gaps.
    band=max(1,min(3,round(ph*.006)))
    profile=alpha[max(0,foot-band+1):foot+1].max(axis=0).astype(float)/255
    contact=np.zeros((h,w),np.uint8)
    for dy,factor in ((0,1.),(1,.75),(2,.25)):
        contact[floor+dy,left:left+pw]=np.round(profile*contact_opacity*255*factor).astype('uint8')
    contact=np.array(Image.fromarray(contact).filter(ImageFilter.GaussianBlur(1.5)))
    # Designed2Dprojection. Higher rows travel farther; do not claim true optics.
    cast=np.zeros((h,w),np.uint8)
    for row in range(foot+1):
        height=foot-row;dy=round(height*depth);dx=round(height*direction*depth)
        target_y=floor+dy
        if target_y>=h:continue
        src0=max(0,-left-dx);src1=min(pw,w-left-dx)
        if src1<=src0:continue
        value=np.round(alpha[row,src0:src1].astype(float)*cast_opacity*math.exp(-height/max(1,foot*.22))).astype('uint8')
        a=left+dx+src0;b=left+dx+src1
        cast[target_y,a:b]=np.maximum(cast[target_y,a:b],value)
    cast=np.array(Image.fromarray(cast).filter(ImageFilter.GaussianBlur(5)))
    # Blur first, then floor clipping. No blurred halo is allowed on the wall.
    contact[:floor]=0;cast[:floor]=0
    combined=np.round(255*(1-(1-contact.astype(float)/255)*(1-cast.astype(float)/255))).astype('uint8')
    return contact,cast,combined,dict(floor_row=floor,support_band_rows=band,
        support_columns=int((profile>0).sum()),direction=direction,depth=depth,
        cast_opacity=cast_opacity,contact_opacity=contact_opacity,
        rows_above_floor_nonzero=int(np.count_nonzero(combined[:floor])),
        physical_geometry_recovered=False)


def replay_contact(candidate:Path,out:Path,*,flat_floor_confirmed=False,direction=0.45,depth=0.12):
    candidate=Path(candidate);out=Path(out)
    if out.exists():raise ValueError('output exists; preserve earlier versions')
    if flat_floor_confirmed is not True:raise ValueError('explicit flat floor declaration required')
    m=json.loads((candidate/'native-studio-manifest.json').read_text())
    if m.get('style') not in FLOOR_STYLES:raise ValueError('platform support geometry requires separate review')
    if m.get('material')!='opaque' or m.get('framing')!='complete' or m.get('source_pose') not in ('front','elevated') or m.get('scale')!=1 or m.get('rotation')!=0:
        raise ValueError('source-native opaque complete candidate required')
    for f in ('background.png','native-product-crop.png'):
        if sha(candidate/f)!=m.get('output_sha256',{}).get(f):raise ValueError('candidate file hash mismatch: '+f)
    with Image.open(candidate/'background.png') as im:bg=im.convert('RGB')
    with Image.open(candidate/'native-product-crop.png') as im:
        if im.mode!='RGBA':raise ValueError('RGBA product required')
        crop=np.array(im)
    alpha=crop[:,:,3];core=alpha==255
    if not np.any(core):raise ValueError('opaque core required')
    contact,cast,combined,d=shadow_layers(alpha,bg.size,m['placement'],direction=direction,depth=depth)
    scene=Image.composite(Image.new('RGB',bg.size,(0,0,0)),bg,Image.fromarray(combined))
    scene.paste(Image.fromarray(crop[:,:,:3]),tuple(m['placement']),Image.fromarray(alpha))
    x,y=m['placement'];ph,pw=alpha.shape;result=np.array(scene)[y:y+ph,x:x+pw]
    d['opaque_core_rgb_max_error']=int(np.abs(result[core].astype('int16')-crop[:,:,:3][core].astype('int16')).max())
    if d['opaque_core_rgb_max_error']:raise ValueError('source-native product invariant failed')
    d.update(style=m['style'],parent_manifest_sha256=sha(candidate/'native-studio-manifest.json'),source_sha256=m['source_sha256'],placement=m['placement'],scale=1,rotation=0,opaque_core_pixels=int(core.sum()),quality='NOT_EVALUATED',mode='DESIGNED_TWO_LAYER_SHADOW_NOT_INFERRED_LIGHTING',warnings=['No product relighting or material repair.','Source alpha errors survive unchanged.','Only fully opaque core pixels remain exact in final RGB.','Flat floor and light direction are designer declarations, not image inference.','Platforms rejected, no attempt to paint a shadow across unsupported geometry.'])
    out.mkdir(parents=True);scene.save(out/'candidate.png')
    for name,a in (('contact-alpha.png',contact),('cast-alpha.png',cast),('shadow-alpha.png',combined)):Image.fromarray(a).save(out/name)
    d['output_sha256']={p.name:sha(p) for p in out.iterdir()}
    (out/'contact-studio-manifest.json').write_text(json.dumps(d,indent=2));return d
