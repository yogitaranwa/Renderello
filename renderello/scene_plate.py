"""Opt-in real background plate composition with declared support geometry.

The image generator makes empty scenery only. No automatic camera/lighting
inference, no product relighting, no glass optics or photographic acceptance.
"""
from pathlib import Path
from io import BytesIO
import hashlib,json
import cv2
import numpy as np
from PIL import Image
from .prepared_compose import load_prepared
from .native_pixel_audit import audit_native_pixels


def support_shadow(alpha, canvas, placement, *, cast_offset=(0,0), opacity=.16):
    """Two-scale contact cue plus optional designer-directed 2D cast projection.

    cast_offset is the chosen floor displacement of the product's highest point.
    It is not recovered light direction or physical geometry.
    """
    if alpha.dtype!=np.uint8 or alpha.ndim!=2 or min(alpha.shape)<1:
        raise ValueError('alpha must be nonempty uint8 L')
    w,h=canvas;x,y=placement;ph,pw=alpha.shape
    if any(type(v) is not int for v in [w,h,x,y]) or not (0<=x<=w-pw and 0<=y<=h-ph):
        raise ValueError('native product must fit inside canvas')
    if not 0<=opacity<=.35 or len(cast_offset)!=2 or not np.isfinite(cast_offset).all():
        raise ValueError('invalid declared shadow parameters')
    visible=alpha>127;yy,xx=np.where(visible)
    if not len(yy):raise ValueError('opaque support contour required')
    bottom=int(yy.max());support=np.zeros((h,w),np.float32)
    band=max(2,round(ph*.025))
    for col in range(pw):
        rows=np.flatnonzero(visible[:,col])
        if len(rows) and rows[-1]>=bottom-band:
            fy=y+int(rows[-1])+1
            if fy<h:support[fy,x+col]=alpha[rows[-1],col]/255.
    # Contact follows actual bottom contours, rather than a full-width ellipse.
    tight=cv2.GaussianBlur(support,(0,0),sigmaX=1.5,sigmaY=1.2)
    broad=cv2.GaussianBlur(support,(0,0),sigmaX=max(3,pw*.025),sigmaY=max(2,ph*.008))
    for layer,peak in [(tight,.24),(broad,.12)]:
        if layer.max()>0:layer*=peak/layer.max()
    contact=1-(1-tight)*(1-broad)
    dx,dy=cast_offset
    if dx or dy:
        # The lowest support row maps to itself. Higher silhouette pixels
        # project onto the explicitly selected support-plane direction.
        denom=max(1,bottom)
        matrix=np.float32([[1,-dx/denom,x+dx*bottom/denom],
                           [0,-dy/denom,y+bottom+dy*bottom/denom]])
        cast=cv2.warpAffine(alpha.astype(np.float32)/255,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cast=cv2.GaussianBlur(cast,(0,0),sigmaX=max(3,pw*.025),sigmaY=max(2,ph*.012))*opacity
        contact=1-(1-contact)*(1-cast)
    return np.clip(contact,0,.45)


def compose_plate(prepared,plate_path,out,*,baseline=None,center_x=512,cast_offset=(0,0),contact=True,
                  support_plane_path=None,contact_strength=1.,scene_exposure_ev=0.,scene_white_balance=(1.,1.,1.),background_kind='unspecified_inspected_background',shadow_model='legacy',shadow_softness=1.,reviewed_contact_span=None,source_origin=None):
    """Native-source placement onto an inspected plate. Refuse any resizing."""
    if background_kind not in ['photograph','generated','procedural','unspecified_inspected_background']:
        raise ValueError('declare photograph/generated/procedural/unspecified background kind')
    if shadow_model not in ['legacy','height_soft_linear']:raise ValueError('unknown shadow model')
    if not np.isfinite(shadow_softness) or not .25<=shadow_softness<=3:raise ValueError('softness must be finite0.25to3')
    out=Path(out)
    if out.exists():raise ValueError('output exists; preserve older candidate')
    if source_origin is None:
        if type(baseline) is not int or type(center_x) is not int:raise ValueError('integer placement required')
    else:
        if baseline is not None or center_x!=512:raise ValueError('source origin cannot combine with baseline/centre placement')
        if not isinstance(source_origin,(list,tuple)) or len(source_origin)!=2 or any(type(v) is not int for v in source_origin):raise ValueError('source origin must be two integers')
    record,files=load_prepared(prepared)
    source=np.array(Image.open(BytesIO(files['normalized-source.png'])))
    alpha=np.array(Image.open(BytesIO(files['reviewed-alpha.png'])))
    raw=Path(plate_path).read_bytes()
    with Image.open(BytesIO(raw)) as im:
        if im.mode!='RGB' or im.size!=(1024,1024):raise ValueError('reviewed plate must be1024x1024RGB')
        im.load();scene=np.array(im)
    from .scene_finish import inspect_support_components
    support_components=inspect_support_components(alpha)
    yy,xx=np.where(alpha>0)
    if not len(yy):raise ValueError('empty alpha')
    sx,sy=int(xx.min()),int(yy.min());sw,sh=int(xx.max())+1-sx,int(yy.max())+1-sy
    a=alpha[sy:sy+sh,sx:sx+sw];rgb=source[sy:sy+sh,sx:sx+sw]
    support=np.where((a>127).any(axis=1))[0]
    if not len(support):raise ValueError('opaque support contour required')
    if source_origin is None:
        x=center_x-sw//2;y=baseline-int(support[-1])-1
    else:
        x=source_origin[0]+sx;y=source_origin[1]+sy
        baseline=y+int(support[-1])+1
    if x<0 or y<0 or x+sw>1024 or y+sh>1024:raise ValueError('native product does not fit declared placement')
    if not np.isfinite(contact_strength) or not 0<=contact_strength<=2:
        raise ValueError("contact strength must be finite0to2")
    from .scene_finish import finish_background,validate_support_plane
    plane=None;plane_hash=None;support_samples=None
    if support_plane_path is not None:
        plane_bytes=Path(support_plane_path).read_bytes()
        with Image.open(BytesIO(plane_bytes)) as im:
            if im.mode!="L" or im.size!=(1024,1024):raise ValueError("support plane must be1024x1024L")
            plane=np.array(im)
        support_samples=validate_support_plane(plane,a,(x,y))
        plane_hash=hashlib.sha256(plane_bytes).hexdigest()
    scene=finish_background(scene,exposure_ev=scene_exposure_ev,white_balance=scene_white_balance)
    from .scene_shadow import projected_soft_shadow,attenuate_linear,reviewed_span_contact
    crop_span=None
    if reviewed_contact_span is not None:
        if not contact or shadow_model!='height_soft_linear':raise ValueError('reviewed contact span requires enabled height-soft contact')
        if not isinstance(reviewed_contact_span,(list,tuple)) or len(reviewed_contact_span)!=2 or any(type(v) is not int for v in reviewed_contact_span):raise ValueError('reviewed contact span must be integer source columns')
        crop_span=(reviewed_contact_span[0]-sx,reviewed_contact_span[1]-sx)
        reviewed_span_contact(a,(1024,1024),(x,y),crop_span)
        if plane is not None:
            for col in range(*crop_span):
                row=int(np.flatnonzero(a[:,col]>127)[-1])+y+1
                if plane[row,x+col]!=255:raise ValueError('reviewed contact span misses support plane')
    shadow_fn=support_shadow if shadow_model=='legacy' else projected_soft_shadow
    extra={} if shadow_model=='legacy' else {'softness':shadow_softness}
    if crop_span is not None:extra['contact_span']=crop_span
    shadow=(shadow_fn(a,(1024,1024),(x,y),cast_offset=cast_offset,**extra) if contact else np.zeros((1024,1024),np.float32))
    shadow=np.clip(shadow*contact_strength,0,.65)
    if plane is not None:shadow*=plane.astype(np.float32)/255
    # Neutral multiplicative attenuation changes only the scene, never RGB
    # inside the source product. The source's existing highlights stay as-is.
    if shadow_model=='legacy':
        shaded=scene.astype(np.float32)*(1-shadow[...,None])
    else:
        # Keep the reviewed near-contact cue while softening only the cast.
        near=(support_shadow(a,(1024,1024),(x,y),cast_offset=(0,0)) if crop_span is None else reviewed_span_contact(a,(1024,1024),(x,y),crop_span)) if contact else np.zeros((1024,1024),np.float32)
        near=np.clip(near*contact_strength,0,.65)
        if plane is not None:near*=plane.astype(np.float32)/255
        cast_only=np.clip((shadow-near)/np.maximum(1-near,1e-6),0,1)
        shaded=attenuate_linear(scene,cast_only).astype(np.float32)*(1-near[...,None])
    opacity=a[...,None].astype(np.float32)/255
    region=shaded[y:y+sh,x:x+sw]
    region[:]=region*(1-opacity)+rgb.astype(np.float32)*opacity
    final=np.clip(shaded,0,255).astype(np.uint8)
    audit=audit_native_pixels(source,alpha,final,(sx,sy,sw,sh),(x,y,sw,sh))
    if audit['status']!='MATCH':raise ValueError('native product opaque RGB failed')
    out.mkdir(parents=True)
    Image.fromarray(final).save(out/'candidate.png');Image.fromarray(np.rint(shadow*255).astype(np.uint8)).save(out/'shadow-alpha.png')
    meta=dict(placement_mode='original_source_origin' if source_origin is not None else 'alpha_bounds_center_baseline',original_source_origin=[x-sx,y-sy],reviewed_contact_span_source_columns=reviewed_contact_span,contact_span_geometry='EXPLICIT_REVIEW_NOT_INFERRED' if crop_span is not None else 'lowest_row_band',support_component_review=support_components,shadow_model=shadow_model,shadow_softness=shadow_softness,shadow_light_geometry='DESIGNER_DECLARED_NOT_INFERRED',shadow_alpha_meaning='geometric_strength_field_NOT_sRGB_transmittance' if shadow_model!='legacy' else 'sRGB_multiplicative_attenuation',support_plane_sha256=plane_hash,support_plane_samples=support_samples,contact_strength=contact_strength,scene_exposure_ev=scene_exposure_ev,scene_white_balance=list(scene_white_balance),scene_finish='declared_scene_only_tone_matching_NOT_physical_relighting',source_record=record,background_sha256=hashlib.sha256(raw).hexdigest(),source_box=[sx,sy,sw,sh],placement=[x,y],baseline=baseline,cast_offset=list(cast_offset),shadow_kind='declared_2D_support_and_optional_projection',background_kind=background_kind,source_scale=1,product_relighting=False,glass_optics=False,native_pixel_audit=audit,photographic_acceptance='NOT_EVALUATED',source_rights='NOT_UPGRADED')
    (out/'scene-plate.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    return meta
