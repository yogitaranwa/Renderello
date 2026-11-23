"""Opt-in source-reviewed affine silhouette cast, not recovered physical light.

Orthographic-like 2D projection on a declared neutral floor. No source RGB,
alpha, camera or material edit. Existing contact is retained separately.
"""
from dataclasses import dataclass,asdict
import math
import cv2
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear,srgb

@dataclass(frozen=True)
class SilhouetteCast:
    floor_row:int
    support_x0:int
    support_x1:int
    cast_x_per_height:float
    cast_y_per_height:float
    blur_sigma:float
    opacity:float
    source_sha256:str
    alpha_sha256:str
    background_sha256:str
    contact_sha256:str
    review_note:str
    reviewed:bool=False
    source_key_side:str='left'
    floor_kind:str='continuous_neutral'


def cast_source_silhouette(rgb,alpha,background,contact,review):
    native_rgb(rgb);native_rgb(background);native_alpha(alpha,rgb.shape[:2])
    h,w=alpha.shape
    if rgb.shape!=background.shape or h*w>2_000_000:raise ValueError('same native canvas within2MP required')
    if not isinstance(review,SilhouetteCast) or review.reviewed is not True or review.floor_kind!='continuous_neutral':
        raise ValueError('typed reviewed continuous neutral floor required')
    if not isinstance(review.review_note,str) or not review.review_note.strip() or len(review.review_note)>2000:
        raise ValueError('source light/support review note required')
    if not isinstance(contact,np.ndarray) or contact.dtype.kind!='f' or contact.shape!=alpha.shape or not np.isfinite(contact).all() or np.any((contact<0)|(contact>.5)):
        raise ValueError('finite native contact attenuation0..0.5required')
    bindings=[(rgb,review.source_sha256),(alpha,review.alpha_sha256),(background,review.background_sha256),(contact,review.contact_sha256)]
    if any(digest(v)!=s for v,s in bindings):raise ValueError('source/alpha/background/contact binding mismatch')
    if any(type(getattr(review,k)) is not int for k in ['floor_row','support_x0','support_x1']):raise ValueError('integer native support required')
    if not 0<=review.support_x0<review.support_x1<=w or not 0<=review.floor_row<h-1:raise ValueError('support outside canvas')
    ranges=[('cast_x_per_height',-.8,.8),('cast_y_per_height',.03,.35),('blur_sigma',.5,12),('opacity',0,.4)]
    for name,lo,hi in ranges:
        v=getattr(review,name)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(name+' outside bounded controls')
    if review.source_key_side not in ('left','right'):raise ValueError('reviewed source key side left/right required')
    if (review.source_key_side=='left' and review.cast_x_per_height<=0) or (review.source_key_side=='right' and review.cast_x_per_height>=0):
        raise ValueError('cast must point away from reviewed side key')
    hard=alpha>=128
    if not hard.any() or hard.all():raise ValueError('nontrivial native silhouette required')
    if any(e.any() for e in [hard[0],hard[-1],hard[:,0],hard[:,-1]]):raise ValueError('complete silhouette must not touch frame')
    if np.any(contact[hard]):raise ValueError('existing contact crosses hard foreground')
    lower=np.where(hard.any(0),h-1-np.argmax(hard[::-1],axis=0),-1)
    cols=np.arange(review.support_x0,review.support_x1);rows=lower[cols]
    if np.any(np.abs(rows-review.floor_row)>2) or not np.all(contact[rows+1,cols]>0):
        raise ValueError('reviewed base/floor/contact not attached')
    yy,xx=np.where(alpha>0)
    if yy.max()>review.floor_row+2:raise ValueError('nonzero source alpha below declared floor beyond2pixel tolerance')
    # y height above support maps to positive floor depth; no hidden clipping.
    kx,ky=review.cast_x_per_height,review.cast_y_per_height
    tx=xx+kx*(review.floor_row-yy);ty=review.floor_row+ky*(review.floor_row-yy)
    margin=math.ceil(3*review.blur_sigma)
    if tx.min()-margin<0 or tx.max()+margin>=w or ty.max()+margin>=h:
        raise ValueError('projected silhouette plus3sigma blur leaves canvas; refuse rather than truncate')
    matrix=np.array([[1,-kx,kx*review.floor_row],[0,-ky,(1+ky)*review.floor_row]],np.float64)
    projection=cv2.warpAffine(alpha.astype(np.float64)/255,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    blurred=cv2.GaussianBlur(projection,(2*margin+1,2*margin+1),review.blur_sigma,borderType=cv2.BORDER_CONSTANT)
    cast=blurred*review.opacity
    cast[:review.floor_row]=0
    cast[hard]=0
    # Multiplicative attenuation keeps contact readable under the cast.
    shade=1-(1-contact)*(1-cast);shade[hard]=0
    out=linear_over(rgb,alpha,srgb(linear(background)*(1-shade[...,None])))
    if not np.array_equal(out[alpha==255],rgb[alpha==255]):raise RuntimeError('opaque source RGB changed')
    return out,cast,shade,dict(schema='reviewed-silhouette-cast.v1',review=asdict(review),
        projection_matrix=matrix.tolist(),cast_sha256=digest(cast),combined_shadow_sha256=digest(shade),
        result_sha256=digest(out),cast_pixels=int((cast>0).sum()),source_resampled=False,
        source_rgb_repainted=False,alpha_repainted=False,opaque_source_pixels_exact=True,
        recovered_geometry=False,physical_relighting=False,accepted=False,default_pipeline_changed=False,
        limitations=['2D affine silhouette projection, not recovered3Dshape or physical illumination.',
          'Source light side, opaque material and floor support are reviewer declarations.',
          'Source highlights/reflections, alpha errors and material mismatch remain unchanged.',
          'Uniform floor blur/opacity is a design approximation, not a measured penumbra.'])
