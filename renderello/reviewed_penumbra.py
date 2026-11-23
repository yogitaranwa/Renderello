"""Bounded height-band penumbra design on reviewed 2D casts, not light recovery."""
from dataclasses import dataclass,asdict
import math
import cv2
import numpy as np
from .reviewed_silhouette_cast import cast_source_silhouette,SilhouetteCast
from .candidate_diagnostics import digest
from .candidate_harmonization import linear_over
from .checked_research.contact_camera import linear,srgb

@dataclass(frozen=True)
class PenumbraProfile:
    near_sigma:float=.75
    far_sigma:float=9.
    far_opacity_ratio:float=.6
    levels:int=8


def cast_reviewed_penumbra(rgb,alpha,background,contact,review,profile):
    if not isinstance(profile,PenumbraProfile):raise ValueError('typed penumbra profile required')
    for name,lo,hi in [('near_sigma',.5,12),('far_sigma',.5,12),('far_opacity_ratio',.2,1.)]:
        v=getattr(profile,name)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(name+' outside bounded design controls')
    if profile.far_sigma<profile.near_sigma:raise ValueError('far sigma must not be smaller than near')
    if type(profile.levels) is not int or not 4<=profile.levels<=12:raise ValueError('4..12integer bands required')
    # Reuse all source/support/contact/light/boundary checks. Margin must cover
    # the largest new kernel, never silently replace the old review controls.
    if not isinstance(review,SilhouetteCast):raise ValueError('typed silhouette review required')
    if review.blur_sigma!=profile.far_sigma:raise ValueError('review blur margin must equal maximum profile sigma')
    _,_,_,meta=cast_source_silhouette(rgb,alpha,background,contact,review)
    h,w=alpha.shape;matrix=np.array(meta['projection_matrix'])
    projected=cv2.warpAffine(alpha.astype(np.float64)/255,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    nonzero=np.where(alpha>0)[0];height=max(1,review.floor_row-int(nonzero.min()))
    # With this affine projection, target depth is proportional to source
    # height. That proxy is not recovered 3D distance or light size.
    depth=np.clip((np.arange(h)-review.floor_row)/(review.cast_y_per_height*height),0,1)[:,None]
    result=np.zeros((h,w),np.float64)
    for i,t in enumerate(np.linspace(0,1,profile.levels)):
        weights=np.maximum(1-np.abs(depth-t)*(profile.levels-1),0)
        sigma=profile.near_sigma+(profile.far_sigma-profile.near_sigma)*t
        margin=math.ceil(3*sigma);atten=1-(1-profile.far_opacity_ratio)*t
        layer=projected*weights*atten
        result+=cv2.GaussianBlur(layer,(2*margin+1,2*margin+1),sigma,borderType=cv2.BORDER_CONSTANT)
    cast=np.clip(result,0,1)*review.opacity
    cast[:review.floor_row]=0;hard=alpha>=128;cast[hard]=0
    shade=1-(1-contact)*(1-cast);shade[hard]=0
    out=linear_over(rgb,alpha,srgb(linear(background)*(1-shade[...,None])))
    if not np.array_equal(out[alpha==255],rgb[alpha==255]):raise RuntimeError('opaque source RGB changed')
    meta.update(schema='reviewed-height-band-penumbra.v1',profile=asdict(profile),
        penumbra_proxy='affine projected height, not measured3Ddistance',cast_sha256=digest(cast),
        combined_shadow_sha256=digest(shade),result_sha256=digest(out),cast_pixels=int((cast>0).sum()))
    meta['limitations']=meta['limitations'][:-1]+['Height-band blur and opacity are authored design approximations, not measured light size or physical penumbra.']
    return out,cast,shade,meta
