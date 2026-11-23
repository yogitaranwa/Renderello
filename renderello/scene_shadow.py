"""Explicit soft-light shadow approximation. Not recovered scene geometry.

A flat silhouette projection is sliced by normalized source height. More distant
slices have wider penumbrae and weaker opacity. This remains a 2D approximation,
not a physically reconstructed product or measured light.
"""
import cv2
import numpy as np


def projected_soft_shadow(alpha, canvas, placement, *, cast_offset=(0,0), opacity=.16, softness=1., contact_span=None):
    from .scene_plate import support_shadow
    if not np.isfinite(opacity) or not 0<=opacity<=.35:raise ValueError('invalid declared opacity')
    # Reuse established shape/placement/opacity checks before allocating layers.
    contact=(support_shadow(alpha,canvas,placement,cast_offset=(0,0),opacity=opacity) if contact_span is None else reviewed_span_contact(alpha,canvas,placement,contact_span))
    if not np.isfinite(softness) or not .25<=softness<=3:
        raise ValueError('softness must be finite0.25to3')
    if len(cast_offset)!=2 or not np.isfinite(cast_offset).all():
        raise ValueError('finite cast offset required')
    dx,dy=cast_offset
    if abs(dx)>canvas[0] or abs(dy)>canvas[1]:
        raise ValueError('cast offset exceeds canvas')
    if not dx and not dy:return contact
    w,h=canvas;x,y=placement;ph,pw=alpha.shape
    bottom=int(np.where(alpha>127)[0].max());denom=max(1,bottom)
    height=np.clip((bottom-np.arange(ph,dtype=np.float32))/denom,0,1)
    matrix=np.float32([[1,-dx/denom,x+dx],[0,-dy/denom,y+bottom+dy]])
    cast=np.zeros((h,w),np.float32)
    centers=np.linspace(0,1,9)
    for level in centers:
        # Linear partition of unity gives smooth height weighting without bands.
        weight=np.maximum(0,1-np.abs(height-level)*8)
        layer=alpha.astype(np.float32)/255*weight[:,None]
        warped=cv2.warpAffine(layer,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        sigma=softness*(1.2+level*max(5,pw*.075))
        layer=cv2.GaussianBlur(warped,(0,0),sigmaX=sigma,sigmaY=max(1,sigma*.6))
        cast+=layer*opacity*(1-.45*level)
    return np.clip(1-(1-contact)*(1-np.clip(cast,0,.35)),0,.45)


def attenuate_linear(rgb, shadow):
    """Neutral linear-light transmittance, with exact no-shadow identity."""
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise ValueError('uint8RGB required')
    if not isinstance(shadow,np.ndarray) or shadow.shape!=rgb.shape[:2] or not np.isfinite(shadow).all() or (shadow<0).any() or (shadow>1).any():
        raise ValueError('finite aligned shadow0to1 required')
    s=rgb.astype(np.float64)/255
    lin=np.where(s<=.04045,s/12.92,((s+.055)/1.055)**2.4)*(1-shadow[...,None])
    out=np.where(lin<=.0031308,lin*12.92,1.055*lin**(1/2.4)-.055)
    result=np.rint(np.clip(out,0,1)*255).astype(np.uint8)
    result[shadow==0]=rgb[shadow==0]
    return result


def reviewed_span_contact(alpha,canvas,placement,span):
    """Explicit half-open native crop-column contact span, never inferred geometry.

    Choose only after image inspection. Follows the lower opaque contour within
    the declared span, including sloped bases. Does not alter alpha or RGB.
    """
    from .scene_plate import support_shadow
    support_shadow(alpha,canvas,placement) # established alpha/canvas checks
    if not isinstance(span,(list,tuple)) or len(span)!=2 or any(type(v) is not int for v in span):
        raise ValueError('contact span must be two integer crop columns')
    left,right=span;ph,pw=alpha.shape;w,h=canvas;x,y=placement
    if not 0<=left<right<=pw or right-left<2:raise ValueError('contact span outside native crop or too narrow')
    support=np.zeros((h,w),np.float32)
    for col in range(left,right):
        rows=np.flatnonzero(alpha[:,col]>127)
        if not len(rows) or y+int(rows[-1])+1>=h:raise ValueError('declared contact column has no supported lower contour')
        support[y+int(rows[-1])+1,x+col]=alpha[rows[-1],col]/255.
    tight=cv2.GaussianBlur(support,(0,0),sigmaX=1.5,sigmaY=1.2)
    broad=cv2.GaussianBlur(support,(0,0),sigmaX=max(3,pw*.025),sigmaY=max(2,ph*.008))
    for layer,peak in [(tight,.24),(broad,.12)]:
        if layer.max()>0:layer*=peak/layer.max()
    return np.clip(1-(1-tight)*(1-broad),0,.45)
