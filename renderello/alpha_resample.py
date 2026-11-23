"""Explicit alpha-aware byte-RGB resampling. No alpha or material recovery.

Filters premultiplied sRGB code values, not calibrated linear scene radiance.
Opaque-only inputs retain OpenCV's exact established byte resampler. Fully
transparent contributors cannot carry hidden RGB into a visible product edge.
"""
import cv2
import numpy as np


def resize_rgba(rgba, size, *, interpolation):
    if not isinstance(rgba,np.ndarray) or rgba.dtype!=np.uint8 or rgba.ndim!=3 or rgba.shape[2]!=4:
        raise ValueError('uint8 RGBA required')
    if min(rgba.shape[:2])<1 or rgba.shape[0]*rgba.shape[1]>4_000_000:
        raise ValueError('nonempty bounded4MP source required')
    if not isinstance(size,(tuple,list)) or len(size)!=2 or any(type(v)is not int or v<1 for v in size) or size[0]*size[1]>4_000_000:
        raise ValueError('integer positive4MP destination required')
    if type(interpolation)is not int or interpolation not in (cv2.INTER_AREA,cv2.INTER_LANCZOS4):
        raise ValueError('explicit area or Lanczos4 resampling required')
    if size[0]==rgba.shape[1] and size[1]==rgba.shape[0]:
        return rgba.copy()
    if np.all(rgba[...,3]==255):
        return cv2.resize(rgba,tuple(size),interpolation=interpolation)
    a=rgba[...,3].astype(np.float32)/255
    p=rgba[...,:3].astype(np.float32)*a[...,None]
    af=cv2.resize(a,tuple(size),interpolation=interpolation)
    pf=cv2.resize(p,tuple(size),interpolation=interpolation)
    # Lanczos has negative lobes. Bound coverage before division and suppress
    # invisible RGB, including coverage that rounds to zero in the stored alpha.
    # Divide by the unbounded filtered coverage to preserve constant straight
    # colour when a negative-lobe filter overshoots coverage above one.
    coverage=np.clip(af,0,1)
    alpha=np.rint(coverage*255).astype(np.uint8)
    straight=np.zeros_like(pf)
    np.divide(pf,af[...,None],out=straight,where=af[...,None]>1e-8)
    rgb=np.clip(np.rint(straight),0,255).astype(np.uint8)
    rgb[alpha==0]=0
    return np.dstack((rgb,alpha))
