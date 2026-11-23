"""Explicit geometry-prior rasterization, not a shadow estimator.

A caller supplies a model's regression values and foreground rotated box.
This module does not load a model, infer light, or modify a product image.
It avoids writing into a temporary ndarray dtype-conversion copy.
"""
import math
import numpy as np
from PIL import Image,ImageDraw


def rasterize_shadow_prior(regression, foreground, size=(512,512)):
    """Return a real uint8 box mask and diagnostics, from GPS-style offsets.

    foreground=(center_x,center_y,width,height,angle_degrees).
    regression=(normalized_dx,normalized_dy,log_width,log_height,angle_radians).
    Mask samples are 0 or 1, as expected by the published control input.
    """
    if not isinstance(size,(tuple,list)) or len(size)!=2 or any(type(n)!=int or not 1<=n<=4096 for n in size):raise ValueError('size must be two integer dimensions 1-4096')
    try:r=np.asarray(regression,dtype=float);f=np.asarray(foreground,dtype=float)
    except (TypeError,ValueError):raise ValueError('regression and foreground must be five finite numbers')
    if r.shape!=(5,) or f.shape!=(5,) or not np.isfinite(r).all() or not np.isfinite(f).all():raise ValueError('regression and foreground must be five finite numbers')
    if np.any(np.abs(r[:2])>16) or np.any(np.abs(r[2:4])>8) or abs(r[4])>math.pi*8:raise ValueError('regression outside bounded geometry range')
    w,h=size
    if f[2]<=0 or f[3]<=0 or f[2]>w*4 or f[3]>h*4:raise ValueError('foreground dimensions must be positive and bounded')
    cx=r[0]*f[2]+f[0];cy=r[1]*f[3]+f[1];bw=f[2]*math.exp(r[2]);bh=f[3]*math.exp(r[3]);angle=math.degrees(r[4])+f[4]
    if angle>0:bw,bh=bh,bw;angle-=90
    a=math.radians(angle);rotation=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    points=np.array([[-bw/2,-bh/2],[bw/2,-bh/2],[bw/2,bh/2],[-bw/2,bh/2]])@rotation.T+np.array([cx,cy])
    points=np.clip(points,[0,0],[w-1,h-1]);integer_points=points.astype('int32')
    im=Image.new('L',(w,h));ImageDraw.Draw(im).polygon([tuple(p) for p in integer_points.tolist()],fill=1)
    mask=np.array(im,dtype=np.uint8)
    return mask,dict(rotated_box=[float(cx),float(cy),float(bw),float(bh),float(angle)],polygon=integer_points.tolist(),nonzero_pixels=int(np.count_nonzero(mask)),geometry_inferred_here=False,quality='NOT_EVALUATED')


def add_shape_prior(base_embeddings, labels, centroids):
    """Return one centroid addition to a copied embedding array, never in place.

    Reuse the ORIGINAL base for every diffusion step. Passing a previously
    conditioned array intentionally adds again and is not idempotent.
    This helper does not classify the mask or load pickled model content.
    """
    base=np.asarray(base_embeddings)
    ids=np.asarray(labels)
    if base.ndim!=3 or not np.issubdtype(base.dtype,np.floating) or not np.isfinite(base).all():raise ValueError('base must be finite floating batch x slots x dimensions')
    if ids.shape!=base.shape[:2] or not np.issubdtype(ids.dtype,np.integer):raise ValueError('labels must be an integer batch x slots array')
    if not isinstance(centroids,dict):raise ValueError('centroids must be an already-loaded dictionary')
    result=base.copy()
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            label=int(ids[i,j])
            if label not in centroids:continue
            value=np.asarray(centroids[label],dtype=base.dtype)
            if value.shape!=(base.shape[2],) or not np.isfinite(value).all():raise ValueError('centroid dimensions or values invalid')
            with np.errstate(over='ignore',invalid='ignore'):result[i,j]+=value
    if not np.isfinite(result).all():raise ValueError('conditioned embeddings overflowed')
    return result
