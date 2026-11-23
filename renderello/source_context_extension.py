"""Opt-in native scene-context extension. Not product staging or outpainting.

Retains the complete input photo including its actual lighting and contact.
Only new outer canvas pixels are synthesized from reviewed scene border colours.
This deliberately cannot remove the old backdrop, change pose or relight glass.
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .candidate_diagnostics import native_rgb,native_alpha,digest


def extend_source_context(rgb,alpha,*,material,reviewed_source_sha256,
                          reviewed_alpha_sha256,canvas,offset,band=8,decay=48,smooth_outer=False):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if rgb.shape[0]*rgb.shape[1]>1_500_000:raise ValueError('bounded1.5MP native source')
    if material!='opaque':raise ValueError('opaque scene-context route only; no glass or material recovery')
    if digest(rgb)!=reviewed_source_sha256 or digest(alpha)!=reviewed_alpha_sha256:
        raise ValueError('exact source/alpha review binding required')
    if not isinstance(canvas,(tuple,list))or len(canvas)!=2 or any(type(v)is not int or v<3 for v in canvas)or math.prod(canvas)>1_500_000:
        raise ValueError('bounded integer canvas<=1.5MP required')
    if not isinstance(offset,(tuple,list))or len(offset)!=2 or any(type(v)is not int for v in offset):raise ValueError('integer source translation required')
    if type(smooth_outer)is not bool:raise ValueError('boolean smooth-outer switch required')
    h,w=alpha.shape;cw,ch=canvas;dx,dy=offset
    if min(dx,dy)<0 or dx+w>cw or dy+h>ch:raise ValueError('entire source frame must fit; no crop or resize')
    if type(band)is not int or not 2<=band<=min(64,min(w,h)//4):raise ValueError('bounded scene border band required')
    if isinstance(decay,bool)or not isinstance(decay,(int,float))or not math.isfinite(decay)or not 4<=decay<=256:raise ValueError('bounded finite extension decay required')
    if not(alpha>=128).any():raise ValueError('nonempty hard foreground required')
    border=np.ones((h,w),bool);border[band:-band,band:-band]=False
    if (alpha[border]>0).any():raise ValueError('source alpha intersects border band; scene-only extension unverified')
    samples=rgb[border].astype(float)
    med=np.median(samples,axis=0);spread=float(np.percentile(np.linalg.norm(samples-med,axis=1),90))
    if spread>32:raise ValueError('nonuniform scene border; no photographic texture or geometry extension')
    sy,sx=np.nonzero(border);design=np.column_stack([np.ones(len(sx)),sx/max(1,w-1),sy/max(1,h-1)])
    coefficients=np.linalg.lstsq(design,samples,rcond=None)[0]
    yy,xx=np.mgrid[:ch,:cw];local_x=(xx-dx)/max(1,w-1);local_y=(yy-dy)/max(1,h-1)
    # Clamp fit coordinates to observed source bounds. Extrapolated slopes can
    # invent impossible light ramps; fade only to the observed affine edge tone.
    plane=coefficients[0]+np.clip(local_x,0,1)[...,None]*coefficients[1]+np.clip(local_y,0,1)[...,None]*coefficients[2]
    sx=np.clip(xx-dx,0,w-1);sy=np.clip(yy-dy,0,h-1)
    edge=rgb[sy,sx].astype(float);edgeplane=coefficients[0]+sx[...,None]/max(1,w-1)*coefficients[1]+sy[...,None]/max(1,h-1)*coefficients[2]
    distance=np.hypot(xx-(sx+dx),yy-(sy+dy))
    residual=edge-edgeplane
    if smooth_outer:
        # Smooth ONLY scene-border sequences tangentially. Never blur across
        # the product or use a mirrored/replicated product as extension data.
        tx=np.arange(w)/max(1,w-1);ty=np.arange(h)/max(1,h-1)
        top=rgb[0].astype(float)-(coefficients[0]+tx[:,None]*coefficients[1])
        bottom=rgb[-1].astype(float)-(coefficients[0]+tx[:,None]*coefficients[1]+coefficients[2])
        left=rgb[:,0].astype(float)-(coefficients[0]+ty[:,None]*coefficients[2])
        right=rgb[:,-1].astype(float)-(coefficients[0]+coefficients[1]+ty[:,None]*coefficients[2])
        vx=np.maximum.reduce([dx-xx,xx-(dx+w-1),np.zeros_like(xx)]).astype(float)
        vy=np.maximum.reduce([dy-yy,yy-(dy+h-1),np.zeros_like(yy)]).astype(float)
        weight=np.divide(vx,vx+vy,out=np.zeros_like(vx),where=vx+vy>0)
        target=np.minimum(distance/2,128);levels=[0,1,2,4,8,16,32,64,128];filtered=np.zeros_like(residual)
        for i,level in enumerate(levels):
            arrays=[v if level==0 else gaussian_filter1d(v,level,axis=0,mode='nearest')for v in [top,bottom,left,right]]
            vertical=np.where((xx<dx)[...,None],arrays[2][sy],arrays[3][sy])
            horizontal=np.where((yy<dy)[...,None],arrays[0][sx],arrays[1][sx])
            field=vertical*weight[...,None]+horizontal*(1-weight[...,None])
            if i==0:blend=np.clip(1-target,0,1)
            elif i==len(levels)-1:blend=np.clip((target-levels[i-1])/(level-levels[i-1]),0,1)
            else:blend=np.minimum(np.clip((target-levels[i-1])/(level-levels[i-1]),0,1),np.clip((levels[i+1]-target)/(levels[i+1]-level),0,1))
            filtered+=field*blend[...,None]
        residual=filtered
    result=np.rint(np.clip(plane+residual*np.exp(-distance[...,None]/decay),0,255)).astype(np.uint8)
    result[dy:dy+h,dx:dx+w]=rgb
    changed=np.any(result[dy:dy+h,dx:dx+w]!=rgb,axis=2)
    if changed.any():raise RuntimeError('source context changed')
    report=dict(schema='source-context-extension.v1',source_rgb_sha256=digest(rgb),alpha_sha256=digest(alpha),
        canvas=list(canvas),integer_translation=list(offset),source_size=[w,h],source_changed_pixels=0,
        source_pixels_preserved=h*w,new_outer_pixels=cw*ch-h*w,border_band=band,border90_rgb_distance=spread,
        decay=decay,smooth_outer=smooth_outer,fit_coefficients=coefficients.tolist(),source_resampled=False,source_contact_retained=True,
        source_lighting_retained=True,new_contact_synthesized=False,background_replaced=False,
        new_template_executed=False,optics_recovered=False,photo_accepted=False,
        limitations=['The complete old scene remains. This is added copy space, not a product staging template.',
            'Smooth outer colour continuation has no recovered physical geometry or photographic texture.',
            'A caller mask is not ground truth; undetected products at the border can contaminate extension.',
            'Joins and repeated edge-colour streaks require native visual review.'])
    return result,report
