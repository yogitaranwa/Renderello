"""Smaller source-sized alternate compositions. No upscale or model call."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
from .compose import finish_scene, flatlay_center


def native_reframe(cache_dir, *, fill=.76):
    """Center an unchanged cached product on a smaller, same-aspect scene crop.

    This deliberately exports fewer pixels rather than inventing product detail.
    Camera angle and illumination are unchanged and still need human review.
    Glass/cover-fit caches are excluded because their layer semantics differ.
    """
    if not math.isfinite(fill) or not .25 <= fill <= .9:
        raise ValueError('fill must be finite and between .25 and .9')
    cd=Path(cache_dir)
    options=json.loads((cd/'layers.json').read_text())
    if options.get('cover_used') or options.get('glass_backdrop') is not None:
        raise ValueError('native reframe requires an ordinary opaque-product cache')
    scene=np.asarray(Image.open(cd/'scene.png').convert('RGB'))
    product=np.asarray(Image.open(cd/'product_rgba.png').convert('RGBA'))
    yy,xx=np.where(product[...,3]>0)
    if not len(xx):raise ValueError('empty product alpha')
    # Trim only fully transparent padding; all nonzero-alpha pixels retained.
    product=product[yy.min():yy.max()+1,xx.min():xx.max()+1].copy()
    ph,pw=product.shape[:2];h,w=scene.shape[:2]
    if pw>w or ph>h:raise ValueError('product larger than scene')
    factor=min(1.,max(pw/(w*fill),ph/(h*fill)))
    nw=min(w,max(pw,math.ceil(w*factor)))
    nh=min(h,max(ph,math.ceil(h*factor)))
    x=(w-nw)//2;y=(h-nh)//2
    scene=scene[y:y+nh,x:x+nw].copy()
    px,py=flatlay_center(product[...,3],(nw,nh))
    out,lift=finish_scene(scene,product,(px,py),
        shadow_style=options.get('shadow_style'),scene_grade=options.get('scene_grade',0.),
        scene_auto_lift=options.get('scene_auto_lift',False))
    inner=product[...,3]==255
    exact=bool(inner.any() and np.array_equal(out[py:py+ph,px:px+pw][inner],product[...,:3][inner]))
    return Image.fromarray(out),dict(output_size=[nw,nh],original_size=[w,h],
        product_size=[pw,ph],placement=[px,py],paste_bit_exact=exact,
        product_resampled=False,scene_resampled=False,source_resolution_limited=True,
        scene_lift=lift,quality_status='REVIEW_REQUIRED',
        warnings=['Smaller export, not a higher-resolution product.',
                  'Camera angle, light direction, mask and contact need visual review.'])
