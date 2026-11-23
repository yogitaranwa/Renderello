"""Checked bilinear correspondence rendering. This is a component, not glass synthesis.
Coordinates are normalized absolute background coordinates, x then y, align_corners=True.
Default refuses out-of-image requested coordinates. Caller may explicitly select mask or
border, with a manifest recording exclusions/clamping. No attenuation/reflections inferred.
"""
import numpy as np


def render_correspondence(background, correspondence, valid, *, out_of_image='reject'):
    if not isinstance(background, np.ndarray) or background.dtype != np.uint8 or background.ndim != 3 or background.shape[2] != 3:
        raise ValueError('uint8 RGB background required')
    if not isinstance(correspondence, np.ndarray) or correspondence.dtype.kind != 'f' or correspondence.ndim != 3 or correspondence.shape[2] != 2 or not np.isfinite(correspondence).all():
        raise ValueError('finite floating HxWx2 normalized correspondence required')
    h, w = correspondence.shape[:2]
    bh, bw = background.shape[:2]
    if min(h, w, bh, bw) < 2 or max(h*w, bh*bw) > 4194304:
        raise ValueError('dimensions >=2 and <=4MP required')
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.shape != (h, w) or not valid.any():
        raise ValueError('nonempty matching boolean valid mask required')
    if out_of_image not in ('reject', 'mask', 'border'):
        raise ValueError('explicit reject, mask or border policy required')
    inside = ((correspondence >= 0) & (correspondence <= 1)).all(2)
    outside = valid & ~inside
    n = int(outside.sum())
    if n and out_of_image == 'reject':
        raise ValueError(f'{n} requested correspondences outside background; choose an explicit policy')
    keep = valid & inside if out_of_image == 'mask' else valid.copy()
    if not keep.any():
        raise ValueError('no renderable pixels after policy')
    c = np.clip(correspondence, 0, 1)
    x = c[:,:,0].astype(np.float64) * (bw-1)
    y = c[:,:,1].astype(np.float64) * (bh-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, bw-1), np.minimum(y0+1, bh-1)
    fx, fy = (x-x0)[:,:,None], (y-y0)[:,:,None]
    sample = (background[y0,x0]*(1-fx)*(1-fy) + background[y0,x1]*fx*(1-fy) + background[y1,x0]*(1-fx)*fy + background[y1,x1]*fx*fy)
    rgb = np.zeros((h,w,3), np.uint8)
    rgb[keep] = np.rint(sample[keep]).clip(0,255).astype(np.uint8)
    alpha = keep.astype(np.uint8)*255
    return rgb, alpha, dict(method='bilinear normalized absolute correspondence, align_corners=True',out_of_image_policy=out_of_image,requested_pixels=int(valid.sum()),out_of_image_requested_pixels=n,masked_out_of_image_pixels=n if out_of_image=='mask' else 0,border_clamped_pixels=n if out_of_image=='border' else 0,rendered_pixels=int(keep.sum()),is_complete_glass_render=False,attenuation_recovered=False,reflections_recovered=False,publication_accepted=False)
