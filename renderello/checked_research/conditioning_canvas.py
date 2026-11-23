"""Experimental shared reference/latent-anchor coordinates. No image generation.

This does not recover a mask, remove the old background, relight the source or
prove the model will respect placement. Native RGB is copied only into the INPUT
canvas, never onto a generated output. Use the returned canvas for both model
reference and latent encoding; verify model output separately.
"""
from __future__ import annotations
import numpy as np
from ..candidate_diagnostics import native_rgb, native_alpha, digest


def conditioning_canvas(source, mask, *, size=(896, 896), offset=None,
                        padding=(127, 127, 127)):
    """Return native source placement, same-coordinate anchor mask, provenance.

    size is (width,height). The mask is caller-supplied experimental protection,
    not evidence of object completeness or physical alpha/transparency. No crop,
    resize, erosion, thresholding or hidden automatic region selection occurs.
    """
    native_rgb(source)
    native_alpha(mask, source.shape[:2])
    if not isinstance(size, (tuple, list)) or len(size) != 2 or any(type(v) is not int for v in size):
        raise ValueError('two integer canvas dimensions required')
    w, h = size
    if min(w, h) < 32 or w % 32 or h % 32 or w*h > 4_000_000:
        raise ValueError('canvas must be positive multiples of32 within4MP')
    sh, sw = source.shape[:2]
    if sw > w or sh > h:
        raise ValueError('native source does not fit; no silent resize/crop')
    if offset is None:
        offset = ((w-sw)//2, (h-sh)//2)
    if not isinstance(offset, (tuple, list)) or len(offset) != 2 or any(type(v) is not int for v in offset):
        raise ValueError('two integer placement coordinates required')
    x, y = offset
    if min(x,y) < 0 or x+sw > w or y+sh > h:
        raise ValueError('source rectangle outside canvas')
    if not isinstance(padding, (tuple, list)) or len(padding) != 3 or any(type(v) is not int or not 0<=v<=255 for v in padding):
        raise ValueError('three integer padding channels0..255 required')
    if not mask.any():
        raise ValueError('empty protection mask')
    canvas = np.empty((h,w,3), np.uint8)
    canvas[:] = padding
    canvas[y:y+sh,x:x+sw] = source
    placed_mask = np.zeros((h,w),np.uint8)
    placed_mask[y:y+sh,x:x+sw] = mask
    yy,xx = np.where(mask>0)
    support_box = [x+int(xx.min()), y+int(yy.min()), x+int(xx.max())+1, y+int(yy.max())+1]
    metadata = dict(method='NATIVE_SHARED_INPUT_CONDITIONING_CANVAS',
        source_shape=list(source.shape), canvas_shape=list(canvas.shape),
        source_rectangle=[x,y,x+sw,y+sh], protection_support_box=support_box,
        protection_height_fraction=(support_box[3]-support_box[1])/h,
        padding=list(padding), source_pixels_sha256=digest(source),
        input_mask_pixels_sha256=digest(mask), canvas_pixels_sha256=digest(canvas),
        placed_mask_pixels_sha256=digest(placed_mask), source_resampled=False,
        source_cropped=False, old_background_removed=False,
        input_rgb_native_exact=True, output_rgb_paste_performed=False,
        geometry_preservation='NOT_EVALUATED', semantic_mask_accuracy='NOT_EVALUATED',
        physical_relighting=False, quality_accepted=False,
        warnings=['Use exactly this canvas for model reference and latent encoding.',
                  'Source rectangle includes its original background.',
                  'Protection mask can preserve wrong parts or miss details.',
                  'Matching input coordinates does not prove generated geometry or text.'])
    return canvas, placed_mask, metadata
