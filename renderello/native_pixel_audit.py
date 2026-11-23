"""Decoded, unresampled opaque-RGB check at explicit source/output coordinates.

This is neither an edge/mask evaluation nor a photographic quality score.
A size mismatch returns unavailable, never an interpolated pixel-exact pass.
"""
import numpy as np


def _box(value, width, height):
    if not isinstance(value,(tuple,list)) or len(value)!=4 or any(type(x) is not int for x in value):
        raise ValueError('Box requires four integer coordinates, not booleans')
    x,y,w,h=value
    if min(x,y)<0 or min(w,h)<1 or x+w>width or y+h>height:
        raise ValueError('Box outside decoded image')
    return x,y,w,h


def audit_native_pixels(source, alpha, output, source_box, output_box):
    """Compare uint8 RGB arrays only where original alpha equals exactly 255."""
    for name,array,channels in [('source',source,3),('output',output,3)]:
        if not isinstance(array,np.ndarray) or array.dtype!=np.uint8 or array.ndim!=3 or array.shape[2]!=channels or min(array.shape[:2])<1:
            raise ValueError(name+' must be a nonempty uint8 RGB array')
    if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=source.shape[:2]:
        raise ValueError('Alpha must be source-sized uint8 grayscale')
    sx,sy,sw,sh=_box(source_box,source.shape[1],source.shape[0])
    ox,oy,ow,oh=_box(output_box,output.shape[1],output.shape[0])
    mask=alpha[sy:sy+sh,sx:sx+sw]==255
    count=int(mask.sum())
    result={'scope':'FULLY_OPAQUE_NATIVE_RGB_ONLY','source_box':[sx,sy,sw,sh],
            'output_box':[ox,oy,ow,oh],'fully_opaque_source_pixels':count,
            'changed_pixels':None,'max_channel_error':None,'native_rgb_exact':None,
            'mask_semantics':'NOT_EVALUATED','edges':'NOT_EVALUATED',
            'lighting_and_grounding':'NOT_EVALUATED','quality_acceptance':'NOT_EVALUATED'}
    if (sw,sh)!=(ow,oh):
        result['status']='NATIVE_SIZE_CHANGED';return result
    if count==0:
        result['status']='NO_FULLY_OPAQUE_PIXELS';return result
    original=source[sy:sy+sh,sx:sx+sw][mask].astype(np.int16)
    placed=output[oy:oy+oh,ox:ox+ow][mask].astype(np.int16)
    difference=np.abs(original-placed)
    changed=int(np.any(difference!=0,axis=1).sum())
    result.update(status='MATCH' if changed==0 else 'MISMATCH',changed_pixels=changed,
                  max_channel_error=int(difference.max()),native_rgb_exact=changed==0)
    return result
