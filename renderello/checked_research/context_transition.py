"""Experimental background-only fade around an exact reviewed context mask.

No alpha solve, relighting, new physical contact, glass optics or model call.
The preserved context includes old-scene RGB around all nonzero product alpha.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from ..candidate_diagnostics import native_rgb,native_alpha,digest
from ..source_context_extension import extend_source_context


def transition_context(rgb,alpha,context,*,source_sha256,alpha_sha256,context_sha256,
                       reviewed=False,note='',material='unknown',canvas=(1024,1024),offset=(0,0),feather=20):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque context review required')
    if not isinstance(note,str)or not note.strip()or len(note)>2000:raise ValueError('bounded review note required')
    if not isinstance(context,np.ndarray)or context.dtype!=np.bool_ or context.shape!=alpha.shape or not context.any():raise ValueError('native boolean context mask required')
    if digest(rgb)!=source_sha256 or digest(alpha)!=alpha_sha256 or digest(context)!=context_sha256:raise ValueError('source/alpha/context binding mismatch')
    if np.any((alpha>0)&~context):raise ValueError('context must retain every nonzero source-alpha pixel')
    if type(feather)is not int or not 2<=feather<=64:raise ValueError('bounded2..64pixel feather required')
    d=distance_transform_edt(~context);weight=np.clip(1-d/feather,0,1);weight=weight*weight*(3-2*weight)
    if any(np.any(v>0)for v in(weight[0],weight[-1],weight[:,0],weight[:,-1])):
        raise ValueError('protected context+feather reaches source frame; no hidden rectangular seam')
    extension,record=extend_source_context(rgb,alpha,material=material,reviewed_source_sha256=source_sha256,
        reviewed_alpha_sha256=alpha_sha256,canvas=canvas,offset=offset,smooth_outer=True)
    h,w=alpha.shape;cw,ch=canvas;dx,dy=offset;coef=np.array(record['fit_coefficients']);yy,xx=np.mgrid[:ch,:cw]
    plane=np.clip(coef[0]+np.clip((xx-dx)/max(1,w-1),0,1)[...,None]*coef[1]+np.clip((yy-dy)/max(1,h-1),0,1)[...,None]*coef[2],0,255)
    output=np.rint(plane).astype(np.uint8);local=output[dy:dy+h,dx:dx+w]
    local[:]=np.rint(rgb*weight[...,None]+plane[dy:dy+h,dx:dx+w]*(1-weight[...,None])).astype(np.uint8)
    local[context]=rgb[context]
    if not np.array_equal(local[context],rgb[context]):raise RuntimeError('protected context changed')
    change=np.any(local!=rgb,axis=2)
    record.update(schema='reviewed-context-transition.v1',context_sha256=context_sha256,review_note=note,
        context_is_caller_hypothesis=True,source_pixels_preserved=int(context.sum()),source_changed_pixels=int(change.sum()),
        background_changed_pixels=int(change.sum()),protected_context_changed_pixels=0,source_alpha_support_changed_pixels=0,
        whole_source_frame_locked=False,source_product_and_context_rgb_locked=True,feather=feather,
        source_contact_retained='ONLY_IF_CALLER_CONTEXT_INCLUDES_IT_NOT_VERIFIED',background_replaced='FAR_BACKGROUND_ONLY',
        photo_accepted=False,new_template_executed=False)
    record['limitations']=[x for x in record['limitations'] if x!='The complete old scene remains. This is added copy space, not a product staging template.']
    record['limitations'].extend(['Old-scene pixels remain around the whole protected product; a halo is still possible.',
        'Far background RGB changes within the original photograph; entire-source RGB is not locked.'])
    return output,record
