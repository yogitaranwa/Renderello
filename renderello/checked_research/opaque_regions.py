"""Explicit identity-core opacity override for research, not model-alpha truth.
Only previously reviewed opaque source rectangles may be restored. This can
create seams, must be inspected, and never turns a candidate into accepted work.
"""
import numpy as np
from .segmentation_candidate import import_rgb_exact_candidate

def restore_reviewed_opaque_regions(source,candidate,reference_alpha,regions,*,pairing_reviewed=False,internal_resampling_declared=False,opaque_regions_reviewed=False):
    out,base=import_rgb_exact_candidate(source,candidate,pairing_reviewed=pairing_reviewed,internal_resampling_declared=internal_resampling_declared)
    if opaque_regions_reviewed is not True:raise ValueError('explicit opaque source-region review required')
    h,w=source.shape[:2]
    if not isinstance(reference_alpha,np.ndarray) or reference_alpha.dtype!=np.uint8 or reference_alpha.shape!=(h,w):raise ValueError('matching reviewed uint8 reference alpha required')
    if not isinstance(regions,dict) or not regions:raise ValueError('named reviewed region rectangles required')
    union=np.zeros((h,w),bool)
    for name,box in regions.items():
        if not isinstance(name,str) or not name.strip():raise ValueError('nonempty region name required')
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('four integer source coordinates required')
        x0,y0,x1,y1=box
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('source region outside bounds')
        if np.any(reference_alpha[y0:y1,x0:x1]!=255):raise ValueError('reference region is not fully opaque')
        if np.any(candidate[y0:y1,x0:x1,3]<240):raise ValueError('model alpha contradicts opaque-region declaration below240; review rather than overwrite')
        union[y0:y1,x0:x1]=True
    changed=union&(out[:,:,3]!=255);out[union,3]=255
    return out,dict(**base,alpha_override='explicit previously reviewed opaque source rectangles only',alpha_override_pixels=int(changed.sum()),unique_region_pixels=int(union.sum()),region_names=list(regions),model_alpha_truth=False,visible_region_exact_only_after_verified_composition=True)
