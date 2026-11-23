"""Native selected-subset assembly. Not automatic enumeration or photo acceptance.

All components come from the same source frame. Max-alpha union preserves that
frame's positions and original RGB, including old lighting and contamination.
It does not infer occlusion, independent objects, contact, reflections or truth.
"""
import hashlib
import numpy as np


def assemble_subset(source, components, *, native_pairing_reviewed=False):
    if native_pairing_reviewed is not True:
        raise ValueError('explicit same-source native crop pairing review required')
    if not isinstance(source, np.ndarray) or source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('native uint8 RGB source required')
    h,w = source.shape[:2]
    if min(h,w)<2 or h*w>2_000_000:
        raise ValueError('native source dimensions 2..2MP required')
    if not isinstance(components,list) or not 1 <= len(components) <= 16:
        raise ValueError('one to sixteen explicitly selected components required')
    union=np.zeros((h,w),np.uint8);count=np.zeros((h,w),np.uint8);names=set();records=[]
    for c in components:
        if not isinstance(c,dict) or set(c)!={'name','alpha','crop','source_crop_rgb_sha256','selection_note'}:
            raise ValueError('explicit component fields required')
        name=c['name'];note=c['selection_note'];crop=c['crop'];alpha=c['alpha']
        if not isinstance(name,str) or not name.strip() or len(name)>128 or name in names:
            raise ValueError('unique nonempty component name required')
        if not isinstance(note,str) or not note.strip() or len(note)>2048:
            raise ValueError('source-review selection note required')
        if not isinstance(crop,(list,tuple)) or len(crop)!=4 or any(type(v) is not int for v in crop):
            raise ValueError('integer source crop bounds required')
        x0,y0,x1,y1=crop
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('crop outside source')
        if not isinstance(alpha,np.ndarray) or alpha.dtype!=np.uint8 or alpha.shape!=(y1-y0,x1-x0):
            raise ValueError('native uint8 crop alpha required; no resizing')
        if not np.any(alpha):
            raise ValueError('empty selected component refused')
        actual=hashlib.sha256(source[y0:y1,x0:x1].tobytes()).hexdigest()
        if c['source_crop_rgb_sha256']!=actual:
            raise ValueError('component source-crop RGB hash differs')
        names.add(name);v=union[y0:y1,x0:x1];np.maximum(v,alpha,out=v);count[y0:y1,x0:x1]+=(alpha>0).astype(np.uint8)
        records.append(dict(name=name,crop=list(crop),selection_note=note,source_crop_rgb_sha256=actual,
                            alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest(),support_pixels=int((alpha>0).sum())))
    rgba=np.dstack([source,union])
    manifest=dict(method='explicit same-source native max-alpha subset union',components=records,
                  overlap_support_pixels=int((count>1).sum()),union_support_pixels=int((union>0).sum()),
                  source_rgb_sha256=hashlib.sha256(source.tobytes()).hexdigest(),source_rgb_exact=True,
                  source_resampled=False,source_positions_unchanged=True,alpha_rule='maximum, not additive opacity',
                  semantic_object_count=None,semantic_completeness=None,independent_boundary_accuracy=None,
                  publication_accepted=False,production_default_changed=False,
                  limitations=['Selected subset only; omitted source items remain omitted.',
                               'Component names and pairing review are caller declarations, not verified identities.',
                               'Max union is for the same photographed frame, not independently layered new arrangements.',
                               'Source lighting, reflections and boundary contamination remain.'])
    return rgba,manifest
