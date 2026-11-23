"""Named native part preservation, not segmentation or independent label truth."""
import numpy as np
from .candidate_diagnostics import native_rgb,native_alpha,digest


def audit_parts(rgb,prior_alpha,candidate_rgb,candidate_alpha,regions,*,source_sha256,
                prior_alpha_sha256,reviewed=False,review_note=''):
    native_rgb(rgb);native_alpha(prior_alpha,rgb.shape[:2]);native_rgb(candidate_rgb)
    native_alpha(candidate_alpha,prior_alpha.shape)
    if candidate_rgb.shape!=rgb.shape:raise ValueError('same native RGB coordinates required')
    if prior_alpha.size>1_000_000:raise ValueError('1MP native budget; no resize')
    if reviewed is not True or not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('explicit region review and bounded note required')
    if source_sha256!=digest(rgb) or prior_alpha_sha256!=digest(prior_alpha):
        raise ValueError('source/prior binding mismatch')
    if not isinstance(regions,list) or not 1<=len(regions)<=32:raise ValueError('one to32 reviewed regions required')
    names=set();rows=[]
    for region in regions:
        if not isinstance(region,dict):raise ValueError('region records required')
        name=region.get('name');mask=region.get('mask')
        if not isinstance(name,str) or not name.strip() or len(name)>100 or name in names:raise ValueError('unique bounded region names required')
        names.add(name)
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=prior_alpha.shape or not mask.any():
            raise ValueError('nonempty native boolean region required')
        if region.get('mask_sha256')!=digest(mask):raise ValueError('region binding mismatch')
        if not np.any(mask&(prior_alpha>0)):raise ValueError('region must include prior foreground')
        before=prior_alpha[mask].astype(np.int16);after=candidate_alpha[mask].astype(np.int16)
        delta=after-before;rgb_changed=np.any(candidate_rgb[mask]!=rgb[mask],axis=1)
        rows.append(dict(name=name,mask_sha256=digest(mask),pixels=int(mask.sum()),
            rgb_changed_pixels=int(rgb_changed.sum()),alpha_changed_pixels=int(np.count_nonzero(delta)),
            alpha_decreased_pixels=int((delta<0).sum()),alpha_increased_pixels=int((delta>0).sum()),
            alpha_loss_units=int(np.maximum(-delta,0).sum()),alpha_gain_units=int(np.maximum(delta,0).sum()),
            prior_covered_pixels_lost=int(((before>0)&(after==0)).sum()),
            exact_prior_preservation=not bool(rgb_changed.any() or delta.any())))
    return dict(schema='reviewed-native-parts.v1',source_sha256=digest(rgb),prior_alpha_sha256=digest(prior_alpha),
        candidate_rgb_sha256=digest(candidate_rgb),candidate_alpha_sha256=digest(candidate_alpha),
        review_note=review_note,regions=rows,all_named_parts_preserved=all(x['exact_prior_preservation'] for x in rows),
        accepted=False,independent_ground_truth=False,semantic_completeness='NOT_VERIFIED',
        limitations=['Exact preservation of reviewed prior pixels only. Prior segmentation may be wrong.',
            'Named parts are reviewer declarations, not model-detected semantics.',
            'Unreviewed parts, visible scene appearance and label transcription are not verified.',
            'A changed alpha can be an improvement; a failed preservation check requires review, not automatic repair.'])
