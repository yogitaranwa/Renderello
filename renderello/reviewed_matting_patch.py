"""Opt-in source-bound transfer of a solver's reviewed trimap-unknown patch.

Scope/protection are human hypotheses. No inference, geometry or quality claim.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .matting import check_trimap


def apply_reviewed_matting_patch(rgb,alpha,estimated_rgb,estimated_alpha,trimap,region,protected,*,
        source_sha256,alpha_sha256,estimated_rgb_sha256,estimated_alpha_sha256,
        trimap_sha256,region_sha256,protected_sha256,reviewed=False,material=None,
        review_note='',color_mode=None,feather=8,max_rgb_delta=24,max_alpha_delta=255,
        support_policy=None):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2]);native_rgb(estimated_rgb);native_alpha(estimated_alpha,alpha.shape)
    if estimated_rgb.shape!=rgb.shape:raise ValueError('native candidate shape required; no resize')
    if alpha.size>1_000_000:raise ValueError('1MP allocation budget')
    check_trimap(trimap,alpha.shape)
    if reviewed is not True or material not in ('opaque','opaque_fibres'):raise ValueError('explicit reviewed opaque or opaque-fibres region required')
    if not isinstance(review_note,str)or not review_note.strip()or len(review_note)>2000:raise ValueError('bounded nonempty review note required')
    for mask in (region,protected):
        if not isinstance(mask,np.ndarray)or mask.dtype!=np.bool_ or mask.shape!=alpha.shape or not mask.any():raise ValueError('nonempty native boolean region/protection required')
    if np.any(region&protected):raise ValueError('region and protection must be disjoint')
    if not np.any(protected&(alpha>0)):raise ValueError('protected source product required')
    for a,h in zip([rgb,alpha,estimated_rgb,estimated_alpha,trimap,region,protected],[source_sha256,alpha_sha256,estimated_rgb_sha256,estimated_alpha_sha256,trimap_sha256,region_sha256,protected_sha256]):
        if digest(a)!=h:raise ValueError('source/solver/review binding mismatch')
    if color_mode not in ('source','bounded-estimate'):raise ValueError('explicit source or bounded-estimate colour mode required')
    if support_policy not in ('retain','reviewed-change'):raise ValueError('explicit retain or reviewed-change support policy required')
    for v,lo,hi in [(feather,1,16),(max_rgb_delta,1,64),(max_alpha_delta,1,255)]:
        if type(v)is not int or not lo<=v<=hi:raise ValueError('bounded integer feather/edit budgets required')
    target=region&(trimap==128)
    if not target.any():raise ValueError('reviewed region has no trimap-unknown pixels')
    weight=np.minimum(1.,distance_transform_edt(np.pad(region,1))[1:-1,1:-1]/feather)*target
    aa=np.rint(alpha*(1-weight)+estimated_alpha*weight).astype(np.uint8)
    if support_policy=='retain':
        aa[alpha==0]=0;aa[alpha==255]=255
        fractional=(alpha>0)&(alpha<255);aa[fractional]=np.clip(aa[fractional],1,254)
    if int(np.abs(aa.astype(np.int16)-alpha).max())>max_alpha_delta:raise ValueError('requested alpha change exceeds budget; no partial silent alpha clamp')
    out=rgb.copy();color_target=target&(aa>0)&(aa<255)&(alpha<255)
    if color_mode=='bounded-estimate':
        delta=(estimated_rgb.astype(float)-rgb)*weight[...,None];peak=np.abs(delta).max(2);delta*=np.minimum(1.,max_rgb_delta/np.maximum(peak,1.))[...,None]
        candidate=np.rint(np.clip(rgb+delta,0,255)).astype(np.uint8);out[color_target]=candidate[color_target]
    changed_rgb=np.any(out!=rgb,2);changed_alpha=aa!=alpha
    if np.any(changed_rgb&~target)or np.any(changed_alpha&~target)or not np.array_equal(out[protected],rgb[protected])or not np.array_equal(aa[protected],alpha[protected]):raise RuntimeError('patch protection invariant failed')
    if np.any(changed_rgb&(alpha==255)):raise RuntimeError('original opaque source RGB changed')
    support_added=(alpha==0)&(aa>0);support_removed=(alpha>0)&(aa==0)
    if support_policy=='retain'and(support_added.any()or support_removed.any()):raise RuntimeError('retained support changed')
    return out,aa,dict(schema='reviewed-matting-patch.v1',source_sha256=digest(rgb),initial_alpha_sha256=digest(alpha),estimated_rgb_sha256=digest(estimated_rgb),estimated_alpha_sha256=digest(estimated_alpha),trimap_sha256=digest(trimap),region_sha256=digest(region),protected_sha256=digest(protected),output_rgb_sha256=digest(out),output_alpha_sha256=digest(aa),review_note=review_note,material=material,color_mode=color_mode,feather=feather,max_rgb_delta=max_rgb_delta,max_alpha_delta=max_alpha_delta,support_policy=support_policy,target_pixels=int(target.sum()),changed_rgb_pixels=int(changed_rgb.sum()),changed_alpha_pixels=int(changed_alpha.sum()),support_added_pixels=int(support_added.sum()),support_removed_pixels=int(support_removed.sum()),observed_max_rgb_delta=int(np.abs(out.astype(int)-rgb).max()),observed_max_alpha_delta=int(np.abs(aa.astype(int)-alpha).max()),original_opaque_rgb_exact=True,protected_rgb_alpha_exact=True,outside_unknown_region_exact=True,no_inference=True,source_resampled=False,accepted=False,default_pipeline_changed=False,physical_recovery=False,solver_provenance='NOT_AUTHENTICATED_BY_HASH',limitations=['Region, protection and trimap are reviewer hypotheses, not semantic or optical truth.','SHA256 binds candidate bytes but does not authenticate the model or its inputs.','Estimated alpha can remove detail or include background when reviewed-change is selected.','Bounded colour changes can flatten fur texture or erase genuine shading.','Region and eligibility boundaries can leave seams; native light/dark inspection remains required.','No camera, illumination, physical contact or transparent-material recovery.'])
