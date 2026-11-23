"""Opt-in near-opaque interior repair, never automatic alpha saturation.

Only an explicit reviewed region on opaque material may approach alpha255. A
bounded existing opacity deficit and interior clearance are refusal safeguards,
not material or segmentation evidence. No default pipeline integration.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest


def restore_reviewed_opaque_core(rgb,alpha,region,*,source_sha256,alpha_sha256,
        region_sha256,reviewed=False,material=None,review_note='',max_deficit=8,
        clearance=8,feather=4):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP native budget; no resize')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque-core source review required')
    if not isinstance(review_note,str)or not review_note.strip()or len(review_note)>2000:raise ValueError('bounded nonempty review note required')
    if not isinstance(region,np.ndarray)or region.dtype!=np.bool_ or region.shape!=alpha.shape or not region.any():raise ValueError('nonempty native boolean reviewed region required')
    for a,h in ((rgb,source_sha256),(alpha,alpha_sha256),(region,region_sha256)):
        if digest(a)!=h:raise ValueError('source/alpha/region binding mismatch')
    if type(max_deficit)is not int or not 1<=max_deficit<=16:raise ValueError('integer opacity deficit1..16 required')
    if type(clearance)is not int or not 4<=clearance<=64:raise ValueError('integer clearance4..64 required')
    if type(feather)is not int or not 2<=feather<=16:raise ValueError('integer feather2..16 required')
    if np.any(alpha[region]<255-max_deficit):raise ValueError('reviewed region exceeds near-opaque deficit budget')
    hard=alpha>=128
    if hard.all()or not hard.any():raise ValueError('nontrivial original foreground/background required')
    # Padding prevents an image-frame crop from falsely looking interior.
    interior=distance_transform_edt(np.pad(hard,1))[1:-1,1:-1]
    if np.any(interior[region]<clearance):raise ValueError('reviewed core lacks boundary/frame clearance')
    ramp=np.minimum(1.,distance_transform_edt(np.pad(region,1))[1:-1,1:-1]/feather)
    core=region&(ramp>=1.)
    if not core.any():raise ValueError('reviewed region has no full-strength interior after feathering')
    out=alpha.copy();gain=np.rint((255-alpha[region].astype(np.float64))*ramp[region]).astype(np.uint8)
    out[region]=alpha[region]+gain
    changed=out!=alpha
    if np.any(changed&~region)or np.any(out<alpha)or not np.all(out[core]==255):raise RuntimeError('opaque-core scope invariant failed')
    return rgb.copy(),out,dict(schema='reviewed-opaque-core.v1',source_sha256=digest(rgb),alpha_sha256=digest(alpha),region_sha256=digest(region),output_alpha_sha256=digest(out),review_note=review_note,material=material,max_deficit=max_deficit,clearance=clearance,feather=feather,minimum_clearance=float(interior[region].min()),reviewed_pixels=int(region.sum()),full_strength_pixels=int(core.sum()),new_opaque_pixels=int(((alpha<255)&(out==255)).sum()),alpha_changed_pixels=int(changed.sum()),maximum_alpha_gain=int(gain.max()),all_source_rgb_exact=True,outside_region_alpha_exact=True,source_resampled=False,accepted=False,default_pipeline_changed=False,independent_ground_truth=False,limitations=['Opaque material and region membership are reviewer hypotheses, not inferred facts.','Near-opaque alpha does not prove opacity; wrong review can erase real transmission.','Feathering can leave a coverage seam. Inspect native light/dark composites.','Boundary alpha, holes, source illumination, material reflectance and scene contact are unchanged.'])
