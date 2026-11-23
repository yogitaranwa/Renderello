"""Opt-in reviewed local two-colour edge fit, not optical or alpha truth.

Explicit foreground/background donors and mixing space are hypotheses. Only a
bounded reviewed fractional edge can change; no support or protected RGB changes.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest


def fit_reviewed_local_edge(rgb,alpha,target,foreground_donor,background_donor,protected,*,
        source_sha256,alpha_sha256,target_sha256,foreground_sha256,background_sha256,protected_sha256,
        reviewed=False,material=None,review_note='',mixing_space=None,strength=.75,feather=3,
        edge_width=4,max_distance=16,max_alpha_delta=96,max_rgb_delta=64,max_residual=12,minimum_contrast=48):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP budget')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque source and donor review required')
    if not isinstance(review_note,str)or not review_note.strip()or len(review_note)>2000:raise ValueError('bounded nonempty review note required')
    masks=[target,foreground_donor,background_donor,protected]
    for m in masks:
        if not isinstance(m,np.ndarray)or m.dtype!=np.bool_ or m.shape!=alpha.shape or not m.any():raise ValueError('nonempty native boolean review masks required')
    for i,m in enumerate(masks):
        if any(np.any(m&n)for n in masks[i+1:]):raise ValueError('review masks must be disjoint')
    for v,h in zip([rgb,alpha]+masks,[source_sha256,alpha_sha256,target_sha256,foreground_sha256,background_sha256,protected_sha256]):
        if digest(v)!=h:raise ValueError('source/alpha/review binding mismatch')
    if np.any((alpha[target]==0)|(alpha[target]==255)):raise ValueError('fractional target only')
    if np.any(alpha[foreground_donor]<250)or np.any(alpha[background_donor]!=0):raise ValueError('foreground donor alpha>=250 and background alpha0 required')
    if not np.any(alpha[protected]>0):raise ValueError('protected product required')
    if mixing_space not in ('srgb','linear'):raise ValueError('explicit mixing-space hypothesis required')
    if isinstance(strength,(bool,np.bool_))or not isinstance(strength,(int,float))or not math.isfinite(strength)or not 0<strength<=1:raise ValueError('strength in(0,1]')
    for v,lo,hi in [(feather,1,8),(edge_width,1,8),(max_distance,1,32),(max_alpha_delta,1,128),(max_rgb_delta,1,96),(max_residual,1,24),(minimum_contrast,16,128)]:
        if type(v)is not int or not lo<=v<=hi:raise ValueError('bounded integer controls required')
    hard=alpha>=128
    if hard.all()or not hard.any()or np.any(distance_transform_edt(hard)[target]>edge_width):raise ValueError('nontrivial support and narrow reviewed edge band required')
    df,iff=distance_transform_edt(~foreground_donor,return_indices=True);db,ib=distance_transform_edt(~background_donor,return_indices=True)
    if np.any(df[target]>max_distance)or np.any(db[target]>max_distance):raise ValueError('every target needs nearby reviewed donors')
    def decode(x):
        x=x.astype(float)/255
        return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)if mixing_space=='linear'else x
    def encode(x):
        x=np.clip(x,0,1)
        return 255*(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)if mixing_space=='linear'else x)
    c=decode(rgb[target]);f=decode(rgb[iff[0][target],iff[1][target]]);b=decode(rgb[ib[0][target],ib[1][target]])
    direction=f-b;denom=np.sum(direction**2,axis=1)
    proposal=np.clip(np.sum((c-b)*direction,axis=1)/np.maximum(denom,1e-12),0,1)
    residual=np.max(np.abs(encode(b+proposal[:,None]*direction)-rgb[target]),axis=1)
    contrast=np.max(np.abs(encode(f)-encode(b)),axis=1)
    foreground_delta=np.max(np.abs(encode(f)-rgb[target]),axis=1)
    alpha_delta=np.abs(proposal*255-alpha[target])
    eligible=(residual<=max_residual)&(contrast>=minimum_contrast)&(foreground_delta<=max_rgb_delta)&(alpha_delta<=max_alpha_delta)
    eligible_map=np.zeros(alpha.shape,bool);eligible_map[target]=eligible
    taper=np.minimum(1.,distance_transform_edt(np.pad(eligible_map,1))[1:-1,1:-1][target]/feather);weight=strength*taper
    out=rgb.copy();aa=alpha.copy();newrgb=np.rint(rgb[target]*(1-weight[:,None])+encode(f)*weight[:,None]).astype(np.uint8);newalpha=np.clip(np.rint(alpha[target]*(1-weight)+proposal*255*weight),1,254).astype(np.uint8)
    values=out[target];values[eligible]=newrgb[eligible];out[target]=values;av=aa[target];av[eligible]=newalpha[eligible];aa[target]=av
    if not np.array_equal(out[~target],rgb[~target])or not np.array_equal(aa[~target],alpha[~target]):raise RuntimeError('outside-target invariant failed')
    report=dict(schema='reviewed-local-edge-fit.v1',source_sha256=digest(rgb),initial_alpha_sha256=digest(alpha),target_sha256=digest(target),foreground_sha256=digest(foreground_donor),background_sha256=digest(background_donor),protected_sha256=digest(protected),output_rgb_sha256=digest(out),output_alpha_sha256=digest(aa),mixing_space=mixing_space,review_note=review_note,strength=strength,feather=feather,edge_width=edge_width,max_distance=max_distance,max_alpha_delta=max_alpha_delta,max_rgb_delta=max_rgb_delta,max_residual=max_residual,minimum_contrast=minimum_contrast,target_pixels=int(target.sum()),eligible_pixels=int(eligible.sum()),rejected_residual=int((residual>max_residual).sum()),rejected_contrast=int((contrast<minimum_contrast).sum()),rejected_rgb_budget=int((foreground_delta>max_rgb_delta).sum()),rejected_alpha_budget=int((alpha_delta>max_alpha_delta).sum()),rgb_changed_pixels=int(np.any(out!=rgb,2).sum()),alpha_changed_pixels=int((aa!=alpha).sum()),observed_max_rgb_delta=int(np.abs(out.astype(int)-rgb).max()),observed_max_alpha_delta=int(np.abs(aa.astype(int)-alpha).max()),support_exact=True,protected_rgb_alpha_exact=True,source_resampled=False,accepted=False,default_pipeline_changed=False,physical_recovery=False,limitations=['Donor correspondence, opacity and mixing space are reviewer hypotheses, not observations under the edge.','RGB and alpha can change at the reviewed boundary, flattening real shading or changing apparent shape.','Fit residual measures agreement with the assumed local model, not optical or photographic accuracy.','Eligibility rejection counts can overlap. Ineligible pixels remain unchanged and may cause transitions.','Lighting, pose, label semantics, contact and glass optics are not recovered.'])
    return out,aa,report
