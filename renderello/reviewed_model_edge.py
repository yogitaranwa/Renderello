"""Opt-in bounded use of an external foreground estimate on reviewed edge strips.

The external estimate is not a source-RGB-locked result. Only the explicit target
can change; original alpha, protected RGB and all other RGB remain exact. No model
is called. This is a colour hypothesis, not recovered foreground or relighting.
"""
import math
from numbers import Real
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest


def import_reviewed_model_edge(rgb,alpha,estimate,target,protected,*,
        source_sha256,alpha_sha256,estimate_sha256,target_sha256,protected_sha256,
        reviewed=False,material=None,review_note='',edge_width=4.,strength=.75,
        max_code_delta=24,feather=3):
    native_rgb(rgb);native_rgb(estimate);native_alpha(alpha,rgb.shape[:2])
    if estimate.shape!=rgb.shape:raise ValueError('native estimate shape must match; no resize')
    if alpha.size>1_000_000:raise ValueError('1MP native allocation budget')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque source/edge review required')
    if not isinstance(review_note,str)or not review_note.strip()or len(review_note)>2000:raise ValueError('bounded nonempty review note required')
    for mask in (target,protected):
        if not isinstance(mask,np.ndarray)or mask.dtype!=np.bool_ or mask.shape!=alpha.shape or not mask.any():raise ValueError('nonempty native boolean target/protected masks required')
    for value,expected in ((rgb,source_sha256),(alpha,alpha_sha256),(estimate,estimate_sha256),(target,target_sha256),(protected,protected_sha256)):
        if digest(value)!=expected:raise ValueError('source/alpha/estimate/review binding mismatch')
    if np.any(target&protected):raise ValueError('target overlaps protected source')
    if not np.any(protected&(alpha>0)):raise ValueError('protected foreground required')
    if np.any(alpha[target]==0):raise ValueError('no zero-alpha background RGB edits')
    for value,lo,hi in ((edge_width,0,8),(strength,0,1)):
        if isinstance(value,(bool,np.bool_))or not isinstance(value,Real)or not math.isfinite(value)or not lo<value<=hi:raise ValueError('finite bounded controls required')
    if type(max_code_delta)is not int or not 1<=max_code_delta<=32:raise ValueError('integer code delta1..32 required')
    if type(feather)is not int or not 1<=feather<=8:raise ValueError('integer feather1..8 required')
    hard=alpha>=128
    if not hard.any()or hard.all():raise ValueError('nontrivial foreground/background support required')
    if np.any(distance_transform_edt(hard)[target]>edge_width):raise ValueError('target outside declared edge band')
    # Preserve the estimate's signed RGB correction direction. Bound its largest
    # channel before feathering; do not clip channels independently or silently
    # adopt its alpha. Zero-pad to treat image-frame target borders consistently.
    delta=estimate[target].astype(np.float64)-rgb[target].astype(np.float64)
    peak=np.abs(delta).max(axis=1)
    gain=np.minimum(1.,max_code_delta/np.maximum(peak,1.))*float(strength)
    taper=np.minimum(1.,distance_transform_edt(np.pad(target,1))[1:-1,1:-1][target]/feather)
    correction=delta*(gain*taper)[:,None]
    out=rgb.copy();out[target]=np.clip(np.rint(rgb[target]+correction),0,255).astype(np.uint8)
    observed=np.abs(out.astype(np.int16)-rgb.astype(np.int16))
    changed=np.any(observed,axis=2)
    if observed.max()>max_code_delta or np.any(changed&~target)or not np.array_equal(out[protected],rgb[protected]):raise RuntimeError('reviewed model edge invariant failed')
    return out,dict(schema='reviewed-model-edge.v1',source_sha256=digest(rgb),alpha_sha256=digest(alpha),estimate_sha256=digest(estimate),target_sha256=digest(target),protected_sha256=digest(protected),output_sha256=digest(out),review_note=review_note,material=material,edge_width=float(edge_width),strength=float(strength),max_code_delta=max_code_delta,feather=feather,target_pixels=int(target.sum()),changed_rgb_pixels=int(changed.sum()),changed_opaque_rgb_pixels=int((changed&(alpha==255)).sum()),observed_max_code_delta=int(observed.max()),bounded_estimate_pixels=int((peak>max_code_delta).sum()),protected_rgb_exact=True,outside_target_rgb_exact=True,alpha_changed=False,source_resampled=False,model_called=False,accepted=False,default_pipeline_changed=False,physical_recovery=False,independent_ground_truth=False,limitations=['External estimate and target review are hypotheses, not foreground truth.','Opaque edge RGB may change; this is not a source-RGB-locked result.','All alpha is retained, including wrong coverage and missing details.','Feathering and code bounds do not prove a seamless or better edge.','Source light, label semantics, material, geometry and contact are not recovered.'])
