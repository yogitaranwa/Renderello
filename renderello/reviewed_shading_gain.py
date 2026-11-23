"""Source-bound smooth shading-gain hypothesis, never intrinsic/physical truth.
Only explicitly reviewed opaque interior pixels change. No albedo replacement.
"""
from numbers import Real
import math
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .checked_research.detail_transfer import decode,encode


def apply_reviewed_shading_gain(rgb,alpha,shading,target,protected,*,source_sha256,
        alpha_sha256,shading_sha256,target_sha256,protected_sha256,review_note,
        reviewed=False,material=None,sigma=8.,feather=8.,strength=.5,
        minimum_gain=.8,max_code_delta=24):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP native allocation cap; no resize')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque interior/source alignment review required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:raise ValueError('bounded review note required')
    if not isinstance(shading,np.ndarray) or shading.dtype.kind!='f' or shading.shape!=alpha.shape or not np.isfinite(shading).all() or np.any(shading<0):raise ValueError('native finite nonnegative floating scalar shading required')
    for m in (target,protected):
        if not isinstance(m,np.ndarray) or m.dtype!=np.bool_ or m.shape!=alpha.shape or not m.any():raise ValueError('nonempty native boolean reviewed masks required')
    if np.any(target&protected) or np.any(alpha[target]!=255):raise ValueError('target must be opaque and disjoint from protected regions')
    if np.any(target[0]) or np.any(target[-1]) or np.any(target[:,0]) or np.any(target[:,-1]):raise ValueError('interior target cannot touch frame')
    for a,h in [(rgb,source_sha256),(alpha,alpha_sha256),(shading,shading_sha256),(target,target_sha256),(protected,protected_sha256)]:
        if digest(a)!=h:raise ValueError('source/shading/mask binding mismatch')
    for v,lo,hi in [(sigma,1,32),(feather,1,32),(strength,0,1)]:
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not math.isfinite(v) or not lo<v<=hi:raise ValueError('finite bounded smooth controls required')
    if isinstance(minimum_gain,(bool,np.bool_)) or not isinstance(minimum_gain,Real) or not math.isfinite(minimum_gain) or not .5<=minimum_gain<1:raise ValueError('minimum gain0.5..below1 required')
    if type(max_code_delta)is not int or not 1<=max_code_delta<=32:raise ValueError('integer delta1..32 required')
    # Normalized smoothing excludes other materials/labels from the reference.
    weights=target.astype(float);den=gaussian_filter(weights,sigma,mode='constant')
    smoothed=gaussian_filter(shading.astype(float)*weights,sigma,mode='constant')/np.maximum(den,1e-12)
    values=smoothed[target]
    if np.any(values<=1e-8):raise ValueError('positive reviewed shading required')
    reference=float(np.median(values));base=np.maximum(minimum_gain,np.minimum(1.,reference/values))
    distance=distance_transform_edt(target)[target]
    # First target row has gain1; smoothstep eases to the full bounded change.
    t=np.clip((distance-1)/feather,0,1);ease=t*t*(3-2*t);gain=1-strength*ease*(1-base)
    result=encode(decode(rgb[target])*gain[:,None]);delta=rgb[target].astype(np.int16)-result.astype(np.int16)
    if np.any(delta<0):raise RuntimeError('darken-only invariant failed')
    if np.any(delta>max_code_delta):raise ValueError('shading hypothesis exceeds channel delta budget')
    out=rgb.copy();out[target]=result;changed=np.any(out!=rgb,axis=2)
    if np.any(changed&~target) or not np.array_equal(out[protected],rgb[protected]):raise RuntimeError('protected source invariant failed')
    return out,dict(schema='reviewed-shading-gain.v1',source_sha256=digest(rgb),alpha_sha256=digest(alpha),shading_sha256=digest(shading),target_sha256=digest(target),protected_sha256=digest(protected),result_sha256=digest(out),review_note=review_note,sigma=float(sigma),feather=float(feather),strength=float(strength),minimum_gain=float(minimum_gain),max_code_delta=max_code_delta,changed_pixels=int(changed.sum()),observed_maximum_code_delta=int(delta.max()),reference_shading=reference,actual_minimum_gain=float(gain.min()),source_rgb_exact=not bool(changed.any()),alpha_changed=False,protected_rgb_exact=True,outside_target_rgb_exact=True,source_resampled=False,albedo_replaced=False,physical_relighting=False,intrinsics_truth=False,accepted=False,default_changed=False,limitations=['Saved model shading is an aligned hypothesis, not recovered physical illumination.','Achromatic darkening can damage real material color, highlights and texture.','Outline, label, pose, contact and background mismatch are not corrected.','Native visual comparison required; no numeric quality acceptance.'])
