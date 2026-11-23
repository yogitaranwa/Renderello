"""Opt-in achromatic edge brightening from reviewed same-material donor RGB.

A local brightness hypothesis, not inverse compositing or physical recovery.
Opaque edge RGB CAN change. Alpha and RGB outside the target stay exact.
Genuine contour shadows can be erased. There is no default pipeline integration.
"""
import math
from numbers import Real
import numpy as np
from scipy.ndimage import distance_transform_edt
from .candidate_diagnostics import native_rgb,native_alpha,digest
from .checked_research.detail_transfer import decode,encode


def lift_reviewed_edge(rgb,alpha,target,donor,protected,*,source_sha256,
        alpha_sha256,target_sha256,donor_sha256,protected_sha256,
        reviewed=False,material=None,review_note='',edge_width=4.,
        donor_distance=24.,strength=.5,maximum_gain=1.25,max_code_delta=24):
    native_rgb(rgb);native_alpha(alpha,rgb.shape[:2])
    if alpha.size>1_000_000:raise ValueError('1MP native budget; no resize')
    if reviewed is not True or material!='opaque':raise ValueError('explicit opaque same-material review required')
    if not isinstance(review_note,str) or not review_note.strip() or len(review_note)>2000:
        raise ValueError('bounded same-material review note required')
    for mask in (target,donor,protected):
        if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=alpha.shape or not mask.any():
            raise ValueError('nonempty native boolean target/donor/protected masks required')
    for a,h in [(rgb,source_sha256),(alpha,alpha_sha256),(target,target_sha256),(donor,donor_sha256),(protected,protected_sha256)]:
        if digest(a)!=h:raise ValueError('source/alpha/region binding mismatch')
    if np.any(target&(donor|protected)) or np.any(donor&protected):raise ValueError('review masks must be disjoint')
    if np.any(alpha[target]<128) or np.any(alpha[donor]!=255) or not np.any(alpha[protected]>0):
        raise ValueError('high-alpha targets, opaque donors and protected foreground required')
    for v,lo,hi in [(edge_width,0,8),(donor_distance,0,48),(strength,0,1)]:
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not math.isfinite(v) or not lo<v<=hi:
            raise ValueError('finite bounded real controls required')
    if isinstance(maximum_gain,(bool,np.bool_)) or not isinstance(maximum_gain,Real) or not math.isfinite(maximum_gain) or not 1<maximum_gain<=1.5:
        raise ValueError('maximum linear gain above1..1.5 required')
    if type(max_code_delta) is not int or not 1<=max_code_delta<=32:raise ValueError('integer code delta1..32 required')
    hard=alpha>=128
    if hard.all():raise ValueError('background required for boundary distance')
    edge=distance_transform_edt(hard)
    if np.any(edge[target]>edge_width):raise ValueError('target outside reviewed edge band')
    dist,idx=distance_transform_edt(~donor,return_indices=True)
    if np.any(dist[target]>donor_distance):raise ValueError('target lacks nearby reviewed same-material donors')
    s=decode(rgb[target]);near=decode(rgb[idx[0][target],idx[1][target]])
    weights=np.array([.2126,.7152,.0722]);ys=s@weights;yd=near@weights
    # Never darken an edge. A brighter donor alone does not prove contamination.
    ratio=np.maximum(1.,yd/np.maximum(ys,1e-12));ratio[ys<=1e-12]=1.
    bounded=np.minimum(ratio,maximum_gain);gain=1+strength*(bounded-1)
    result=encode(s*gain[:,None]);delta=result.astype(np.int16)-rgb[target].astype(np.int16)
    if np.any(delta<0):raise RuntimeError('brighten-only invariant failed')
    if np.any(delta>max_code_delta):raise ValueError('requested edge correction exceeds code-delta budget')
    out=rgb.copy();out[target]=result;changed=np.any(out!=rgb,axis=2)
    if np.any(changed&~target) or not np.array_equal(out[protected],rgb[protected]):raise RuntimeError('source protection invariant failed')
    return out,dict(schema='reviewed-edge-lift.v1',source_sha256=digest(rgb),alpha_sha256=digest(alpha),
        target_sha256=digest(target),donor_sha256=digest(donor),protected_sha256=digest(protected),
        output_sha256=digest(out),review_note=review_note,material=material,edge_width=float(edge_width),
        donor_distance=float(donor_distance),strength=float(strength),maximum_gain=float(maximum_gain),
        max_code_delta=max_code_delta,observed_maximum_code_delta=int(delta.max()),
        target_pixels=int(target.sum()),changed_pixels=int(changed.sum()),
        changed_opaque_pixels=int((changed&(alpha==255)).sum()),
        donor_gain_capped_pixels=int((ratio>maximum_gain).sum()),
        actual_maximum_linear_gain=float(gain.max()),brighten_only=True,
        alpha_changed=False,protected_rgb_exact=True,outside_target_rgb_exact=True,
        fully_source_rgb_exact=not bool(changed.any()),source_resampled=False,
        preencode_gamut_clipped_pixels=int(np.any(s*gain[:,None]>1,axis=1).sum()),
        accepted=False,default_pipeline_changed=False,physical_recovery=False,
        independent_ground_truth=False,
        limitations=['Same-material donors and spill are reviewer hypotheses, not detected facts.',
         'Brightening can erase genuine contour shadows, material variation and edge texture.',
         'Source foreground RGB is edited, including opaque pixels; it is not source-RGB locked.',
         'Gamut clipping can alter chromatic ratios; this is not a color-recovery guarantee.',
         'Alpha errors, geometry, background ghosts and label semantics are not corrected.',
         'Native light/dark inspection is required; numeric bounds are not quality acceptance.'])
