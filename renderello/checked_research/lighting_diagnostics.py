"""Screen-space left/right gain diagnostic, not physical light estimation.

Requires declared paired prompts and aligned gain maps. It does not infer surface
normals, light position, scene geometry, source identity or publication readiness.
"""
import numpy as np


def paired_horizontal_gain(left_prompt_gain, right_prompt_gain, opaque, *, aligned_pair_reviewed=False):
    if aligned_pair_reviewed is not True:
        raise ValueError('explicit aligned pair review required; shapes cannot prove alignment')
    gains = [np.asarray(left_prompt_gain), np.asarray(right_prompt_gain)]
    if not isinstance(opaque,np.ndarray) or opaque.dtype!=np.bool_ or opaque.ndim!=2:
        raise ValueError('native boolean opaque mask required')
    if opaque.size>4_194_304 or min(opaque.shape)<4 or opaque.sum()<16:
        raise ValueError('nontrivial native support within4MP required')
    if any(g.shape!=opaque.shape or g.dtype.kind!='f' or not np.isfinite(g).all() or np.any(g<=0) for g in gains):
        raise ValueError('finite positive native floating gain maps required')
    yy,xx=np.indices(opaque.shape)
    centre=float(xx[opaque].mean());lo,hi=np.quantile(yy[opaque],[.15,.85])
    left=opaque&(xx<centre)&(yy>=lo)&(yy<=hi)
    right=opaque&(xx>=centre)&(yy>=lo)&(yy<=hi)
    if min(left.sum(),right.sum())<4:
        raise ValueError('both screen-space half regions need support')
    rows=[]
    for name,g in zip(['left_prompt','right_prompt'],gains):
        l,r=float(g[left].mean()),float(g[right].mean())
        rows.append(dict(prompt=name,left_mean_gain=l,right_mean_gain=r,left_minus_right=l-r))
    return dict(method='mean gain on opaque horizontal halves, central15..85percent vertical support',
                left_pixels=int(left.sum()),right_pixels=int(right.sum()),centre_x=centre,
                vertical_limits=[float(lo),float(hi)],rows=rows,
                contrast_change_right_minus_left=rows[1]['left_minus_right']-rows[0]['left_minus_right'],
                gain_mean_abs_difference=float(np.abs(gains[0]-gains[1])[opaque].mean()),
                physical_lighting_established=False,independent_direction_accuracy=None,
                publication_accepted=False,alignment_review_is_caller_declaration=True)
