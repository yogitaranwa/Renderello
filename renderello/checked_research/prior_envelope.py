"""Declared reviewed-support envelope, not automatic segmentation or true alpha.
A candidate can only lose alpha outside a dilated prior-positive support. A prior
omission cannot be recovered. Native source coordinates and source RGB stay fixed.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

def constrain_alpha(candidate,reviewed_prior,*,native_pairing_reviewed=False,prior_support_reviewed=False,radius=2):
    if native_pairing_reviewed is not True or prior_support_reviewed is not True:
        raise ValueError('native pairing and prior support review required')
    if not isinstance(candidate,np.ndarray) or candidate.dtype!=np.uint8 or candidate.ndim!=2:
        raise ValueError('native uint8 candidate alpha required')
    if not isinstance(reviewed_prior,np.ndarray) or reviewed_prior.dtype!=np.uint8 or reviewed_prior.shape!=candidate.shape:
        raise ValueError('matching uint8 reviewed prior alpha required')
    h,w=candidate.shape
    if min(h,w)<2 or h*w>2_000_000:raise ValueError('native dimensions2..2MPrequired')
    if type(radius) is not int or not 0<=radius<=12:raise ValueError('integer radius0..12required')
    support=reviewed_prior>0
    if not support.any() or support.all() or not (candidate>0).any():raise ValueError('nontrivial reviewed support and candidate required')
    keep=distance_transform_edt(~support)<=radius
    out=np.where(keep,candidate,0).astype('uint8')
    return out,dict(method='candidate alpha restricted to declared prior-support distance envelope',radius=radius,
        candidate_support_pixels=int((candidate>0).sum()),result_support_pixels=int((out>0).sum()),
        removed_support_pixels=int(((candidate>0)&~keep).sum()),prior_support_pixels=int(support.sum()),
        prior_positive_lost_pixels=int((support&(out==0)).sum()),alpha_never_increased=True,
        native_coordinates_unchanged=True,prior_omission_recovery=False,semantic_completeness=None,
        boundary_accuracy=None,publication_accepted=False,production_promoted=False)
